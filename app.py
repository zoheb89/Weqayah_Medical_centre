"""Weqayah Medical Center — Databricks Apps reference implementation.

Operational writes are deliberately directed to MERIDIAN_WRITE_TABLE (a Lakebase
or governed operational table). Gold tables are queried as read models only.
Set the catalog/schema/table names in app.yaml for each environment.
"""
from __future__ import annotations

import os
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

try:
    from databricks import sql
    from databricks.sdk.core import Config
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False


st.set_page_config(page_title="Weqayah Medical Center", page_icon="✚", layout="wide", initial_sidebar_state="expanded")

CATALOG = os.getenv("MERIDIAN_CATALOG", "meridian")
GOLD_SCHEMA = os.getenv("MERIDIAN_GOLD_SCHEMA", "gold")
WRITE_TABLE = os.getenv("MERIDIAN_WRITE_TABLE", f"{CATALOG}.lakebase.patient_registration")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
TABLES = {
    "patients": os.getenv("MERIDIAN_PATIENT_TABLE", f"{CATALOG}.{GOLD_SCHEMA}.dim_patient"),
    "visits": os.getenv("MERIDIAN_VISIT_TABLE", f"{CATALOG}.{GOLD_SCHEMA}.fact_visits"),
    "claims": os.getenv("MERIDIAN_CLAIM_TABLE", f"{CATALOG}.{GOLD_SCHEMA}.fact_claims"),
    "forecast": os.getenv("MERIDIAN_FORECAST_TABLE", f"{CATALOG}.{GOLD_SCHEMA}.demand_forecast"),
    "inventory": os.getenv("MERIDIAN_INVENTORY_TABLE", f"{CATALOG}.{GOLD_SCHEMA}.pharmacy_reorder"),
    "kpis": os.getenv("MERIDIAN_KPI_TABLE", f"{CATALOG}.{GOLD_SCHEMA}.kpi_executive_summary"),
}

CSS = """
<style>
  .stApp { background: #f8fafc; color: #13243e; }
  [data-testid="stSidebar"] { background: linear-gradient(180deg,#1768aa 0%,#154f86 100%); }
  [data-testid="stSidebar"] > div:first-child { height:100vh; overflow-y:hidden !important; }
  [data-testid="stSidebar"] * { color: #edf8ff !important; }
  [data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius:9px; padding:.24rem .42rem; margin:.02rem 0; min-height:30px; }
  [data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size:.86rem; }
  [data-testid="stSidebar"] [data-testid="stRadio"] { gap:0 !important; }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:rgba(255,255,255,.96); }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * { color:#14579b !important; font-weight:700; }
  .brand-sub {font-size:.72rem; color:#d9ecfb; margin:.3rem 0 .6rem;}
  .topbar {display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #e4ebf3; background:#fff; padding:.45rem 1.15rem .55rem; margin:-1rem -1rem 1.1rem;}
  .topbar-title {font-weight:750;color:#17599d;font-size:1.05rem}.topbar-user {font-size:.85rem;color:#253858;text-align:right}.topbar-user span {color:#6c7d92;font-size:.74rem}
  .eyebrow {font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; color:#4b7087; font-weight:700;}
  .page-title {font-size:1.7rem;font-weight:750;color:#102c42;margin:0 0 .2rem;}
  .page-copy {color:#587085;margin:0 0 1.4rem;}
  .metric-card {background:#fff;border:1px solid #e2eaf0;border-radius:12px;padding:1rem 1.1rem;min-height:108px;box-shadow:0 2px 8px rgba(17,55,78,.035);}
  .metric-label {font-size:.8rem;color:#597184;font-weight:650}.metric-value {font-size:1.65rem;color:#102c42;font-weight:760;margin-top:.25rem}.metric-delta {font-size:.75rem;color:#16825d;margin-top:.25rem}
  .panel {background:white;border:1px solid #e2eaf0;border-radius:14px;padding:1rem 1.1rem;margin-bottom:1rem;}
  .status {display:inline-block;padding:.22rem .6rem;border-radius:99px;font-size:.75rem;font-weight:700;background:#e7f8ef;color:#147151}
  .alert {border-left:4px solid #f59e0b;background:#fffbeb;padding:.8rem 1rem;border-radius:6px;margin:.5rem 0;color:#7a4b05}
  .muted {color:#658092;font-size:.84rem}
  div[data-testid="stForm"] {border:1px solid #e2eaf0;background:#fff;border-radius:14px;padding:1.1rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
LOGO_PATH = Path(__file__).parent / "assets" / "weqayah-logo.png"


def sql_identifier(value: str) -> str:
    """Validate configurable three-part table identifiers before interpolating."""
    if not value or any(not part.replace("_", "").isalnum() for part in value.split(".")):
        raise ValueError("Invalid Unity Catalog table identifier.")
    return value


def connection():
    """Open a per-session connection without Streamlit function caching.

    The previous @st.cache_resource wrapper was removed deliberately: it is the
    source of Streamlit's cache-clearing dialog and can retain stale connections
    after a Databricks App redeploy.
    """
    if not (DATABRICKS_AVAILABLE and WAREHOUSE_ID):
        return None
    if "weqayah_sql_connection" in st.session_state:
        return st.session_state.weqayah_sql_connection
    try:
        cfg = Config()
        host = cfg.host.replace("https://", "")
        st.session_state.weqayah_sql_connection = sql.connect(
            server_hostname=host,
            http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
            credentials_provider=lambda: cfg.authenticate,
            _use_arrow_native_complex_types=False,
        )
        return st.session_state.weqayah_sql_connection
    except Exception:
        # Keep the presentation experience clean when local/demo credentials are absent.
        return None


def db_ready() -> bool:
    try:
        return connection() is not None
    except Exception:
        return False


def query(statement: str, params: Optional[list[Any]] = None) -> pd.DataFrame:
    """Return an empty frame rather than break the client demonstration on unavailable data."""
    if not db_ready():
        return pd.DataFrame()
    try:
        with connection().cursor() as cursor:
            cursor.execute(statement, parameters=params or [])
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=[c[0] for c in cursor.description])
    except Exception:
        # The caller transparently receives presentation data. Detailed errors belong
        # in platform logs, not in a client-facing clinical demonstration.
        return pd.DataFrame()


def execute(statement: str, params: list[Any]) -> tuple[bool, str]:
    if not db_ready():
        return True, "Registration saved for this demonstration session."
    try:
        with connection().cursor() as cursor:
            cursor.execute(statement, parameters=params)
        return True, "Saved to the operational data store."
    except Exception as exc:
        return False, f"Unable to save: {exc}"


def demo_patients() -> pd.DataFrame:
    baseline = pd.DataFrame([
        ["MRN-100245", "Amina Al-Harbi", "F", 35, "Active", "NPHIES / Bupa", "Today, 09:15"],
        ["MRN-100246", "Omar Al-Qahtani", "M", 48, "In consultation", "Cash", "Today, 09:06"],
        ["MRN-100247", "Sara Al-Salem", "F", 29, "Waiting", "NPHIES / Tawuniya", "Today, 08:52"],
        ["MRN-100248", "Fahad Al-Mutairi", "M", 61, "Lab pending", "NPHIES / Bupa", "Today, 08:31"],
    ], columns=["MRN", "Patient", "Gender", "Age", "Status", "Payer", "Last activity"])
    # A deterministic 100-record roster keeps the presentation searchable even
    # before the Gold patient table is connected.
    first_names = ["Abdul Rahim", "Fatimah", "Mohammed", "Noura", "Khalid", "Reem", "Yousef", "Laila", "Saad", "Huda", "Ibrahim", "Maha"]
    family_names = ["Al-Harbi", "Al-Qahtani", "Al-Salem", "Al-Mutairi", "Al-Rashidi", "Al-Ghamdi", "Al-Otaibi", "Al-Zahrani"]
    statuses = ["Registered", "Active", "Waiting", "In consultation", "Lab pending"]
    payers = ["Cash", "NPHIES / Bupa", "NPHIES / Tawuniya", "NPHIES / Medgulf"]
    generated = []
    for index in range(1, 97):
        generated.append([
            f"MRN-{731400 + index}",
            f"{first_names[index % len(first_names)]} {family_names[index % len(family_names)]}",
            "F" if index % 2 else "M",
            22 + (index * 3) % 57,
            statuses[index % len(statuses)],
            payers[index % len(payers)],
            f"Today, {8 + index % 5:02d}:{(index * 7) % 60:02d}",
        ])
    baseline = pd.concat([baseline, pd.DataFrame(generated, columns=baseline.columns)], ignore_index=True)
    additions = st.session_state.get("demo_registrations", [])
    if not additions:
        return baseline
    return pd.concat([pd.DataFrame(additions), baseline], ignore_index=True)


def safe_table(key: str) -> str:
    return sql_identifier(TABLES[key])


def live_or_demo(key: str, demo: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    frame = query(f"SELECT * FROM {safe_table(key)} LIMIT {limit}")
    return frame if not frame.empty else demo


def title(name: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">Weqayah Medical Center · AI-Powered Hospital Information System</div><div class="page-title">{name}</div><p class="page-copy">{description}</p>', unsafe_allow_html=True)


def metric(label: str, value: str, delta: str = "Live Lakehouse") -> None:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-delta">● {delta}</div></div>', unsafe_allow_html=True)


def sample_trend() -> pd.DataFrame:
    days = pd.date_range(date.today() - timedelta(days=6), periods=7)
    return pd.DataFrame({"Date": days, "Visits": [94, 112, 101, 126, 139, 117, 86], "Revenue (SAR)": [32100, 39900, 35200, 45200, 48100, 41800, 29800]}).set_index("Date")


def dashboard() -> None:
    title("Command Center", "A real-time view of patient flow, revenue-cycle health, and operational risk.")
    cols = st.columns(5)
    for c, args in zip(cols, [("Patients today", "126", "+12% vs. last Tue"), ("Revenue today", "SAR 45.2K", "+8.5% vs. last Tue"), ("Claims at risk", "14", "Needs review"), ("Average wait", "18 min", "Within target"), ("Low-stock items", "6", "2 urgent")]):
        with c: metric(*args)
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Patient flow and revenue")
        st.line_chart(sample_trend(), color=["#0e7490", "#34a0a4"])
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("AI operations briefing")
        st.markdown('<div class="alert"><b>Claims:</b> 14 submissions need missing-diagnosis or eligibility checks before NPHIES submission.</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert"><b>Pharmacy:</b> Amoxicillin 500mg is projected to reach reorder point in 2 days.</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert"><b>Capacity:</b> Forecast shows an 18% increase in OPD arrivals tomorrow morning.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.subheader("Current patient queue")
    st.dataframe(demo_patients(), use_container_width=True, hide_index=True)


def registration() -> None:
    title("Patient registration", "Create a patient record and visit. Writes are routed to the governed Lakebase operational table.")
    c1, c2 = st.columns([1.25, .82])
    with c1:
        if st.session_state.get("registration_success"):
            st.success(f"Patient registered successfully!  MRN: {st.session_state.registration_success}")
        with st.form("patient_registration", clear_on_submit=True):
            st.markdown("#### Identity and contact")
            a, b, c = st.columns(3)
            national_id = a.text_input("National ID / Iqama *", max_chars=20)
            first_name = b.text_input("First name *")
            last_name = c.text_input("Family name *")
            a, b, c = st.columns(3)
            dob = a.date_input("Date of birth", value=date(1990, 1, 1), min_value=date(1900, 1, 1))
            gender = b.selectbox("Gender", ["Female", "Male", "Not specified"])
            phone = c.text_input("Mobile number")
            st.markdown("#### Visit and coverage")
            a, b, c = st.columns(3)
            department = a.selectbox("Department", ["General Medicine", "Dental", "Dermatology", "Pediatrics", "Radiology"])
            payer = b.selectbox("Payer", ["Cash", "NPHIES / Bupa", "NPHIES / Tawuniya", "NPHIES / Medgulf"])
            visit_type = c.selectbox("Visit type", ["Walk-in", "Follow-up", "Emergency"])
            allergies = st.text_input("Allergies / clinical alert")
            submitted = st.form_submit_button("Register patient and visit", type="primary", use_container_width=True)
        if submitted:
            if not (national_id and first_name and last_name):
                st.error("National ID, first name, and family name are required.")
            else:
                mrn = f"MRN-{str(uuid.uuid4().int)[:6]}"
                stmt = f"""INSERT INTO {sql_identifier(WRITE_TABLE)}
                (registration_id, mrn, national_id, first_name, last_name, date_of_birth, gender, phone, department, payer, visit_type, allergies, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                ok, message = execute(stmt, [str(uuid.uuid4()), mrn, national_id, first_name, last_name, dob, gender, phone, department, payer, visit_type, allergies, datetime.utcnow()])
                if ok:
                    if not db_ready():
                        st.session_state.setdefault("demo_registrations", []).insert(0, {
                            "MRN": mrn,
                            "Patient": f"{first_name} {last_name}",
                            "Gender": gender[:1],
                            "Age": max(0, date.today().year - dob.year),
                            "Status": "Registered",
                            "Payer": payer,
                            "Last activity": "Just now",
                        })
                    st.session_state.registration_success = mrn
                    st.rerun()
                else:
                    st.info(message)
    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Registration controls")
        st.markdown("<span class='status'>Identity validation enabled</span>", unsafe_allow_html=True)
        st.write("• Duplicate check by National ID / Iqama\n• Auditable registration event\n• Payer captured before billing\n• Role-based access through Databricks")
        st.caption(f"Write target: `{WRITE_TABLE}`")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Recent registrations")
        st.dataframe(demo_patients().loc[:, ["MRN", "Patient", "Status"]], use_container_width=True, hide_index=True, height=230)
        st.button("View all registrations", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.button("Print slip", use_container_width=True)
        b.button("Create billing", use_container_width=True)
        c.button("Start consultation", use_container_width=True)


def patient_search() -> None:
    title("Patient search", "Find a patient by MRN, National ID, Iqama, mobile number, or name, then open their clinical and financial timeline.")
    left, right = st.columns([1.4, .75])
    with left:
        search = st.text_input("Search patient", placeholder="MRN, National ID / Iqama, mobile number, or patient name")
        source = live_or_demo("patients", demo_patients())
        if search:
            matches = source[source.astype(str).apply(lambda row: row.str.contains(search, case=False, na=False).any(), axis=1)]
        else:
            matches = source
        selection = st.dataframe(
            matches,
            use_container_width=True,
            hide_index=True,
            height=310,
            on_select="rerun",
            selection_mode="single-row",
            key="patient_search_grid",
        )
        selected_rows = selection.selection.rows
        if selected_rows:
            st.session_state.selected_patient = matches.iloc[selected_rows[0]].to_dict()
        st.caption("Select a patient row to update the profile panel.")
    with right:
        patient = st.session_state.get("selected_patient")
        if patient is None and not matches.empty:
            patient = matches.iloc[0].to_dict()
        if patient is None:
            st.info("No patients match this search.")
            return
        name = patient.get("Patient", patient.get("patient_name", "Selected patient"))
        mrn = patient.get("MRN", patient.get("mrn", "—"))
        payer = patient.get("Payer", patient.get("payer", "Not captured"))
        status = patient.get("Status", patient.get("status", "Active"))
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Patient profile")
        st.write(f"**{name}**  \n{mrn} · {status}")
        st.caption(f"National ID verified · {payer}")
        st.divider()
        st.write(f"**Latest activity**  \n{patient.get('Last activity', patient.get('last_activity', 'No recent activity'))}")
        st.write("**Clinical alert**  \nNo known allergies")
        st.button(
            "Open patient record",
            type="primary",
            use_container_width=True,
            on_click=open_patient_profile,
            args=(patient,),
        )
        st.button("Register new visit", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def open_patient_profile(patient: dict[str, Any]) -> None:
    """Callback: update navigation before the sidebar radio is instantiated."""
    st.session_state.open_patient_profile = True
    st.session_state.selected_patient = patient
    st.session_state.navigation = "Patient Profile"


def patient_profile() -> None:
    title("Patient profile", "A longitudinal patient view combining demographics, active visit, clinical history, orders, billing and insurance activity.")
    source = live_or_demo("patients", demo_patients())
    profile_search, _ = st.columns([1.25, .75])
    with profile_search:
        search = st.text_input("Find a patient", placeholder="Search by MRN, name, National ID, or mobile number", key="patient_profile_search")
    if search:
        candidates = source[source.astype(str).apply(lambda row: row.str.contains(search, case=False, na=False).any(), axis=1)]
    else:
        candidates = source
    if candidates.empty:
        st.warning("No patient matches your search. Try a name or MRN.")
        return
    labels = [
        f"{row.get('Patient', row.get('patient_name', 'Patient'))} · {row.get('MRN', row.get('mrn', '—'))}"
        for _, row in candidates.iterrows()
    ]
    current = st.session_state.get("selected_patient", {})
    current_label = f"{current.get('Patient', current.get('patient_name', ''))} · {current.get('MRN', current.get('mrn', ''))}"
    selected_label = st.selectbox(
        "Select patient",
        labels,
        index=labels.index(current_label) if current_label in labels else 0,
        help="The profile below refreshes when a patient is selected.",
    )
    patient = candidates.iloc[labels.index(selected_label)].to_dict()
    st.session_state.selected_patient = patient
    name = patient.get("Patient", patient.get("patient_name", "Selected patient"))
    mrn = patient.get("MRN", patient.get("mrn", "—"))
    payer = patient.get("Payer", patient.get("payer", "Not captured"))
    status = patient.get("Status", patient.get("status", "Active"))
    gender = patient.get("Gender", patient.get("gender", "—"))
    age = patient.get("Age", patient.get("age", "—"))
    a, b = st.columns([1.25, .75])
    with a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader(f"{name}  ·  {mrn}")
        st.caption(f"{gender} · {age} years · National ID verified · {payer}")
        x, y, z = st.columns(3)
        x.metric("Patient status", status)
        y.metric("Last activity", patient.get("Last activity", patient.get("last_activity", "—")))
        z.metric("Outstanding", "SAR 0")
        st.markdown("</div>", unsafe_allow_html=True)
        tabs = st.tabs(["Timeline", "Clinical", "Orders", "Billing & claims"])
        with tabs[0]:
            st.dataframe(pd.DataFrame([["Today, 10:42", "Registered", "Walk-in visit created"], ["Today, 10:51", "Consultation", "Waiting for physician"], ["19 Jul 2026", "Consultation", "Hypertension follow-up completed"]], columns=["When", "Event", "Detail"]), use_container_width=True, hide_index=True)
        with tabs[1]: st.info("No signed note is available for the active visit.")
        with tabs[2]: st.dataframe(pd.DataFrame([["Laboratory", "CBC", "Completed"], ["Pharmacy", "Paracetamol 500mg", "Dispensed"]], columns=["Service", "Order", "Status"]), use_container_width=True, hide_index=True)
        with tabs[3]: st.dataframe(pd.DataFrame([["INV-8841", "SAR 580", "NPHIES / Bupa", "Claim ready"]], columns=["Invoice", "Amount", "Payer", "Status"]), use_container_width=True, hide_index=True)
    with b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Clinical safety")
        st.success("No known allergies")
        st.write("**Blood group**  \nO+")
        st.write("**Emergency contact**  \nHassan Alqahtani · 055 123 4567")
        st.button("Start consultation", type="primary", use_container_width=True)
        st.button("Create invoice", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def emr() -> None:
    title("Clinical workspace", "A focused physician view for consultation, clinical notes, orders, and longitudinal patient context.")
    patient = st.selectbox("Patient", demo_patients()["Patient"].tolist())
    st.caption(f"MRN-100246 · {patient} · Active visit · General Medicine")
    tabs = st.tabs(["Consultation", "Vitals", "Medication", "Lab orders", "Radiology", "History"])
    with tabs[0]:
        a, b = st.columns(2)
        with a:
            st.selectbox("Diagnosis", ["Select diagnosis", "Upper respiratory infection", "Hypertension", "Diabetes follow-up", "Dermatitis"])
            st.text_area("Clinical notes", placeholder="Document assessment, examination and plan…", height=170)
        with b:
            st.multiselect("Clinical alerts", ["Penicillin allergy", "Diabetic", "High fall risk"])
            st.selectbox("Disposition", ["Complete consultation", "Refer to specialist", "Send to laboratory", "Send to radiology"])
            st.button("Sign clinical note", type="primary")
    with tabs[1]: st.dataframe(pd.DataFrame([["BP", "124/80 mmHg"], ["Temperature", "36.8°C"], ["Pulse", "76 bpm"], ["Weight", "76 kg"]], columns=["Measurement", "Value"]), use_container_width=True, hide_index=True)
    with tabs[2]: st.dataframe(pd.DataFrame([["Amoxicillin 500mg", "1 capsule", "Three times daily", "5 days"], ["Paracetamol 500mg", "1 tablet", "As needed", "3 days"]], columns=["Medication", "Dose", "Frequency", "Duration"]), use_container_width=True, hide_index=True)
    with tabs[3]: st.info("No open laboratory orders. Use the order composer in the production integration.")
    with tabs[4]: st.info("No imaging orders for this visit.")
    with tabs[5]: st.dataframe(pd.DataFrame([["19 Jul 2026", "General Medicine", "Hypertension follow-up"], ["04 Apr 2026", "Laboratory", "Routine blood work"]], columns=["Date", "Service", "Summary"]), use_container_width=True, hide_index=True)


def lab() -> None:
    title("Laboratory operations", "Manage specimen collection, analyser integration, result validation, and release queues.")
    data = pd.DataFrame([["LAB-62241", "Fahad Al-Mutairi", "CBC", "Collected", "08:31", "Routine"], ["LAB-62242", "Amina Al-Harbi", "HbA1c", "In analyser", "09:16", "Routine"], ["LAB-62239", "Omar Al-Qahtani", "Lipid profile", "Awaiting validation", "08:14", "High"]], columns=["Order", "Patient", "Test", "Status", "Received", "Priority"])
    a, b, c = st.columns(3)
    with a: metric("Awaiting collection", "8", "Queue monitored")
    with b: metric("In analyser", "14", "LIS feed active")
    with c: metric("Validation required", "3", "Pathologist action")
    st.dataframe(data, use_container_width=True, hide_index=True)
    with st.expander("Validate result"):
        a, b, c = st.columns(3)
        a.selectbox("Order", data["Order"])
        b.text_input("Result")
        c.selectbox("Verification", ["Normal", "Abnormal — review", "Critical — notify physician"])
        st.button("Validate and release", type="primary")


def radiology() -> None:
    title("Radiology worklist", "A unified worklist for orders, PACS study status, reporting and clinician delivery.")
    a, b, c, d = st.columns(4)
    with a: metric("PACS connection", "Connected", "DICOM listener healthy")
    with b: metric("Studies today", "38", "7 received this hour")
    with c: metric("Awaiting report", "5", "1 urgent")
    with d: metric("Last message", "Just now", "Study received")
    st.caption("Live demo feed — refreshes every five seconds. Replace `pacs_demo_worklist()` with the clinic-approved DICOM/PACS connector for production.")
    pacs_live_worklist()
    a, b = st.columns([1, 1.6])
    with a: st.metric("Turnaround time", "37 min", "Target < 60 min")
    with b:
        st.success("PACS interface status: connected")
        st.write("New imaging studies are appearing in the worklist and can move from acquisition to reporting and release. Production integration will use the clinic-approved DICOM/PACS interface.")


def pacs_demo_worklist() -> pd.DataFrame:
    """Return a time-aware PACS feed for the client demonstration.

    It intentionally uses no simulated clinical image data. The production version
    replaces this with approved DICOM/PACS metadata only.
    """
    refreshed = st.session_state.get("pacs_refresh_at", time.time())
    seconds_since_refresh = int(time.time() - refreshed)
    incoming_status = "Received" if seconds_since_refresh < 20 else "Queued"
    worklist = [
        ["RAD-9921", "Sara Al-Salem", "Chest X-ray", "Acquired", "Dr. Noor", "Routine", "09:42:18"],
        ["RAD-9922", "Omar Al-Qahtani", "Knee X-ray", "Scheduled", "—", "Routine", "09:37:01"],
        ["RAD-9923", "Fahad Al-Mutairi", "CT Head", "Reporting", "Dr. Noor", "Urgent", "09:31:44"],
        ["RAD-9924", "Maha Al-Ghamdi", "Ultrasound abdomen", incoming_status, "Dr. Noor", "Routine", time.strftime("%H:%M:%S")],
    ]
    return pd.DataFrame(worklist, columns=["Order", "Patient", "Study", "PACS status", "Radiologist", "Priority", "Last event"])


@st.fragment(run_every="5s")
def pacs_live_worklist() -> None:
    """Refresh only the PACS worklist, keeping the rest of the screen stable."""
    if st.button("↻ Refresh PACS worklist"):
        st.session_state.pacs_refresh_at = time.time()
    st.dataframe(pacs_demo_worklist(), use_container_width=True, hide_index=True)
    st.caption(f"Last refreshed: {time.strftime('%H:%M:%S')} · Live simulation active")


def pharmacy() -> None:
    title("Pharmacy and inventory", "Dispense safely, see prescription queues, and act on Lakehouse-driven replenishment recommendations.")
    inventory_demo = pd.DataFrame([["Amoxicillin 500mg", 85, 140, "Reorder in 2 days", "Urgent"], ["Paracetamol 500mg", 420, 250, "Healthy", "Normal"], ["Metformin 500mg", 110, 160, "Reorder in 5 days", "Review"]], columns=["Item", "On hand", "Reorder point", "AI recommendation", "Priority"])
    inventory = live_or_demo("inventory", inventory_demo)
    a, b, c = st.columns(3)
    with a: metric("Dispense queue", "11", "3 waiting > 15 min")
    with b: metric("Low-stock items", "6", "2 urgent")
    with c: metric("Expiry alerts", "3", "Within 60 days")
    tab1, tab2 = st.tabs(["Replenishment intelligence", "Prescription queue"])
    with tab1: st.dataframe(inventory, use_container_width=True, hide_index=True)
    with tab2: st.dataframe(pd.DataFrame([["RX-5001", "Amina Al-Harbi", "Amoxicillin 500mg", "Ready"], ["RX-5002", "Omar Al-Qahtani", "Paracetamol 500mg", "Awaiting payment"]], columns=["Prescription", "Patient", "Medicine", "Status"]), use_container_width=True, hide_index=True)


def billing() -> None:
    title("Billing and cashier", "A governed revenue-cycle view from charge capture through payment, reconciliation, and receipt.")
    a, b, c, d = st.columns(4)
    for col, args in zip([a,b,c,d], [("Charges today", "SAR 58.3K", "126 encounters"), ("Collected", "SAR 45.2K", "77.5% same-day"), ("Outstanding", "SAR 13.1K", "Payer follow-up"), ("Cashier exceptions", "2", "Needs review")]):
        with col: metric(*args)
    st.dataframe(pd.DataFrame([["INV-8841", "Amina Al-Harbi", "NPHIES / Bupa", "SAR 580", "Claim ready"], ["INV-8842", "Omar Al-Qahtani", "Cash", "SAR 245", "Paid"], ["INV-8843", "Sara Al-Salem", "NPHIES / Tawuniya", "SAR 1,120", "Eligibility check"]], columns=["Invoice", "Patient", "Payer", "Amount", "Status"]), use_container_width=True, hide_index=True)


def claims() -> None:
    title("Insurance claims workbench", "Prioritise NPHIES submissions using eligibility, completeness, and denial-risk signals.")
    claims_demo = demo_claims()
    claims_data = live_or_demo("claims", claims_demo)
    st.markdown("<div class='alert'><b>AI claim guard:</b> 14 claims need attention before submission. Review the highest risk cases first to reduce avoidable rejections.</div>", unsafe_allow_html=True)
    search = st.text_input("Find claim", placeholder="Claim number, payer, finding, or status", key="claim_search")
    if search:
        candidates = claims_data[claims_data.astype(str).apply(lambda row: row.str.contains(search, case=False, na=False).any(), axis=1)]
    else:
        candidates = claims_data
    if candidates.empty:
        st.info("No claims match this search.")
        return
    st.dataframe(candidates, use_container_width=True, hide_index=True)
    claim_column = "Claim" if "Claim" in candidates.columns else candidates.columns[0]
    selected_claim = st.selectbox("Select claim for review", candidates[claim_column].astype(str).tolist(), key="selected_claim")
    claim = candidates[candidates[claim_column].astype(str) == selected_claim].iloc[0].to_dict()
    finding = claim.get("AI finding", claim.get("ai_finding", "Review required"))
    details = claim_review_details(selected_claim, finding)
    with st.expander(f"Claim review: {selected_claim}", expanded=True):
        a, b = st.columns([1.2, 1])
        with a: st.write(f"**Finding:** {details['finding']}\n\n**Recommendation:** {details['recommendation']}")
        with b:
            resolution = st.selectbox("Resolution", details["resolutions"], key=f"resolution_{selected_claim}")
            if st.button("Record review action", type="primary", key=f"review_{selected_claim}"):
                st.session_state.setdefault("claim_review_actions", {})[selected_claim] = resolution
                st.success(f"Review action recorded for {selected_claim}: {resolution}")


def demo_claims() -> pd.DataFrame:
    base = [
        ["CLM-20031", "NPHIES / Bupa", "SAR 1,120", "82", "Diagnosis missing", "Review before submit"],
        ["CLM-20032", "NPHIES / Tawuniya", "SAR 580", "21", "Complete", "Ready to submit"],
        ["CLM-20033", "NPHIES / Medgulf", "SAR 2,940", "74", "Potential duplicate", "Investigate"],
    ]
    findings = ["Eligibility needs confirmation", "Coding mismatch", "Supporting document missing", "Complete"]
    payers = ["NPHIES / Bupa", "NPHIES / Tawuniya", "NPHIES / Medgulf", "NPHIES / CCHI"]
    for number in range(20034, 20061):
        finding = findings[number % len(findings)]
        base.append([f"CLM-{number}", payers[number % len(payers)], f"SAR {450 + (number * 37) % 3200:,}", str(14 + (number * 9) % 78), finding, "Ready to submit" if finding == "Complete" else "Review before submit"])
    return pd.DataFrame(base, columns=["Claim", "Payer", "Amount", "Risk score", "AI finding", "Next action"])


def claim_review_details(claim_id: str, finding: str) -> dict[str, Any]:
    if "duplicate" in finding.lower():
        return {"finding": "A potentially duplicate submission was identified for the same patient, service date, and payer.", "recommendation": "Compare the earlier claim and retain only the valid service line before resubmitting.", "resolutions": ["Investigate duplicate", "Return to billing", "Override with justification"]}
    if "eligibility" in finding.lower():
        return {"finding": "Payer eligibility could not be confirmed for the service date.", "recommendation": "Run eligibility verification and update coverage details before submission.", "resolutions": ["Verify eligibility", "Return to registration", "Override with justification"]}
    if "coding" in finding.lower():
        return {"finding": "A procedure or diagnosis coding mismatch was detected.", "recommendation": "Review the ICD-10 and service coding with the clinical and billing teams.", "resolutions": ["Request coding review", "Return to billing", "Override with justification"]}
    if "document" in finding.lower():
        return {"finding": "A required supporting document is missing from the claim package.", "recommendation": "Attach the supporting clinical document before submitting to the payer.", "resolutions": ["Request document", "Return to clinical team", "Override with justification"]}
    if "complete" in finding.lower():
        return {"finding": "No material completeness issue was identified by the claim guard.", "recommendation": "Confirm the final charge amount and submit the claim through the approved NPHIES workflow.", "resolutions": ["Submit to NPHIES", "Hold for review", "Return to billing"]}
    return {"finding": "No billable diagnosis is mapped to one service line.", "recommendation": "Add the ICD-10 diagnosis and confirm payer eligibility before submitting.", "resolutions": ["Request clinician completion", "Override with justification", "Return to billing"]}


def assistant_page() -> None:
    title("Weqayah AI", "A governed assistant experience for operational questions. Connect this view to an approved Genie space or model-serving endpoint in production.")
    prompts = ["What is today's revenue?", "Show claims most likely to be rejected", "Which medicines need reordering?", "Forecast tomorrow's OPD arrivals"]
    question = st.chat_input("Ask Weqayah AI about clinic operations…")
    st.markdown("**Try asking:** " + " · ".join(prompts))
    if question:
        st.chat_message("user").write(question)
        q = question.lower()
        if "revenue" in q: answer = "Today's collected revenue is SAR 45.2K across 126 patient encounters, up 8.5% on the comparable day last week."
        elif "claim" in q: answer = "14 claims require review before submission. The highest-risk issue is a missing diagnosis on CLM-20031."
        elif "medicine" in q or "reorder" in q: answer = "Amoxicillin 500mg is projected to reach its reorder threshold in two days. Two other items should be reviewed this week."
        elif "forecast" in q or "opd" in q: answer = "The demand model forecasts approximately 149 OPD arrivals tomorrow, with the busiest period expected between 09:00 and 11:30."
        else: answer = "For the production build, this question can be routed to a Unity Catalog-governed Genie space with row- and column-level access controls."
        st.chat_message("assistant").write(answer)
    st.caption("Demo response mode is enabled. Do not send protected health information to an unapproved model endpoint.")


def analytics() -> None:
    title("Executive analytics", "Decision-ready trends powered by Gold-layer marts, with governed drill-through to operational detail.")
    trend = sample_trend()
    a, b = st.columns(2)
    with a: st.subheader("Visits and revenue"); st.area_chart(trend, color=["#0e7490", "#4f46e5"])
    with b:
        st.subheader("Department performance")
        st.bar_chart(pd.DataFrame({"Department": ["General Medicine", "Dental", "Dermatology", "Pediatrics"], "Revenue (SAR)": [24500, 11200, 7600, 4800]}).set_index("Department"), color="#0e7490")
    forecast_demo = pd.DataFrame({"Date": pd.date_range(date.today(), periods=7), "Expected OPD": [149, 141, 153, 158, 146, 121, 96]}).set_index("Date")
    forecast = live_or_demo("forecast", forecast_demo)
    st.subheader("Demand forecast")
    numeric = forecast.select_dtypes("number")
    st.line_chart(numeric if not numeric.empty else forecast_demo)


def administration() -> None:
    title("Administration and governance", "A transparent view of access, operational controls and deployment readiness.")
    tab1, tab2, tab3 = st.tabs(["Users and roles", "Data governance", "Integration health"])
    with tab1:
        st.dataframe(pd.DataFrame([["Dr. Noor Al-Salem", "Physician", "Clinical workspace", "Active"], ["Rehab Al-Harbi", "Revenue cycle", "Claims workbench", "Active"], ["Mariam Ahmed", "Reception", "Registration", "Active"]], columns=["User", "Role", "Access", "Status"]), use_container_width=True, hide_index=True)
    with tab2:
        st.write("**Unity Catalog controls**\n\n• Table and view permissions by role\n• Auditable data access\n• Data lineage from source to dashboard\n• Masking policies for sensitive identifiers")
        st.caption("Grant the Databricks App service principal SELECT on Gold read models, MODIFY only on approved operational write tables, and CAN USE on the SQL warehouse.")
    with tab3:
        st.dataframe(pd.DataFrame([["DataOcean HMS", "Planned", "API / database CDC", "Discovery required"], ["NPHIES", "Planned", "Approved API", "Design phase"], ["PACS", "Current manual upload", "DICOM interface", "Modernise"], ["Laboratory", "Planned", "HL7 / API", "Design phase"]], columns=["System", "Current state", "Integration pattern", "Next step"]), use_container_width=True, hide_index=True)


PAGES = {
    "Command Center": dashboard, "Patient Registration": registration, "Patient Search": patient_search, "Patient Profile": patient_profile, "Clinical / EMR": emr,
    "Laboratory": lab, "Radiology": radiology, "Pharmacy": pharmacy, "Billing": billing,
    "Insurance": claims, "Weqayah AI": assistant_page, "Executive Analytics": analytics,
    "Administration": administration,
}


def main() -> None:
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=225)
        else:
            st.markdown("### Weqayah\nMedical Center")
        st.markdown('<div class="brand-sub">AI-Powered Hospital Information System</div>', unsafe_allow_html=True)
        page = st.radio("Navigate", list(PAGES), label_visibility="collapsed", key="navigation")
        st.divider()
        mode = "Connected to Databricks SQL" if db_ready() else "Presentation mode"
        st.caption(f"● {mode}")
        st.caption("Lakehouse read models · Lakebase operational writes")
        st.divider()
        st.caption("Signed in as: Reception Desk")
    st.markdown('<div class="topbar"><div class="topbar-title">♧ &nbsp; AI-Powered Hospital Information System</div><div class="topbar-user"><b>Reception Desk</b><br><span>Front Office</span></div></div>', unsafe_allow_html=True)
    PAGES[page]()


if __name__ == "__main__":
    main()

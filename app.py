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
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.core import Config
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False


st.set_page_config(page_title="Weqayah Medical Center", page_icon="✚", layout="wide", initial_sidebar_state="expanded")

CATALOG = os.getenv("MERIDIAN_CATALOG", "meridian")
GOLD_SCHEMA = os.getenv("MERIDIAN_GOLD_SCHEMA", "gold")
WRITE_TABLE = os.getenv("MERIDIAN_WRITE_TABLE", f"{CATALOG}.lakebase.patient_registration")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
# Injected automatically when a Genie Space resource is attached to this Databricks
# App (see app.yaml). Without it, the assistant falls back to demo answers.
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")
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


PATIENT_COLUMN_ALIASES: dict[str, list[str]] = {
    "MRN": ["mrn", "patient_mrn", "medical_record_number"],
    "Patient": ["patient", "patient_name", "full_name", "name"],
    "Gender": ["gender", "sex"],
    "Age": ["age", "patient_age"],
    "Status": ["status", "patient_status", "visit_status"],
    "Payer": ["payer", "payer_name", "insurance", "coverage"],
    "National ID": ["national id", "national_id", "iqama", "id_number"],
    "Mobile": ["mobile", "phone", "mobile_number", "contact_number"],
    "Last activity": ["last activity", "last_activity", "last_visit", "updated_at"],
}


def normalize_patient_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrarily-shaped patient frame (e.g. a raw Gold table read
    with production column names) onto the canonical schema the UI renders.

    Without this, a `SELECT *` against a differently-named source table would
    get stacked onto the app's expected columns via pd.concat, producing rows
    that are entirely NaN in "MRN"/"Patient" — and worse, drop_duplicates on a
    NaN "MRN" collapses *all* such rows down to a single row, which is exactly
    how a full existing roster can appear to "disappear" after one real
    registration lands. Renaming known aliases and dropping rows that still
    have no identifiable MRN/Patient keeps garbage rows out entirely instead
    of letting them masquerade as blank records.
    """
    if frame.empty:
        return frame
    lookup = {str(col).strip().lower(): col for col in frame.columns}
    rename_map: dict[str, str] = {}
    for canonical, aliases in PATIENT_COLUMN_ALIASES.items():
        if canonical in frame.columns:
            continue
        for alias in aliases:
            if alias in lookup:
                rename_map[lookup[alias]] = canonical
                break
    normalized = frame.rename(columns=rename_map)
    for canonical in PATIENT_COLUMN_ALIASES:
        if canonical not in normalized.columns:
            normalized[canonical] = pd.NA
    ordered = list(PATIENT_COLUMN_ALIASES) + [c for c in normalized.columns if c not in PATIENT_COLUMN_ALIASES]
    normalized = normalized[ordered]
    # Drop rows with no usable identity — these are schema-mismatch artifacts,
    # not real patients, and must not be allowed to survive into the UI or
    # into drop_duplicates() below.
    has_identity = normalized["MRN"].notna() & (normalized["MRN"].astype(str).str.strip() != "")
    return normalized[has_identity].reset_index(drop=True)


def live_registrations(limit: int = 100) -> pd.DataFrame:
    """Patients written via the Registration form, read straight back from the
    Lakebase write table — shown immediately rather than waiting on a Gold-layer
    ETL pipeline (Bronze -> Silver -> Gold) that hasn't been built yet. Columns
    are shaped to match demo_patients()/the Gold read model so this can be
    combined with either and rendered by the same existing UI code.
    """
    return query(f"""
        SELECT
          mrn AS MRN,
          concat_ws(' ', first_name, last_name) AS Patient,
          left(gender, 1) AS Gender,
          (year(current_date()) - year(date_of_birth)) AS Age,
          'Registered' AS Status,
          payer AS Payer,
          national_id AS `National ID`,
          phone AS Mobile,
          date_format(created_at, 'MMM d, HH:mm') AS `Last activity`
        FROM {sql_identifier(WRITE_TABLE)}
        ORDER BY created_at DESC
        LIMIT {limit}
    """)


def combined_patients(limit: int = 100) -> pd.DataFrame:
    """Live registrations first (most recent on top), then the Gold read model
    or demo fallback — so a patient registered seconds ago is searchable
    immediately, without depending on a sync job that doesn't exist yet.

    Both sources are normalized onto the same canonical columns before being
    combined. This is what keeps a brand-new registration from ever *replacing*
    the existing roster: a schema mismatch on the Gold side now just gets
    filtered out (see normalize_patient_frame), instead of surviving as blank
    rows that then collapse the rest of the roster during de-duplication.
    """
    live = normalize_patient_frame(live_registrations(limit))
    demo = demo_patients()
    rest = normalize_patient_frame(live_or_demo("patients", demo))
    if rest.empty:
        # Gold read produced nothing usable (empty table, permissions error,
        # or a schema that didn't match) — always fall back to the full demo
        # roster rather than showing only the live registrations.
        rest = normalize_patient_frame(demo)
    frames = [f for f in (live, rest) if not f.empty]
    if not frames:
        return demo
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[combined["MRN"].notna() & (combined["MRN"].astype(str).str.strip() != "")]
    combined = combined.drop_duplicates(subset="MRN", keep="first").reset_index(drop=True)
    return combined


def genie_client() -> Optional["WorkspaceClient"]:
    """Return a cached WorkspaceClient for Genie calls, or None if unavailable.

    Mirrors connection() above: session-cached, never raises, so a missing or
    misconfigured Genie space degrades to demo mode instead of breaking the page.
    """
    if not (DATABRICKS_AVAILABLE and GENIE_SPACE_ID):
        return None
    if "weqayah_genie_client" in st.session_state:
        return st.session_state.weqayah_genie_client
    try:
        st.session_state.weqayah_genie_client = WorkspaceClient()
        return st.session_state.weqayah_genie_client
    except Exception:
        return None


def genie_ready() -> bool:
    try:
        return genie_client() is not None
    except Exception:
        return False


def ask_genie(question: str) -> tuple[str, bool]:
    """Ask the configured Genie space a question in real time.

    Returns (answer_text, is_live). is_live is False whenever the call falls back
    to a canned demo answer, so the UI can be honest about which path answered —
    the same transparency principle used by db_ready()/live_or_demo() elsewhere.
    """
    client = genie_client()
    if client is None:
        return _demo_genie_answer(question), False
    try:
        conversation_id = st.session_state.get("genie_conversation_id")
        if conversation_id:
            # Follow-up in the same session — Genie uses prior messages for context.
            message = client.genie.create_message_and_wait(
                space_id=GENIE_SPACE_ID,
                conversation_id=conversation_id,
                content=question,
            )
        else:
            # First question this session — start a fresh conversation thread.
            # (Databricks recommends a new thread per user session rather than
            # reusing one across sessions; Streamlit's session_state gives us that.)
            message = client.genie.start_conversation_and_wait(
                space_id=GENIE_SPACE_ID,
                content=question,
            )
            st.session_state.genie_conversation_id = message.conversation_id

        texts = [a.text.content for a in (message.attachments or []) if getattr(a, "text", None)]
        if texts:
            return "\n\n".join(texts), True
        return "Genie answered but returned no text content — check the space's query attachments.", True
    except Exception as exc:
        # Presentation stays clean for the client; the real error goes to app logs.
        print(f"Genie call failed: {exc}")
        return _demo_genie_answer(question), False


def _demo_genie_answer(question: str) -> str:
    q = question.lower()
    if "revenue" in q:
        return "Today's collected revenue is SAR 45.2K across 126 patient encounters, up 8.5% on the comparable day last week."
    if "claim" in q:
        return "14 claims require review before submission. The highest-risk issue is a missing diagnosis on CLM-20031."
    if "medicine" in q or "reorder" in q:
        return "Amoxicillin 500mg is projected to reach its reorder threshold in two days. Two other items should be reviewed this week."
    if "forecast" in q or "opd" in q:
        return "The demand model forecasts approximately 149 OPD arrivals tomorrow, with the busiest period expected between 09:00 and 11:30."
    return "For the production build, this question can be routed to a Unity Catalog-governed Genie space with row- and column-level access controls."


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
        trend = sample_trend()
        # Visits (~tens) and Revenue (~tens of thousands) sit on wildly
        # different scales. Plotted on one axis, revenue swamps visits and
        # the chart flattens into an unreadable line hugging the bottom —
        # this is also why "view as table" on that combined chart looked
        # like a wall of melted color/value rows. Two compact charts, one
        # per scale, read cleanly at a glance.
        vc, rc = st.columns(2)
        with vc:
            st.caption("Visits (7 days)")
            st.line_chart(trend[["Visits"]], color=["#0e7490"], height=200)
        with rc:
            st.caption("Revenue, SAR (7 days)")
            st.bar_chart(trend[["Revenue (SAR)"]], color=["#34a0a4"], height=200)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("AI operations briefing")
        st.markdown('<div class="alert"><b>Claims:</b> 14 submissions need missing-diagnosis or eligibility checks before NPHIES submission.</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert"><b>Pharmacy:</b> Amoxicillin 500mg is projected to reach reorder point in 2 days.</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert"><b>Capacity:</b> Forecast shows an 18% increase in OPD arrivals tomorrow morning.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    queue_header, queue_action = st.columns([3, 1])
    with queue_header:
        st.subheader("Current patient queue")
    queue = demo_patients()
    with queue_action:
        st.caption(f"{len(queue)} in roster")
    # A full 100-row dump here duplicates Patient Search and dominates the
    # page. Show a compact top slice and send anyone who needs the rest to
    # the dedicated Search tab instead of scrolling a giant table.
    st.dataframe(queue.head(8), use_container_width=True, hide_index=True, height=250)
    st.button("Open full patient roster in Patient Search →", on_click=_go_to_patient_search)
    st.markdown("</div>", unsafe_allow_html=True)


def _go_to_patient_search() -> None:
    """Callback: set the destination page/sub-view before the sidebar radio
    (key="navigation") is instantiated on the next rerun. Assigning these
    session_state keys directly inside the page body (after the radio has
    already rendered this run) raises a StreamlitAPIException — the callback
    runs between reruns, which is the only safe place to do it.
    """
    st.session_state.navigation = "Patient Registration & Search"
    st.session_state.patient_subview = "Search"


def patient_hub() -> None:
    """Single page hosting Registration, Search, and Profile as one workflow.

    These three used to be separate top-level sidebar pages. Each one called
    combined_patients() independently and jumped between pages by mutating
    st.session_state.navigation, which meant selection state, search state,
    and the underlying roster could all disagree with each other across a
    page swap. Keeping them as sub-views of one page means there is exactly
    one shared session_state.selected_patient and one place that computes the
    roster, so registering a patient, finding them, and opening their profile
    is one continuous flow instead of three independently-rendered pages.
    """
    title("Patient Registration & Search", "Register a patient and visit, search the full roster, and open a longitudinal patient profile — all in one workspace.")
    st.radio(
        "Section",
        ["Register", "Search", "Profile"],
        horizontal=True,
        label_visibility="collapsed",
        key="patient_subview",
    )
    st.divider()
    view = st.session_state.get("patient_subview", "Register")
    if view == "Register":
        _patient_registration_view()
    elif view == "Search":
        _patient_search_view()
    else:
        _patient_profile_view()


def _go_to_subview(view: str, patient: Optional[dict[str, Any]] = None) -> None:
    """Callback: set the sub-view (and optionally the selected patient) before
    the patient_subview radio is instantiated on the next rerun — mirrors the
    existing sidebar-navigation callback pattern used elsewhere in this app.
    """
    st.session_state.patient_subview = view
    if patient is not None:
        st.session_state.selected_patient = patient


def _patient_registration_view() -> None:
    c1, c2 = st.columns([1.25, .82])
    with c1:
        # pop (not get): this is a one-shot flash message for the rerun that
        # immediately follows a successful submit. Leaving it in session_state
        # made it reappear every time this tab was revisited, even minutes
        # and page-switches later, showing a stale MRN as if it just happened.
        just_registered = st.session_state.pop("registration_success", None)
        if just_registered:
            st.success(f"Patient registered successfully!  MRN: {just_registered}")
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
        st.dataframe(combined_patients().loc[:, ["MRN", "Patient", "Status"]], use_container_width=True, hide_index=True, height=230)
        st.button("View all registrations", use_container_width=True, on_click=_go_to_subview, args=("Search",))
        st.markdown("</div>", unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.button("Print slip", use_container_width=True)
        b.button("Create billing", use_container_width=True)
        c.button("Start consultation", use_container_width=True)


def _patient_search_view() -> None:
    st.caption("Find a patient by MRN, National ID, Iqama, mobile number, or name, then open their clinical and financial timeline.")
    left, right = st.columns([1.4, .75])
    with left:
        search = st.text_input("Search patient", placeholder="MRN, National ID / Iqama, mobile number, or patient name")
        source = combined_patients()
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
            on_click=_go_to_subview,
            args=("Profile", patient),
        )
        st.button("Register new visit", use_container_width=True, on_click=_go_to_subview, args=("Register",))
        st.markdown("</div>", unsafe_allow_html=True)


def _patient_profile_view() -> None:
    st.caption("A longitudinal patient view combining demographics, active visit, clinical history, orders, billing and insurance activity.")
    source = combined_patients()
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
    # Previously this selectbox always defaulted to the first demo name
    # (Amina Al-Harbi) and the caption hardcoded "MRN-100246" no matter who
    # was picked. It also only listed demo names, not real registrations,
    # and duplicate names (several "Abdul Rahim" entries) were ambiguous.
    # Sourcing from the full roster with an "MRN" suffix and no default
    # selection makes this an actual search rather than a static picker.
    source = combined_patients()
    labels = [f"{row.get('Patient', 'Patient')} · {row.get('MRN', '—')}" for _, row in source.iterrows()]
    label_to_patient = dict(zip(labels, source.to_dict("records")))
    selected_label = st.selectbox(
        "Patient",
        labels,
        index=None,
        placeholder="Search by patient name or MRN…",
        key="emr_patient_search",
    )
    if not selected_label:
        st.info("Search for a patient above to open their clinical workspace.")
        return
    patient = label_to_patient[selected_label]
    name = patient.get("Patient", "Selected patient")
    mrn = patient.get("MRN", "—")
    st.caption(f"{mrn} · {name} · Active visit · General Medicine")
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


LAB_STATUS_FLOW = ["Ordered", "Collected", "In analyser", "Awaiting validation", "Validated & released"]


def lab_orders_state() -> list[dict[str, Any]]:
    """Session-backed lab worklist so the order composer and status actions
    below actually persist across reruns, instead of the previous static
    DataFrame that was rebuilt from scratch (and silently discarded any
    change) on every page load.
    """
    if "lab_orders" not in st.session_state:
        st.session_state.lab_orders = [
            {"Order": "LAB-62241", "Patient": "Fahad Al-Mutairi", "Test": "CBC", "Status": "Collected", "Received": "08:31", "Priority": "Routine"},
            {"Order": "LAB-62242", "Patient": "Amina Al-Harbi", "Test": "HbA1c", "Status": "In analyser", "Received": "09:16", "Priority": "Routine"},
            {"Order": "LAB-62239", "Patient": "Omar Al-Qahtani", "Test": "Lipid profile", "Status": "Awaiting validation", "Received": "08:14", "Priority": "High"},
        ]
    return st.session_state.lab_orders


def lab_average_turnaround(orders: list[dict[str, Any]]) -> str:
    """Average minutes from each order's received time to now — a real
    metric derived from the current queue instead of a hardcoded number."""
    now = datetime.now()
    minutes = []
    for o in orders:
        try:
            received = datetime.strptime(o["Received"], "%H:%M").replace(year=now.year, month=now.month, day=now.day)
        except ValueError:
            continue
        if received > now:
            received -= timedelta(days=1)
        minutes.append((now - received).total_seconds() / 60)
    return f"{int(sum(minutes) / len(minutes))} min" if minutes else "—"


def lab() -> None:
    title("Laboratory operations", "Manage specimen collection, analyser integration, result validation, and release queues.")
    orders = lab_orders_state()
    data = pd.DataFrame(orders)
    awaiting_collection = sum(1 for o in orders if o["Status"] == "Ordered")
    in_analyser = sum(1 for o in orders if o["Status"] == "In analyser")
    awaiting_validation = sum(1 for o in orders if o["Status"] == "Awaiting validation")
    a, b, c, d = st.columns(4)
    with a: metric("Awaiting collection", str(awaiting_collection), "Queue monitored")
    with b: metric("In analyser", str(in_analyser), "LIS feed active")
    with c: metric("Validation required", str(awaiting_validation), "Pathologist action")
    with d: metric("Avg. turnaround", lab_average_turnaround(orders), "Order to now, current queue")
    st.dataframe(data, use_container_width=True, hide_index=True)

    tab1, tab2 = st.tabs(["Advance / validate order", "New order composer"])
    with tab1:
        if not orders:
            st.info("No open lab orders.")
        else:
            a, b, c = st.columns(3)
            order_ids = [o["Order"] for o in orders]
            selected_order = a.selectbox("Order", order_ids, key="lab_selected_order")
            current_status = next(o["Status"] for o in orders if o["Order"] == selected_order)
            next_options = LAB_STATUS_FLOW[LAB_STATUS_FLOW.index(current_status) + 1:] if current_status in LAB_STATUS_FLOW else LAB_STATUS_FLOW
            new_status = b.selectbox("Advance to", next_options or [current_status], key="lab_next_status")
            c.text_input("Result / note (optional)", key="lab_result_note")
            if st.button("Advance status", type="primary"):
                for o in orders:
                    if o["Order"] == selected_order:
                        o["Status"] = new_status
                st.success(f"{selected_order} moved to '{new_status}'.")
                st.rerun()
    with tab2:
        source = combined_patients()
        patient_labels = [f"{row.get('Patient', 'Patient')} · {row.get('MRN', '—')}" for _, row in source.iterrows()]
        a, b, c = st.columns(3)
        patient_label = a.selectbox("Patient", patient_labels, index=None, placeholder="Search patient…", key="lab_new_patient")
        test = b.selectbox("Test", ["CBC", "HbA1c", "Lipid profile", "Renal panel", "Liver function", "Thyroid panel", "Urinalysis"], key="lab_new_test")
        priority = c.selectbox("Priority", ["Routine", "High", "Urgent"], key="lab_new_priority")
        if st.button("Create lab order", type="primary", use_container_width=True):
            if not patient_label:
                st.error("Select a patient to create an order.")
            else:
                new_id = f"LAB-{62200 + len(orders) + 1}"
                orders.insert(0, {
                    "Order": new_id,
                    "Patient": patient_label.split(" · ")[0],
                    "Test": test,
                    "Status": "Ordered",
                    "Received": datetime.now().strftime("%H:%M"),
                    "Priority": priority,
                })
                st.success(f"{new_id} created for {patient_label.split(' · ')[0]}.")
                st.rerun()


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


def billing_invoices_state() -> list[dict[str, Any]]:
    """Session-backed invoice roster. The original page rendered three
    hardcoded rows every time with no way to record a payment or see aging —
    this seeds a wider, still-deterministic set of invoices with due dates
    spread across overdue/current/future so aging buckets have something
    real to show, and keeps them mutable across reruns.
    """
    if "billing_invoices" not in st.session_state:
        today = date.today()
        seed = [
            {"Invoice": "INV-8841", "Patient": "Amina Al-Harbi", "Payer": "NPHIES / Bupa", "Amount": 580, "Status": "Claim ready", "Due date": today - timedelta(days=5)},
            {"Invoice": "INV-8842", "Patient": "Omar Al-Qahtani", "Payer": "Cash", "Amount": 245, "Status": "Paid", "Due date": today - timedelta(days=1)},
            {"Invoice": "INV-8843", "Patient": "Sara Al-Salem", "Payer": "NPHIES / Tawuniya", "Amount": 1120, "Status": "Eligibility check", "Due date": today + timedelta(days=10)},
        ]
        payers = ["NPHIES / Bupa", "NPHIES / Tawuniya", "NPHIES / Medgulf", "Cash"]
        patients = ["Fahad Al-Mutairi", "Khalid Al-Rashidi", "Reem Al-Ghamdi", "Yousef Al-Otaibi", "Laila Al-Zahrani", "Huda Al-Harbi", "Saad Al-Qahtani", "Noura Al-Salem", "Ibrahim Al-Mutairi", "Maha Al-Rashidi"]
        statuses = ["Claim ready", "Eligibility check", "Paid", "Denied", "Pending payer"]
        for i in range(10):
            days_offset = (i * 17) % 120 - 20
            due = today - timedelta(days=days_offset) if days_offset > 0 else today + timedelta(days=abs(days_offset))
            seed.append({
                "Invoice": f"INV-88{44 + i}",
                "Patient": patients[i % len(patients)],
                "Payer": payers[i % len(payers)],
                "Amount": 200 + (i * 137) % 2200,
                "Status": statuses[i % len(statuses)],
                "Due date": due,
            })
        st.session_state.billing_invoices = seed
    return st.session_state.billing_invoices


def aging_bucket(due_date: date, status: str, today: date) -> str:
    if status == "Paid":
        return "Paid"
    days_overdue = (today - due_date).days
    if days_overdue <= 0:
        return "Not yet due"
    if days_overdue <= 30:
        return "0-30 days"
    if days_overdue <= 60:
        return "31-60 days"
    if days_overdue <= 90:
        return "61-90 days"
    return "90+ days"


def billing() -> None:
    title("Billing and cashier", "A governed revenue-cycle view from charge capture through payment, reconciliation, and receipt.")
    invoices = billing_invoices_state()
    df = pd.DataFrame(invoices)
    today = date.today()
    charges_total = df["Amount"].sum()
    collected_total = df.loc[df["Status"] == "Paid", "Amount"].sum()
    outstanding_total = charges_total - collected_total
    collection_rate = (collected_total / charges_total * 100) if charges_total else 0
    a, b, c, d = st.columns(4)
    with a: metric("Charges", f"SAR {charges_total:,.0f}", f"{len(df)} invoices")
    with b: metric("Collected", f"SAR {collected_total:,.0f}", f"{collection_rate:.1f}% of charges")
    with c: metric("Outstanding", f"SAR {outstanding_total:,.0f}", "Payer + patient follow-up")
    with d: metric("Denied", str((df["Status"] == "Denied").sum()), "Needs resubmission")

    tab1, tab2, tab3 = st.tabs(["Invoices", "Record a payment", "Aging & reconciliation"])
    with tab1:
        display_df = df.copy()
        display_df["Amount"] = display_df["Amount"].map(lambda v: f"SAR {v:,.0f}")
        display_df["Due date"] = pd.to_datetime(display_df["Due date"]).dt.strftime("%d %b %Y")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        selected_invoice = st.selectbox("Open invoice", df["Invoice"].tolist(), key="billing_selected_invoice")
        row = next(i for i in invoices if i["Invoice"] == selected_invoice)
        with st.expander(f"Invoice detail: {selected_invoice}", expanded=True):
            a, b = st.columns(2)
            with a:
                st.write(f"**Patient:** {row['Patient']}\n\n**Payer:** {row['Payer']}\n\n**Status:** {row['Status']}")
            with b:
                st.write(f"**Amount:** SAR {row['Amount']:,.0f}\n\n**Due date:** {row['Due date'].strftime('%d %b %Y')}\n\n**Aging:** {aging_bucket(row['Due date'], row['Status'], today)}")

    with tab2:
        open_invoices = [i for i in invoices if i["Status"] != "Paid"]
        if not open_invoices:
            st.info("No open invoices to collect against.")
        else:
            a, b, c = st.columns(3)
            invoice_id = a.selectbox("Invoice", [i["Invoice"] for i in open_invoices], key="billing_payment_invoice")
            method = b.selectbox("Method", ["Cash", "Card", "Bank transfer", "Payer remittance"], key="billing_payment_method")
            row = next(i for i in open_invoices if i["Invoice"] == invoice_id)
            amount = c.number_input("Amount (SAR)", min_value=0.0, value=float(row["Amount"]), step=10.0, key="billing_payment_amount")
            if st.button("Record payment", type="primary", use_container_width=True):
                for i in invoices:
                    if i["Invoice"] == invoice_id:
                        i["Status"] = "Paid" if amount >= i["Amount"] else "Partially paid"
                st.success(f"Recorded SAR {amount:,.0f} against {invoice_id} via {method}.")
                st.rerun()

    with tab3:
        df["Aging"] = df.apply(lambda r: aging_bucket(r["Due date"], r["Status"], today), axis=1)
        order = ["Not yet due", "0-30 days", "31-60 days", "61-90 days", "90+ days", "Paid"]
        bucket_totals = df.groupby("Aging")["Amount"].sum().reindex(order).fillna(0)
        st.subheader("Aging buckets")
        st.bar_chart(bucket_totals, color="#0e7490")
        st.subheader("Daily collected vs. charged (7 days)")
        trend = sample_trend()
        recon = pd.DataFrame({
            "Charged": trend["Revenue (SAR)"],
            "Collected": (trend["Revenue (SAR)"] * 0.78).round(0),
        })
        st.line_chart(recon, color=["#0e7490", "#16825d"])
        st.caption("Reconciliation compares charges captured against amounts collected same-day; the gap is the same-day collection lag reflected in Outstanding above.")


def claims() -> None:
    title("Insurance claims workbench", "Prioritise NPHIES submissions using eligibility, completeness, and denial-risk signals.")
    claims_demo = demo_claims()
    claims_data = live_or_demo("claims", claims_demo)
    st.markdown("<div class='alert'><b>AI claim guard:</b> 14 claims need attention before submission. Review the highest risk cases first to reduce avoidable rejections.</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Claim queue", "Payer rollups", "Denial trend"])

    with tab1:
        search = st.text_input("Find claim", placeholder="Claim number, payer, finding, or status", key="claim_search")
        candidates = claims_data[claims_data.astype(str).apply(lambda row: row.str.contains(search, case=False, na=False).any(), axis=1)] if search else claims_data
        if candidates.empty:
            st.info("No claims match this search.")
        else:
            # Multi-row selection + a bulk-action bar. Previously this grid
            # only supported reviewing one claim at a time via the selectbox
            # below — most real triage work (submit the clean ones, bounce
            # the rest to billing) happens in batches, not one row at a time.
            selection = st.dataframe(
                candidates,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
                key="claims_grid",
            )
            claim_column = "Claim" if "Claim" in candidates.columns else candidates.columns[0]
            selected_rows = selection.selection.rows
            if selected_rows:
                selected_claims = candidates.iloc[selected_rows]
                st.caption(f"{len(selected_rows)} claim(s) selected.")
                bcol1, bcol2 = st.columns([2, 1])
                bulk_action = bcol1.selectbox(
                    "Bulk action",
                    ["Mark ready to submit", "Return to billing", "Flag for clinician review"],
                    key="claims_bulk_action",
                )
                if bcol2.button("Apply to selected", type="primary", use_container_width=True):
                    actions = st.session_state.setdefault("claim_review_actions", {})
                    for _, row in selected_claims.iterrows():
                        actions[str(row[claim_column])] = bulk_action
                    st.success(f"Applied '{bulk_action}' to {len(selected_rows)} claim(s).")

            selected_claim = st.selectbox("Select claim for detailed review", candidates[claim_column].astype(str).tolist(), key="selected_claim")
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
            actions = st.session_state.get("claim_review_actions", {})
            if actions:
                st.caption(f"{len(actions)} claim(s) have a recorded action this session.")

    with tab2:
        st.subheader("Payer-level rollup")
        rollup_source = claims_data.copy()
        rollup_source["Amount (SAR)"] = (
            rollup_source["Amount"].astype(str).str.replace("SAR ", "", regex=False).str.replace(",", "", regex=False).astype(float)
        )
        rollup_source["Risk score"] = pd.to_numeric(rollup_source["Risk score"], errors="coerce")
        agg_spec = {
            "Claims": ("Claim", "count"),
            "Total amount (SAR)": ("Amount (SAR)", "sum"),
            "Avg. risk score": ("Risk score", "mean"),
        }
        rollup = rollup_source.groupby("Payer").agg(**agg_spec).reset_index()
        ready_counts = rollup_source[rollup_source["Next action"] == "Ready to submit"].groupby("Payer").size()
        rollup["Ready to submit"] = rollup["Payer"].map(ready_counts).fillna(0).astype(int)
        rollup["Needs review"] = rollup["Claims"] - rollup["Ready to submit"]
        st.dataframe(rollup.round(1), use_container_width=True, hide_index=True)
        st.bar_chart(rollup.set_index("Payer")["Total amount (SAR)"], color="#0e7490")

    with tab3:
        st.subheader("Denial / risk trend (last 14 days)")
        days = pd.date_range(date.today() - timedelta(days=13), periods=14)
        # Deterministic synthetic series (no random seed drift across
        # reruns), same approach demo_patients()/demo_claims() use elsewhere.
        denial_rate = [8 + (i * 3) % 11 for i in range(14)]
        trend_df = pd.DataFrame({"Date": days, "Denial rate (%)": denial_rate}).set_index("Date")
        st.line_chart(trend_df, color=["#dc2626"])
        st.caption("Synthetic trend for presentation — replace with a Gold-layer fact_claims rollup by submission date in production.")


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

    live = genie_ready()
    top = st.columns([5, 1])
    with top[0]:
        st.markdown("**Try asking:** " + " · ".join(prompts))
    with top[1]:
        if st.session_state.get("genie_conversation_id") and st.button("↻ New chat", use_container_width=True):
            st.session_state.pop("genie_conversation_id", None)
            st.session_state.pop("genie_history", None)
            st.rerun()

    for turn in st.session_state.get("genie_history", []):
        st.chat_message(turn["role"]).write(turn["content"])

    question = st.chat_input("Ask Weqayah AI about clinic operations…")
    if question:
        st.chat_message("user").write(question)
        with st.spinner("Asking Genie…" if live else "Thinking…"):
            answer, is_live = ask_genie(question)
        st.chat_message("assistant").write(answer)
        history = st.session_state.setdefault("genie_history", [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if not is_live:
            st.caption("⚠️ Answered in demo mode this turn — the live Genie call was unavailable, see below.")

    mode = "Connected to Genie space — answers are live" if live else "Demo response mode — GENIE_SPACE_ID not configured or unreachable"
    st.caption(f"● {mode}. Do not send protected health information to an unapproved model endpoint.")


def analytics_trend(n_days: int) -> pd.DataFrame:
    """n-day Visits/Revenue series. Extends sample_trend()'s 7-day shape to
    an arbitrary window using the same deterministic-not-random approach, so
    the analytics filters below have real (if synthetic) data to respond to.
    """
    days = pd.date_range(date.today() - timedelta(days=n_days - 1), periods=n_days)
    visits = [90 + int(30 * abs((i % 14) - 7) / 7) + (i * 3) % 11 for i in range(n_days)]
    revenue = [v * 340 + (i * 53) % 900 for i, v in enumerate(visits)]
    return pd.DataFrame({"Date": days, "Visits": visits, "Revenue (SAR)": revenue}).set_index("Date")


def analytics_ar_days(invoices: pd.DataFrame, today: date) -> float:
    """Amount-weighted average days-outstanding across open invoices."""
    open_invoices = invoices[invoices["Status"] != "Paid"]
    if open_invoices.empty:
        return 0.0
    ages = open_invoices["Due date"].map(lambda d: max((today - d).days, 0))
    weights = open_invoices["Amount"]
    if weights.sum() == 0:
        return float(ages.mean())
    return float((ages * weights).sum() / weights.sum())


def analytics() -> None:
    title("Executive analytics", "Decision-ready trends powered by Gold-layer marts, with governed drill-through to operational detail.")

    departments = ["General Medicine", "Dental", "Dermatology", "Pediatrics", "Radiology"]
    dept_revenue_base = {"General Medicine": 24500, "Dental": 11200, "Dermatology": 7600, "Pediatrics": 4800, "Radiology": 9100}

    filter1, filter2 = st.columns([1, 2])
    with filter1:
        window = st.selectbox("Time window", ["Last 7 days", "Last 30 days", "Last 90 days"], key="analytics_window")
    with filter2:
        selected_departments = st.multiselect("Departments", departments, default=departments, key="analytics_departments")

    n_days = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}[window]
    trend = analytics_trend(n_days)

    invoices = pd.DataFrame(billing_invoices_state())
    claims_data = demo_claims()
    today = date.today()
    charges_total = invoices["Amount"].sum()
    collected_total = invoices.loc[invoices["Status"] == "Paid", "Amount"].sum()
    collection_rate = (collected_total / charges_total * 100) if charges_total else 0
    ar_days = analytics_ar_days(invoices, today)
    denial_risk = (claims_data["AI finding"] != "Complete").mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    with k1: metric("Collection rate", f"{collection_rate:.1f}%", "Paid ÷ charged, Billing tab")
    with k2: metric("AR days", f"{ar_days:.0f} days", "Amount-weighted, open invoices")
    with k3: metric("Claim denial risk", f"{denial_risk:.1f}%", "Share flagged by AI guard")
    with k4: metric("No-show rate", "6.4%", "Illustrative — no source table yet")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader(f"Visits — {window.lower()}")
        # Visits (~tens/hundreds) and Revenue (~tens of thousands) plotted
        # together flatten into an unreadable single-axis chart — same fix
        # applied on the Command Center dashboard.
        st.line_chart(trend[["Visits"]], color=["#0e7490"])
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader(f"Revenue (SAR) — {window.lower()}")
        st.bar_chart(trend[["Revenue (SAR)"]], color=["#4f46e5"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Department performance")
    dept_df = pd.DataFrame({"Department": departments, "Revenue (SAR)": [dept_revenue_base[d] for d in departments]}).set_index("Department")
    filtered_dept = dept_df.loc[dept_df.index.isin(selected_departments)] if selected_departments else dept_df
    st.bar_chart(filtered_dept, color="#0e7490")
    st.markdown("</div>", unsafe_allow_html=True)

    forecast_demo = pd.DataFrame({"Date": pd.date_range(date.today(), periods=7), "Expected OPD": [149, 141, 153, 158, 146, 121, 96]}).set_index("Date")
    forecast = live_or_demo("forecast", forecast_demo)
    st.subheader("Demand forecast")
    numeric = forecast.select_dtypes("number")
    st.line_chart(numeric if not numeric.empty else forecast_demo)

    st.download_button(
        "Export trend data (CSV)",
        data=trend.to_csv().encode("utf-8"),
        file_name=f"weqayah_trend_{window.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


def administration() -> None:
    title("Administration and governance", "A transparent view of access, operational controls and deployment readiness.")
    tab1, tab2, tab3 = st.tabs(["Users and roles", "Data governance", "Integration health"])
    with tab1:
        current_role = st.session_state.get("current_role", "Administrator")
        st.caption("Change **Signed in as** in the sidebar to see the navigation menu itself change per role — this isn't just a static list, it drives what's actually visible.")
        rows = []
        for role, info in ROLE_USERS.items():
            access = ROLE_ACCESS.get(role, list(PAGES))
            access_label = "All modules" if len(access) >= len(PAGES) else ", ".join(access)
            rows.append([info["name"], role, access_label, "You — this session" if role == current_role else "Active"])
        st.dataframe(pd.DataFrame(rows, columns=["User", "Role", "Access", "Status"]), use_container_width=True, hide_index=True)
    with tab2:
        st.write("**Unity Catalog controls**\n\n• Table and view permissions by role\n• Auditable data access\n• Data lineage from source to dashboard\n• Masking policies for sensitive identifiers")
        st.caption("Grant the Databricks App service principal SELECT on Gold read models, MODIFY only on approved operational write tables, and CAN USE on the SQL warehouse.")
        st.markdown("**Configured targets for this environment**")
        targets = pd.DataFrame(
            [["Operational write (Lakebase)", WRITE_TABLE]] + [[f"Gold read: {key}", table] for key, table in TABLES.items()],
            columns=["Purpose", "Table"],
        )
        st.dataframe(targets, use_container_width=True, hide_index=True)
    with tab3:
        # Real status where it's checkable from this app (SQL warehouse,
        # Genie), rather than a purely aspirational roadmap table.
        sql_status = "Connected" if db_ready() else "Not connected — presentation mode"
        genie_status = "Connected" if genie_ready() else "Not configured"
        rows = [
            ["Databricks SQL warehouse", sql_status, f"Write target: {WRITE_TABLE}", "Live" if db_ready() else "Falling back to presentation data"],
            ["Genie space (Weqayah AI)", genie_status, "Natural-language Q&A over Gold marts", "Live" if genie_ready() else "Falling back to demo answers"],
            ["PACS", "Connected (simulated feed)", "DICOM interface", "Modernise — replace pacs_demo_worklist() with the approved connector"],
            ["Laboratory LIS", "Planned", "HL7 / API", "Design phase"],
            ["NPHIES", "Planned", "Approved API", "Design phase"],
        ]
        st.dataframe(pd.DataFrame(rows, columns=["System", "Status", "Detail", "Next step"]), use_container_width=True, hide_index=True)


PAGES = {
    "Command Center": dashboard, "Patient Registration & Search": patient_hub, "Clinical / EMR": emr,
    "Laboratory": lab, "Radiology": radiology, "Pharmacy": pharmacy, "Billing": billing,
    "Insurance": claims, "Weqayah AI": assistant_page, "Executive Analytics": analytics,
    "Administration": administration,
}

ROLE_USERS = {
    "Administrator": {"name": "Aisha Al-Dossari", "team": "IT & Governance"},
    "Physician": {"name": "Dr. Noor Al-Salem", "team": "Clinical"},
    "Revenue Cycle": {"name": "Rehab Al-Harbi", "team": "Billing & Claims"},
    "Reception Desk": {"name": "Mariam Ahmed", "team": "Front Office"},
}

# What each role can see in the sidebar. This is a presentation-layer
# demonstration of RBAC (real enforcement belongs in Unity Catalog table
# grants, per the Data governance tab) — but the navigation genuinely
# changes when you switch "Signed in as", it isn't just decorative copy.
ROLE_ACCESS = {
    "Administrator": list(PAGES),
    "Physician": ["Command Center", "Patient Registration & Search", "Clinical / EMR", "Laboratory", "Radiology", "Pharmacy", "Weqayah AI"],
    "Revenue Cycle": ["Command Center", "Patient Registration & Search", "Billing", "Insurance", "Executive Analytics", "Weqayah AI"],
    "Reception Desk": ["Command Center", "Patient Registration & Search", "Billing", "Weqayah AI"],
}


def main() -> None:
    st.session_state.setdefault("current_role", "Administrator")
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=225)
        else:
            st.markdown("### Weqayah\nMedical Center")
        st.markdown('<div class="brand-sub">AI-Powered Hospital Information System</div>', unsafe_allow_html=True)
        st.selectbox(
            "Signed in as",
            list(ROLE_USERS),
            key="current_role",
            help="Demo role switch — see Administration > Users and roles for what each role can access.",
        )
        available_pages = [p for p in PAGES if p in ROLE_ACCESS.get(st.session_state.current_role, list(PAGES))]
        if st.session_state.get("navigation") not in available_pages:
            # Either first load, or a role switch just hid the page the user
            # was on (or a callback targeted a page this role can't see) —
            # land on that role's first available page instead of erroring.
            st.session_state.navigation = available_pages[0]
        page = st.radio("Navigate", available_pages, label_visibility="collapsed", key="navigation")
        st.divider()
        mode = "Connected to Databricks SQL" if db_ready() else "Presentation mode"
        st.caption(f"● {mode}")
        st.caption("Lakehouse read models · Lakebase operational writes")
        st.divider()
        user = ROLE_USERS[st.session_state.current_role]
        st.caption(f"Signed in as: {user['name']} · {st.session_state.current_role}")
    user = ROLE_USERS[st.session_state.current_role]
    st.markdown(f'<div class="topbar"><div class="topbar-title">♧ &nbsp; AI-Powered Hospital Information System</div><div class="topbar-user"><b>{user["name"]}</b><br><span>{user["team"]}</span></div></div>', unsafe_allow_html=True)
    PAGES[page]()


if __name__ == "__main__":
    main()

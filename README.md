# Weqayah Medical Center — Databricks App

This Streamlit application is a client-facing demonstration of the Weqayah HIS modernization target state. It uses Databricks SQL for Unity Catalog Gold-layer read models and routes patient-registration writes to a separate governed operational table (Lakebase or an approved Delta operational table).

## Configured Gold read models

| Purpose | Default table |
|---|---|
| Patient context | `meridian.gold.dim_patient` |
| Visits | `meridian.gold.fact_visits` |
| Claims | `meridian.gold.fact_claims` |
| Demand forecast | `meridian.gold.demand_forecast` |
| Pharmacy recommendations | `meridian.gold.pharmacy_reorder` |
| KPI summary | `meridian.gold.kpi_executive_summary` |

All table names are environment-variable overrides, so the notebook catalog/schema can be aligned at deployment without changing code.

## Deploy

1. Create the destination write table, for example `meridian.lakebase.patient_registration`, with the columns used in the `INSERT` statement in `app.py`.
2. Update `app.yaml` with the SQL warehouse ID and the actual catalog/schema/table names from the notebooks.
3. Deploy the folder as a Databricks App.
4. Grant the App service principal `CAN USE` on the SQL warehouse, `SELECT` on the Gold tables, and `MODIFY` only on the operational write table.

The app runs in a polished presentation mode when no warehouse ID is configured, allowing a complete executive demo without access to live clinical data or developer-facing connection errors. The user-provided Weqayah logo is stored in `assets/weqayah-logo.png`.

The same source is compatible with Streamlit Cloud: configure the `DATABRICKS_*` and `MERIDIAN_*` environment variables as Streamlit secrets/environment variables. `app.yaml` is used only by Databricks Apps.

## Important production controls

- Do not write operational transactions to Gold tables.
- Apply Unity Catalog row/column controls and audit policy before using real patient data.
- Replace the demo Weqayah AI response handler with an approved Genie space or model-serving endpoint.
- Validate the actual notebook table schemas; fields differ across implementations.

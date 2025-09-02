# app.py
import re
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="DOST Grants-in-Aid (GIA) Dashboard",
    page_icon="⚛️",
    layout="wide",
)
st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      .metric {text-align:center;}
    </style>
    """,
    unsafe_allow_html=True,
)
# Altair: allow large datasets
alt.data_transformers.disable_max_rows()

# =========================
# Helpers
# =========================
def parse_money(x):
    """Parse strings like '7,557,581.20' or '₱1,234,567.89' to float."""
    if pd.isna(x):
        return np.nan
    s = re.sub(r"[₱,]", "", str(x))
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan

def to_date(x):
    if pd.isna(x) or str(x).strip() == "":
        return pd.NaT
    return pd.to_datetime(x, errors="coerce")

@st.cache_data(show_spinner=True)
def load_data():
    """
    Expects: data/dost_gia.xlsx with columns:
      - Program Title, Project Title, Key Result Areas (KRA),
        Description..., Expected Output..., Implementing Agency, Beneficiaries,
        Start, End, Status, Total Project Cost, PCAARRD GIA, Year
    """
    df = pd.read_excel("data/dost_gia.xlsx").copy()

    # Normalize column names to friendly snake_case
    colmap = {
        "Program Title": "program_title",
        "Project Title": "project_title",
        "Key Result Areas (KRA)": "kra",
        "Description of Program/Project/Objectives": "description",
        "Expected Output/Target": "expected_output",
        "Implementing Agency": "implementing_agency",
        "Beneficiaries": "beneficiaries",
        "Start": "start",
        "End": "end",
        "Status": "status",
        "Total Project Cost": "total_project_cost",
        "PCAARRD GIA": "pcaarrd_gia",
        "Year": "year",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

    # Dates
    if "start" in df: df["start_dt"] = df["start"].apply(to_date)
    if "end" in df: df["end_dt"] = df["end"].apply(to_date)

    # Official Year column (already present in your data)
    if "year" in df:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        # fallback if ever missing: infer from Start → End
        if "start_dt" in df and df["start_dt"].notna().any():
            df["year"] = df["start_dt"].dt.year
        elif "end_dt" in df and df["end_dt"].notna().any():
            df["year"] = df["end_dt"].dt.year
        else:
            df["year"] = np.nan

    # Funding preference: use PCAARRD GIA if present else Total Project Cost
    pca = df["pcaarrd_gia"].apply(parse_money) if "pcaarrd_gia" in df else pd.Series(np.nan, index=df.index)
    tpc = df["total_project_cost"].apply(parse_money) if "total_project_cost" in df else pd.Series(np.nan, index=df.index)
    df["funding_num"] = pca.fillna(tpc).fillna(0.0)

    # Duration (days) for timeline
    if {"start_dt", "end_dt"}.issubset(df.columns):
        df["duration_days"] = (df["end_dt"] - df["start_dt"]).dt.days

    # Clean text cols
    for c in ["program_title","project_title","kra","description",
              "expected_output","implementing_agency","beneficiaries","status"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": ""}).str.strip()

    return df

# =========================
# Load & Year Filter (ONLY)
# =========================
df = load_data()

# UI year bounds (enforced) and data-aware min/max inside that window
UI_MIN, UI_MAX = 2016, 2024
data_years = df["year"].dropna().astype(int)
if data_years.empty:
    min_year, max_year = UI_MIN, UI_MAX
else:
    min_year = max(UI_MIN, int(data_years.min()))
    max_year = min(UI_MAX, int(data_years.max()))

st.sidebar.title("Filter")
if min_year == max_year:
    # single-year slider when only one year exists
    selected_year = st.sidebar.slider("Year", min_value=min_year, max_value=max_year, value=min_year)
    fdf = df[df["year"] == selected_year].copy()
else:
    year_range = st.sidebar.slider("Year", min_value=min_year, max_value=max_year, value=(min_year, max_year))
    fdf = df[(df["year"].fillna(min_year) >= year_range[0]) & (df["year"].fillna(max_year) <= year_range[1])].copy()

# =========================
# KPIs
# =========================
st.title("⚛️ DOST Grants-in-Aid (GIA) Dashboard")
st.caption("Analyst-friendly view of projects, funding, agencies, beneficiaries, and timelines. Year-filtered.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Projects", f"{len(fdf):,}")
c2.metric("Agencies", f"{fdf.get('implementing_agency', pd.Series(dtype=str)).dropna().nunique():,}")
c3.metric("Beneficiaries", f"{fdf.get('beneficiaries', pd.Series(dtype=str)).dropna().nunique():,}")
c4.metric("Total Funding (₱)", f"{fdf['funding_num'].sum():,.0f}")

# =========================
# Insights (auto-narrative)
# =========================
with st.expander("🔎 Quick Insights"):
    lines = []
    # Funding concentration among agencies
    if "implementing_agency" in fdf and not fdf.empty:
        adf = fdf.groupby("implementing_agency", dropna=False)["funding_num"].sum().reset_index()
        adf = adf.sort_values("funding_num", ascending=False)
        total = adf["funding_num"].sum()
        if total > 0:
            adf["cum_share"] = adf["funding_num"].cumsum() / total * 100
            cutoff_idx = (adf["cum_share"] >= 80).idxmax()
            try:
                top_k = int(adf.loc[cutoff_idx].name) + 1  # display rank count
            except:
                top_k = int((adf["cum_share"] >= 80).sum())
            top_3 = ", ".join(adf.head(3)["implementing_agency"].astype(str).tolist())
            lines.append(f"- Top agencies: {top_3}.")
            lines.append(f"- Approximately **top {top_k} agencies** account for **~80%** of funding.")
    # KRA leader
    if "kra" in fdf and not fdf.empty:
        kdf = fdf.groupby("kra", dropna=False)["funding_num"].sum().reset_index().sort_values("funding_num", ascending=False)
        if not kdf.empty and kdf["funding_num"].sum() > 0:
            lines.append(f"- **Leading KRA by funding:** {kdf.iloc[0]['kra']}.")
    # Timeline insight
    if "duration_days" in fdf:
        durations = fdf["duration_days"].dropna()
        if len(durations) > 3:
            lines.append(f"- Median project duration: **{int(durations.median()):,} days**.")
    if not lines:
        lines = ["No strong signals yet (add more years/projects for richer insights)."]
    st.write("\n".join(lines))

# =========================
# Visualizations
# =========================
tabs = st.tabs([
    "📈 Projects & Funding by Year",
    "💰 Funding by KRA",
    "🏛️ Agencies (Pareto)",
    "👥 Beneficiaries",
    "🟦 Status Distribution",
    "📅 Project Timeline (Gantt)"
])

# 1) Projects & Funding by Year (dual-layer)
with tabs[0]:
    if not fdf.empty:
        by_year = (
            fdf.groupby("year", dropna=False)
               .agg(Projects=("project_title", "count"),
                    Funding=("funding_num", "sum"))
               .reset_index()
               .sort_values("year")
        )
        bars = alt.Chart(by_year).mark_bar().encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("Funding:Q", title="Funding (₱)"),
            tooltip=[alt.Tooltip("year:O"),
                     alt.Tooltip("Funding:Q", format=",.0f"),
                     alt.Tooltip("Projects:Q")]
        )
        line = alt.Chart(by_year).mark_line(point=True).encode(
            x="year:O",
            y=alt.Y("Projects:Q", axis=alt.Axis(title="Projects")),
            tooltip=["year:O", "Projects:Q"]
        )
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=360), use_container_width=True)
    else:
        st.info("No data for selected year(s).")

# 2) Funding by KRA
with tabs[1]:
    if "kra" in fdf and not fdf.empty:
        kdf = fdf.groupby("kra")["funding_num"].sum().reset_index().sort_values("funding_num", ascending=False)
        st.altair_chart(
            alt.Chart(kdf).mark_bar().encode(
                x=alt.X("kra:N", sort="-y", title="KRA"),
                y=alt.Y("funding_num:Q", title="Funding (₱)"),
                tooltip=["kra:N", alt.Tooltip("funding_num:Q", format=",.0f", title="₱")]
            ).properties(height=340),
            use_container_width=True
        )
    else:
        st.info("KRA column not available or no rows after filtering.")

# 3) Agencies Pareto (bar + cumulative line)
with tabs[2]:
    if "implementing_agency" in fdf and not fdf.empty:
        adf = (
            fdf.groupby("implementing_agency")["funding_num"]
               .sum()
               .reset_index()
               .sort_values("funding_num", ascending=False)
        )
        total = adf["funding_num"].sum()
        if total > 0:
            adf["cum_share"] = adf["funding_num"].cumsum() / total * 100
        top_n = st.slider("Show top N agencies", 5, min(30, len(adf)), min(15, len(adf)))
        view = adf.head(top_n)

        bars = alt.Chart(view).mark_bar().encode(
            x=alt.X("implementing_agency:N", sort="-y", title="Implementing Agency"),
            y=alt.Y("funding_num:Q", title="Funding (₱)"),
            tooltip=["implementing_agency:N", alt.Tooltip("funding_num:Q", format=",.0f")]
        )
        line = alt.Chart(view).mark_line(point=True).encode(
            x="implementing_agency:N",
            y=alt.Y("cum_share:Q", axis=alt.Axis(title="Cumulative %")),
            tooltip=[alt.Tooltip("cum_share:Q", format=".1f", title="Cum %")]
        )
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=360), use_container_width=True)
    else:
        st.info("No agency data to display.")

# 4) Beneficiaries (frequency)
with tabs[3]:
    if "beneficiaries" in fdf and not fdf.empty:
        vals = fdf["beneficiaries"].dropna().astype(str)
        exploded = []
        for v in vals:
            # split on commas, semicolons, slashes, 'and'
            parts = re.split(r"(?:,|;|/| and )+", v, flags=re.I)
            exploded.extend([p.strip() for p in parts if p and p.strip()])
        if exploded:
            ben = pd.Series(exploded).value_counts().reset_index()
            ben.columns = ["Beneficiary", "Projects"]
            st.altair_chart(
                alt.Chart(ben.head(20)).mark_bar().encode(
                    x=alt.X("Projects:Q"),
                    y=alt.Y("Beneficiary:N", sort="-x"),
                    tooltip=["Beneficiary:N", "Projects:Q"]
                ).properties(height=420),
                use_container_width=True
            )
        else:
            st.info("No beneficiary tokens extracted.")
    else:
        st.info("No beneficiary data to display.")

# 5) Status Distribution (pie)
with tabs[4]:
    if "status" in fdf and not fdf.empty:
        sdf = fdf["status"].fillna("Unspecified").value_counts().reset_index()
        sdf.columns = ["Status", "Count"]
        st.altair_chart(
            alt.Chart(sdf).mark_arc(innerRadius=60).encode(
                theta="Count:Q",
                color=alt.Color("Status:N", legend=alt.Legend(title="Status")),
                tooltip=["Status:N", "Count:Q"]
            ).properties(height=360),
            use_container_width=True
        )
    else:
        st.info("No status column available.")

# 6) Project Timeline (Gantt)
with tabs[5]:
    if {"project_title", "start_dt", "end_dt"}.issubset(fdf.columns):
        tdf = fdf.dropna(subset=["start_dt", "end_dt"]).copy()
        if not tdf.empty:
            max_rows = st.slider("Max projects to display", 50, 1000, min(200, len(tdf)), step=50)
            tdf = tdf.sort_values("start_dt").head(max_rows)
            tooltip_cols = [c for c in ["project_title","program_title","implementing_agency","kra","start_dt","end_dt","funding_num","status"] if c in tdf.columns]
            st.altair_chart(
                alt.Chart(tdf).mark_bar().encode(
                    x=alt.X("start_dt:T", title="Start"),
                    x2=alt.X2("end_dt:T", title="End"),
                    y=alt.Y("project_title:N", sort="-x", title=None),
                    color=alt.Color("kra:N", legend=alt.Legend(title="KRA")) if "kra" in tdf.columns else alt.value("#888"),
                    tooltip=tooltip_cols
                ).properties(height=min(28*len(tdf), 720)),
                use_container_width=True
            )
        else:
            st.info("No projects with both Start and End dates in the selected year(s).")
    else:
        st.info("Need `Project Title`, `Start`, and `End` to draw the timeline.")

# =========================
# Data Explorer + Export
# =========================
st.subheader("📑 Data Explorer")
show_cols = [c for c in [
    "program_title","project_title","kra","implementing_agency","beneficiaries",
    "start","end","status","year","pcaarrd_gia","total_project_cost","funding_num","duration_days"
] if c in fdf.columns]
st.dataframe(fdf[show_cols], use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered CSV",
    data=fdf[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="dost_gia_filtered.csv",
    mime="text/csv",
)

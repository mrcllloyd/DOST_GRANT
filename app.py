# app.py
import os
import re
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ----------------------
# App Config
# ----------------------
st.set_page_config(
    page_title="DOST Grants-in-Aid (GIA) Dashboard",
    page_icon="⚛️",   # Atom icon
    layout="wide",
)

# (Optional) streamlined chrome
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .metric {text-align:center;}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------
# Column Auto-Mapping
# ----------------------
EXPECTED_KEYS = {
    "program_title": ["program title","program","program_name","programtitle"],
    "project_title": ["project title","project","project_name","projecttitle"],
    "kra": ["key result areas (kra)","kra","key result area","key result areas"],
    "description": ["description of program/project/objectives","description","objectives"],
    "expected_output": ["expected output/target","expected output","target"],
    "implementing_agency": ["implementing agency","proponent","lead agency","implementer"],
    "beneficiaries": ["beneficiaries","beneficiary","target beneficiaries"],
    "start": ["start","date start","start date","project start"],
    "end": ["end","date end","end date","project end"],
    "status": ["status 'as of december 31, 2024'","status","status as of dec 31, 2024"],
    "total_cost": ["total project cost","total cost","project cost"],
    "pcaarrd_2024": ["2024 pcaarrd gia","pcaarrd 2024 gia","2024 gia","2024 allocation"],
}

def _slug(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for raw in df.columns:
        mapped = False
        for canon, alts in EXPECTED_KEYS.items():
            if _slug(raw) in {_slug(x) for x in (alts + [canon])}:
                col_map[raw] = canon
                mapped = True
                break
        if not mapped:
            col_map[raw] = _slug(raw).replace(" ", "_")
    return df.rename(columns=col_map)

def parse_money(x):
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
def load_data(path: str | None = None) -> pd.DataFrame:
    candidates = []
    if path:
        candidates.append(path)
    envp = os.getenv("STREAMLIT_DOST_GIA_PATH")
    if envp:
        candidates.append(envp)
    candidates.append("data/dost_gia.xlsx")

    last_err = None
    for p in [c for c in candidates if c]:
        try:
            df = pd.read_excel(p).copy()
            df = normalize_columns(df)

            if "total_cost" in df:
                df["total_cost_num"] = df["total_cost"].apply(parse_money)
            if "pcaarrd_2024" in df:
                df["pcaarrd_2024_num"] = df["pcaarrd_2024"].apply(parse_money)

            if "start" in df:
                df["start_dt"] = df["start"].apply(to_date)
                df["year_start"] = df["start_dt"].dt.year
            if "end" in df:
                df["end_dt"] = df["end"].apply(to_date)
                df["year_end"] = df["end_dt"].dt.year

            for c in [
                "program_title","project_title","kra","description",
                "expected_output","implementing_agency","beneficiaries","status"
            ]:
                if c in df:
                    df[c] = df[c].astype(str).replace({"nan": ""})

            # computed duration (in days) for timeline viz
            if "start_dt" in df and "end_dt" in df:
                df["duration_days"] = (df["end_dt"] - df["start_dt"]).dt.days

            # unified funding column for charts
            if "total_cost_num" in df and df["total_cost_num"].notna().any():
                df["funding_num"] = df["total_cost_num"]
            elif "pcaarrd_2024_num" in df and df["pcaarrd_2024_num"].notna().any():
                df["funding_num"] = df["pcaarrd_2024_num"]
            else:
                df["funding_num"] = np.nan

            return df
        except Exception as e:
            last_err = e
    raise FileNotFoundError(f"Could not load dataset. Tried: {candidates}. Last error: {last_err}")

# ----------------------
# Sidebar Controls
# ----------------------
st.sidebar.title("Filters")

with st.sidebar.expander("Data Source", expanded=False):
    user_path = st.text_input(
        "Optional: custom path (Excel)",
        value="",
        help="Override default path. Repo default is data/dost_gia.xlsx",
    )

df = load_data(user_path if user_path.strip() else None)

kra_opts = sorted(df.get("kra", pd.Series(dtype=str)).dropna().unique().tolist())
agency_opts = sorted(df.get("implementing_agency", pd.Series(dtype=str)).dropna().unique().tolist())

if "year_start" in df:
    min_year = int(np.nanmin(df["year_start"])) if df["year_start"].notna().any() else 2000
    max_year = int(np.nanmax(df["year_start"])) if df["year_start"].notna().any() else 2030
else:
    min_year, max_year = 2000, 2030

year_range = st.sidebar.slider(
    "Start Year",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

kra_sel = st.sidebar.multiselect("KRA", options=kra_opts, default=kra_opts if kra_opts else [])
agency_sel = st.sidebar.multiselect("Implementing Agency", options=agency_opts, default=[])
search_q = st.sidebar.text_input("Search (title/desc/beneficiaries)", value="")
show_only_funded = st.sidebar.checkbox("Only projects with ₱ amount", value=False)

# ----------------------
# Apply Filters
# ----------------------
mask = (df.get("year_start", pd.Series([min_year])).fillna(min_year) >= year_range[0]) & \
       (df.get("year_start", pd.Series([max_year])).fillna(max_year) <= year_range[1])

if kra_sel and "kra" in df:
    mask &= df["kra"].isin(kra_sel)

if agency_sel and "implementing_agency" in df:
    mask &= df["implementing_agency"].isin(agency_sel)

if search_q.strip():
    pat = re.compile(re.escape(search_q.strip()), flags=re.I)
    cols = [c for c in ["project_title","program_title","description","beneficiaries","expected_output"] if c in df.columns]
    if cols:
        sub = df[cols].astype(str).apply(lambda s: s.str.contains(pat))
        mask &= sub.any(axis=1)

if show_only_funded and "funding_num" in df.columns:
    mask &= df["funding_num"].fillna(0) > 0

fdf = df[mask].copy()

# ----------------------
# Header & KPIs
# ----------------------
st.title("⚛️ DOST Grants-in-Aid (GIA) Dashboard")
st.caption("Interactive overview of programs, projects, beneficiaries, and funding.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Projects", f"{len(fdf):,}")
c2.metric("Beneficiaries (unique)", f"{fdf.get('beneficiaries', pd.Series(dtype=str)).dropna().nunique():,}")
c3.metric("Implementing Agencies", f"{fdf.get('implementing_agency', pd.Series(dtype=str)).dropna().nunique():,}")
total_cost_sum = fdf.get("funding_num", pd.Series([0.0])).fillna(0.0).sum()
c4.metric("Total Funding (₱)", f"{total_cost_sum:,.0f}" if total_cost_sum else "—")

# ----------------------
# Core Charts
# ----------------------
core_tabs = st.tabs(["Projects by Year", "Funding by KRA", "Top Beneficiaries"])

with core_tabs[0]:
    if "year_start" in fdf.columns:
        by_year = (
            fdf.groupby("year_start", dropna=False)
               .size()
               .reset_index(name="Projects")
               .sort_values("year_start")
        )
        st.altair_chart(
            alt.Chart(by_year).mark_bar().encode(
                x=alt.X("year_start:O", title="Start Year"),
                y=alt.Y("Projects:Q"),
                tooltip=["year_start:O","Projects:Q"]
            ).properties(height=300),
            use_container_width=True
        )
    else:
        st.info("No 'Start' year information available to chart.")

with core_tabs[1]:
    if "kra" in fdf.columns and "funding_num" in fdf.columns:
        kdf = (
            fdf.groupby("kra", dropna=False)["funding_num"]
               .sum()
               .reset_index()
        )
        kdf["funding_num"] = kdf["funding_num"].fillna(0)
        st.altair_chart(
            alt.Chart(kdf).mark_bar().encode(
                x=alt.X("kra:N", sort="-y", title="KRA"),
                y=alt.Y("funding_num:Q", title="₱"),
                tooltip=["kra:N", alt.Tooltip("funding_num:Q", format=",.0f", title="Amount (₱)")]
            ).properties(height=300),
            use_container_width=True
        )
    else:
        st.info("Need KRA and funding columns to render this chart.")

with core_tabs[2]:
    if "beneficiaries" in fdf.columns:
        vals = fdf["beneficiaries"].dropna().astype(str)
        exploded = []
        for v in vals:
            parts = re.split(r"[;,/]+", v)
            exploded.extend([p.strip() for p in parts if p.strip()])
        if exploded:
            ben = pd.Series(exploded).value_counts().reset_index()
            ben.columns = ["Beneficiary","Projects"]
            ben = ben.head(20)
            st.altair_chart(
                alt.Chart(ben).mark_bar().encode(
                    x=alt.X("Projects:Q"),
                    y=alt.Y("Beneficiary:N", sort="-x"),
                    tooltip=["Beneficiary:N","Projects:Q"]
                ).properties(height=400),
                use_container_width=True
            )
        else:
            st.info("No beneficiary entries to display.")
    else:
        st.info("No Beneficiaries column available.")

# ----------------------
# Insightful Visualizations
# ----------------------
st.subheader("🔎 Insightful Visualizations")
viz_tabs = st.tabs([
    "Project Timeline (Gantt-style)",
    "KRA × Year Heatmap",
    "Agency Pareto (80/20 View)",
    "Funding Distribution by KRA"
])

# 1) Project Timeline (Gantt-style)
with viz_tabs[0]:
    if {"project_title","start_dt","end_dt"}.issubset(fdf.columns):
        # Focus on projects with both dates
        tdf = fdf.dropna(subset=["start_dt","end_dt"]).copy()
        # Limit rows for readability (toggleable)
        max_rows = st.slider("Max projects to display", 50, 1000, 200, step=50)
        tdf = tdf.sort_values("start_dt").head(max_rows)
        tooltip_cols = [c for c in ["project_title","program_title","implementing_agency","kra","start_dt","end_dt","funding_num","status"] if c in tdf.columns]

        chart = alt.Chart(tdf).mark_bar().encode(
            x=alt.X("start_dt:T", title="Start"),
            x2=alt.X2("end_dt:T", title="End"),
            y=alt.Y("project_title:N", sort="-x", title=None),
            color=alt.Color("kra:N", legend=alt.Legend(title="KRA")) if "kra" in tdf.columns else alt.value("#888"),
            tooltip=tooltip_cols
        ).properties(height=min(28*len(tdf), 700))
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Need `project_title`, `start`, and `end` columns to draw the timeline.")

# 2) KRA × Year Heatmap (funding intensity over time)
with viz_tabs[1]:
    if {"kra","year_start","funding_num"}.issubset(fdf.columns):
        hdf = (
            fdf.dropna(subset=["kra","year_start"])
               .groupby(["kra","year_start"], dropna=False)["funding_num"]
               .sum()
               .reset_index()
        )
        st.altair_chart(
            alt.Chart(hdf).mark_rect().encode(
                x=alt.X("year_start:O", title="Start Year"),
                y=alt.Y("kra:N", title="KRA"),
                color=alt.Color("funding_num:Q", title="₱", scale=alt.Scale(type="linear")),
                tooltip=[alt.Tooltip("kra:N"), alt.Tooltip("year_start:O"), alt.Tooltip("funding_num:Q", format=",.0f", title="Amount (₱)")]
            ).properties(height=300),
            use_container_width=True
        )
    else:
        st.info("Need KRA, Start Year, and funding to render heatmap.")

# 3) Agency Pareto (bar + cumulative line)
with viz_tabs[2]:
    if {"implementing_agency","funding_num"}.issubset(fdf.columns):
        adf = (
            fdf.dropna(subset=["implementing_agency"])
               .groupby("implementing_agency", dropna=False)["funding_num"]
               .sum()
               .reset_index()
               .sort_values("funding_num", ascending=False)
        )
        adf["funding_num"] = adf["funding_num"].fillna(0)
        adf["rank"] = np.arange(1, len(adf)+1)
        total = adf["funding_num"].sum()
        adf["cum_share"] = (adf["funding_num"].cumsum() / total * 100) if total else 0
        top_n = st.slider("Show top N agencies", 5, min(50, len(adf)), min(15, len(adf)))
        view = adf.head(top_n)

        bars = alt.Chart(view).mark_bar().encode(
            x=alt.X("implementing_agency:N", sort="-y", title="Implementing Agency"),
            y=alt.Y("funding_num:Q", title="₱"),
            tooltip=[alt.Tooltip("implementing_agency:N", title="Agency"),
                     alt.Tooltip("funding_num:Q", title="Amount (₱)", format=",.0f")]
        )
        line = alt.Chart(view).mark_line(point=True).encode(
            x=alt.X("implementing_agency:N", sort="-y"),
            y=alt.Y("cum_share:Q", axis=alt.Axis(title="Cumulative %", grid=False)),
            tooltip=[alt.Tooltip("cum_share:Q", title="Cum %", format=".1f")]
        )
        st.altair_chart(alt.layer(bars, line).resolve_scale(y='independent').properties(height=380), use_container_width=True)
        if len(adf):
            eighty_cut = (adf["cum_share"] >= 80).idxmax() if total else None
            if total and pd.notna(eighty_cut):
                cutoff_rank = int(adf.loc[eighty_cut, "rank"])
                st.caption(f"Approx. **Top {cutoff_rank} agencies** account for **~80%** of total funding.")
    else:
        st.info("Need Implementing Agency and funding to render Pareto.")

# 4) Funding Distribution by KRA (boxplot)
with viz_tabs[3]:
    if {"kra","funding_num"}.issubset(fdf.columns):
        bdf = fdf.dropna(subset=["kra","funding_num"]).copy()
        if len(bdf):
            st.altair_chart(
                alt.Chart(bdf).mark_boxplot().encode(
                    x=alt.X("kra:N", title="KRA"),
                    y=alt.Y("funding_num:Q", title="Funding (₱)"),
                    tooltip=[alt.Tooltip("kra:N"), alt.Tooltip("funding_num:Q", format=",.0f")]
                ).properties(height=350),
                use_container_width=True
            )
            st.caption("Boxplot shows median, quartiles, and outliers per KRA to reveal dispersion and high-value projects.")
        else:
            st.info("No rows with both KRA and numeric funding.")
    else:
        st.info("Need KRA and funding to render distribution.")

# ----------------------
# Data Explorer
# ----------------------
st.subheader("Data Explorer")
st.caption("Browse and download the filtered records.")

show_cols = [c for c in [
    "program_title","project_title","kra","implementing_agency","beneficiaries",
    "start","end","status","total_cost","pcaarrd_2024","funding_num","total_cost_num","pcaarrd_2024_num",
    "year_start","year_end","description","expected_output","duration_days"
] if c in fdf.columns]

st.dataframe(fdf[show_cols], use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered CSV",
    data=fdf[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="dost_gia_filtered.csv",
    mime="text/csv",
)

# ----------------------
# About
# ----------------------
with st.expander("About this dashboard"):
    st.markdown("""
**Purpose.** Explore the DOST GIA portfolio—projects, beneficiaries, KRAs, timelines, and funding—interactively.

**How to use.** Filter by year, KRA, agency, or text search. Toggle **Only projects with ₱ amount** to focus on entries with numeric funding.

**Data path.** Default is `data/dost_gia.xlsx`. Override via `STREAMLIT_DOST_GIA_PATH` or the sidebar input.
""")

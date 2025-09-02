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
    page_icon="⚛️",
    layout="wide",
)

# (Optional) remove Streamlit chrome for a cleaner look
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
    """Lowercase, collapse spaces."""
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw columns to canonical names using EXPECTED_KEYS. Fallback: snake_case of header."""
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
    """Parse peso strings like '₱1,234,567.89' → float."""
    if pd.isna(x):
        return np.nan
    s = re.sub(r"[₱,]", "", str(x))
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan

def to_date(x):
    """Best-effort date coercion."""
    if pd.isna(x) or str(x).strip() == "":
        return pd.NaT
    return pd.to_datetime(x, errors="coerce")

@st.cache_data(show_spinner=True)
def load_data(path: str | None = None) -> pd.DataFrame:
    """
    Load Excel dataset with fallback order:
      1) explicit path (sidebar)
      2) env STREAMLIT_DOST_GIA_PATH
      3) data/dost_gia.xlsx
    """
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

if show_only_funded:
    money_col = "total_cost_num" if "total_cost_num" in df.columns else ("pcaarrd_2024_num" if "pcaarrd_2024_num" in df.columns else None)
    if money_col:
        mask &= df[money_col].fillna(0) > 0

fdf = df[mask].copy()

# ----------------------
# Header & KPIs
# ----------------------
st.title("📊 DOST Grants-in-Aid (GIA) Dashboard")
st.caption("Interactive overview of programs, projects, beneficiaries, and funding.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Projects", f"{len(fdf):,}")
c2.metric("Beneficiaries (unique)", f"{fdf.get('beneficiaries', pd.Series(dtype=str)).dropna().nunique():,}")
c3.metric("Implementing Agencies", f"{fdf.get('implementing_agency', pd.Series(dtype=str)).dropna().nunique():,}")
total_cost_sum = fdf.get("total_cost_num", pd.Series([0.0])).fillna(0.0).sum()
c4.metric("Total Cost (₱)", f"{total_cost_sum:,.0f}" if total_cost_sum else "—")

# ----------------------
# Charts
# ----------------------
tabs = st.tabs(["Projects by Year", "Funding by KRA", "Top Beneficiaries"])

with tabs[0]:
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

with tabs[1]:
    if "kra" in fdf.columns:
        money_col = "total_cost_num" if "total_cost_num" in fdf.columns else ("pcaarrd_2024_num" if "pcaarrd_2024_num" in fdf.columns else None)
        if money_col:
            kdf = (
                fdf.groupby("kra", dropna=False)[money_col]
                   .sum()
                   .reset_index()
            )
            kdf[money_col] = kdf[money_col].fillna(0)
            st.altair_chart(
                alt.Chart(kdf).mark_bar().encode(
                    x=alt.X("kra:N", sort="-y", title="KRA"),
                    y=alt.Y(f"{money_col}:Q", title="₱"),
                    tooltip=["kra:N", alt.Tooltip(f"{money_col}:Q", format=",.0f", title="Amount (₱)")]
                ).properties(height=300),
                use_container_width=True
            )
        else:
            st.info("No funding column found (Total Project Cost or 2024 PCAARRD GIA).")
    else:
        st.info("No KRA column available.")

with tabs[2]:
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
# Data Explorer
# ----------------------
st.subheader("Data Explorer")
st.caption("Browse and download the filtered records.")

show_cols = [c for c in [
    "program_title","project_title","kra","implementing_agency","beneficiaries",
    "start","end","status","total_cost","pcaarrd_2024","total_cost_num","pcaarrd_2024_num",
    "year_start","year_end","description","expected_output"
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

**How to use.** Filter by year, KRA, agency, or text search. Toggle **Only projects with ₱ amount** to focus on entries with peso values.

**Data path.** Default is `data/dost_gia.xlsx`. You can override via the `STREAMLIT_DOST_GIA_PATH` environment variable or the sidebar input box.
""")

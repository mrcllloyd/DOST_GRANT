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
# Helpers
# ----------------------
def _slug(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    expected = {
        "program_title": ["program title"],
        "project_title": ["project title"],
        "kra": ["key result areas (kra)", "kra"],
        "implementing_agency": ["implementing agency"],
        "beneficiaries": ["beneficiaries"],
        "start": ["start", "start date"],
        "end": ["end", "end date"],
        "status": ["status", "status 'as of december 31, 2024'"],
        "total_project_cost": ["total project cost", "total cost", "project cost"],
        "year": ["year"],
    }
    for raw in df.columns:
        mapped = False
        for canon, alts in expected.items():
            if _slug(raw) in {_slug(x) for x in (alts+[canon])}:
                col_map[raw] = canon
                mapped = True
                break
        if not mapped:
            col_map[raw] = _slug(raw).replace(" ", "_")
    return df.rename(columns=col_map)

def parse_money(x):
    if pd.isna(x): return np.nan
    s = re.sub(r"[₱,]", "", str(x))
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan

def to_date(x):
    if pd.isna(x) or str(x).strip()=="" : return pd.NaT
    return pd.to_datetime(x, errors="coerce")

def first_present(*series):
    """Return the first non-null series among inputs."""
    for s in series:
        if s is not None and hasattr(s, "notna") and s.notna().any():
            return s
    return None

@st.cache_data(show_spinner=True)
def load_data():
    # Fixed path inside repo
    path = "data/dost_gia.xlsx"
    df = pd.read_excel(path).copy()
    df = normalize_columns(df)

    # Dates
    if "start" in df:
        df["start_dt"] = df["start"].apply(to_date)
    if "end" in df:
        df["end_dt"] = df["end"].apply(to_date)

    # Derive a robust 'year' column:
    # Priority: explicit 'year' column → start year → end year
    year_explicit = pd.to_numeric(df.get("year"), errors="coerce") if "year" in df else None
    start_year = df["start_dt"].dt.year if "start_dt" in df else None
    end_year = df["end_dt"].dt.year if "end_dt" in df else None
    year_choice = first_present(year_explicit, start_year, end_year)
    df["year"] = year_choice if year_choice is not None else np.nan

    # Funding detection:
    # 1) total_project_cost if present
    # 2) any column containing "PCAARRD GIA"
    funding = None
    if "total_project_cost" in df:
        funding = df["total_project_cost"].apply(parse_money)
    else:
        # find any PCAARRD GIA column (e.g., "2016 PCAARRD GIA")
        gia_cols = [c for c in df.columns if re.search(r"pcaarrd\s*gia", c, flags=re.I)]
        if gia_cols:
            # Sum across any found GIA columns after parsing numeric
            tmp = pd.DataFrame({c: df[c].apply(parse_money) for c in gia_cols})
            funding = tmp.sum(axis=1)
    if funding is None:
        funding = pd.Series(np.nan, index=df.index)

    df["funding_num"] = funding.fillna(0)

    # Duration (optional, used by Gantt)
    if {"start_dt", "end_dt"}.issubset(df.columns):
        df["duration_days"] = (df["end_dt"] - df["start_dt"]).dt.days

    return df

# ----------------------
# Load
# ----------------------
df = load_data()

# Compute actual data year bounds
valid_years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
data_min = int(valid_years.min()) if not valid_years.empty else 2016
data_max = int(valid_years.max()) if not valid_years.empty else 2024

# Force UI slider to 2016–2024 window, intersected with actual data
UI_MIN, UI_MAX = 2016, 2024
min_year = max(UI_MIN, data_min)
max_year = min(UI_MAX, max(data_max, UI_MIN))  # ensure at least UI_MIN

# ----------------------
# Sidebar (Year only)
# ----------------------
st.sidebar.title("Filter")
year_range = st.sidebar.slider(
    "Year",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

mask = (df["year"].fillna(min_year) >= year_range[0]) & (df["year"].fillna(max_year) <= year_range[1])
fdf = df[mask].copy()

# ----------------------
# KPIs
# ----------------------
st.title("⚛️ DOST Grants-in-Aid (GIA) Dashboard")
st.caption("Interactive overview of projects, beneficiaries, agencies, and funding (Year-filtered).")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Projects", f"{len(fdf):,}")
c2.metric("Unique Beneficiaries", f"{fdf.get('beneficiaries', pd.Series(dtype=str)).dropna().nunique():,}")
c3.metric("Agencies", f"{fdf.get('implementing_agency', pd.Series(dtype=str)).dropna().nunique():,}")
c4.metric("Total Funding (₱)", f"{fdf['funding_num'].sum():,.0f}")

# ----------------------
# Visualizations
# ----------------------
tabs = st.tabs(["Projects per Year", "Funding by KRA", "Top Agencies", "Project Timeline"])

with tabs[0]:
    by_year = (
        fdf.dropna(subset=["year"])
           .groupby("year")
           .size()
           .reset_index(name="Projects")
           .sort_values("year")
    )
    st.altair_chart(
        alt.Chart(by_year).mark_bar().encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("Projects:Q"),
            tooltip=["year:O","Projects:Q"]
        ).properties(height=300),
        use_container_width=True
    )

with tabs[1]:
    if "kra" in fdf and "funding_num" in fdf:
        kdf = (
            fdf.dropna(subset=["kra"])
               .groupby("kra")["funding_num"]
               .sum()
               .reset_index()
               .sort_values("funding_num", ascending=False)
        )
        st.altair_chart(
            alt.Chart(kdf).mark_bar().encode(
                x=alt.X("kra:N", sort="-y", title="KRA"),
                y=alt.Y("funding_num:Q", title="Funding (₱)"),
                tooltip=["kra:N", alt.Tooltip("funding_num:Q", format=",.0f", title="₱")]
            ).properties(height=320),
            use_container_width=True
        )
    else:
        st.info("KRA column not found in the dataset.")

with tabs[2]:
    if "implementing_agency" in fdf and "funding_num" in fdf:
        adf = (
            fdf.dropna(subset=["implementing_agency"])
               .groupby("implementing_agency")["funding_num"]
               .sum()
               .reset_index()
               .sort_values("funding_num", ascending=False)
               .head(15)
        )
        st.altair_chart(
            alt.Chart(adf).mark_bar().encode(
                x=alt.X("implementing_agency:N", sort="-y", title="Implementing Agency"),
                y=alt.Y("funding_num:Q", title="Funding (₱)"),
                tooltip=["implementing_agency:N", alt.Tooltip("funding_num:Q", format=",.0f", title="₱")]
            ).properties(height=320),
            use_container_width=True
        )
    else:
        st.info("Implementing Agency column not found in the dataset.")

with tabs[3]:
    if {"project_title","start_dt","end_dt"}.issubset(fdf.columns):
        tdf = fdf.dropna(subset=["start_dt","end_dt"]).copy()
        max_rows = st.slider("Max projects to display", 50, 1000, 200, step=50)
        tdf = tdf.sort_values("start_dt").head(max_rows)
        tooltip_cols = [c for c in ["project_title","program_title","implementing_agency","kra","start_dt","end_dt","funding_num","status"] if c in tdf.columns]
        st.altair_chart(
            alt.Chart(tdf).mark_bar().encode(
                x=alt.X("start_dt:T", title="Start"),
                x2=alt.X2("end_dt:T", title="End"),
                y=alt.Y("project_title:N", sort="-x", title=None),
                color=alt.Color("kra:N", legend=alt.Legend(title="KRA")) if "kra" in tdf.columns else alt.value("#888"),
                tooltip=tooltip_cols
            ).properties(height=min(28*len(tdf), 700)),
            use_container_width=True
        )
    else:
        st.info("Need `project_title`, `Start`, and `End` to draw the timeline.")

# ----------------------
# Data Explorer
# ----------------------
st.subheader("Data Explorer")
show_cols = [c for c in [
    "program_title","project_title","kra","implementing_agency","beneficiaries",
    "start","end","status","funding_num","total_project_cost",
    "year","duration_days","start_dt","end_dt"
] if c in fdf.columns]

st.dataframe(fdf[show_cols], use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered CSV",
    data=fdf[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="dost_gia_filtered.csv",
    mime="text/csv",
)

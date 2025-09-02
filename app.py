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
        "status": ["status"],
        "total_cost": ["total project cost"],
        "pcaarrd_2024": ["2024 pcaarrd gia"],
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

@st.cache_data(show_spinner=True)
def load_data():
    # fixed path inside repo
    path = "data/dost_gia.xlsx"
    df = pd.read_excel(path).copy()
    df = normalize_columns(df)

    if "total_cost" in df:
        df["funding_num"] = df["total_cost"].apply(parse_money)
    elif "pcaarrd_2024" in df:
        df["funding_num"] = df["pcaarrd_2024"].apply(parse_money)
    else:
        df["funding_num"] = np.nan

    if "start" in df:
        df["start_dt"] = df["start"].apply(to_date)
    if "end" in df:
        df["end_dt"] = df["end"].apply(to_date)

    # Year column
    if "start_dt" in df and df["start_dt"].notna().any():
        df["year"] = df["start_dt"].dt.year
    elif "end_dt" in df and df["end_dt"].notna().any():
        df["year"] = df["end_dt"].dt.year
    else:
        df["year"] = np.nan

    return df

# ----------------------
# Sidebar (Year filter only)
# ----------------------
st.sidebar.title("Filter")
df = load_data()

min_year = int(df["year"].min()) if df["year"].notna().any() else 2000
max_year = int(df["year"].max()) if df["year"].notna().any() else 2030

year_range = st.sidebar.slider(
    "Year",
    min_value=min_year,
    max_value=max_year,
    value=(min_year,max_year)
)

mask = (df["year"].fillna(min_year) >= year_range[0]) & (df["year"].fillna(max_year) <= year_range[1])
fdf = df[mask].copy()

# ----------------------
# KPIs
# ----------------------
st.title("⚛️ DOST Grants-in-Aid (GIA) Dashboard")
st.caption("Interactive overview of projects, beneficiaries, agencies, and funding.")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Projects", f"{len(fdf):,}")
c2.metric("Unique Beneficiaries", f"{fdf.get('beneficiaries', pd.Series(dtype=str)).dropna().nunique():,}")
c3.metric("Agencies", f"{fdf.get('implementing_agency', pd.Series(dtype=str)).dropna().nunique():,}")
c4.metric("Total Funding (₱)", f"{fdf['funding_num'].sum():,.0f}")

# ----------------------
# Visualizations
# ----------------------
tabs = st.tabs(["Projects per Year", "Funding by KRA", "Top Agencies"])

with tabs[0]:
    by_year = fdf.groupby("year").size().reset_index(name="Projects")
    st.altair_chart(
        alt.Chart(by_year).mark_bar().encode(
            x=alt.X("year:O"),
            y="Projects:Q",
            tooltip=["year:O","Projects:Q"]
        ).properties(height=300),
        use_container_width=True
    )

with tabs[1]:
    if "kra" in fdf and "funding_num" in fdf:
        kdf = fdf.groupby("kra")["funding_num"].sum().reset_index()
        st.altair_chart(
            alt.Chart(kdf).mark_bar().encode(
                x=alt.X("kra:N", sort="-y"),
                y=alt.Y("funding_num:Q", title="₱"),
                tooltip=["kra:N", alt.Tooltip("funding_num:Q", format=",.0f")]
            ).properties(height=300),
            use_container_width=True
        )

with tabs[2]:
    if "implementing_agency" in fdf and "funding_num" in fdf:
        adf = fdf.groupby("implementing_agency")["funding_num"].sum().reset_index().sort_values("funding_num", ascending=False).head(15)
        st.altair_chart(
            alt.Chart(adf).mark_bar().encode(
                x=alt.X("implementing_agency:N", sort="-y"),
                y=alt.Y("funding_num:Q", title="₱"),
                tooltip=["implementing_agency:N", alt.Tooltip("funding_num:Q", format=",.0f")]
            ).properties(height=300),
            use_container_width=True
        )

# ----------------------
# Data Explorer
# ----------------------
st.subheader("Data Explorer")
st.dataframe(fdf, use_container_width=True, hide_index=True)

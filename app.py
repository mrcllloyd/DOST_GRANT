# app.py
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
    path = "data/dost_gia.xlsx"
    df = pd.read_excel(path).copy()

    # Normalize columns
    df = df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))

    # Dates
    if "start" in df: df["start_dt"] = df["start"].apply(to_date)
    if "end" in df: df["end_dt"] = df["end"].apply(to_date)

    # Year (priority: explicit year → start → end)
    if "year" in df:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    elif "start_dt" in df and df["start_dt"].notna().any():
        df["year"] = df["start_dt"].dt.year
    elif "end_dt" in df and df["end_dt"].notna().any():
        df["year"] = df["end_dt"].dt.year
    else:
        df["year"] = np.nan

    # Funding (try multiple cols)
    if "total_project_cost" in df:
        df["funding_num"] = df["total_project_cost"].apply(parse_money)
    else:
        gia_cols = [c for c in df.columns if "pcaarrd" in c.lower() and "gia" in c.lower()]
        if gia_cols:
            tmp = pd.DataFrame({c: df[c].apply(parse_money) for c in gia_cols})
            df["funding_num"] = tmp.sum(axis=1)
        else:
            df["funding_num"] = np.nan

    # Duration
    if {"start_dt","end_dt"}.issubset(df.columns):
        df["duration_days"] = (df["end_dt"] - df["start_dt"]).dt.days

    return df

# ----------------------
# Load & Filter
# ----------------------
df = load_data()

# Year filter (fixed 2016–2024)
UI_MIN, UI_MAX = 2016, 2024
data_years = df["year"].dropna().astype(int)
min_year = max(UI_MIN, int(data_years.min())) if not data_years.empty else UI_MIN
max_year = min(UI_MAX, int(data_years.max())) if not data_years.empty else UI_MAX

st.sidebar.title("Filter")
year_range = st.sidebar.slider("Year", min_value=min_year, max_value=max_year, value=(min_year,max_year))
mask = (df["year"].fillna(min_year) >= year_range[0]) & (df["year"].fillna(max_year) <= year_range[1])
fdf = df[mask].copy()

# ----------------------
# KPIs
# ----------------------
st.title("⚛️ DOST Grants-in-Aid (GIA) Dashboard")
st.caption("Interactive analysis of projects, funding, beneficiaries, and agencies.")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Projects", f"{len(fdf):,}")
c2.metric("Agencies", f"{fdf.get('implementing_agency', pd.Series(dtype=str)).dropna().nunique():,}")
c3.metric("Beneficiaries", f"{fdf.get('beneficiaries', pd.Series(dtype=str)).dropna().nunique():,}")
c4.metric("Total Funding (₱)", f"{fdf['funding_num'].sum():,.0f}")

# ----------------------
# Visualizations
# ----------------------
tabs = st.tabs([
    "📈 Projects per Year",
    "💰 Funding by KRA",
    "🏛️ Top Agencies (Pareto)",
    "👥 Top Beneficiaries",
    "📊 Funding Distribution",
    "📅 Project Timeline"
])

# 1) Projects per Year
with tabs[0]:
    by_year = fdf.groupby("year").size().reset_index(name="Projects")
    st.altair_chart(
        alt.Chart(by_year).mark_bar().encode(
            x=alt.X("year:O", title="Year"),
            y="Projects:Q",
            tooltip=["year:O","Projects:Q"]
        ).properties(height=320),
        use_container_width=True
    )

# 2) Funding by KRA
with tabs[1]:
    if "key_result_areas_(kra)" in df.columns or "kra" in df.columns:
        col = "kra" if "kra" in fdf else "key_result_areas_(kra)"
        kdf = fdf.groupby(col)["funding_num"].sum().reset_index()
        st.altair_chart(
            alt.Chart(kdf).mark_bar().encode(
                x=alt.X(col+":N", sort="-y", title="KRA"),
                y=alt.Y("funding_num:Q", title="Funding (₱)"),
                tooltip=[col+":N", alt.Tooltip("funding_num:Q", format=",.0f", title="₱")]
            ).properties(height=320),
            use_container_width=True
        )
    else:
        st.info("KRA column not available.")

# 3) Agencies Pareto
with tabs[2]:
    if "implementing_agency" in fdf:
        adf = (
            fdf.groupby("implementing_agency")["funding_num"]
               .sum()
               .reset_index()
               .sort_values("funding_num", ascending=False)
        )
        adf["cum_share"] = adf["funding_num"].cumsum() / adf["funding_num"].sum() * 100
        top_n = st.slider("Top agencies to display", 5, min(30, len(adf)), 15)
        view = adf.head(top_n)

        bars = alt.Chart(view).mark_bar().encode(
            x=alt.X("implementing_agency:N", sort="-y"),
            y=alt.Y("funding_num:Q", title="₱"),
            tooltip=["implementing_agency:N", alt.Tooltip("funding_num:Q", format=",.0f")]
        )
        line = alt.Chart(view).mark_line(point=True, color="red").encode(
            x="implementing_agency:N",
            y=alt.Y("cum_share:Q", axis=alt.Axis(title="Cumulative %")),
            tooltip=["implementing_agency:N", alt.Tooltip("cum_share:Q", format=".1f")]
        )
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=360), use_container_width=True)

# 4) Beneficiaries
with tabs[3]:
    if "beneficiaries" in fdf:
        vals = fdf["beneficiaries"].dropna().astype(str)
        exploded = []
        for v in vals:
            exploded.extend([p.strip() for p in re.split(r"[;,/]+", v) if p.strip()])
        if exploded:
            ben = pd.Series(exploded).value_counts().reset_index()
            ben.columns = ["Beneficiary","Projects"]
            st.altair_chart(
                alt.Chart(ben.head(20)).mark_bar().encode(
                    x=alt.X("Projects:Q"),
                    y=alt.Y("Beneficiary:N", sort="-x"),
                    tooltip=["Beneficiary:N","Projects:Q"]
                ).properties(height=400),
                use_container_width=True
            )
        else:
            st.info("No beneficiary info to display.")

# 5) Funding Distribution (boxplot)
with tabs[4]:
    if "kra" in fdf:
        st.altair_chart(
            alt.Chart(fdf).mark_boxplot().encode(
                x="kra:N",
                y=alt.Y("funding_num:Q", title="Funding (₱)"),
                tooltip=["kra:N","funding_num:Q"]
            ).properties(height=350),
            use_container_width=True
        )
        st.caption("Boxplot shows median, quartiles, and outliers per KRA.")
    else:
        st.info("KRA column not available.")

# 6) Timeline (Gantt)
with tabs[5]:
    if {"project_title","start_dt","end_dt"}.issubset(fdf.columns):
        tdf = fdf.dropna(subset=["start_dt","end_dt"]).copy()
        max_rows = st.slider("Max projects to display", 50, 1000, 200, step=50)
        tdf = tdf.sort_values("start_dt").head(max_rows)
        st.altair_chart(
            alt.Chart(tdf).mark_bar().encode(
                x="start_dt:T",
                x2="end_dt:T",
                y=alt.Y("project_title:N", sort="-x", title=None),
                color=alt.Color("kra:N", legend=alt.Legend(title="KRA")) if "kra" in tdf.columns else alt.value("#888"),
                tooltip=["project_title","implementing_agency","kra","start_dt","end_dt","funding_num"]
            ).properties(height=min(28*len(tdf), 700)),
            use_container_width=True
        )
    else:
        st.info("Need Start/End dates for timeline.")

# ----------------------
# Data Explorer
# ----------------------
st.subheader("📑 Data Explorer")
show_cols = [c for c in ["program_title","project_title","kra","implementing_agency",
                         "beneficiaries","start","end","status","year","funding_num"] if c in fdf.columns]
st.dataframe(fdf[show_cols], use_container_width=True, hide_index=True)
st.download_button("Download filtered CSV",
    data=fdf[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="dost_gia_filtered.csv",
    mime="text/csv")

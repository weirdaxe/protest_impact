import streamlit as st
import pandas as pd
import numpy as np
import math
import ast
import os
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt

# =========================================================
# 1. GLOBAL CONSTANTS / HELPERS
# =========================================================

GFC_START = pd.Timestamp("2007-07-01")
GFC_END = pd.Timestamp("2010-12-31")
COVID_START = pd.Timestamp("2019-12-01")
COVID_END = pd.Timestamp("2021-12-31")
BASE22_START = pd.Timestamp("2022-01-01")
BASE22_END = pd.Timestamp("2023-12-31")


def is_in_gfc(ts: pd.Timestamp) -> bool:
    if pd.isna(ts):
        return False
    return GFC_START <= ts <= GFC_END


def is_in_covid(ts: pd.Timestamp) -> bool:
    if pd.isna(ts):
        return False
    return COVID_START <= ts <= COVID_END


def is_in_2223(ts: pd.Timestamp) -> bool:
    if pd.isna(ts):
        return False
    return BASE22_START <= ts <= BASE22_END


def is_excluded_date(
    ts: pd.Timestamp,
    exclude_gfc: bool,
    exclude_covid: bool,
    exclude_2223: bool,
) -> bool:
    if pd.isna(ts):
        return False
    if exclude_gfc and is_in_gfc(ts):
        return True
    if exclude_covid and is_in_covid(ts):
        return True
    if exclude_2223 and is_in_2223(ts):
        return True
    return False


def parse_group(x):
    if isinstance(x, str) and x.startswith("["):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []


VARIANT_TO_MASTER = {
    "Afghanistan, Islamic Republic of": "Afghanistan",
    "Armenia, Republic of": "Armenia",
    "Azerbaijan, Republic of": "Azerbaijan",
    "Bahrain, Kingdom of": "Bahrain",
    "Belarus, Republic of": "Belarus",
    "China, People's Republic of": "China (inc. Hong Kong SAR results)",
    "Egypt, Arab Republic of": "Egypt",
    "Estonia, Republic of": "Estonia",
    "Eswatini, Kingdom of": "Eswatini",
    "Ethiopia, The Federal Democratic Republic of": "Ethiopia",
    "Hong Kong Special Administrative Region, People's Republic of China":
        "Hong Kong SAR (inc. China results)",
    "Iran, Islamic Republic of": "Iran",
    "Kazakhstan, Republic of": "Kazakhstan",
    "Korea, Republic of": "Korea",
    "Kosovo, Republic of": "Kosovo",
    "Latvia, Republic of": "Latvia",
    "Lithuania, Republic of": "Lithuania",
    "Madagascar, Republic of": "Madagascar",
    "Mauritania, Islamic Republic of": "Mauritania",
    "Moldova, Republic of": "Moldova",
    "Montenegro": "Montenegro, Rep. of",
    "Netherlands, The": "Netherlands",
    "North Macedonia, Republic of": "North Macedonia",
    "Poland, Republic of": "Poland",
    "Russian Federation": "Russia",
    "Serbia, Republic of": "Serbia",
    "Slovenia, Republic of": "Slovenia",
    "Syrian Arab Republic": "Syria",
    "Tajikistan, Republic of": "Tajikistan",
    "Tanzania, United Republic of": "Tanzania",
    "Timor-Leste, Democratic Republic of": "Timor-Leste, Dem. Rep. of",
    "Türkiye, Republic of": "Turkey",
    "Uzbekistan, Republic of": "Uzbekistan",
    "Venezuela, República Bolivariana de": "Venezuela",
    "West Bank and Gaza": "West Bank & Gaza",
    "Yemen Arab Republic": "Yemen",
    "Yemen, People's Democratic Republic of": "Yemen",
    "Yemen, Republic of": "Yemen",
}


def canonicalize_factory(master_names: List[str]):
    master_set = set(master_names)

    def canonicalize(name):
        if name is None or (isinstance(name, float) and math.isnan(name)):
            return None
        s = str(name)
        if s in master_set:
            return s
        if s in VARIANT_TO_MASTER:
            return VARIANT_TO_MASTER[s]
        if "," in s:
            base = s.split(",", 1)[0]
            if base in master_set:
                return base
        return None

    return canonicalize


def winsorize_panel(df: pd.DataFrame,
                    lower_q: float = 0.005,
                    upper_q: float = 0.995) -> pd.DataFrame:
    out = df.copy()
    vals = pd.to_numeric(out["value"], errors="coerce").dropna()
    if vals.empty:
        return out
    lo = vals.quantile(lower_q)
    hi = vals.quantile(upper_q)
    out["value"] = pd.to_numeric(out["value"], errors="coerce").clip(lo, hi)
    return out


def zscore_by_country(df: pd.DataFrame) -> pd.DataFrame:
    def _z(g):
        v = pd.to_numeric(g["value"], errors="coerce")
        mu = v.mean()
        sigma = v.std(ddof=0)
        g = g.copy()
        if sigma == 0 or pd.isna(sigma):
            g["value"] = 0.0
        else:
            g["value"] = (v - mu) / sigma
        return g

    return df.groupby("Country_standard", group_keys=False).apply(_z)


def minmax_normalize_event_paths(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Min–max normalisation per event over the full T-window:
    (value - min) / (max - min) in that event's window.
    """
    if panel is None or panel.empty:
        return panel

    def _mm(g: pd.DataFrame) -> pd.DataFrame:
        v = pd.to_numeric(g["value"], errors="coerce")
        vmin = v.min()
        vmax = v.max()
        g = g.copy()
        if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
            g["value"] = 0.0
        else:
            g["value"] = (v - vmin) / (vmax - vmin)
        return g

    return panel.groupby("event_id", group_keys=False).apply(_mm)


# ---------------------------------------------------------
# Region mapping
# ---------------------------------------------------------

SSA_COUNTRIES = {
    "South Africa", "Nigeria", "Kenya", "Ghana", "Tanzania", "Uganda", "Zambia",
    "Zimbabwe", "Angola", "Namibia", "Botswana", "Mozambique", "Rwanda",
    "Ethiopia", "Senegal", "Ivory Coast", "Côte d'Ivoire", "Cameroon",
    "Mauritius", "Madagascar", "Malawi", "Burkina Faso", "Mali", "Niger",
    "Sierra Leone", "Liberia", "Benin", "Togo", "Gabon", "Congo", "Congo, Dem. Rep.",
    "Eswatini", "Lesotho", "Cape Verde"
}

MENA_COUNTRIES = {
    "Algeria", "Bahrain", "Egypt", "Iraq", "Jordan", "Kuwait", "Lebanon",
    "Libya", "Morocco", "Oman", "Qatar", "Saudi Arabia", "Syria", "Tunisia",
    "United Arab Emirates", "UAE", "Yemen", "Israel", "Turkey", "Iran"
}

CEE_CIS_COUNTRIES = {
    "Poland", "Czech Republic", "Czechia", "Hungary", "Slovakia", "Slovenia",
    "Croatia", "Bosnia and Herzegovina", "Serbia", "Montenegro, Rep. of",
    "North Macedonia", "Albania", "Bulgaria", "Romania", "Estonia", "Latvia",
    "Lithuania", "Russia", "Ukraine", "Belarus", "Moldova", "Georgia",
    "Armenia", "Azerbaijan", "Kazakhstan", "Kyrgyz Republic", "Kyrgyzstan",
    "Uzbekistan", "Tajikistan", "Turkmenistan"
}

LATAM_COUNTRIES = {
    "Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru",
    "Venezuela", "Uruguay", "Paraguay", "Bolivia", "Ecuador",
    "Costa Rica", "Panama", "Guatemala", "Honduras", "El Salvador",
    "Nicaragua", "Dominican Republic", "Jamaica", "Trinidad and Tobago",
    "Bahamas", "Barbados"
}

ASIA_EM_COUNTRIES = {
    "China (inc. Hong Kong SAR results)", "Hong Kong SAR (inc. China results)",
    "China", "India", "Indonesia", "Malaysia", "Philippines", "Thailand",
    "Vietnam", "Pakistan", "Bangladesh", "Sri Lanka", "Mongolia",
    "Cambodia", "Laos", "Myanmar", "Fiji"
}

DEVELOPED_COUNTRIES = {
    "United States", "United States of America", "US", "USA",
    "Canada",
    "Germany", "France", "United Kingdom", "UK", "Italy", "Spain",
    "Netherlands", "Belgium", "Austria", "Sweden", "Norway", "Denmark",
    "Finland", "Ireland", "Switzerland", "Portugal", "Greece",
    "Iceland", "Luxembourg",
    "Japan", "Australia", "New Zealand",
    "Korea", "Singapore"
}


def country_to_region(name: Optional[str]) -> Optional[str]:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    n = str(name)
    if n in SSA_COUNTRIES:
        return "SSA"
    if n in MENA_COUNTRIES:
        return "MENA"
    if n in CEE_CIS_COUNTRIES:
        return "CEE+CIS"
    if n in LATAM_COUNTRIES:
        return "LATAM"
    if n in ASIA_EM_COUNTRIES:
        return "Asia"
    if n in DEVELOPED_COUNTRIES:
        return "Developed"
    return None


# =========================================================
# 2. INDICATOR PANEL BUILDERS
# =========================================================

def make_gdp_growth_panels(df_gdp_raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = df_gdp_raw.copy()
    df = df[df["Country_standard"].notna()].copy()
    df = df[df["INDICATOR"] == "Gross domestic product (GDP)"].copy()
    if "PRICE_TYPE" in df.columns:
        df = df[df["PRICE_TYPE"] == "Constant prices"].copy()
    if "FREQUENCY" in df.columns:
        df = df[df["FREQUENCY"] == "Quarterly"].copy()

    df["period"] = pd.PeriodIndex(df["TIME_PERIOD"], freq="Q")
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])

    if "S_ADJUSTMENT" in df.columns:
        def choose_sa(grp):
            if (grp["S_ADJUSTMENT"] == "Seasonally adjusted (SA)").any():
                return grp[grp["S_ADJUSTMENT"] == "Seasonally adjusted (SA)"]
            return grp
        df = df.groupby("Country_standard", group_keys=False).apply(choose_sa)

    df = df.sort_values(["Country_standard", "period"])
    df["gdp_yoy"] = df.groupby("Country_standard")["OBS_VALUE"].pct_change(4) * 100.0
    df["gdp_qoq"] = df.groupby("Country_standard")["OBS_VALUE"].pct_change(1) * 100.0

    gdp_yoy = df[["Country_standard", "period", "gdp_yoy"]].rename(
        columns={"gdp_yoy": "value"}
    ).dropna()
    gdp_qoq = df[["Country_standard", "period", "gdp_qoq"]].rename(
        columns={"gdp_qoq": "value"}
    ).dropna()

    gdp_yoy = winsorize_panel(gdp_yoy)
    gdp_qoq = winsorize_panel(gdp_qoq)

    return {"gdp_yoy": gdp_yoy, "gdp_qoq": gdp_qoq}


def make_gdp_level_panel(df_gdp_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_gdp_raw.copy()
    df = df[df["Country_standard"].notna()].copy()
    df = df[df["INDICATOR"] == "Gross domestic product (GDP)"].copy()
    if "PRICE_TYPE" in df.columns:
        df = df[df["PRICE_TYPE"] == "Constant prices"].copy()
    if "FREQUENCY" in df.columns:
        df = df[df["FREQUENCY"] == "Quarterly"].copy()

    df["period"] = pd.PeriodIndex(df["TIME_PERIOD"], freq="Q")
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])
    df = df.sort_values(["Country_standard", "period"])
    df = df.rename(columns={"OBS_VALUE": "value"})
    return df[["Country_standard", "period", "value"]]


def make_cpi_yoy_panel(df_cpi_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_cpi_raw.copy()
    df = df[df["Country_standard"].notna()].copy()
    df = df[
        (df["INDEX_TYPE"] == "Consumer price index (CPI)") &
        (df["COICOP_1999"] == "All Items") &
        (df["FREQUENCY"] == "Monthly") &
        (df["TYPE_OF_TRANSFORMATION"].astype(str)
           .str.contains("Year-over-year", na=False, regex=False))
    ].copy()

    df["period"] = df["TIME_PERIOD"].astype(str).str.replace("-M", "-", regex=False)
    df["period"] = pd.PeriodIndex(df["period"], freq="M")
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])
    df = df.sort_values(["Country_standard", "period"])
    df = df.rename(columns={"OBS_VALUE": "value"})

    df = winsorize_panel(df)
    return df[["Country_standard", "period", "value"]]


def make_ca_panel(df_ca_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Current account balance, net (credits less debits), USD, quarterly.
    Raw level; winsorised.
    """
    df = df_ca_raw.copy()
    df = df[df["Country_standard"].notna()].copy()

    if "INDICATOR" in df.columns:
        df = df[
            df["INDICATOR"].astype(str).str.contains(
                "Current account balance (credit less debit)",
                na=False,
                regex=False,
            )
        ].copy()

    if "BOP_ACCOUNTING_ENTRY" in df.columns:
        df = df[
            df["BOP_ACCOUNTING_ENTRY"].astype(str).str.contains(
                "Net (credits less debits)",
                na=False,
                regex=False,
            )
        ].copy()

    if "UNIT" in df.columns:
        df = df[df["UNIT"] == "US dollar"].copy()

    if "FREQUENCY" in df.columns:
        df = df[df["FREQUENCY"] == "Quarterly"].copy()

    df["period"] = pd.PeriodIndex(df["TIME_PERIOD"].astype(str), freq="Q")
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])
    df = df.rename(columns={"OBS_VALUE": "value"})
    df = df.sort_values(["Country_standard", "period"])
    df = winsorize_panel(df)
    return df[["Country_standard", "period", "value"]]


def make_fdi_panel(df_fdi_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Direct investment, net (assets-liabilities), USD, quarterly.
    Raw level; winsorised.
    """
    df = df_fdi_raw.copy()
    df = df[df["Country_standard"].notna()].copy()

    if "INDICATOR" in df.columns:
        df = df[
            df["INDICATOR"].astype(str).str.contains(
                "Direct investment, Total financial assets/liabilities",
                na=False,
                regex=False,
            )
        ].copy()

    if "BOP_ACCOUNTING_ENTRY" in df.columns:
        df = df[
            df["BOP_ACCOUNTING_ENTRY"].astype(str).str.contains(
                "Net (net acquisition of financial assets less net incurrence of liabilities)",
                na=False,
                regex=False,
            )
        ].copy()

    if "UNIT" in df.columns:
        df = df[df["UNIT"] == "US dollar"].copy()

    if "FREQUENCY" in df.columns:
        df = df[df["FREQUENCY"] == "Quarterly"].copy()

    df["period"] = pd.PeriodIndex(df["TIME_PERIOD"].astype(str), freq="Q")
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])
    df = df.rename(columns={"OBS_VALUE": "value"})
    df = df.sort_values(["Country_standard", "period"])
    df = winsorize_panel(df)
    return df[["Country_standard", "period", "value"]]


def make_reserves_panel(df_reserves_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Official reserve assets, monthly.
    Returns level, y/y %, and z-score; app uses the level.
    """
    df = df_reserves_raw.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    if "Country_standard" in df.columns:
        df["Country_standard"] = df["Country_standard"].astype(str)
    elif "COUNTRY" in df.columns:
        df["Country_standard"] = df["COUNTRY"].astype(str)
    else:
        raise KeyError("Expected 'Country_standard' or 'COUNTRY' column in reserves data.")

    df = df[df["Country_standard"].notna()].copy()

    if "INDICATOR" in df.columns:
        mask_indicator = df["INDICATOR"].astype(str).str.contains(
            "Official Reserve Assets", na=False, regex=False
        )
        df = df[mask_indicator].copy()

    if "FREQUENCY" in df.columns:
        df = df[df["FREQUENCY"] == "Monthly"].copy()

    period_str = df["TIME_PERIOD"].astype(str).str.replace("-M", "-", regex=False)
    df["period"] = pd.PeriodIndex(period_str, freq="M")

    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])
    df = df.rename(columns={"OBS_VALUE": "value"})

    df = df.sort_values(["Country_standard", "period"])

    # y/y %
    df["yoy_reserves"] = (
        df.groupby("Country_standard")["value"]
          .pct_change(12)
          .mul(100)
    )

    # z-score of level
    tmp_z = df[["Country_standard", "period", "value"]].copy()
    tmp_z = zscore_by_country(tmp_z)
    df["z_reserves"] = tmp_z["value"].values

    return df[["Country_standard", "period", "value", "yoy_reserves", "z_reserves"]]


def make_budget_panel(df_budget_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Budget balance (net lending / net borrowing), domestic currency, quarterly.
    Raw level; OBS_VALUE scaled by SCALE (Billions/Millions/...).
    """
    df = df_budget_raw.copy()
    df = df[df["Country_standard"].notna()].copy()

    if "ACCOUNTS" in df.columns:
        df = df[df["ACCOUNTS"] == "Statement of operations"].copy()
    if "SECTOR" in df.columns:
        df = df[df["SECTOR"] == "Budgetary central government"].copy()
    if "INDICATOR" in df.columns:
        df = df[
            df["INDICATOR"].astype(str).str.contains(
                "Net lending (+) / net borrowing (-), Transactions",
                na=False,
                regex=False,
            )
        ].copy()
    if "FREQUENCY" in df.columns:
        df = df[df["FREQUENCY"] == "Quarterly"].copy()

    df["period"] = pd.PeriodIndex(df["TIME_PERIOD"].astype(str), freq="Q")

    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

    if "SCALE" in df.columns:
        scale_map = {
            "Billions": 1e9,
            "Millions": 1e6,
            "Thousands": 1e3,
        }
        df["scale_factor"] = df["SCALE"].map(scale_map).fillna(1.0)
        df["OBS_VALUE"] = df["OBS_VALUE"] * df["scale_factor"]
        df = df.drop(columns=["scale_factor"])

    df = df.dropna(subset=["OBS_VALUE"])
    df = df.rename(columns={"OBS_VALUE": "value"})
    df = df.sort_values(["Country_standard", "period"])
    df = winsorize_panel(df)
    return df[["Country_standard", "period", "value"]]


def make_fx_panel(df_fx_raw: pd.DataFrame) -> pd.DataFrame:
    """
    FX panel with:
      - value: DC per USD level (EoP)
      - mm_fx: m/m % change
      - yy_fx: y/y % change
      - mm_fx_ra: risk-adjusted m/m (mm_fx / vol)
    App uses only level (value); returns are kept but not used in UI.
    """
    df = df_fx_raw.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    if "Country_standard" in df.columns:
        df["Country_standard"] = df["Country_standard"].astype(str)
    elif "COUNTRY" in df.columns:
        df["Country_standard"] = df["COUNTRY"].astype(str)
    else:
        raise KeyError("Expected 'Country_standard' or 'COUNTRY' column in FX data.")

    df = df[df["Country_standard"].notna()].copy()

    if "INDICATOR" in df.columns:
        mask_indicator = df["INDICATOR"].astype(str).str.contains(
            "Domestic currency per US Dollar", na=False, regex=False
        )
        df = df[mask_indicator].copy()

    if "FREQUENCY" in df.columns:
        df = df[df["FREQUENCY"] == "Monthly"].copy()

    period_str = df["TIME_PERIOD"].astype(str).str.replace("-M", "-", regex=False)
    df["period"] = pd.PeriodIndex(period_str, freq="M")

    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])
    df = df.rename(columns={"OBS_VALUE": "value"})

    df = df.sort_values(["Country_standard", "period"])

    df["mm_fx"] = (
        df.groupby("Country_standard")["value"]
          .pct_change(1)
          .mul(100)
    )
    df["yy_fx"] = (
        df.groupby("Country_standard")["value"]
          .pct_change(12)
          .mul(100)
    )

    def _add_ra(g: pd.DataFrame) -> pd.DataFrame:
        s = g["mm_fx"].dropna()
        sigma = s.std(ddof=0)
        g = g.copy()
        if pd.isna(sigma) or sigma == 0:
            g["mm_fx_ra"] = np.nan
        else:
            g["mm_fx_ra"] = g["mm_fx"] / sigma
        return g

    df = df.groupby("Country_standard", group_keys=False).apply(_add_ra)

    return df[["Country_standard", "period", "value", "mm_fx", "yy_fx", "mm_fx_ra"]]


# =========================================================
# 3. LOAD DATA (CACHED)
# =========================================================

@st.cache_data
def load_data():
    # Protest data
    protest = pd.read_csv("protest.csv")

    protest["event_date"] = pd.to_datetime(protest["Date"], errors="coerce")
    protest["Country_standard"] = protest["cty.name"]
    protest["Group_list"] = protest["Group"].apply(parse_group)

    for col in ["Violent_0_1", "GovtChange_0_1", "Revolution_0_1", "Duration_days"]:
        if col in protest.columns:
            protest[col] = pd.to_numeric(protest[col], errors="coerce")

    protest["Region"] = protest["Country_standard"].apply(country_to_region)

    canonicalize = canonicalize_factory(list(protest["cty.name"].unique()))

    # IMF data: GDP
    df_gdp = pd.read_csv("imf_gdp_all.csv")
    df_gdp.columns = df_gdp.columns.str.replace("\ufeff", "", regex=False).str.strip()
    df_gdp["Country_standard"] = df_gdp["COUNTRY"].map(canonicalize)

    # CPI
    df_cpi = pd.read_csv("imf_cpi_all.csv")
    df_cpi.columns = df_cpi.columns.str.replace("\ufeff", "", regex=False).str.strip()
    df_cpi["Country_standard"] = df_cpi["COUNTRY"].map(canonicalize)

    # CA
    df_ca = pd.read_csv("imf_ca_all.csv")
    df_ca.columns = df_ca.columns.str.replace("\ufeff", "", regex=False).str.strip()
    df_ca["Country_standard"] = df_ca["COUNTRY"].map(canonicalize)

    # FDI
    df_fdi = pd.read_csv("imf_fdi_all.csv")
    df_fdi.columns = df_fdi.columns.str.replace("\ufeff", "", regex=False).str.strip()
    df_fdi["Country_standard"] = df_fdi["COUNTRY"].map(canonicalize)

    # Reserves
    df_res = pd.read_csv("imf_reserves_all.csv")
    df_res.columns = df_res.columns.str.replace("\ufeff", "", regex=False).str.strip()
    df_res["Country_standard"] = df_res["COUNTRY"].map(canonicalize)

    # Budget
    df_budget = pd.read_csv("imf_budget_all.csv")
    df_budget.columns = df_budget.columns.str.replace("\ufeff", "", regex=False).str.strip()
    df_budget["Country_standard"] = df_budget["COUNTRY"].map(canonicalize)

    indicator_panels: Dict[str, pd.DataFrame] = {}
    indicator_freq: Dict[str, str] = {}
    indicator_labels: Dict[str, str] = {}

    # GDP growth
    gdp_panels = make_gdp_growth_panels(df_gdp)
    indicator_panels["gdp_yoy"] = gdp_panels["gdp_yoy"]
    indicator_freq["gdp_yoy"] = "Q"
    indicator_labels["gdp_yoy"] = "GDP y/y (%)"

    indicator_panels["gdp_qoq"] = gdp_panels["gdp_qoq"]
    indicator_freq["gdp_qoq"] = "Q"
    indicator_labels["gdp_qoq"] = "GDP q/q (%)"

    # GDP levels (for 4q sums)
    gdp_level_q = make_gdp_level_panel(df_gdp)
    indicator_panels["gdp_level"] = gdp_level_q
    indicator_freq["gdp_level"] = "Q"
    indicator_labels["gdp_level"] = "GDP level (internal)"

    # 4-quarter rolling sum of GDP (for % of GDP scaling)
    gdp_level_q = gdp_level_q.sort_values(["Country_standard", "period"]).copy()
    gdp_level_q["gdp_4q"] = (
        gdp_level_q
        .groupby("Country_standard")["value"]
        .transform(lambda s: s.rolling(window=4, min_periods=4).sum())
    )

    gdp_4q_q = gdp_level_q[["Country_standard", "period", "gdp_4q"]].rename(
        columns={"gdp_4q": "value"}
    )
    indicator_panels["gdp_4q_q"] = gdp_4q_q
    indicator_freq["gdp_4q_q"] = "Q"
    indicator_labels["gdp_4q_q"] = "GDP 4-quarter sum (internal)"

    # Expand 4-quarter GDP sum to monthly for monthly indicators (% of GDP)
    rows = []
    for _, row in gdp_4q_q.iterrows():
        c = row["Country_standard"]
        p = row["period"]
        val4 = row["value"]
        if pd.isna(val4):
            continue
        start_m = p.start_time.to_period("M")
        end_m = p.end_time.to_period("M")
        for m in pd.period_range(start_m, end_m, freq="M"):
            rows.append({"Country_standard": c, "period": m, "value": val4})
    gdp_4q_m = pd.DataFrame(rows)
    indicator_panels["gdp_4q_m"] = gdp_4q_m
    indicator_freq["gdp_4q_m"] = "M"
    indicator_labels["gdp_4q_m"] = "GDP 4-quarter sum monthly (internal)"

    # CPI y/y (%)
    cpi_panel = make_cpi_yoy_panel(df_cpi)
    indicator_panels["cpi_yoy"] = cpi_panel
    indicator_freq["cpi_yoy"] = "M"
    indicator_labels["cpi_yoy"] = "CPI y/y (%)"

    # CA – USD level
    ca_panel = make_ca_panel(df_ca)
    indicator_panels["ca_level"] = ca_panel
    indicator_freq["ca_level"] = "Q"
    indicator_labels["ca_level"] = "Current account balance, net (USD)"

    # FDI level – USD
    fdi_panel = make_fdi_panel(df_fdi)
    indicator_panels["fdi_level"] = fdi_panel
    indicator_freq["fdi_level"] = "Q"
    indicator_labels["fdi_level"] = "FDI, net (USD)"

    # Reserves – level (USD, monthly)
    res_panel = make_reserves_panel(df_res)
    indicator_panels["res_level"] = res_panel[["Country_standard", "period", "value"]]
    indicator_freq["res_level"] = "M"
    indicator_labels["res_level"] = "Official reserves (USD)"

    # Budget balance (domestic currency)
    budget_panel = make_budget_panel(df_budget)
    indicator_panels["budget_level"] = budget_panel
    indicator_freq["budget_level"] = "Q"
    indicator_labels["budget_level"] = "Budget balance (net lending/borrowing, dom. currency)"

    # FX: use your working make_fx_panel, but app only uses level
    if os.path.exists("imf_fx_all.csv"):
        df_fx = pd.read_csv("imf_fx_all.csv")
        df_fx.columns = df_fx.columns.str.replace("\ufeff", "", regex=False).str.strip()
        df_fx["Country_standard"] = df_fx["COUNTRY"].map(canonicalize)
        fx_panel = make_fx_panel(df_fx)
        indicator_panels["fx_level"] = fx_panel[["Country_standard", "period", "value"]]
        indicator_freq["fx_level"] = "M"
        indicator_labels["fx_level"] = "FX level (DC per USD, EoM)"

    return protest, indicator_panels, indicator_freq, indicator_labels


# =========================================================
# 4. PSEUDO CANDIDATES
# =========================================================

def _build_pseudo_candidates_core(
    series_df: pd.DataFrame,
    protest_df: pd.DataFrame,
    freq: str,
    pre_periods: int,
    post_periods: int,
    no_protest_years_pre: int = 3,
    no_protest_years_post: int = 1,
) -> pd.DataFrame:
    sdf = series_df.sort_values(["Country_standard", "period"])
    sdf_idx = sdf.set_index(["Country_standard", "period"]).index

    ev = protest_df.copy()
    ev = ev[ev["Country_standard"].isin(sdf["Country_standard"].unique())].copy()
    ev = ev[ev["event_date"].notna()].copy()

    if freq == "Q":
        ev["event_period"] = ev["event_date"].dt.to_period("Q")
        per_year = 4
    else:
        ev["event_period"] = ev["event_date"].dt.to_period("M")
        per_year = 12

    protest_by_cty: Dict[str, np.ndarray] = {}
    for c, grp in ev.groupby("Country_standard"):
        ps = grp["event_period"].dropna().unique()
        protest_by_cty[c] = np.array([p.ordinal for p in ps], dtype=int)

    rows = []

    for c, grp in sdf.groupby("Country_standard"):
        if c not in protest_by_cty:
            continue
        periods = np.sort(grp["period"].unique())
        if len(periods) == 0:
            continue
        protest_ord = protest_by_cty[c]

        for p in periods:
            ok = True
            for k in range(-pre_periods, post_periods + 1):
                pt = p + k
                if (c, pt) not in sdf_idx:
                    ok = False
                    break
            if not ok:
                continue

            idx = p.ordinal
            if protest_ord.size > 0:
                lower = idx - no_protest_years_pre * per_year
                upper = idx + no_protest_years_post * per_year
                if ((protest_ord >= lower) & (protest_ord <= upper)).any():
                    continue

            rows.append(
                {
                    "Country_standard": c,
                    "event_period": p,
                    "event_date": p.to_timestamp(),
                }
            )

    return pd.DataFrame(rows)


@st.cache_data
def get_pseudo_candidates(
    indicator_key: str,
    exclude_gfc: bool,
    exclude_covid: bool,
    exclude_2223: bool,
) -> pd.DataFrame:
    protest, indicator_panels, indicator_freq, _ = load_data()

    if indicator_key not in indicator_panels:
        return pd.DataFrame(columns=["Country_standard", "event_period", "event_date"])

    series_df = indicator_panels[indicator_key].copy()
    freq = indicator_freq[indicator_key]

    if exclude_gfc or exclude_covid or exclude_2223:
        series_df["ts"] = series_df["period"].dt.to_timestamp()
        series_df = series_df[
            ~series_df["ts"].apply(
                lambda x: is_excluded_date(x, exclude_gfc, exclude_covid, exclude_2223)
            )
        ].copy()
        series_df = series_df.drop(columns=["ts"])
        protest_used = protest[
            ~protest["event_date"].apply(
                lambda x: is_excluded_date(x, exclude_gfc, exclude_covid, exclude_2223)
            )
        ].copy()
    else:
        protest_used = protest.copy()

    if freq == "Q":
        pre = post = 4
    else:
        pre = post = 12

    pseudo = _build_pseudo_candidates_core(
        series_df,
        protest_used,
        freq=freq,
        pre_periods=pre,
        post_periods=post,
        no_protest_years_pre=3,
        no_protest_years_post=1,
    )
    return pseudo


# =========================================================
# 5. EVENT-STUDY HELPERS
# =========================================================

def build_event_panel(
    series_df: pd.DataFrame,
    events_df: pd.DataFrame,
    freq: str,
    pre_periods: int,
    post_periods: int,
) -> pd.DataFrame:
    """
    Build event panel and then drop all event_ids that have any NaN in value
    over the full [-pre_periods, +post_periods] window.
    """
    sdf = series_df.set_index(["Country_standard", "period"]).sort_index()
    events = events_df.copy()
    if freq == "Q":
        events["event_period"] = events["event_date"].dt.to_period("Q")
    else:
        events["event_period"] = events["event_date"].dt.to_period("M")

    rows = []
    for _, ev in events.iterrows():
        c = ev["Country_standard"]
        t0 = ev["event_period"]
        eid = ev["event_id"]
        if pd.isna(t0):
            continue
        for k in range(-pre_periods, post_periods + 1):
            t = t0 + k
            try:
                v = sdf.loc[(c, t), "value"]
            except KeyError:
                v = np.nan
            rows.append(
                {
                    "event_id": eid,
                    "Country_standard": c,
                    "event_period": t0,
                    "rel_t": k,
                    "period": t,
                    "value": v,
                }
            )
    panel = pd.DataFrame(rows)
    # drop all event_ids that have any NaN in value
    if not panel.empty:
        valid_mask = panel.groupby("event_id")["value"].transform(lambda s: s.notna().all())
        panel = panel[valid_mask].copy()

    return panel


def mean_path(panel: pd.DataFrame,
              lower_q: float = 0.05,
              upper_q: float = 0.95,
              agg: str = "mean") -> pd.DataFrame:
    if panel is None or panel.empty:
        return pd.DataFrame(columns=["rel_t", "mean_value", "n_events"])

    panel = panel.copy()

    def _trim_group(g: pd.DataFrame) -> pd.DataFrame:
        vals = pd.to_numeric(g["value"], errors="coerce").to_numpy()
        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            return g.iloc[0:0]

        v_valid = vals[valid]
        try:
            lo = float(np.nanquantile(v_valid, lower_q))
            hi = float(np.nanquantile(v_valid, upper_q))
        except Exception:
            return g.loc[valid]

        keep = valid & (vals >= lo) & (vals <= hi)
        return g.loc[keep]

    trimmed = panel.groupby("rel_t", group_keys=False).apply(_trim_group)

    if trimmed.empty:
        trimmed = panel.copy()

    if agg == "median":
        mean_val = trimmed.groupby("rel_t")["value"].median()
    else:
        mean_val = trimmed.groupby("rel_t")["value"].mean()

    n_events = (
        trimmed[trimmed["value"].notna()]
        .groupby("rel_t")["event_id"]
        .nunique()
        .reindex(mean_val.index)
        .fillna(0)
        .astype(int)
        .values
    )

    path = pd.DataFrame(
        {"rel_t": mean_val.index, "mean_value": mean_val.values}
    )
    path["n_events"] = n_events

    return path


def compute_pre_trend_metrics(
    series_df: pd.DataFrame,
    events_df: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    sdf = series_df.set_index(["Country_standard", "period"]).sort_index()

    ev = events_df.copy()
    if freq == "Q":
        ev["event_period"] = ev["event_date"].dt.to_period("Q")
        pre_periods = 4
        min_obs = 3
    else:
        ev["event_period"] = ev["event_date"].dt.to_period("M")
        pre_periods = 12
        min_obs = 6

    rows = []

    for _, e in ev.iterrows():
        c = e["Country_standard"]
        t0 = e["event_period"]
        eid = e["event_id"]

        if pd.isna(t0):
            continue

        vals = []
        for k in range(-pre_periods, 0):
            t = t0 + k
            try:
                v = sdf.loc[(c, t), "value"]
            except KeyError:
                continue
            if not pd.isna(v):
                vals.append(v)

        if len(vals) < min_obs:
            continue

        y = np.asarray(vals, dtype=float)
        mu = y.mean()
        if mu != 0:
            y = y / abs(mu)

        t_idx = np.arange(len(y), dtype=float)

        if len(y) > 1 and np.var(t_idx, ddof=1) > 0:
            cov_ty = np.cov(t_idx, y, bias=False)[0, 1]
            slope = cov_ty / np.var(t_idx, ddof=1)
        else:
            slope = np.nan

        rows.append(
            {
                "event_id": eid,
                "Country_standard": c,
                "event_period": t0,
                "slope_per_t": slope,
            }
        )

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics["abs_slope"] = metrics["slope_per_t"].abs()
    return metrics


# =========================================================
# 6. NORMALISATION AND RATE-TRANSFORM HELPERS
# =========================================================

def get_window_lengths(freq: str, rate_option: str) -> Tuple[int, int, int, int]:
    if freq == "Q":
        base_pre = base_post = 4
        per_year = 4
    else:
        base_pre = base_post = 12
        per_year = 12

    extra_pre = 0
    extra_post = 0
    if rate_option == "y/y % change":
        extra_pre = per_year
    elif rate_option == "obs/obs % change":
        extra_pre = 1

    pre_full = base_pre + extra_pre
    post_full = base_post + extra_post
    return base_pre, base_post, pre_full, post_full


def apply_scale_normalisation(
    panel: pd.DataFrame,
    freq: str,
    scale_option: str,
    gdp_level_panel: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if panel is None or panel.empty:
        return panel

    out = panel.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    if scale_option == "Raw level":
        return out

    if scale_option == "Min-max per event (T-window)":
        return minmax_normalize_event_paths(out)

    if scale_option == "As % of GDP":
        # Use rolling 4-quarter GDP sum (annualised GDP flow)
        if gdp_level_panel is None or gdp_level_panel.empty:
            return out

        gdf = gdp_level_panel.copy()
        gdf = gdf[["Country_standard", "period", "value"]].rename(
            columns={"value": "gdp_4q"}
        )

        merged = out.merge(
            gdf,
            on=["Country_standard", "period"],
            how="left",
        )
        merged["gdp_4q"] = pd.to_numeric(merged["gdp_4q"], errors="coerce")
        mask = merged["gdp_4q"].notna() & (merged["gdp_4q"] != 0)
        merged.loc[mask, "value"] = (
            merged.loc[mask, "value"] / merged.loc[mask, "gdp_4q"]
        ) * 100.0
        merged.loc[~mask, "value"] = np.nan
        return merged.drop(columns=["gdp_4q"])

    return out


def apply_rate_transform(
    panel: pd.DataFrame,
    freq: str,
    rate_option: str,
) -> pd.DataFrame:
    if panel is None or panel.empty:
        return panel

    if rate_option == "Raw":
        return panel.copy()

    out = panel.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    if freq == "Q":
        per_year = 4
    else:
        per_year = 12

    if rate_option == "y/y % change":
        shift_n = per_year
    elif rate_option == "obs/obs % change":
        shift_n = 1
    else:
        return out

    def _rate_event(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("period").copy()
        v = pd.to_numeric(g["value"], errors="coerce")
        v_shift = v.shift(shift_n)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = (v / v_shift - 1.0) * 100.0
        res[(v_shift == 0) | pd.isna(v_shift)] = np.nan
        g["value"] = res
        return g

    return out.groupby("event_id", group_keys=False).apply(_rate_event)


def transform_event_panel(
    panel: pd.DataFrame,
    freq: str,
    scale_option: str,
    rate_option: str,
    gdp_level_panel: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if panel is None or panel.empty:
        return panel

    scaled = apply_scale_normalisation(panel, freq, scale_option, gdp_level_panel)
    transformed = apply_rate_transform(scaled, freq, rate_option)
    return transformed


def complete_rel_t(path: pd.DataFrame, base_pre: int, base_post: int) -> pd.DataFrame:
    if path is None or path.empty:
        return path
    rel_range = list(range(-base_pre, base_post + 1))
    path = path.set_index("rel_t").reindex(rel_range)
    path.index.name = "rel_t"
    path["mean_value"] = path["mean_value"]
    path["n_events"] = path["n_events"].fillna(0).astype(int)
    path = path.reset_index()
    return path


def normalize_path_T0(path: pd.DataFrame) -> pd.DataFrame:
    out = path.copy()
    if out.empty:
        out["y_plot"] = np.nan
        return out
    base_series = out.loc[out["rel_t"] == 0, "mean_value"]
    if base_series.empty or pd.isna(base_series.iloc[0]):
        base = out["mean_value"].iloc[0]
    else:
        base = base_series.iloc[0]
    out["y_plot"] = out["mean_value"] - base
    return out


def align_pair_pre_level(
    path_protest: pd.DataFrame,
    path_pseudo: Optional[pd.DataFrame],
    pre_start: int,
    pre_end: int,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if path_protest is None or path_protest.empty:
        return path_protest, path_pseudo

    p = path_protest.copy()
    s = None if (path_pseudo is None or path_pseudo.empty) else path_pseudo.copy()

    mask_p = (p["rel_t"] >= pre_start) & (p["rel_t"] <= pre_end)
    if s is not None:
        mask_s = (s["rel_t"] >= pre_start) & (s["rel_t"] <= pre_end)
        combined_pre = pd.concat(
            [p.loc[mask_p, "mean_value"], s.loc[mask_s, "mean_value"]]
        ).dropna()
    else:
        combined_pre = p.loc[mask_p, "mean_value"].dropna()

    if combined_pre.empty:
        mu = p["mean_value"].mean()
    else:
        mu = combined_pre.mean()

    p["y_plot"] = p["mean_value"] - mu
    if s is not None:
        s["y_plot"] = s["mean_value"] - mu
    return p, s


# =========================================================
# 7. CORE EVENT-STUDY
# =========================================================

def run_event_study(
    indicator_key: str,
    indicator_panels: Dict[str, pd.DataFrame],
    indicator_freq: Dict[str, str],
    protest_df: pd.DataFrame,
    violent_only: bool,
    revolution_only: bool,
    govchange_only: bool,
    selected_regions: List[str],
    selected_countries: List[str],
    selected_groups: List[str],
    duration_range: Optional[Tuple[float, float]],
    exclude_gfc: bool,
    exclude_covid: bool,
    exclude_2223: bool,
    methodology: str,
    vol_q_low: float,
    vol_q_high: float,
    slope_q: float,
    pre_years_stats: int = 3,
    use_pseudo: bool = True,
    scale_option: str = "Raw level",
    rate_option: str = "Raw",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:

    if indicator_key not in indicator_panels:
        return None, None, None

    series_df = indicator_panels[indicator_key].copy()
    freq = indicator_freq[indicator_key]

    if exclude_gfc or exclude_covid or exclude_2223:
        series_df["ts"] = series_df["period"].dt.to_timestamp()
        series_df = series_df[
            ~series_df["ts"].apply(
                lambda ts: is_excluded_date(ts, exclude_gfc, exclude_covid, exclude_2223)
            )
        ].copy()
        series_df = series_df.drop(columns=["ts"])

    events = protest_df.copy()
    events = events[events["Country_standard"].isin(series_df["Country_standard"].unique())].copy()
    events = events[events["event_date"].notna()].copy()
    if exclude_gfc or exclude_covid or exclude_2223:
        events = events[
            ~events["event_date"].apply(
                lambda ts: is_excluded_date(ts, exclude_gfc, exclude_covid, exclude_2223)
            )
        ].copy()

    if selected_regions and "Region" in events.columns:
        events = events[events["Region"].isin(selected_regions)].copy()

    if selected_countries:
        events = events[events["Country_standard"].isin(selected_countries)].copy()

    if violent_only and "Violent_0_1" in events.columns:
        events = events[events["Violent_0_1"] == 1]
    if revolution_only and "Revolution_0_1" in events.columns:
        events = events[events["Revolution_0_1"] == 1]
    if govchange_only and "GovtChange_0_1" in events.columns:
        events = events[events["GovtChange_0_1"] == 1]

    if selected_groups:
        sel = set(selected_groups)
        events = events[
            events["Group_list"].apply(
                lambda gl: any(g in sel for g in (gl if isinstance(gl, list) else []))
            )
        ]

    if duration_range is not None and "Duration_days" in events.columns:
        dmin, dmax = duration_range
        events = events[
            (events["Duration_days"] >= dmin) &
            (events["Duration_days"] <= dmax)
        ]

    events = events.reset_index(drop=True)
    events["event_id"] = np.arange(len(events))

    if events.empty:
        return None, None, None

    apply_trend = (methodology == "Trend filter")

    events_current = events.copy()

    if apply_trend and not events_current.empty:
        trend_metrics = compute_pre_trend_metrics(series_df, events_current, freq)
        trend_metrics = trend_metrics.dropna(subset=["abs_slope"])
        if not trend_metrics.empty:
            thr = trend_metrics["abs_slope"].quantile(slope_q)
            keep_ids = trend_metrics[trend_metrics["abs_slope"] <= thr]["event_id"].unique()
            events_current = events_current[events_current["event_id"].isin(keep_ids)].copy()

    if events_current.empty:
        return None, None, None

    base_pre, base_post, pre_full, post_full = get_window_lengths(freq, rate_option)

    # Protest panel (full window, then drop events with any NaN inside)
    panel_protest = build_event_panel(series_df, events_current, freq, pre_full, post_full)
    if panel_protest is None or panel_protest.empty:
        return None, None, None

    # Use 4-quarter GDP sum panels for % of GDP scaling
    gdp_level_panel = indicator_panels.get("gdp_4q_q" if freq == "Q" else "gdp_4q_m")

    panel_protest = transform_event_panel(
        panel_protest,
        freq=freq,
        scale_option=scale_option,
        rate_option=rate_option,
        gdp_level_panel=gdp_level_panel,
    )

    # Keep only the plotting window
    panel_protest = panel_protest[
        (panel_protest["rel_t"] >= -base_pre) & (panel_protest["rel_t"] <= base_post)
    ].copy()

    if panel_protest.empty or not panel_protest["value"].notna().any():
        return None, None, None

    # Event IDs that actually contribute (at least one non-NaN in plotting window)
    used_ids = panel_protest[panel_protest["value"].notna()]["event_id"].unique()
    events_used_stats = events_current[events_current["event_id"].isin(used_ids)].copy()

    # Median for FX, mean for others
    agg_choice = "median" if indicator_key == "fx_level" else "mean"
    path_protest = mean_path(
        panel_protest,
        lower_q=vol_q_low,
        upper_q=vol_q_high,
        agg=agg_choice,
    )

    path_pseudo = None
    if use_pseudo:
        base_pseudo = get_pseudo_candidates(
            indicator_key, exclude_gfc, exclude_covid, exclude_2223
        )
        if not base_pseudo.empty:
            ctys = events_current["Country_standard"].unique()
            pseudo_events = base_pseudo[base_pseudo["Country_standard"].isin(ctys)].copy()
            if not pseudo_events.empty:
                pseudo_events = pseudo_events.reset_index(drop=True)
                pseudo_events["event_id"] = np.arange(len(pseudo_events))

                panel_pseudo = build_event_panel(series_df, pseudo_events, freq, pre_full, post_full)
                if not panel_pseudo.empty:
                    panel_pseudo = transform_event_panel(
                        panel_pseudo,
                        freq=freq,
                        scale_option=scale_option,
                        rate_option=rate_option,
                        gdp_level_panel=gdp_level_panel,
                    )
                    panel_pseudo = panel_pseudo[
                        (panel_pseudo["rel_t"] >= -base_pre) & (panel_pseudo["rel_t"] <= base_post)
                    ].copy()
                    if not panel_pseudo.empty and panel_pseudo["value"].notna().any():
                        path_pseudo = mean_path(
                            panel_pseudo,
                            lower_q=vol_q_low,
                            upper_q=vol_q_high,
                            agg=agg_choice,
                        )

    return path_protest, path_pseudo, events_used_stats


# =========================================================
# 8. STREAMLIT UI
# =========================================================

def main():
    st.title("Protest Event Study Dashboard")

    protest_df, indicator_panels, indicator_freq, indicator_labels = load_data()

    st.sidebar.header("Controls")

    preferred_order = [
        "gdp_yoy",
        "gdp_qoq",
        "cpi_yoy",
        "ca_level",
        "fdi_level",
        "res_level",
        "budget_level",
        "fx_level",
    ]
    ind_options = [k for k in preferred_order if k in indicator_panels]

    ind_key = st.sidebar.selectbox(
        "Indicator",
        options=ind_options,
        format_func=lambda k: indicator_labels.get(k, k),
        help=(
            "Macro/market series to study around protest dates "
            "(e.g. GDP growth, CPI, CA, FDI, reserves, budget, FX). "
            "All subsequent calculations use this indicator."
        ),
    )

    freq = indicator_freq[ind_key]

    # Scale options: only allow "% of GDP" where it makes sense
    if ind_key in ("ca_level", "fdi_level", "res_level", "budget_level"):
        scale_opts = ["Raw level", "Min-max per event (T-window)", "As % of GDP"]
    else:
        scale_opts = ["Raw level", "Min-max per event (T-window)"]

    scale_option = st.sidebar.radio(
        "Value normalisation (Group 1)",
        options=scale_opts,
        index=0,
        help=(
            "How to scale the indicator before averaging:\n"
            "• Raw level: keep original units (USD, %, domestic currency, etc.).\n"
            "• Min-max per event: rescale each event window to [0,1].\n"
            "• As % of GDP: divide by rolling 4-quarter GDP sum and express as % "
            "(where available)."
        ),
    )

    # For pre-computed growth, only "Raw" is sensible
    if ind_key in ("gdp_yoy", "gdp_qoq", "cpi_yoy"):
        rate_choices = ["Raw"]
    else:
        rate_choices = ["Raw", "y/y % change", "obs/obs % change"]

    rate_option = st.sidebar.radio(
        "Rate transformation (Group 2)",
        options=rate_choices,
        index=0,
        help=(
            "Optional additional transformation on the series:\n"
            "• Raw: no extra change.\n"
            "• y/y % change: (t vs t-4 or t-12) in %.\n"
            "• obs/obs % change: one-step % change (q/q or m/m)."
        ),
    )

    norm_option = st.sidebar.radio(
        "Plot centring",
        options=["None", "Shift T=0 to 0", "Align pre (1y before event)"],
        index=1,
        help=(
            "How to align the plotted paths vertically:\n"
            "• None: plot raw transformed levels.\n"
            "• Shift T=0 to 0: subtract value at event month/quarter.\n"
            "• Align pre (1y): subtract common mean over T-12..-1 (or T-4..-1)."
        ),
    )

    methodology = st.sidebar.selectbox(
        "Event subset / methodology",
        options=["All", "Trend filter"],
        index=0,
        help=(
            "Which protest events to keep:\n"
            "• All: use every event that passes your filters.\n"
            "• Trend filter: drop events with steep pre-trends "
            "(based on pre-event slope quantile)."
        ),
    )

    col1, col2 = st.sidebar.columns(2)
    vol_q_low = col1.number_input(
        "Trim q_low",
        min_value=0.0,
        max_value=0.49,
        value=0.05,
        step=0.01,
        help=(
            "Lower cross-sectional trim quantile at each relative period. "
            "Events below this quantile are dropped as outliers."
        ),
    )
    vol_q_high = col2.number_input(
        "Trim q_high",
        min_value=0.51,
        max_value=1.0,
        value=0.95,
        step=0.01,
        help=(
            "Upper cross-sectional trim quantile at each relative period. "
            "Events above this quantile are dropped as outliers."
        ),
    )

    slope_q = st.sidebar.number_input(
        "Max |trend| quantile (Trend filter)",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01,
        help=(
            "Used only when 'Trend filter' is selected. "
            "Events with absolute pre-trend slope above this quantile "
            "are excluded."
        ),
    )

    st.sidebar.markdown("### Excluded periods")
    exclude_gfc = st.sidebar.checkbox(
        "Exclude GFC (2007–2010)",
        value=True,
        help="Drop macro observations and protest events that fall inside the GFC window.",
    )
    exclude_covid = st.sidebar.checkbox(
        "Exclude Covid (2019–2021)",
        value=True,
        help="Drop macro observations and protest events that fall inside the Covid window.",
    )
    exclude_2223 = st.sidebar.checkbox(
        "Exclude 2022–2023",
        value=False,
        help="Drop macro observations and protest events during 2022–2023 (e.g. to avoid base effects).",
    )

    show_pseudo = st.sidebar.checkbox(
        "Show no-protest pseudo line",
        value=True,
        help=(
            "If checked, also plot a synthetic path built from periods "
            "with similar macro coverage but no nearby protests."
        ),
    )

    # Country filter
    st.sidebar.markdown("### Country filter")
    countries_available = sorted(
        protest_df["Country_standard"].dropna().unique()
    )
    selected_countries = st.sidebar.multiselect(
        "Countries",
        options=countries_available,
        default=[],
        help=(
            "Restrict to protests in these specific countries. "
            "Leave empty to use all countries in the sample."
        ),
    )

    # Region filter
    st.sidebar.markdown("### Region filter")
    regions_available = sorted(
        [r for r in protest_df["Region"].dropna().unique()]
    )
    selected_regions = st.sidebar.multiselect(
        "Regions",
        options=regions_available,
        default=[],
        help=(
            "Restrict to protests in these regions (custom EM/DM buckets). "
            "Leave empty to include all regions."
        ),
    )

    st.sidebar.markdown("### Protest filters")
    violent_only = st.sidebar.checkbox(
        "Violent protests only",
        value=False,
        help="Keep only protests flagged as violent (Violent_0_1 = 1).",
    )
    revolution_only = st.sidebar.checkbox(
        "Revolutions only",
        value=False,
        help="Keep only protests classified as revolutions (Revolution_0_1 = 1).",
    )
    govchange_only = st.sidebar.checkbox(
        "Govt change only",
        value=False,
        help="Keep only protests associated with government change (GovtChange_0_1 = 1).",
    )

    all_groups = sorted(
        {
            g
            for gl in protest_df["Group_list"]
            for g in (gl if isinstance(gl, list) else [])
        }
    )
    selected_groups = st.sidebar.multiselect(
        "Filter by groups",
        options=all_groups,
        help="Restrict protests to those involving any of the selected protester groups.",
    )

    duration_range = None
    if "Duration_days" in protest_df.columns:
        dmin = float(np.nanmin(protest_df["Duration_days"]))
        dmax = float(np.nanmax(protest_df["Duration_days"]))
        if np.isfinite(dmin) and np.isfinite(dmax):
            duration_range = st.sidebar.slider(
                "Duration (days)",
                min_value=float(dmin),
                max_value=float(dmax),
                value=(float(dmin), float(dmax)),
                help=(
                    "Restrict protests by event duration in days "
                    "(based on the duration field in the protest data)."
                ),
            )

    path_protest, path_pseudo, events_used_stats = run_event_study(
        indicator_key=ind_key,
        indicator_panels=indicator_panels,
        indicator_freq=indicator_freq,
        protest_df=protest_df,
        violent_only=violent_only,
        revolution_only=revolution_only,
        govchange_only=govchange_only,
        selected_regions=selected_regions,
        selected_countries=selected_countries,
        selected_groups=selected_groups,
        duration_range=duration_range,
        exclude_gfc=exclude_gfc,
        exclude_covid=exclude_covid,
        exclude_2223=exclude_2223,
        methodology=methodology,
        vol_q_low=vol_q_low,
        vol_q_high=vol_q_high,
        slope_q=slope_q,
        use_pseudo=show_pseudo,
        scale_option=scale_option,
        rate_option=rate_option,
    )

    if path_protest is None or path_protest.empty:
        st.warning("No protest events for this configuration.")
        return

    freq = indicator_freq[ind_key]
    base_pre, base_post, _, _ = get_window_lengths(freq, rate_option)

    path_protest = complete_rel_t(path_protest, base_pre, base_post)
    if path_pseudo is not None and not path_pseudo.empty:
        path_pseudo = complete_rel_t(path_pseudo, base_pre, base_post)

    if norm_option == "None":
        p_plot = path_protest.copy()
        p_plot["y_plot"] = p_plot["mean_value"]
        if path_pseudo is not None and not path_pseudo.empty:
            s_plot = path_pseudo.copy()
            s_plot["y_plot"] = s_plot["mean_value"]
        else:
            s_plot = None
        ylabel = indicator_labels.get(ind_key, "Indicator value")
    elif norm_option == "Shift T=0 to 0":
        p_plot = normalize_path_T0(path_protest)
        if path_pseudo is not None and not path_pseudo.empty:
            s_plot = normalize_path_T0(path_pseudo)
        else:
            s_plot = None
        ylabel = f"{indicator_labels.get(ind_key, 'Indicator')} (Δ vs T=0)"
    else:
        if freq == "Q":
            pre_start, pre_end = -4, -1
        else:
            pre_start, pre_end = -12, -1
        p_plot, s_plot = align_pair_pre_level(
            path_protest, path_pseudo, pre_start=pre_start, pre_end=pre_end
        )
        ylabel = f"{indicator_labels.get(ind_key, 'Indicator')} (dev from common pre-mean)"

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        p_plot["rel_t"],
        p_plot["y_plot"],
        label=f"Protests ({methodology})",
        linewidth=2,
    )
    if s_plot is not None and not s_plot.empty:
        ax.plot(
            s_plot["rel_t"],
            s_plot["y_plot"],
            label="No-protest pseudo (all candidates, same countries)",
            linewidth=2,
            linestyle="--",
            alpha=0.5,
        )

    ax.axvline(0, linewidth=1, linestyle="--")
    ax.set_xlim(-base_pre, base_post)
    ax.set_xlabel("Periods relative to event (T = 0)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{indicator_labels.get(ind_key, ind_key)} around protests vs no-protest")
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)

    # Basic stats
    protest_n = int(p_plot["n_events"].max()) if "n_events" in p_plot.columns else None
    pseudo_n = (
        int(s_plot["n_events"].max())
        if (s_plot is not None and "n_events" in s_plot.columns)
        else 0
    )

    if freq == "Q":
        pre_mask = (p_plot["rel_t"] >= -4) & (p_plot["rel_t"] <= -1)
        post_mask = (p_plot["rel_t"] >= 1) & (p_plot["rel_t"] <= 4)
    else:
        pre_mask = (p_plot["rel_t"] >= -12) & (p_plot["rel_t"] <= -1)
        post_mask = (p_plot["rel_t"] >= 1) & (p_plot["rel_t"] <= 12)

    pre_avg = float(p_plot.loc[pre_mask, "y_plot"].mean()) if pre_mask.any() else float("nan")
    post_avg = float(p_plot.loc[post_mask, "y_plot"].mean()) if post_mask.any() else float("nan")

    stats = {
        "protest_events_used_in_path": protest_n,
        "pseudo_events_used_in_path": pseudo_n,
        "mean_pre_window_y": pre_avg,
        "mean_post_window_y": post_avg,
    }

    # Detailed breakdown of protest events used
    breakdown = {}
    if events_used_stats is not None and not events_used_stats.empty:
        evu = events_used_stats.copy()
        total_e = len(evu)
        breakdown["total_events_used"] = int(total_e)

        if "Region" in evu.columns:
            region_counts = evu["Region"].value_counts(dropna=True)
            breakdown["events_by_region"] = {
                str(k): int(v) for k, v in region_counts.to_dict().items()
            }

        if "Violent_0_1" in evu.columns:
            v = evu["Violent_0_1"].fillna(0)
            v_n = int(v.sum())
            breakdown["violent_events"] = {
                "count": v_n,
                "share_of_events": float(v_n / total_e) if total_e > 0 else None,
            }

        if "Revolution_0_1" in evu.columns:
            r = evu["Revolution_0_1"].fillna(0)
            r_n = int(r.sum())
            breakdown["revolution_events"] = {
                "count": r_n,
                "share_of_events": float(r_n / total_e) if total_e > 0 else None,
            }

        if "GovtChange_0_1" in evu.columns:
            g = evu["GovtChange_0_1"].fillna(0)
            g_n = int(g.sum())
            breakdown["gov_change_events"] = {
                "count": g_n,
                "share_of_events": float(g_n / total_e) if total_e > 0 else None,
            }

        if "Group_list" in evu.columns:
            group_series = evu["Group_list"].explode()
            group_series = group_series.dropna()
            if not group_series.empty:
                group_counts = group_series.value_counts()
                top_groups = group_counts.head(10)
                breakdown["top_groups_by_events"] = {
                    str(k): int(v) for k, v in top_groups.to_dict().items()
                }

    stats["protest_event_breakdown"] = breakdown

    st.markdown("#### Event statistics used in averages")
    st.write(stats)


if __name__ == "__main__":
    main()

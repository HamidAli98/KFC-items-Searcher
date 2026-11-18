# -*- coding: utf-8 -*-
"""Nutrition menu analysis and calorie prediction app.

Place these CSV files in the same folder as this script:
  - McDonald Nutrient.csv
  - nutrition-booklet.csv
Run with: streamlit run kfcmenusearcher_fixed.py
"""

from pathlib import Path
import difflib
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Menu Nutrition Searcher", page_icon="🍔", layout="wide")
st.title("🍔 Menu Nutrition Searcher")
st.caption("Explore nutrition data, compare models, and estimate menu-item calories.")

BASE_DIR = Path(__file__).resolve().parent


def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV beside this script, with a Colab path fallback."""
    candidates = [BASE_DIR / filename, Path("/content") / filename]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Could not find '{filename}'. Upload it to the same GitHub repository "
        "as this Python file."
    )


@st.cache_data(show_spinner="Loading nutrition datasets...")
def load_datasets():
    mcd = load_csv("McDonald Nutrient.csv")
    booklet = load_csv("nutrition-booklet.csv")
    return mcd, booklet


try:
    mcdonald_df, nutrition_booklet_df = load_datasets()
except Exception as exc:
    st.error(str(exc))
    st.info("Make sure both CSV files are committed to your repository next to this script.")
    st.stop()

st.subheader("Dataset overview")
c1, c2 = st.columns(2)
c1.metric("McDonald's dataset rows", f"{len(mcdonald_df):,}")
c1.metric("McDonald's dataset columns", f"{len(mcdonald_df.columns):,}")
c2.metric("Nutrition booklet rows", f"{len(nutrition_booklet_df):,}")
c2.metric("Nutrition booklet columns", f"{len(nutrition_booklet_df.columns):,}")

with st.expander("Preview source datasets"):
    st.markdown("**McDonald's nutrient dataset**")
    st.dataframe(mcdonald_df.head(), use_container_width=True)
    st.markdown("**Nutrition booklet dataset**")
    st.dataframe(nutrition_booklet_df.head(), use_container_width=True)

# Clean the booklet dataset while tolerating slightly different column headers.
booklet_clean = nutrition_booklet_df.copy()
booklet_clean.columns = (
    booklet_clean.columns.astype(str)
    .str.replace("\\n", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
if len(booklet_clean) > 2:
    booklet_clean = booklet_clean.iloc[2:].reset_index(drop=True)
booklet_clean = booklet_clean.dropna(axis=1, how="all")
for col in booklet_clean.columns:
    if col != "Products":
        cleaned = booklet_clean[col].astype(str).str.replace("\\n", "", regex=True).str.strip()
        converted = pd.to_numeric(cleaned.str.replace(",", "", regex=False), errors="coerce")
        # Only replace a column if it contains at least some numeric values.
        if converted.notna().any():
            booklet_clean[col] = converted
if "Products" in booklet_clean.columns:
    booklet_clean = booklet_clean.dropna(subset=["Products"])

st.subheader("Nutrition data exploration")
numeric_cols = mcdonald_df.select_dtypes(include=[np.number]).columns.tolist()

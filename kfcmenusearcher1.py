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
if len(numeric_cols) > 1:
    with st.expander("McDonald's nutrient correlation heatmap"):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(mcdonald_df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Matrix of Nutrients")
        st.pyplot(fig)
        plt.close(fig)

energy_col = "Energy (kCal)"
if energy_col in mcdonald_df.columns:
    with st.expander("McDonald's calorie distribution"):
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(data=mcdonald_df, x=energy_col, kde=True, bins=20, ax=ax)
        ax.set_title("McDonald's Energy Distribution")
        st.pyplot(fig)
        plt.close(fig)

# -------------------- Decision Tree model (retained) --------------------
st.header("Decision Tree calorie prediction")
features = [
    '"Protein (g)"', '"Total fat (g)"', 'Sat Fat (g)', 'Trans fat (g)',
    'Cholesterols (mg)', 'Total carbohydrate (g)', 'Total Sugars (g)',
    'Added Sugars (g)', '"Sodium (mg)"'
]
target = "Energy (kCal)"
required = features + [target]
missing = [col for col in required if col not in mcdonald_df.columns]
if missing:
    st.error("The McDonald's CSV is missing columns required by the model: " + ", ".join(missing))
    st.write("Columns found:", list(mcdonald_df.columns))
    st.stop()

model_df = mcdonald_df.copy()
for col in required:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
model_df[features] = model_df[features].replace([np.inf, -np.inf], np.nan)
model_df[target] = model_df[target].replace([np.inf, -np.inf], np.nan)
model_df[features] = model_df[features].fillna(model_df[features].median())
model_df = model_df.dropna(subset=[target])
if len(model_df) < 5:
    st.error("Not enough valid rows to train the prediction models (need at least 5).")
    st.stop()

X = model_df[features]
y = model_df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# The Decision Tree is intentionally retained as the primary prediction model.
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_train = dt_model.predict(X_train)
y_pred_test = dt_model.predict(X_test)
train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
train_r2 = float(r2_score(y_train, y_pred_train))
test_r2 = float(r2_score(y_test, y_pred_test))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Train RMSE", f"{train_rmse:.2f} kCal")
m2.metric("Test RMSE", f"{test_rmse:.2f} kCal")
m3.metric("Train R²", f"{train_r2:.4f}")
m4.metric("Test R²", f"{test_r2:.4f}")

with st.expander("Decision Tree feature importance"):
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": dt_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax)
    ax.set_title("Decision Tree Feature Importance")
    st.pyplot(fig)
    plt.close(fig)

with st.expander("Decision Tree actual vs. predicted values"):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_pred_test, s=65, ax=ax)
    low = min(float(y_test.min()), float(np.min(y_pred_test)))
    high = max(float(y_test.max()), float(np.max(y_pred_test)))
    ax.plot([low, high], [low, high], "r--", linewidth=2)
    ax.set_xlabel("Actual Calories (kCal)")
    ax.set_ylabel("Predicted Calories (kCal)")
    ax.set_title(f"Decision Tree: Actual vs Predicted (Test R²: {test_r2:.4f})")
    st.pyplot(fig)
    plt.close(fig)

# Keep the original Random Forest comparison as an additional model.
st.subheader("Decision Tree vs. Random Forest")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf_train = rf_model.predict(X_train)
y_pred_rf_test = rf_model.predict(X_test)
rf_train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_rf_train)))
rf_test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_rf_test)))
rf_train_r2 = float(r2_score(y_train, y_pred_rf_train))
rf_test_r2 = float(r2_score(y_test, y_pred_rf_test))
comparison_df = pd.DataFrame({
    "Metric": ["Train RMSE (kCal)", "Test RMSE (kCal)", "Train R²", "Test R²"],
    "Decision Tree": [train_rmse, test_rmse, train_r2, test_r2],
    "Random Forest": [rf_train_rmse, rf_test_rmse, rf_train_r2, rf_test_r2],
})
st.dataframe(comparison_df, hide_index=True, use_container_width=True)

# -------------------- Menu search --------------------
st.header("Search a menu item")
if "Menu Items" not in model_df.columns:
    st.warning("Menu search is unavailable because the CSV has no 'Menu Items' column.")
else:
    query = st.text_input("Enter a menu item", placeholder="e.g. McVeggie")
    if query.strip():
        menu_items = model_df["Menu Items"].dropna().astype(str).tolist()
        q = query.strip().lower()
        matches = [item for item in menu_items if q in item.lower()]
        if not matches:
            matches = difflib.get_close_matches(query, menu_items, n=3, cutoff=0.3)
            if matches:
                st.info("No direct match found. Did you mean one of these?")
            else:
                st.warning("No matching or similar menu items were found.")
        else:
            st.success(f"Found {len(matches)} matching item(s).")
        for match in matches[:10]:
            row = model_df[model_df["Menu Items"].astype(str) == match].iloc[0]
            prediction = float(dt_model.predict(pd.DataFrame([row[features]], columns=features))[0])
            with st.container(border=True):
                st.markdown(f"**{match}**")
                info_cols = st.columns(3)
                if "Per Serve Size" in row.index:
                    info_cols[0].write(f"Serving size: {row['Per Serve Size']}")
                info_cols[1].write(f"Actual energy: {row[target]:.2f} kCal")
                info_cols[2].write(f"Decision Tree estimate: {prediction:.2f} kCal")
                detail_cols = st.columns(3)
                detail_cols[0].write(f"Protein: {row['\"Protein (g)\"']} g")
                detail_cols[1].write(f"Total fat: {row['\"Total fat (g)\"']} g")
                detail_cols[2].write(f"Carbohydrates: {row['Total carbohydrate (g)']} g")

# Save a downloadable copy of the trained Decision Tree model.
model_bytes = joblib.dump(dt_model, BASE_DIR / "decision_tree_model.joblib")
st.download_button(
    "Download trained Decision Tree model",
    data=(BASE_DIR / "decision_tree_model.joblib").read_bytes(),
    file_name="decision_tree_model.joblib",
    mime="application/octet-stream",
)

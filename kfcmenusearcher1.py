# -*- coding: utf-8 -*-
"""Clean, search-first Streamlit menu nutrition app.

Keep these files in the same GitHub repository folder:
  - McDonald Nutrient.csv
  - nutrition-booklet.csv
Run with: streamlit run kfcmenusearcher_clean.py
"""
from pathlib import Path
import difflib
import warnings

import joblib
import io
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="Menu Nutrition Searcher", page_icon="🍔", layout="centered")

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data(show_spinner="Loading menu data...")
def load_datasets():
    paths = {
        "mcd": BASE_DIR / "McDonald Nutrient.csv",
        "booklet": BASE_DIR / "nutrition-booklet.csv",
    }
    missing = [str(path.name) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s): " + ", ".join(missing)
            + ". Add both CSV files to the same GitHub folder as this app."
        )
    return pd.read_csv(paths["mcd"]), pd.read_csv(paths["booklet"])

@st.cache_resource(show_spinner="Preparing calorie prediction models...")
def train_models(dataframe):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor

    features = [
        '"Protein (g)"', '"Total fat (g)"', "Sat Fat (g)", "Trans fat (g)",
        "Cholesterols (mg)", "Total carbohydrate (g)", "Total Sugars (g)",
        "Added Sugars (g)", '"Sodium (mg)"',
    ]
    target = "Energy (kCal)"
    missing = [col for col in features + [target] if col not in dataframe.columns]
    if missing:
        raise ValueError("McDonald's CSV is missing model columns: " + ", ".join(missing))

    prepared = dataframe.copy()
    for col in features + [target]:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    prepared[features] = prepared[features].replace([np.inf, -np.inf], np.nan)
    prepared[target] = prepared[target].replace([np.inf, -np.inf], np.nan)
    prepared[features] = prepared[features].fillna(prepared[features].median())
    prepared = prepared.dropna(subset=[target]).copy()
    if len(prepared) < 5:
        raise ValueError("At least 5 valid menu rows are required to train the models.")

    X_train, X_test, y_train, y_test = train_test_split(
        prepared[features], prepared[target], test_size=0.2, random_state=42
    )
    # Keep Decision Tree as the primary calorie prediction model.
    dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_train_pred = dt_model.predict(X_train)
    dt_test_pred = dt_model.predict(X_test)

    # Keep Random Forest for the original model comparison.
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)

    metrics = {
        "dt_train_rmse": float(np.sqrt(mean_squared_error(y_train, dt_train_pred))),
        "dt_test_rmse": float(np.sqrt(mean_squared_error(y_test, dt_test_pred))),
        "dt_train_r2": float(r2_score(y_train, dt_train_pred)),
        "dt_test_r2": float(r2_score(y_test, dt_test_pred)),
        "rf_train_rmse": float(np.sqrt(mean_squared_error(y_train, rf_train_pred))),
        "rf_test_rmse": float(np.sqrt(mean_squared_error(y_test, rf_test_pred))),
        "rf_train_r2": float(r2_score(y_train, rf_train_pred)),
        "rf_test_r2": float(r2_score(y_test, rf_test_pred)),
    }
    return prepared, features, target, dt_model, rf_model, metrics, X_test, y_test, dt_test_pred, rf_test_pred

# -------------------- Clean, search-first interface --------------------
st.title("🍔 Menu Nutrition Searcher")
st.caption("Search menu items to see their nutrition and estimated calories.")

try:
    mcdonald_df, nutrition_booklet_df = load_datasets()
    (model_df, features, target, dt_model, rf_model, metrics,
     X_test, y_test, dt_test_pred, rf_test_pred) = train_models(mcdonald_df)
except Exception as exc:
    st.error(str(exc))
    st.info("Check that both CSV files are committed beside the Streamlit main Python file.")
    st.stop()

if "Menu Items" not in model_df.columns:
    st.error("The McDonald's CSV does not contain the required 'Menu Items' column.")
    st.stop()

# Search is deliberately the first and primary interactive section.
st.subheader("Search a menu item")
query = st.text_input(
    "Enter a menu item name",
    placeholder="For example: McChicken, McVeggie, Fries...",
    label_visibility="collapsed",
)

if query.strip():
    menu_items = model_df["Menu Items"].dropna().astype(str).tolist()
    q = query.strip().lower()
    matches = list(dict.fromkeys(item for item in menu_items if q in item.lower()))
    if not matches:
        matches = difflib.get_close_matches(query.strip(), menu_items, n=5, cutoff=0.3)
        if matches:
            st.info("No exact text match. Here are the closest menu items:")
        else:
            st.warning("No matching menu items found. Try a shorter or different search term.")
    else:
        st.success(f"{len(matches)} matching menu item(s) found.")

    for match in matches[:10]:
        row = model_df[model_df["Menu Items"].astype(str) == match].iloc[0]
        feature_row = pd.DataFrame([[row[col] for col in features]], columns=features)
        predicted_calories = float(dt_model.predict(feature_row)[0])
        with st.container(border=True):
            st.markdown(f"**{match}**")
            c1, c2, c3 = st.columns(3)
            if "Per Serve Size" in row.index and pd.notna(row["Per Serve Size"]):
                c1.metric("Serving size", str(row["Per Serve Size"]))
            c2.metric("Actual calories", f"{float(row[target]):.1f} kcal")
            c3.metric("Estimated calories", f"{predicted_calories:.1f} kcal")
            with st.expander("Nutrition details"):
                n1, n2, n3 = st.columns(3)
                n1.write(f"**Protein:** {row['\"Protein (g)\"']} g")
                n2.write(f"**Total fat:** {row['\"Total fat (g)\"']} g")
                n3.write(f"**Carbohydrates:** {row['Total carbohydrate (g)']} g")
else:
    st.caption("Type a menu item above to see its nutrition information and Decision Tree calorie estimate.")

# All dataset/model diagnostics are retained but hidden unless the user asks for them.
with st.expander("Advanced analysis (optional)"):
    st.subheader("Dataset overview")
    a, b = st.columns(2)
    a.metric("McDonald's dataset rows", f"{len(mcdonald_df):,}")
    a.metric("McDonald's dataset columns", f"{len(mcdonald_df.columns):,}")
    b.metric("Nutrition booklet rows", f"{len(nutrition_booklet_df):,}")
    b.metric("Nutrition booklet columns", f"{len(nutrition_booklet_df.columns):,}")
    with st.expander("Preview source datasets"):
        st.markdown("**McDonald's nutrient dataset**")
        st.dataframe(mcdonald_df.head(), use_container_width=True)
        st.markdown("**Nutrition booklet dataset**")
        st.dataframe(nutrition_booklet_df.head(), use_container_width=True)

    st.subheader("Decision Tree model performance")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Train RMSE", f"{metrics['dt_train_rmse']:.2f} kcal")
    p2.metric("Test RMSE", f"{metrics['dt_test_rmse']:.2f} kcal")
    p3.metric("Train R²", f"{metrics['dt_train_r2']:.4f}")
    p4.metric("Test R²", f"{metrics['dt_test_r2']:.4f}")

    with st.expander("Decision Tree feature importance"):
        import matplotlib.pyplot as plt
        importance = pd.DataFrame({
            "Feature": features,
            "Importance": dt_model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(importance["Feature"][::-1], importance["Importance"][::-1])
        ax.set_xlabel("Relative importance")
        ax.set_title("Decision Tree feature importance")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Decision Tree actual vs. predicted values"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, dt_test_pred, alpha=0.8)
        low = min(float(y_test.min()), float(np.min(dt_test_pred)))
        high = max(float(y_test.max()), float(np.max(dt_test_pred)))
        ax.plot([low, high], [low, high], "r--", linewidth=2)
        ax.set_xlabel("Actual calories (kcal)")
        ax.set_ylabel("Predicted calories (kcal)")
        ax.set_title("Decision Tree: actual vs. predicted")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Decision Tree vs. Random Forest")
    comparison = pd.DataFrame({
        "Metric": ["Train RMSE (kcal)", "Test RMSE (kcal)", "Train R²", "Test R²"],
        "Decision Tree": [metrics["dt_train_rmse"], metrics["dt_test_rmse"],
                          metrics["dt_train_r2"], metrics["dt_test_r2"]],
        "Random Forest": [metrics["rf_train_rmse"], metrics["rf_test_rmse"],
                           metrics["rf_train_r2"], metrics["rf_test_r2"]],
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.subheader("Nutrition data exploration")
    numeric_cols = mcdonald_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        with st.expander("McDonald's nutrient correlation heatmap"):
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(mcdonald_df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation matrix of nutrients")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    if target in mcdonald_df.columns:
        with st.expander("McDonald's calorie distribution"):
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.histplot(data=mcdonald_df, x=target, kde=True, bins=20, ax=ax)
            ax.set_title("McDonald's calorie distribution")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    _model_buffer = io.BytesIO()
    joblib.dump(dt_model, _model_buffer)
    st.download_button(
        "Download trained Decision Tree model",
        data=_model_buffer.getvalue(),
        file_name="decision_tree_model.joblib",
        mime="application/octet-stream",
    )

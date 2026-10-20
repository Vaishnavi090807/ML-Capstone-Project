
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# ML INSIGHT LAB
# Evaluator-facing Streamlit dashboard for the capstone project
# ============================================================

st.set_page_config(
    page_title="ML Insight Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
st.markdown("""
<style>
    :root { --ink:#1f2937; --muted:#687385; --accent:#587b76; --line:#e5e1d8; --paper:#fbfaf7; }
    .stApp { background: #fbfaf7; color: var(--ink); }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1380px; }
    [data-testid="stSidebar"] { background: #f1f0eb; border-right: 1px solid #e5e1d8; }
    [data-testid="stSidebar"] h2 { color:#24313f; letter-spacing:-.02em; }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e5e1d8; padding: 16px 18px;
        border-radius: 14px; box-shadow: 0 2px 10px rgba(31,41,55,.025);
    }
    [data-testid="stMetricLabel"] { color:#687385; }
    [data-testid="stMetricValue"] { color:#24313f; }
    .hero {
        padding: 30px 32px; border-radius: 18px;
        background: linear-gradient(118deg, #202d3b 0%, #2d3c4d 68%, #526f6a 100%);
        color: #fff; margin: 2px 0 24px 0; border: 1px solid rgba(255,255,255,.15);
    }
    .hero h1 { font-size: 2.15rem; font-weight: 650; letter-spacing:-.035em; margin: 0 0 8px 0; }
    .hero p { color: #e2e8e8; margin: 0; font-size: 1.02rem; line-height:1.6; }
    .eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.74rem; font-weight:700; color:#66827c; margin-bottom:7px; }
    .section-card {
        padding: 20px; border: 1px solid #e5e1d8; border-radius: 16px;
        background: #fff; box-shadow: 0 2px 10px rgba(31,41,55,.025);
    }
    .insight {
        padding: 15px 17px; border-left: 3px solid #6b8d85;
        background: #edf3f0; border-radius: 8px; margin: 10px 0; color:#35464a;
    }
    .winner {
        padding: 18px; border: 1px solid #cbdcd5; background: #f1f6f2;
        border-radius: 14px; color:#263d38;
    }
    .small-muted {color:#687385; font-size:.9rem;}
    div[data-testid="stTabs"] button { font-weight: 550; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color:#3d6861; border-color:#6b8d85; }
    div[data-testid="stExpander"] { border:1px solid #e5e1d8; border-radius:12px; background:#fff; }
    .stButton > button { border-radius: 10px; border:1px solid #cbd5d1; }
    h1,h2,h3 { letter-spacing:-.025em; color:#263442; }
    [data-testid="stDataFrame"] { border:1px solid #e5e1d8; border-radius:12px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
DATA_DIR = ROOT / "data"

# Also support running from repository root.
if not DATA_DIR.exists():
    DATA_DIR = Path.cwd() / "data"


@st.cache_data(show_spinner=False)
def load_market_arrivals_data():
    candidates = [
        DATA_DIR / "MarketArrivals.csv",
        Path.cwd() / "data" / "MarketArrivals.csv",
        ROOT.parent / "data" / "MarketArrivals.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            frame["_source_file"] = path.name
            return frame
    return None


@st.cache_data(show_spinner=False)
def prepare_cluster_data():
    data = load_market_arrivals_data()
    if data is None:
        return None, None, None, None, None

    data = data.copy()
    required = {"market", "month", "year", "quantity", "priceMin", "priceMax", "priceMod", "state", "city"}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"MarketArrivals.csv is missing required columns: {sorted(missing_columns)}")

    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    month_to_num = {month: i + 1 for i, month in enumerate(month_order)}
    data["month_num"] = data["month"].map(month_to_num)
    data["month_sin"] = np.sin(2 * np.pi * data["month_num"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month_num"] / 12)

    for col in ["quantity", "priceMin", "priceMax", "priceMod", "year"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["quantity", "priceMin", "priceMax", "priceMod"]:
        data[f"log_{col}"] = np.log1p(data[col].clip(lower=0))

    cluster_features = ["log_quantity", "log_priceMin", "log_priceMax", "log_priceMod",
                        "year", "month_sin", "month_cos"]
    X = data[cluster_features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return data, X, X_scaled, cluster_features, scaler


@st.cache_data(show_spinner=False)
def cluster_experiment(X_scaled, n_clusters=5):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    ward_labels = ward.fit_predict(X_scaled)

    results = pd.DataFrame({
        "Algorithm": ["K-Means", "Agglomerative Ward"],
        "Configuration": [f"k={n_clusters}", f"k={n_clusters}, Ward linkage"],
        "Silhouette": [silhouette_score(X_scaled, km_labels), silhouette_score(X_scaled, ward_labels)],
        "Davies-Bouldin": [davies_bouldin_score(X_scaled, km_labels), davies_bouldin_score(X_scaled, ward_labels)],
        "Calinski-Harabasz": [calinski_harabasz_score(X_scaled, km_labels), calinski_harabasz_score(X_scaled, ward_labels)],
    })
    return km, km_labels, ward, ward_labels, results


@st.cache_data(show_spinner=False)
def elbow_data(X_scaled):
    ks = list(range(2, 11))
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertias.append(model.inertia_)
    return pd.DataFrame({"k": ks, "inertia": inertias})


@st.cache_data(show_spinner=False)
def pca_data(X_scaled, labels):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    out = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": labels.astype(str)})
    return out, pca.explained_variance_ratio_


@st.cache_data(show_spinner=False)
def tsne_data(X_scaled, labels, sample_size=2500):
    rng = np.random.RandomState(42)
    n = min(sample_size, len(X_scaled))
    idx = np.sort(rng.choice(len(X_scaled), n, replace=False))
    perplexity = min(30, max(5, (n - 1) // 3))
    model = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                 init="pca", learning_rate="auto", max_iter=1000)
    coords = model.fit_transform(X_scaled[idx])
    return pd.DataFrame({"tSNE-1": coords[:, 0], "tSNE-2": coords[:, 1], "Cluster": labels[idx].astype(str)})


@st.cache_data(show_spinner=False)
def linkage_sample_comparison(X_scaled, n_clusters=5, sample_size=2000):
    n = min(sample_size, len(X_scaled))
    idx = np.sort(np.random.RandomState(42).choice(len(X_scaled), n, replace=False))
    X_sample = X_scaled[idx]
    rows = []
    for method in ["ward", "complete", "average", "single"]:
        labels = AgglomerativeClustering(n_clusters=n_clusters, linkage=method).fit_predict(X_sample)
        rows.append({
            "Linkage": method.title(), "Sample size": n,
            "Silhouette": silhouette_score(X_sample, labels),
            "Davies-Bouldin": davies_bouldin_score(X_sample, labels),
            "Calinski-Harabasz": calinski_harabasz_score(X_sample, labels),
            "Smallest cluster": int(pd.Series(labels).value_counts().min()),
            "Largest cluster": int(pd.Series(labels).value_counts().max()),
        })
    return pd.DataFrame(rows)



REGRESSION_RESULTS = pd.DataFrame([
    {"Model":"Lasso Regression", "R²":0.521071, "RMSE":22.179859, "MAE":16.033245},
    {"Model":"ElasticNet Regression", "R²":0.521068, "RMSE":22.179946, "MAE":16.033651},
    {"Model":"Linear Regression", "R²":0.521067, "RMSE":22.179950, "MAE":16.033412},
    {"Model":"Ridge Regression", "R²":0.521067, "RMSE":22.179960, "MAE":16.033495},
    {"Model":"Random Forest Regressor", "R²":0.508848, "RMSE":22.461116, "MAE":17.048734},
    {"Model":"Gradient Boosting Regressor", "R²":0.416830, "RMSE":24.474894, "MAE":17.645983},
    {"Model":"Polynomial Regression", "R²":0.308951, "RMSE":26.642680, "MAE":19.814270},
    {"Model":"SVR", "R²":-0.105395, "RMSE":33.696291, "MAE":24.809200},
    {"Model":"KNN Regressor", "R²":-0.132908, "RMSE":34.113063, "MAE":24.968947},
    {"Model":"Decision Tree Regressor", "R²":-0.146484, "RMSE":34.316850, "MAE":24.672516},
])
REGRESSION_CV = pd.DataFrame([
    {"Model":"Lasso Regression", "Mean CV R²":0.521296, "CV R² Std":0.048116},
    {"Model":"ElasticNet Regression", "Mean CV R²":0.521315, "CV R² Std":0.048076},
])

@st.cache_data(show_spinner=False)
def fit_regression_diagnostics(csv_path):
    """Reproduce the notebook's cleaning/features to draw actual-vs-predicted diagnostics."""
    df = pd.read_csv(csv_path)
    required = {"society", "location", "size", "bath", "balcony", "total_sqft", "price",
                "area_type", "availability"}
    if not required.issubset(df.columns):
        raise ValueError("The selected file does not look like Bengaluru_House_Data.csv.")

    def sqft_to_num(value):
        try:
            value = str(value)
            if "-" in value:
                parts = value.split("-")
                return (float(parts[0]) + float(parts[1])) / 2
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    clean = df.drop(columns=["society"]).copy()
    clean = clean.dropna(subset=["location", "size"])
    clean["bath"] = clean["bath"].fillna(clean["bath"].median())
    clean["balcony"] = clean["balcony"].fillna(clean["balcony"].median())
    clean = clean.drop_duplicates()
    clean["bhk"] = clean["size"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    clean["new_total_sqft"] = clean["total_sqft"].apply(sqft_to_num)
    clean["sqft_per_bhk"] = clean["new_total_sqft"] / clean["bhk"]
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=["bhk", "new_total_sqft", "price", "sqft_per_bhk"])
    clean = clean[(clean["new_total_sqft"] > 0) & (clean["price"] > 0) & (clean["bhk"] > 0)]
    for col in ["new_total_sqft", "bhk", "bath", "balcony", "price"]:
        q1, q3 = clean[col].quantile([.25, .75])
        iqr = q3 - q1
        clean = clean[clean[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
    features = ["area_type", "availability", "location", "bhk", "new_total_sqft",
                "bath", "balcony", "sqft_per_bhk"]
    X = clean[features].copy()
    y = clean["price"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42
    )
    categorical = ["area_type", "availability", "location"]
    numeric = ["bhk", "new_total_sqft", "bath", "balcony", "sqft_per_bhk"]
    encoders = {}
    for col in categorical:
        encoder = LabelEncoder()
        # Replace the original string-typed columns with numeric Series.
        # Assigning through .loc can fail on pandas StringDtype columns.
        train_values = encoder.fit_transform(X_train[col].astype(str))
        X_train[col] = pd.Series(train_values, index=X_train.index, dtype="int64")
        known = set(encoder.classes_)
        test_values = X_test[col].astype(str).map(
            lambda v: int(encoder.transform([v])[0]) if v in known else -1
        )
        X_test[col] = test_values.astype("int64")
        encoders[col] = encoder
    scaler = StandardScaler()
    X_train[numeric] = scaler.fit_transform(X_train[numeric]).astype(float)
    X_test[numeric] = scaler.transform(X_test[numeric]).astype(float)
    lasso = Lasso(alpha=.001, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    pred = pd.DataFrame({
        "Actual price (lakhs)": y_test.to_numpy(),
        "Predicted price (lakhs)": lasso_pred,
        "Residual": y_test.to_numpy() - lasso_pred,
    })
    importance = pd.DataFrame({
        "Feature": features, "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)
    return pred, importance, int(len(clean)), int(len(X_train)), int(len(X_test))


def metric_card_explanation():
    st.markdown("""
    <div class="insight">
    <b>How to read these metrics:</b>
    Silhouette → higher is better · Davies–Bouldin → lower is better ·
    Calinski–Harabasz → higher is better.
    </div>
    """, unsafe_allow_html=True)


# ---------- Sidebar ----------
st.sidebar.markdown("## 🧠 ML Insight Lab")
st.sidebar.caption("Your capstone, made explorable")

page = st.sidebar.radio(
    "Explore",
    [
        "🏠 Overview",
        "📊 Classification",
        "📈 Regression",
        "🔵 Clustering",
        "🗂️ Dataset Explorer",
        "🏆 Final Decision",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Reproducibility")
st.sidebar.code("random_state = 42")
st.sidebar.caption("Explore · Compare · Understand")


# ============================================================
# OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow" style="color:#c6d8d2">Machine learning · Capstone studio</div>
        <h1>ML Insight Lab</h1>
        <p>One calm, interactive place to explore your data, compare models, and understand the results.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Your project, at a glance")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Classification", "10 algorithms", "Logistic Regression selected")
    with c2:
        st.metric("Regression", "10 algorithms", "Compare R² / RMSE / MAE")
    with c3:
        st.metric("Clustering", "Market arrivals", "K-Means vs Agglomerative")

    st.markdown("### Choose a track to explore")
    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        **📊 Classification**

        See which model classifies match scores best, compare performance, and inspect where predictions go wrong.
        """)

    with b:
        st.markdown("""
        **📈 Regression**

        Explore house-price estimates, compare errors, and see how close predictions come to actual prices.
        """)

    with c:
        st.markdown("""
        **🔵 Clustering**

        Discover patterns in market arrivals and prices with visual cluster maps and simple quality comparisons.
        """)

    st.markdown("### 🎯 Presentation story")
    st.markdown("""
    <div class="section-card">
        <b>1. Understand the data</b> → <b>2. Prepare the features</b> →
        <b>3. Train competing algorithms</b> → <b>4. Compare evidence</b> →
        <b>5. Select the strongest model</b> → <b>6. Interpret the result</b>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "This is one integrated dashboard for all three ML tracks. Classification and regression "
        "retain their own datasets/results, while clustering uses MarketArrivals.csv. The notebooks "
        "remain the source of the actual ML experiments."
    )


# ============================================================
# CLASSIFICATION
# ============================================================
elif page == "📊 Classification":
    st.markdown("""
    <div class="hero">
        <h1>📊 Classification Arena</h1>
        <p>Compare algorithms and explain the evidence behind the final classifier.</p>
    </div>
    """, unsafe_allow_html=True)

    # Grounded results from the current classification notebook.
    results = pd.DataFrame({
        "Model": [
            "Logistic Regression", "KNN", "Gaussian NB", "Decision Tree", "SVC",
            "Random Forest", "AdaBoost", "Gradient Boosting", "Bagging", "MLP"
        ],
        "Accuracy": [0.7061,0.6301,0.6026,0.6668,0.7042,0.6965,0.6499,0.6939,0.6903,0.6587],
        "Precision": [0.7038,0.6338,0.6010,0.6758,0.7033,0.6969,0.6552,0.6938,0.6892,0.6576],
        "Recall": [0.7061,0.6301,0.6026,0.6668,0.7042,0.6965,0.6499,0.6939,0.6903,0.6587],
        "Weighted F1": [0.7044,0.6316,0.6005,0.6697,0.7036,0.6965,0.6512,0.6939,0.6893,0.6581],
        "ROC-AUC": [0.8785,0.8054,0.7798,0.8219,0.8493,0.8682,0.8166,0.8690,0.8667,0.8475],
    })

    best = results.loc[results["Weighted F1"].idxmax()]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🏆 Best model", best["Model"])
    c2.metric("Accuracy", f"{best['Accuracy']:.2%}")
    c3.metric("Weighted F1", f"{best['Weighted F1']:.4f}")
    c4.metric("ROC-AUC", f"{best['ROC-AUC']:.4f}")

    st.markdown("### ⚔️ Algorithm Arena")
    metric = st.selectbox("Compare using", ["Accuracy", "Weighted F1", "ROC-AUC"])

    chart_df = results.sort_values(metric, ascending=True)
    fig = px.bar(
        chart_df,
        x=metric,
        y="Model",
        orientation="h",
        text=chart_df[metric].map(lambda x: f"{x:.4f}"),
        title=f"Model comparison — {metric}",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=480, margin=dict(l=10,r=30,t=60,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🏆 Why Logistic Regression?")
    st.markdown("""
    <div class="winner">
    <b>Selected model: Logistic Regression</b><br><br>
    It achieved the highest accuracy, highest weighted F1-score and highest
    ROC-AUC among the evaluated models in the current classification notebook.
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        results.style.format({
            "Accuracy":"{:.4f}", "Precision":"{:.4f}",
            "Recall":"{:.4f}", "Weighted F1":"{:.4f}", "ROC-AUC":"{:.4f}"
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Classification results are based on the current notebook's final comparison table."
    )


# ============================================================
# REGRESSION
# ============================================================
elif page == "📈 Regression":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow" style="color:#c6d8d2">Track 02 · Price prediction</div>
        <h1>Regression, made clear</h1>
        <p>Compare how closely each model estimates house prices, then explore where its predictions miss.</p>
    </div>
    """, unsafe_allow_html=True)

    results = REGRESSION_RESULTS.copy()
    best = results.sort_values("R²", ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top model", best["Model"])
    c2.metric("R² score", f"{best['R²']:.3f}")
    c3.metric("Typical absolute error", f"{best['MAE']:.2f} lakhs")
    c4.metric("Models compared", f"{len(results)}")

    st.markdown("### Find the model that fits best")
    left, right = st.columns([1.25, 1])
    with left:
        metric = st.selectbox(
            "Compare models by",
            ["R²", "RMSE", "MAE"],
            format_func=lambda x: {"R²":"Explained variation · higher is better",
                                   "RMSE":"Larger errors count more · lower is better",
                                   "MAE":"Average absolute error · lower is better"}[x],
            key="regression_metric",
        )
        ascending = metric != "R²"
        chart_df = results.sort_values(metric, ascending=ascending)
        fig = px.bar(
            chart_df, x=metric, y="Model", orientation="h",
            text=chart_df[metric].map(lambda value: f"{value:.3f}"),
            title=f"Model comparison · {metric}",
            template="plotly_white",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=470, margin=dict(l=10, r=55, t=60, b=15),
                          yaxis_title="", xaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("""
        <div class="winner">
          <div class="eyebrow">Notebook result</div>
          <h3 style="margin-top:0">Lasso Regression</h3>
          <p>The strongest test-set R² in the current regression notebook, narrowly ahead of ElasticNet and the other linear models.</p>
          <p><b>R²</b> 0.5211 &nbsp; · &nbsp; <b>RMSE</b> 22.18 &nbsp; · &nbsp; <b>MAE</b> 16.03</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### A quick guide")
        st.markdown("""
        - **R²** tells you how much target variation the model explains.
        - **RMSE** penalizes larger misses more heavily.
        - **MAE** gives the average size of the prediction error.
        """)
        st.caption("Price values and errors are in lakhs, matching the notebook target.")

    with st.expander("See the full comparison table", expanded=False):
        display = results.copy()
        for col in ["R²", "RMSE", "MAE"]:
            display[col] = display[col].map(lambda value: round(value, 4))
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("### Explore the model")
    tab_diagnostics, tab_cv, tab_features = st.tabs(
        ["Prediction check", "Consistency check", "What matters most?"]
    )

    with tab_diagnostics:
        st.write("Choose a sample of test-set predictions and see how estimates compare with actual prices.")
        house_candidates = sorted(DATA_DIR.glob("*Bengaluru*House*.csv")) if DATA_DIR.exists() else []
        if not house_candidates and DATA_DIR.exists():
            house_candidates = sorted(DATA_DIR.glob("*house*.csv"))
        if house_candidates:
            selected_house_file = st.selectbox(
                "Housing dataset", house_candidates,
                format_func=lambda p: p.name, key="regression_house_file"
            )
            try:
                with st.spinner("Preparing the notebook-style test split for diagnostics…"):
                    pred_df, feature_importance, clean_count, train_count, test_count = fit_regression_diagnostics(str(selected_house_file))
                plot_sample = pred_df.sample(min(900, len(pred_df)), random_state=42)
                mode = st.radio("View", ["Actual vs predicted", "Prediction errors"], horizontal=True)
                if mode == "Actual vs predicted":
                    fig = px.scatter(
                        plot_sample, x="Actual price (lakhs)", y="Predicted price (lakhs)",
                        opacity=.62, template="plotly_white",
                        title="Actual prices compared with Lasso estimates",
                        hover_data={"Residual":":.2f"},
                    )
                    low = float(min(plot_sample["Actual price (lakhs)"].min(), plot_sample["Predicted price (lakhs)"].min()))
                    high = float(max(plot_sample["Actual price (lakhs)"].max(), plot_sample["Predicted price (lakhs)"].max()))
                    fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high,
                                  line=dict(dash="dash", color="#6b8d85", width=2))
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Points closer to the diagonal line indicate more accurate estimates.")
                else:
                    fig = px.scatter(
                        plot_sample, x="Predicted price (lakhs)", y="Residual",
                        opacity=.62, template="plotly_white",
                        title="Residuals around the zero-error line",
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="#6b8d85")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("A random spread around zero is generally preferable to a clear pattern.")
                with st.expander("How many records were used?"):
                    st.write(f"After cleaning: **{clean_count:,}** rows · Training: **{train_count:,}** · Test: **{test_count:,}**")
            except Exception as exc:
                st.warning(f"Prediction diagnostics could not be prepared from this file: {exc}")
        else:
            st.info("Add Bengaluru_House_Data.csv to the project's data/ folder to enable prediction plots.")

    with tab_cv:
        st.write("The two leading baseline models were checked with five-fold cross-validation on the training data.")
        cv_display = REGRESSION_CV.copy()
        cv_display["Mean CV R²"] = cv_display["Mean CV R²"].map(lambda v: f"{v:.4f}")
        cv_display["CV R² Std"] = cv_display["CV R² Std"].map(lambda v: f"{v:.4f}")
        st.dataframe(cv_display, use_container_width=True, hide_index=True)
        cv_chart = px.bar(REGRESSION_CV, x="Model", y="Mean CV R²",
                          error_y="CV R² Std", template="plotly_white",
                          title="Average cross-validated R² (± standard deviation)")
        cv_chart.update_layout(height=390, yaxis_title="Mean CV R²", xaxis_title="")
        st.plotly_chart(cv_chart, use_container_width=True)
        st.caption("Cross-validation provides a stability check across training folds; the held-out test set remains a separate evaluation.")

    with tab_features:
        house_candidates = sorted(DATA_DIR.glob("*Bengaluru*House*.csv")) if DATA_DIR.exists() else []
        if not house_candidates and DATA_DIR.exists():
            house_candidates = sorted(DATA_DIR.glob("*house*.csv"))
        if house_candidates:
            try:
                with st.spinner("Estimating Random Forest feature importance…"):
                    _, feature_importance, _, _, _ = fit_regression_diagnostics(str(house_candidates[0]))
                fig = px.bar(feature_importance.sort_values("Importance"), x="Importance", y="Feature",
                             orientation="h", template="plotly_white",
                             title="Features used by Random Forest")
                fig.update_layout(height=420, yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Feature importance indicates how much the fitted forest relied on each feature for reducing split error; it does not imply causation.")
            except Exception as exc:
                st.info(f"Feature importance is not available yet: {exc}")
        else:
            st.info("Add the Bengaluru housing CSV to enable feature-importance exploration.")

elif page == "🔵 Clustering":
    st.markdown("""
    <div class="hero">
        <h1>🔵 Market Arrivals Cluster Explorer</h1>
        <p>Explore market quantity, price and seasonal patterns through unsupervised learning.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        data, X, X_scaled, cluster_features, scaler = prepare_cluster_data()
    except Exception as exc:
        st.error(f"Could not prepare MarketArrivals.csv: {exc}")
        st.stop()

    if X_scaled is None:
        st.error("MarketArrivals.csv was not found. Place it in the project's data/ folder.")
        st.stop()

    st.caption(f"Dataset: MarketArrivals.csv · {len(data):,} rows · {len(cluster_features)} clustering features")
    with st.expander("Features used for clustering", expanded=False):
        st.write("The model uses log-transformed quantity/prices, year, and cyclic month features:")
        st.code(", ".join(cluster_features))
        st.write("Market, city, and state are retained for interpretation but excluded from the distance matrix to avoid arbitrary category encoding.")

    n_clusters = st.slider("Number of clusters (k)", 2, 8, 5)
    with st.spinner("Fitting K-Means and Agglomerative Ward on the dataset…"):
        km, km_labels, ward, ward_labels, results = cluster_experiment(X_scaled, n_clusters)

    km_row = results.iloc[0]
    ward_row = results.iloc[1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records analysed", f"{len(data):,}")
    c2.metric("K-Means Silhouette", f"{km_row['Silhouette']:.4f}")
    c3.metric("K-Means DB Index", f"{km_row['Davies-Bouldin']:.4f}")
    c4.metric("K-Means CH Index", f"{km_row['Calinski-Harabasz']:.2f}")

    metric_card_explanation()
    tabs = st.tabs(["🎯 Elbow", "⚔️ Algorithm Battle", "🧭 PCA Explorer", "🔬 t-SNE Discovery", "🌳 Dendrogram", "🧺 Cluster Profiles", "🗺️ Market Explorer"])

    with tabs[0]:
        elbow = elbow_data(X_scaled)
        fig = px.line(elbow, x="k", y="inertia", markers=True, title="Elbow Method — Inertia vs Number of Clusters")
        fig.add_vline(x=n_clusters, line_dash="dash", annotation_text=f"Selected k = {n_clusters}")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.info("Choose k near the point where additional clusters yield diminishing reductions in inertia. Treat the elbow as evidence, not an automatic guarantee.")

    with tabs[1]:
        st.subheader("K-Means vs Agglomerative Ward — same feature space")
        st.dataframe(results.round(4), use_container_width=True, hide_index=True)
        winner = "K-Means" if (km_row["Silhouette"] > ward_row["Silhouette"] and km_row["Davies-Bouldin"] < ward_row["Davies-Bouldin"] and km_row["Calinski-Harabasz"] > ward_row["Calinski-Harabasz"]) else "Mixed evidence"
        st.markdown(f"""
        <div class="winner">
            <h3>Current metric leader: {winner}</h3>
            <p>Compare all three metrics and cluster sizes before making a final decision. Internal metrics describe geometric separation; they do not by themselves prove business usefulness.</p>
        </div>
        """, unsafe_allow_html=True)
        metric_to_plot = st.selectbox("Metric to compare", ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"], key="cluster_metric")
        fig = px.bar(results, x="Algorithm", y=metric_to_plot, color="Algorithm", title=f"Algorithm comparison — {metric_to_plot}")
        st.plotly_chart(fig, use_container_width=True)
        if st.button("Compare hierarchical linkage strategies", type="secondary"):
            with st.spinner("Evaluating four linkage strategies on a reproducible sample of up to 2,000 rows…"):
                link_results = linkage_sample_comparison(X_scaled, n_clusters)
            st.dataframe(link_results.round(4), use_container_width=True, hide_index=True)
            st.caption("Linkage comparison scores are calculated on the displayed sample; final K-Means/Ward scores above use the full dataset.")

    with tabs[2]:
        algorithm = st.radio("Visualize clusters", ["K-Means", "Agglomerative Ward"], horizontal=True, key="pca_algo")
        labels = km_labels if algorithm == "K-Means" else ward_labels
        pca_df, variance = pca_data(X_scaled, labels)
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", opacity=0.65,
                         title=f"{algorithm} — PCA 2D Cluster Visualization", hover_data=["Cluster"])
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Explained variance: PC1 = {variance[0]:.2%}, PC2 = {variance[1]:.2%}, total = {(variance[0] + variance[1]):.2%}")
        st.caption("PCA is used for visualization only; clustering is fitted on all seven standardized features.")

    with tabs[3]:
        st.info("t-SNE is computed on a reproducible sample of up to 2,500 observations to keep the dashboard responsive.")
        algorithm = st.radio("Color by algorithm", ["K-Means", "Agglomerative Ward"], horizontal=True, key="tsne_algo")
        labels = km_labels if algorithm == "K-Means" else ward_labels
        if st.button("Run t-SNE", type="primary"):
            with st.spinner("Computing t-SNE projection…"):
                tsne_df = tsne_data(X_scaled, labels)
            fig = px.scatter(tsne_df, x="tSNE-1", y="tSNE-2", color="Cluster", opacity=0.7,
                             title=f"{algorithm} — t-SNE Pattern Discovery")
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("t-SNE visualizes local neighbourhoods; it is not used to fit the clusters or calculate the metrics.")

    with tabs[4]:
        linkage_method = st.selectbox("Linkage strategy", ["ward", "complete", "average", "single"], key="dendro_linkage")
        sample_n = min(500, len(X_scaled))
        idx = np.sort(np.random.RandomState(42).choice(len(X_scaled), sample_n, replace=False))
        Z = linkage(X_scaled[idx], method=linkage_method)
        d = dendrogram(Z, no_plot=True, truncate_mode="lastp", p=30)
        fig = go.Figure()
        for xs, ys in zip(d["icoord"], d["dcoord"]):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=1), showlegend=False))
        fig.update_layout(title=f"Hierarchical Clustering — {linkage_method.title()} Linkage (sample)",
                          xaxis_title="Sample / merge", yaxis_title="Distance", height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Dendrogram uses a reproducible sample of up to 500 rows for readability.")

    with tabs[5]:
        profile = data.copy()
        profile["Cluster"] = km_labels
        numeric_profile = profile.groupby("Cluster").agg(
            Records=("Cluster", "size"),
            Mean_Quantity=("quantity", "mean"),
            Median_Quantity=("quantity", "median"),
            Mean_Min_Price=("priceMin", "mean"),
            Mean_Max_Price=("priceMax", "mean"),
            Mean_Modal_Price=("priceMod", "mean"),
            Mean_Year=("year", "mean"),
        ).round(2)
        def mode_or_unknown(series):
            modes = series.mode(dropna=True)
            return modes.iloc[0] if not modes.empty else "Unknown"
        meta = profile.groupby("Cluster").agg(
            Most_Common_State=("state", mode_or_unknown),
            Most_Common_City=("city", mode_or_unknown),
            Most_Common_Market=("market", mode_or_unknown),
            Most_Common_Month=("month", mode_or_unknown),
        )
        summary = numeric_profile.join(meta)
        st.subheader("K-Means cluster profiles")
        st.dataframe(summary, use_container_width=True)
        fig = px.bar(summary.reset_index(), x="Cluster", y="Records", title="Records per K-Means cluster")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Cluster IDs are arbitrary. Use the quantity/price averages and the most common location/month to interpret each group.")
        st.download_button("Download cluster profile CSV", summary.to_csv().encode("utf-8"), file_name="market_arrivals_cluster_profiles.csv", mime="text/csv")

    with tabs[6]:
        st.subheader("Explore observations by market and time")
        states = ["All"] + sorted(data["state"].dropna().astype(str).unique().tolist())
        selected_state = st.selectbox("State", states)
        filtered = data if selected_state == "All" else data[data["state"].astype(str) == selected_state]
        markets = ["All"] + sorted(filtered["market"].dropna().astype(str).unique().tolist())
        selected_market = st.selectbox("Market", markets)
        if selected_market != "All":
            filtered = filtered[filtered["market"].astype(str) == selected_market]
        if filtered.empty:
            st.warning("No rows match these filters.")
        else:
            a, b, c = st.columns(3)
            a.metric("Rows", f"{len(filtered):,}")
            b.metric("Mean quantity", f"{filtered['quantity'].mean():,.0f}")
            c.metric("Median modal price", f"{filtered['priceMod'].median():,.0f}")
            chart = filtered.groupby("year", as_index=False).agg(Quantity=("quantity", "mean"), Modal_Price=("priceMod", "median"))
            metric = st.radio("Trend", ["Quantity", "Modal_Price"], horizontal=True)
            fig = px.line(chart, x="year", y=metric, markers=True, title=f"{metric.replace('_', ' ')} over time")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(filtered[["market", "state", "city", "month", "year", "quantity", "priceMin", "priceMax", "priceMod"]].head(100), use_container_width=True, hide_index=True)

elif page == "🗂️ Dataset Explorer":
    st.markdown("""
    <div class="hero">
        <h1>🗂️ Project Dataset Explorer</h1>
        <p>Inspect the CSV datasets used across classification, regression and clustering.</p>
    </div>
    """, unsafe_allow_html=True)

    csv_files = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
    if not csv_files:
        st.warning(f"No CSV files were found in {DATA_DIR}. Place your project datasets in the data/ folder.")
    else:
        st.markdown("### Available datasets")
        inventory = []
        for path in csv_files:
            try:
                frame = pd.read_csv(path, nrows=5)
                with path.open("rb") as handle:
                    row_count = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))
                inventory.append({"File": path.name, "Rows (approx.)": max(0, row_count - 1),
                                  "Columns": len(frame.columns), "Preview columns": ", ".join(map(str, frame.columns[:6]))})
            except Exception as exc:
                inventory.append({"File": path.name, "Rows (approx.)": "—", "Columns": "—", "Preview columns": str(exc)})
        st.dataframe(pd.DataFrame(inventory), use_container_width=True, hide_index=True)

        selected_path = st.selectbox("Open a dataset", csv_files, format_func=lambda path: path.name, key="dataset_explorer_file")
        try:
            dataset = pd.read_csv(selected_path)
            st.markdown(f"### {selected_path.name}")
            a, b, c = st.columns(3)
            a.metric("Rows", f"{len(dataset):,}")
            b.metric("Columns", f"{len(dataset.columns):,}")
            c.metric("Missing cells", f"{int(dataset.isna().sum().sum()):,}")
            tab_preview, tab_schema, tab_stats = st.tabs(["Data preview", "Schema & missing values", "Numeric summary"])
            with tab_preview:
                st.dataframe(dataset.head(200), use_container_width=True, hide_index=True)
            with tab_schema:
                schema = pd.DataFrame({"Column": dataset.columns, "Dtype": dataset.dtypes.astype(str).values,
                                       "Missing values": dataset.isna().sum().values,
                                       "Unique values": dataset.nunique(dropna=True).values})
                st.dataframe(schema, use_container_width=True, hide_index=True)
            with tab_stats:
                numeric = dataset.select_dtypes(include=np.number)
                if numeric.empty:
                    st.info("This dataset has no numeric columns to summarize.")
                else:
                    st.dataframe(numeric.describe().T, use_container_width=True)
                    num_col = st.selectbox("Plot numeric feature", numeric.columns.tolist(), key="dataset_numeric_plot")
                    fig = px.histogram(dataset, x=num_col, nbins=40, title=f"Distribution of {num_col}")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not open {selected_path.name}: {exc}")


else:
    st.markdown("""
    <div class="hero">
        <h1>🏆 Final Decision Board</h1>
        <p>Evidence-backed model selection across the complete ML pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Final model snapshot")

    a,b,c = st.columns(3)
    with a:
        st.markdown("""
        <div class="winner">
        <h3>📊 Classification</h3>
        <b>Logistic Regression</b><br>
        Accuracy: <b>70.61%</b><br>
        Weighted F1: <b>0.7044</b><br>
        ROC-AUC: <b>0.8785</b>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="winner">
        <h3>📈 Regression</h3>
        <b>Lasso Regression</b><br>
        R²: <b>0.5211</b><br>
        RMSE: <b>22.18</b><br>
        MAE: <b>16.03 lakhs</b>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="winner">
        <h3>🔵 Clustering</h3>
        <b>K-Means (k=5 baseline)</b><br>
        Dataset: <b>MarketArrivals.csv</b><br>
        Features: <b>quantity, prices, time</b><br>
        Metrics: <b>available in Cluster Explorer</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧠 The evaluator's one-screen story")
    st.markdown("""
    <div class="section-card">
    <p><b>Data</b> → cleaned and transformed into model-ready features.</p>
    <p><b>Competition</b> → multiple algorithms evaluated using consistent metrics.</p>
    <p><b>Evidence</b> → quantitative metrics supported by visual diagnostics.</p>
    <p><b>Decision</b> → the strongest model is selected using evidence, not complexity.</p>
    <p><b>Interpretation</b> → results are translated into understandable insights.</p>
    </div>
    """, unsafe_allow_html=True)


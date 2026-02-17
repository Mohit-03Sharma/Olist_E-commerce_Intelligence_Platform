"""
dashboard.py — Streamlit Interactive Dashboard for Olist E-Commerce Intelligence

Sections:
  1. EDA Explorer — Interactive charts with filters
  2. Model Performance — ROC, PR curves, confusion matrix
  3. Seller Leaderboard & Geographic Map
  4. Risk Predictor — Input order details, get live prediction

Run: streamlit run app/dashboard.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# ── Page config ──
st.set_page_config(
    page_title="Olist E-Commerce Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, BASE)
DATA_PATH = os.path.join(BASE, "data", "processed", "master_orders.csv")
MODEL_PATH = os.path.join(BASE, "models", "best_model.pkl")
FEATURES_PATH = os.path.join(BASE, "models", "feature_names.json")


# ── Load data ──
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=[
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ])
    delivered = df[df["order_status"] == "delivered"].copy()
    return df, delivered


@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH) as f:
            feat_names = json.load(f)
        return model, feat_names, True
    except Exception as e:
        return None, [], False


df, delivered = load_data()
model, feature_names, model_loaded = load_model()


# ── Sidebar ──
st.sidebar.title("🛒 Olist Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 EDA Explorer", "🤖 Model Performance", "🏪 Seller & Geography", "🔮 Risk Predictor"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(delivered):,} delivered orders")
if model_loaded:
    st.sidebar.caption(f"Model: {type(model).__name__}")
else:
    st.sidebar.warning("Model not loaded. Train first with `python src/train.py`")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: EDA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

if page == "📊 EDA Explorer":
    st.title("📊 Exploratory Data Analysis")

    # ── KPI Row ──
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Orders", f"{len(delivered):,}")
    col2.metric("Revenue", f"R${delivered['total_price'].sum():,.0f}")
    col3.metric("Avg Order Value", f"R${delivered['total_price'].mean():,.2f}")
    col4.metric("Avg Review", f"{delivered['review_score'].mean():.2f} ⭐")
    col5.metric("Negative Rate", f"{(delivered['review_score'] <= 2).mean()*100:.1f}%")

    st.markdown("---")

    # ── Filters ──
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        states = ["All"] + sorted(delivered["customer_state"].dropna().unique().tolist())
        sel_state = st.selectbox("Customer State", states)
    with fcol2:
        categories = ["All"] + sorted(delivered["product_category"].dropna().unique().tolist())
        sel_cat = st.selectbox("Product Category", categories)
    with fcol3:
        date_range = st.date_input(
            "Date Range",
            value=(delivered["order_purchase_timestamp"].min().date(),
                   delivered["order_purchase_timestamp"].max().date()),
        )

    # Apply filters
    filt = delivered.copy()
    if sel_state != "All":
        filt = filt[filt["customer_state"] == sel_state]
    if sel_cat != "All":
        filt = filt[filt["product_category"] == sel_cat]
    if len(date_range) == 2:
        filt = filt[
            (filt["order_purchase_timestamp"].dt.date >= date_range[0]) &
            (filt["order_purchase_timestamp"].dt.date <= date_range[1])
        ]

    st.caption(f"Showing {len(filt):,} orders after filters")

    # ── Charts Row 1: Trend + Review Distribution ──
    ch1, ch2 = st.columns(2)

    with ch1:
        monthly = (
            filt.set_index("order_purchase_timestamp")
            .resample("ME")
            .agg(orders=("order_id", "count"), revenue=("total_price", "sum"))
            .reset_index()
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=monthly["order_purchase_timestamp"], y=monthly["orders"],
                   name="Orders", marker_color="#3b82f6", opacity=0.7),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=monthly["order_purchase_timestamp"], y=monthly["revenue"],
                       name="Revenue", line=dict(color="#ef4444", width=2.5), mode="lines+markers"),
            secondary_y=True,
        )
        fig.update_layout(title="Monthly Orders & Revenue", height=400,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_yaxes(title_text="Orders", secondary_y=False)
        fig.update_yaxes(title_text="Revenue (R$)", secondary_y=True)
        st.plotly_chart(fig, width="stretch")

    with ch2:
        score_dist = filt["review_score"].dropna().value_counts().sort_index()
        colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#16a34a"]
        fig = go.Figure(go.Bar(
            x=score_dist.index, y=score_dist.values,
            marker_color=colors, text=[f"{v/score_dist.sum()*100:.1f}%" for v in score_dist.values],
            textposition="outside",
        ))
        fig.update_layout(title="Review Score Distribution", height=400,
                          xaxis_title="Review Score", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")

    # ── Charts Row 2: Delivery Impact + Payment ──
    ch3, ch4 = st.columns(2)

    with ch3:
        dlv = filt.dropna(subset=["delivery_delta_days", "review_score"])
        fig = px.box(dlv, x="review_score", y="delivery_delta_days",
                     color="review_score",
                     color_discrete_sequence=colors,
                     title="Delivery Delta by Review Score")
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="On-time")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with ch4:
        pay = filt["payment_type"].value_counts()
        fig = px.pie(values=pay.values, names=pay.index,
                     title="Payment Method Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    # ── Charts Row 3: Hourly + Day of Week ──
    ch5, ch6 = st.columns(2)

    with ch5:
        filt_copy = filt.copy()
        filt_copy["hour"] = filt_copy["order_purchase_timestamp"].dt.hour
        hourly = filt_copy.groupby("hour")["order_id"].count().reset_index()
        fig = px.bar(hourly, x="hour", y="order_id", title="Orders by Hour of Day",
                     color="order_id", color_continuous_scale="Blues")
        fig.update_layout(height=350, xaxis_title="Hour", yaxis_title="Orders",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    with ch6:
        dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        filt_copy = filt.copy()
        filt_copy["dow"] = filt_copy["order_purchase_timestamp"].dt.dayofweek.map(dow_map)
        dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = filt_copy.groupby("dow")["order_id"].count().reindex(dow_order).reset_index()
        daily.columns = ["day", "orders"]
        fig = px.bar(daily, x="day", y="orders", title="Orders by Day of Week",
                     color="orders", color_continuous_scale="Blues")
        fig.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    if not model_loaded:
        st.error("Model not found. Run `python src/train.py` first.")
        st.stop()

    # Build features for evaluation
    from src.feature_engineering import build_feature_matrix
    X_train, X_test, y_train, y_test, feat_names = build_feature_matrix(apply_smote=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # ── Metrics Row ──
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
    m2.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
    m3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    m4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")

    st.markdown("---")

    # ── ROC + PR Curves ──
    rc1, rc2 = st.columns(2)

    with rc1:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model",
                                 line=dict(color="#3b82f6", width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=400)
        st.plotly_chart(fig, width="stretch")

    with rc2:
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="Model",
                                 line=dict(color="#ef4444", width=2.5)))
        baseline = y_test.mean()
        fig.add_hline(y=baseline, line_dash="dash", line_color="gray",
                      annotation_text=f"Baseline ({baseline:.2f})")
        fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall",
                          yaxis_title="Precision", height=400)
        st.plotly_chart(fig, width="stretch")

    # ── Confusion Matrix + Feature Importance ──
    cm1, cm2 = st.columns(2)

    with cm1:
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual"),
                        x=["Positive", "Negative"], y=["Positive", "Negative"])
        fig.update_layout(title="Confusion Matrix", height=400)
        st.plotly_chart(fig, width="stretch")

    with cm2:
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({
                "feature": feat_names,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=True).tail(15)
            fig = px.bar(imp, x="importance", y="feature", orientation="h",
                         title="Top 15 Feature Importances",
                         color="importance", color_continuous_scale="Blues")
            fig.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Feature importances not available for this model type.")

    # ── Business Metric: Top 20% Capture ──
    st.markdown("---")
    st.subheader("Business Metric: Risk Flag Capture Rate")

    threshold_pcts = [10, 15, 20, 25, 30]
    capture_data = []
    for pct in threshold_pcts:
        idx = int(len(y_prob) * (1 - pct / 100))
        thresh = np.sort(y_prob)[idx]
        flagged = y_prob >= thresh
        capture = y_test[flagged].sum() / y_test.sum() * 100 if y_test.sum() > 0 else 0
        capture_data.append({"Flag Top %": f"{pct}%", "Negative Reviews Captured": round(capture, 1)})

    cap_df = pd.DataFrame(capture_data)
    fig = px.bar(cap_df, x="Flag Top %", y="Negative Reviews Captured",
                 text="Negative Reviews Captured", color="Negative Reviews Captured",
                 color_continuous_scale="Reds")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=350, yaxis_title="% Captured", coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: SELLER & GEOGRAPHY
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🏪 Seller & Geography":
    st.title("🏪 Seller Leaderboard & Geographic Analysis")

    tab1, tab2 = st.tabs(["📍 Geographic Map", "🏆 Seller Leaderboard"])

    with tab1:
        st.subheader("Order Distribution Across Brazil")

        state_stats = (
            delivered.dropna(subset=["review_score"])
            .groupby("customer_state")
            .agg(
                orders=("order_id", "count"),
                revenue=("total_price", "sum"),
                avg_review=("review_score", "mean"),
                negative_pct=("review_score", lambda x: (x <= 2).mean() * 100),
                avg_delivery=("delivery_time_days", "mean"),
            )
            .round(2)
            .reset_index()
        )

        map_metric = st.selectbox("Color by", ["orders", "avg_review", "negative_pct", "avg_delivery"])

        state_coords = {
            "SP": (-23.55, -46.63), "RJ": (-22.91, -43.17), "MG": (-19.92, -43.94),
            "RS": (-30.03, -51.23), "PR": (-25.43, -49.27), "SC": (-27.59, -48.55),
            "BA": (-12.97, -38.51), "DF": (-15.79, -47.88), "ES": (-20.32, -40.34),
            "GO": (-16.69, -49.25), "PE": (-8.05, -34.87), "CE": (-3.72, -38.53),
            "PA": (-1.46, -48.50), "MT": (-15.60, -56.10), "MA": (-2.53, -44.28),
            "MS": (-20.44, -54.65), "PB": (-7.12, -34.86), "RN": (-5.79, -35.21),
            "AL": (-9.67, -35.74), "PI": (-5.09, -42.80), "SE": (-10.91, -37.07),
            "RO": (-8.76, -63.90), "TO": (-10.18, -48.33), "AM": (-3.12, -60.02),
            "AC": (-9.97, -67.81), "AP": (0.03, -51.07), "RR": (2.82, -60.67),
        }
        state_stats["lat"] = state_stats["customer_state"].map(lambda s: state_coords.get(s, (0, 0))[0])
        state_stats["lng"] = state_stats["customer_state"].map(lambda s: state_coords.get(s, (0, 0))[1])

        color_scale = "Blues" if map_metric == "orders" else "RdYlGn" if map_metric == "avg_review" else "Reds"

        fig = px.scatter_mapbox(
            state_stats, lat="lat", lon="lng", size="orders",
            color=map_metric, hover_name="customer_state",
            hover_data=["orders", "revenue", "avg_review", "negative_pct"],
            color_continuous_scale=color_scale,
            size_max=40, zoom=3.3,
            mapbox_style="open-street-map",
            center={"lat": -14.24, "lon": -51.93},  # center of Brazil
        )
        fig.update_layout(
            height=550, margin=dict(l=0, r=0, t=0, b=0),
            mapbox=dict(
                bounds={"west": -75, "east": -34, "south": -34, "north": 6},  # Brazil bounds
            ),
        )
        st.plotly_chart(fig, width="stretch")

        # Same vs cross state comparison
        st.subheader("Same-State vs Cross-State Delivery")
        sc1, sc2, sc3 = st.columns(3)
        same = delivered[delivered["same_state"] == 1]
        cross = delivered[delivered["same_state"] == 0]
        sc1.metric("Same-State Avg Delivery", f"{same['delivery_time_days'].mean():.1f} days",
                    delta=f"{same['delivery_time_days'].mean() - cross['delivery_time_days'].mean():.1f} days faster")
        sc2.metric("Same-State Avg Review", f"{same['review_score'].mean():.2f}",
                    delta=f"+{same['review_score'].mean() - cross['review_score'].mean():.2f}")
        sc3.metric("Same-State Negative %",
                    f"{(same['review_score'] <= 2).mean()*100:.1f}%",
                    delta=f"{(same['review_score'] <= 2).mean()*100 - (cross['review_score'] <= 2).mean()*100:.1f}%")

    with tab2:
        st.subheader("Seller Performance Leaderboard")

        min_orders = st.slider("Minimum orders for seller", 5, 100, 10)

        seller_stats = (
            delivered.dropna(subset=["review_score"])
            .groupby("seller_id")
            .agg(
                orders=("order_id", "count"),
                avg_review=("review_score", "mean"),
                negative_count=("review_score", lambda x: (x <= 2).sum()),
                revenue=("total_price", "sum"),
                avg_delivery=("delivery_time_days", "mean"),
            )
            .round(2)
        )
        seller_stats["negative_pct"] = (seller_stats["negative_count"] / seller_stats["orders"] * 100).round(1)
        active = seller_stats[seller_stats["orders"] >= min_orders].sort_values("avg_review", ascending=False)

        sort_col = st.selectbox("Sort by", ["avg_review", "orders", "revenue", "negative_pct", "avg_delivery"])
        ascending = sort_col in ["negative_pct", "avg_delivery"]
        active = active.sort_values(sort_col, ascending=ascending)

        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**🏆 Top 10 Sellers**")
            top10 = active.head(10)[["orders", "avg_review", "negative_pct", "revenue"]].reset_index()
            top10["seller_id"] = top10["seller_id"].str[:12] + "..."
            st.dataframe(top10, width="stretch", hide_index=True)

        with t2:
            st.markdown("**⚠️ Bottom 10 Sellers**")
            bot10 = active.tail(10)[["orders", "avg_review", "negative_pct", "revenue"]].reset_index()
            bot10["seller_id"] = bot10["seller_id"].str[:12] + "..."
            st.dataframe(bot10, width="stretch", hide_index=True)

        fig = px.scatter(active.reset_index(), x="orders", y="avg_review",
                         size="revenue", color="negative_pct",
                         color_continuous_scale="RdYlGn_r",
                         hover_data=["orders", "avg_review", "negative_pct"],
                         title="Seller Volume vs Quality")
        fig.add_hline(y=3.5, line_dash="dash", line_color="red",
                      annotation_text="Risk threshold (3.5)")
        fig.update_layout(height=450)
        st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: RISK PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🔮 Risk Predictor":
    st.title("🔮 Order Satisfaction Risk Predictor")
    st.markdown("Input order details below to predict the risk of a negative review.")

    if not model_loaded:
        st.error("Model not found. Run `python src/train.py` first.")
        st.stop()

    # ── Input Form ──
    st.subheader("Order Details")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        delivery_time = st.number_input("Delivery Time (days)", 1, 60, 12)
        delivery_delta = st.number_input("Days Late (neg=early)", -30, 30, -5)
    with c2:
        is_late = st.selectbox("Is Late?", [0, 1], index=0)
        total_price = st.number_input("Order Value (R$)", 1, 5000, 120)
    with c3:
        total_freight = st.number_input("Freight Cost (R$)", 0, 500, 20)
        seller_avg = st.number_input("Seller Avg Review", 1.0, 5.0, 4.0, step=0.1)
    with c4:
        distance = st.number_input("Distance (km)", 0, 4000, 500)
        installments = st.number_input("Installments", 1, 24, 1)

    # ── Explanation of inputs ──
    with st.expander("ℹ️ What inputs lead to LOW vs HIGH risk?"):
        st.markdown("""
        | Factor | 🟢 Low Risk | 🔴 High Risk |
        |--------|------------|-------------|
        | **Delivery Time** | Short (5-10 days) | Long (20+ days) |
        | **Days Late** | Early delivery (-5 or less) | Late delivery (+5 or more) |
        | **Is Late** | 0 (on time) | 1 (past estimated date) |
        | **Order Value** | Moderate (R$50-200) | Very high (R$500+, higher expectations) |
        | **Freight Cost** | Low relative to order | High relative to order |
        | **Seller Avg Review** | High (4.5+) | Low (below 3.5) |
        | **Distance** | Short (<200 km, same state) | Long (1000+ km, cross-state) |
        | **Installments** | Few (1-3) | Many (8+) |
        
        **The #1 driver is delivery performance.** An order that arrives early from a well-reviewed seller 
        will almost always score LOW risk, regardless of other factors.
        """)

    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        defaults = {
            "delivery_time_days": delivery_time, "estimated_delivery_days": 24,
            "delivery_delta_days": delivery_delta, "is_late": is_late,
            "order_item_count": 1, "total_price": total_price, "total_freight": total_freight,
            "freight_ratio": total_freight / max(total_price, 1), "avg_item_price": total_price,
            "payment_installments": installments, "payment_methods_count": 1,
            "product_weight_g": 1500, "product_length_cm": 30,
            "product_height_cm": 10, "product_width_cm": 20,
            "product_photos_qty": 2, "product_description_length": 600,
            "seller_avg_review": seller_avg, "seller_order_count": 50,
            "seller_avg_delivery_time": 12, "seller_negative_pct": 0.1,
            "seller_customer_distance_km": distance, "same_state": 1 if distance < 200 else 0,
            "purchase_hour": 14, "purchase_dayofweek": 2,
            "purchase_month": 6, "is_weekend": 0,
        }

        vector = []
        for feat in feature_names:
            vector.append(float(defaults.get(feat, 0.0)))

        X = np.array(vector).reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        score = prob * 100

        st.markdown("---")

        if score >= 60:
            tier, color, emoji = "HIGH", "red", "🔴"
            action = "Proactive outreach needed — contact customer about delivery status"
        elif score >= 30:
            tier, color, emoji = "MEDIUM", "orange", "🟡"
            action = "Monitor closely — ensure delivery stays on track"
        else:
            tier, color, emoji = "LOW", "green", "🟢"
            action = "No action needed — order is on track"

        r1, r2, r3 = st.columns(3)
        r1.metric("Risk Score", f"{score:.1f}%")
        r2.metric("Risk Tier", f"{emoji} {tier}")
        r3.metric("Prediction", "⚠️ Negative" if prob >= 0.5 else "✅ Positive")

        st.info(f"💡 **Recommended Action:** {action}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Negative Review Risk"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fecaca"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": score},
            },
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")


# ── Footer ──
st.sidebar.markdown("---")
st.sidebar.markdown("[View on GitHub](https://Mohit-03Sharma/olist-ecommerce-intelligence)")
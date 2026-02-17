"""
app.py — Production-Grade Flask REST API for Order Satisfaction Prediction.

Features:
  - RESTful endpoints with proper HTTP status codes
  - Input validation with descriptive error messages
  - Request/response logging
  - CORS support
  - API documentation endpoint
  - Health check with model metadata
  - Single and batch prediction
  - Rate-limiting aware design

Endpoints:
  GET  /health          → model metadata, status, uptime
  GET  /docs            → API documentation (interactive)
  POST /predict         → predict satisfaction for a single order
  POST /predict/batch   → predict for multiple orders (max 100)
  GET  /features        → list of model features with descriptions
  GET  /stats           → summary stats from training data

Usage:
  python api/app.py
  Then visit http://localhost:5000/docs
"""

import os
import sys
import json
import time
import logging
import numpy as np
import joblib
from functools import wraps
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string

# ── Setup ──
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_orders.csv")

app = Flask(__name__)
START_TIME = datetime.utcnow()

# ── Logging ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Load model & feature names ──
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        feature_names = json.load(f)
    MODEL_LOADED = True
    MODEL_INFO = {
        "model_type": type(model).__name__,
        "n_features": len(feature_names),
        "status": "loaded",
    }
    logger.info(f"Model loaded: {MODEL_INFO['model_type']} with {MODEL_INFO['n_features']} features")
except Exception as e:
    MODEL_LOADED = False
    MODEL_INFO = {"status": "error", "error": str(e)}
    model, feature_names = None, []
    logger.error(f"Model failed to load: {e}")


# ── Load summary stats for /stats endpoint ──
try:
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    delivered = df[df["order_status"] == "delivered"]
    SUMMARY_STATS = {
        "total_orders": int(len(delivered)),
        "total_revenue": round(float(delivered["total_price"].sum()), 2),
        "avg_order_value": round(float(delivered["total_price"].mean()), 2),
        "avg_review_score": round(float(delivered["review_score"].mean()), 2),
        "negative_review_rate": round(float((delivered["review_score"] <= 2).mean() * 100), 2),
        "avg_delivery_days": round(float(delivered["delivery_time_days"].mean()), 1),
        "late_delivery_rate": round(float(delivered["is_late"].mean() * 100), 1),
    }
except Exception:
    SUMMARY_STATS = {"error": "Could not load training data stats"}


# ── Feature descriptions ──
FEATURE_DOCS = {
    "delivery_time_days": {"desc": "Actual delivery duration in days", "type": "float", "example": 12.0},
    "estimated_delivery_days": {"desc": "Promised delivery window in days", "type": "float", "example": 24.0},
    "delivery_delta_days": {"desc": "Days late (negative=early, positive=late)", "type": "float", "example": -5.0},
    "is_late": {"desc": "Binary: 1 if delivered after estimated date", "type": "int", "example": 0},
    "order_item_count": {"desc": "Number of items in order", "type": "int", "example": 1},
    "total_price": {"desc": "Total order value in BRL", "type": "float", "example": 120.0},
    "total_freight": {"desc": "Total shipping cost in BRL", "type": "float", "example": 20.0},
    "freight_ratio": {"desc": "Freight / total_price ratio", "type": "float", "example": 0.17},
    "avg_item_price": {"desc": "Average price per item in BRL", "type": "float", "example": 120.0},
    "payment_installments": {"desc": "Max installments used", "type": "int", "example": 3},
    "payment_methods_count": {"desc": "Number of payment methods used", "type": "int", "example": 1},
    "seller_avg_review": {"desc": "Seller's historical average review score", "type": "float", "example": 4.0},
    "seller_order_count": {"desc": "Seller's total historical orders", "type": "int", "example": 50},
    "seller_avg_delivery_time": {"desc": "Seller's average delivery time", "type": "float", "example": 12.0},
    "seller_negative_pct": {"desc": "Seller's historical negative review %", "type": "float", "example": 0.1},
    "seller_customer_distance_km": {"desc": "Haversine distance seller→customer", "type": "float", "example": 500.0},
    "same_state": {"desc": "1 if seller and customer in same state", "type": "int", "example": 0},
    "purchase_hour": {"desc": "Hour of purchase (0-23)", "type": "int", "example": 14},
    "purchase_dayofweek": {"desc": "Day of week (0=Mon, 6=Sun)", "type": "int", "example": 2},
    "purchase_month": {"desc": "Month of purchase (1-12)", "type": "int", "example": 6},
    "is_weekend": {"desc": "1 if purchased on Saturday/Sunday", "type": "int", "example": 0},
    "product_weight_g": {"desc": "Product weight in grams", "type": "float", "example": 1500.0},
    "product_length_cm": {"desc": "Product length in cm", "type": "float", "example": 30.0},
    "product_height_cm": {"desc": "Product height in cm", "type": "float", "example": 10.0},
    "product_width_cm": {"desc": "Product width in cm", "type": "float", "example": 20.0},
    "product_photos_qty": {"desc": "Number of product listing photos", "type": "float", "example": 2.0},
    "product_description_length": {"desc": "Character count of product description", "type": "float", "example": 600.0},
}

DEFAULTS = {k: v["example"] for k, v in FEATURE_DOCS.items()}


# ── Helpers ──

def validate_input(data, required_fields=None):
    """Validate input data and return errors if any."""
    errors = []
    if not data:
        return ["Request body is empty. Send JSON with order features."]

    if required_fields:
        for f in required_fields:
            if f not in data:
                errors.append(f"Missing required field: '{f}'")

    for key, val in data.items():
        if key in FEATURE_DOCS:
            try:
                float(val)
            except (ValueError, TypeError):
                errors.append(f"Field '{key}' must be numeric, got: {val}")

    return errors


def build_feature_vector(input_data: dict) -> np.ndarray:
    """Build a full feature vector from partial input, filling defaults."""
    vector = []
    for feat in feature_names:
        if feat in input_data:
            vector.append(float(input_data[feat]))
        elif feat in DEFAULTS:
            vector.append(DEFAULTS[feat])
        else:
            vector.append(0.0)
    return np.array(vector).reshape(1, -1)


def risk_tier(prob):
    """Map probability to a risk tier with action."""
    if prob >= 0.6:
        return {"tier": "HIGH", "emoji": "🔴", "color": "#ef4444",
                "action": "Proactive outreach needed — contact customer about delivery status"}
    elif prob >= 0.3:
        return {"tier": "MEDIUM", "emoji": "🟡", "color": "#eab308",
                "action": "Monitor closely — ensure delivery stays on track"}
    else:
        return {"tier": "LOW", "emoji": "🟢", "color": "#22c55e",
                "action": "No action needed — order is on track"}


def log_request(endpoint):
    """Decorator to log API requests."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            elapsed = (time.time() - start) * 1000
            logger.info(f"{request.method} {endpoint} — {elapsed:.0f}ms")
            return result
        return wrapper
    return decorator


# ── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
@log_request("/health")
def health():
    """Health check: model status, uptime, metadata."""
    uptime = str(datetime.utcnow() - START_TIME)
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "degraded",
        "uptime": uptime,
        "model": MODEL_INFO,
        "timestamp": datetime.utcnow().isoformat(),
    })


@app.route("/features", methods=["GET"])
@log_request("/features")
def features():
    """List all model features with descriptions and example values."""
    docs = {}
    for feat in feature_names:
        if feat in FEATURE_DOCS:
            docs[feat] = FEATURE_DOCS[feat]
        else:
            docs[feat] = {"desc": "Encoded feature", "type": "float", "example": 0.0}
    return jsonify({
        "total_features": len(feature_names),
        "features": docs,
        "note": "Pass any subset of these in /predict. Missing features use default values.",
    })


@app.route("/stats", methods=["GET"])
@log_request("/stats")
def stats():
    """Summary statistics from the training dataset."""
    return jsonify(SUMMARY_STATS)


@app.route("/predict", methods=["POST"])
@log_request("/predict")
def predict():
    """
    Predict satisfaction risk for a single order.
    
    Accepts JSON body with any subset of order features.
    Returns risk score (0-100%), tier, and recommended action.
    """
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded", "hint": "Run: python src/train.py"}), 503

    data = request.get_json(silent=True)
    errors = validate_input(data)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    try:
        X = build_feature_vector(data)
        prob = float(model.predict_proba(X)[0][1])
        tier = risk_tier(prob)

        return jsonify({
            "risk_score": round(prob * 100, 1),
            "risk_probability": round(prob, 4),
            "risk_tier": tier["tier"],
            "risk_emoji": tier["emoji"],
            "recommended_action": tier["action"],
            "prediction": "negative" if prob >= 0.5 else "positive",
            "features_used": len(feature_names),
            "features_provided": len([k for k in data if k in FEATURE_DOCS]),
            "features_defaulted": len(feature_names) - len([k for k in data if k in FEATURE_DOCS]),
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
@log_request("/predict/batch")
def predict_batch():
    """
    Predict satisfaction risk for multiple orders (max 100).
    
    Accepts JSON: {"orders": [{...}, {...}, ...]}
    Returns predictions array with risk scores.
    """
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data or "orders" not in data:
        return jsonify({
            "error": "Invalid format",
            "expected": {"orders": [{"delivery_time_days": 15, "...": "..."}]},
        }), 400

    orders = data["orders"]
    if len(orders) > 100:
        return jsonify({"error": f"Batch size {len(orders)} exceeds max of 100"}), 400

    results = []
    for i, order in enumerate(orders):
        try:
            X = build_feature_vector(order)
            prob = float(model.predict_proba(X)[0][1])
            tier = risk_tier(prob)
            results.append({
                "order_index": i,
                "risk_score": round(prob * 100, 1),
                "risk_tier": tier["tier"],
                "recommended_action": tier["action"],
            })
        except Exception as e:
            results.append({"order_index": i, "error": str(e)})

    summary = {
        "high_risk": sum(1 for r in results if r.get("risk_tier") == "HIGH"),
        "medium_risk": sum(1 for r in results if r.get("risk_tier") == "MEDIUM"),
        "low_risk": sum(1 for r in results if r.get("risk_tier") == "LOW"),
        "errors": sum(1 for r in results if "error" in r),
    }

    return jsonify({"predictions": results, "total": len(results), "summary": summary})


@app.route("/docs", methods=["GET"])
def docs():
    """Interactive API documentation page."""
    return render_template_string(DOCS_HTML, model_info=MODEL_INFO, stats=SUMMARY_STATS)


# ── API Documentation HTML ──

DOCS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Olist API Documentation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }
        .header { background: linear-gradient(135deg, #1e293b, #334155); padding: 2.5rem; text-align: center; border-bottom: 2px solid #3b82f6; }
        .header h1 { font-size: 2rem; color: #60a5fa; }
        .header p { color: #94a3b8; margin-top: 0.5rem; }
        .container { max-width: 900px; margin: 2rem auto; padding: 0 1.5rem; }
        .endpoint { background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; }
        .endpoint h3 { margin-bottom: 0.8rem; }
        .method { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 700; margin-right: 0.5rem; }
        .get { background: #22c55e; color: #000; }
        .post { background: #3b82f6; color: #fff; }
        .path { font-family: monospace; font-size: 1rem; color: #60a5fa; }
        .desc { color: #94a3b8; margin: 0.5rem 0; }
        pre { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 1rem; overflow-x: auto; font-size: 0.85rem; margin-top: 0.5rem; color: #a5f3fc; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }
        .stat-box { background: #1e293b; border-radius: 8px; padding: 1rem; text-align: center; border: 1px solid #334155; }
        .stat-box .val { font-size: 1.4rem; font-weight: 700; color: #60a5fa; }
        .stat-box .lbl { font-size: 0.75rem; color: #64748b; text-transform: uppercase; margin-top: 0.3rem; }
        .try-it { background: #3b82f6; color: #fff; border: none; padding: 0.5rem 1.2rem; border-radius: 6px; cursor: pointer; font-size: 0.9rem; margin-top: 0.8rem; }
        .try-it:hover { background: #2563eb; }
        .response { display: none; margin-top: 0.8rem; }
        .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 12px; font-size: 0.75rem; margin-left: 0.5rem; }
        .badge-ok { background: #22c55e22; color: #22c55e; border: 1px solid #22c55e44; }
        .badge-model { background: #3b82f622; color: #60a5fa; border: 1px solid #3b82f644; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛒 Olist Satisfaction Prediction API</h1>
        <p>REST API for predicting customer review risk on e-commerce orders</p>
        <p style="margin-top: 0.5rem;">
            <span class="badge badge-ok">Status: {{ model_info.status }}</span>
            <span class="badge badge-model">Model: {{ model_info.model_type }}</span>
        </p>
    </div>

    <div class="container">
        <div class="stats-grid">
            <div class="stat-box"><div class="val">{{ stats.total_orders }}</div><div class="lbl">Training Orders</div></div>
            <div class="stat-box"><div class="val">{{ stats.avg_review_score }}</div><div class="lbl">Avg Review</div></div>
            <div class="stat-box"><div class="val">{{ stats.negative_review_rate }}%</div><div class="lbl">Negative Rate</div></div>
            <div class="stat-box"><div class="val">{{ model_info.n_features }}</div><div class="lbl">Features</div></div>
        </div>

        <div class="endpoint">
            <h3><span class="method get">GET</span><span class="path">/health</span></h3>
            <p class="desc">Model health check — returns status, uptime, and metadata.</p>
            <button class="try-it" onclick="tryEndpoint('/health', 'health-res')">Try it</button>
            <pre class="response" id="health-res"></pre>
        </div>

        <div class="endpoint">
            <h3><span class="method get">GET</span><span class="path">/features</span></h3>
            <p class="desc">List all model features with descriptions and example values.</p>
            <button class="try-it" onclick="tryEndpoint('/features', 'feat-res')">Try it</button>
            <pre class="response" id="feat-res"></pre>
        </div>

        <div class="endpoint">
            <h3><span class="method get">GET</span><span class="path">/stats</span></h3>
            <p class="desc">Summary statistics from the training dataset.</p>
            <button class="try-it" onclick="tryEndpoint('/stats', 'stats-res')">Try it</button>
            <pre class="response" id="stats-res"></pre>
        </div>

        <div class="endpoint">
            <h3><span class="method post">POST</span><span class="path">/predict</span></h3>
            <p class="desc">Predict satisfaction risk for a single order. Send any subset of features as JSON.</p>
            <pre>{
  "delivery_time_days": 20,
  "delivery_delta_days": 8,
  "is_late": 1,
  "total_price": 250,
  "seller_avg_review": 3.0,
  "seller_customer_distance_km": 1200
}</pre>
            <button class="try-it" onclick="tryPredict()">Try with sample data</button>
            <pre class="response" id="predict-res"></pre>
        </div>

        <div class="endpoint">
            <h3><span class="method post">POST</span><span class="path">/predict/batch</span></h3>
            <p class="desc">Predict risk for multiple orders (max 100). Returns individual predictions + summary.</p>
            <pre>{
  "orders": [
    {"delivery_delta_days": 8, "is_late": 1, "seller_avg_review": 3.0},
    {"delivery_delta_days": -5, "is_late": 0, "seller_avg_review": 4.8}
  ]
}</pre>
            <button class="try-it" onclick="tryBatch()">Try with sample data</button>
            <pre class="response" id="batch-res"></pre>
        </div>
    </div>

    <script>
        async function tryEndpoint(url, resId) {
            const el = document.getElementById(resId);
            el.style.display = 'block';
            el.textContent = 'Loading...';
            try {
                const r = await fetch(url);
                const d = await r.json();
                el.textContent = JSON.stringify(d, null, 2);
            } catch (e) { el.textContent = 'Error: ' + e.message; }
        }

        async function tryPredict() {
            const el = document.getElementById('predict-res');
            el.style.display = 'block';
            el.textContent = 'Loading...';
            try {
                const r = await fetch('/predict', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({delivery_time_days:20, delivery_delta_days:8, is_late:1, total_price:250, seller_avg_review:3.0, seller_customer_distance_km:1200})
                });
                el.textContent = JSON.stringify(await r.json(), null, 2);
            } catch (e) { el.textContent = 'Error: ' + e.message; }
        }

        async function tryBatch() {
            const el = document.getElementById('batch-res');
            el.style.display = 'block';
            el.textContent = 'Loading...';
            try {
                const r = await fetch('/predict/batch', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({orders: [
                        {delivery_delta_days:8, is_late:1, seller_avg_review:3.0},
                        {delivery_delta_days:-5, is_late:0, seller_avg_review:4.8}
                    ]})
                });
                el.textContent = JSON.stringify(await r.json(), null, 2);
            } catch (e) { el.textContent = 'Error: ' + e.message; }
        }
    </script>
</body>
</html>
"""


# ── Error handlers ──

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "docs": "Visit /docs for API documentation"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed", "docs": "Visit /docs for API documentation"}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ── Run ──

if __name__ == "__main__":
    from waitress import serve
    print("\n🚀 Olist Satisfaction Prediction API")
    print("=" * 45)
    print(f"   Docs:      http://localhost:5000/docs")
    print(f"   Health:    http://localhost:5000/health")
    print(f"   Features:  http://localhost:5000/features")
    print(f"   Stats:     http://localhost:5000/stats")
    print(f"   Predict:   POST http://localhost:5000/predict")
    print(f"   Batch:     POST http://localhost:5000/predict/batch")
    print("=" * 45 + "\n")
    serve(app, host="0.0.0.0", port=5000)
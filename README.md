# 🛒 Olist E-Commerce Intelligence Platform

An end-to-end machine learning project that predicts customer satisfaction risk on a Brazilian e-commerce marketplace — from raw data through deployed API and interactive dashboard.

Built on the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) containing 100K+ real orders from 2016–2018.

---

## The Problem

Olist connects small Brazilian merchants to major online marketplaces. After each purchase, customers leave a review (1–5 stars). About **11.5% of orders receive negative reviews** (1–2 stars), which hurt seller visibility and platform trust.

The challenge: **Can we predict which orders will get a negative review before the customer submits one?** If yes, the operations team can intervene proactively — reaching out to the customer, expediting a delayed shipment, or flagging a problematic seller.

## Key Findings

- **Late deliveries are the #1 problem.** Orders that arrive past the estimated delivery date are roughly 5x more likely to receive a 1–2 star review. This single factor dominates all other predictors.
- **A small group of sellers drives most complaints.** The worst-performing 10% of sellers are responsible for approximately 40% of all negative reviews.
- **Geography impacts satisfaction.** Same-state deliveries (seller and customer in the same state) have about half the negative review rate compared to cross-state shipments, largely because they arrive faster.
- **The model can catch most problems early.** By flagging the top 20% riskiest orders at checkout, the model catches 59% of all orders that will eventually receive negative reviews — enabling targeted intervention without overwhelming the support team.

---

## Project Structure

```
olist-ecommerce-intelligence/
│
├── data/
│   ├── raw/                          # 8 original Olist CSVs + translation file
│   ├── processed/                    # Cleaned master table (generated)
│   └── data_dictionary.md            # Column documentation with decisions
│
├── notebooks/
│   ├── 02_EDA_business_insights      # 9-section EDA, every chart answers a business question
│   ├── 03_feature_engineering        # Feature pipeline walkthrough
│   ├── 05_model_interpretation       # SHAP analysis + business recommendations
│   └── 06_nlp_review_analysis        # Portuguese text analysis with topic modeling
│
├── src/
│   ├── data_loader.py                # Load 8 CSVs, clean, join into master table
│   ├── feature_engineering.py        # 20+ features, encoding, scaling, SMOTE
│   ├── train.py                      # 4 models, hyperparameter tuning, MLflow tracking
│   └── evaluate.py                   # ROC, PR curves, confusion matrices, comparisons
│
├── sql/
│   ├── 01_schema_exploration.sql     # Table profiling and key validation
│   ├── 02_master_join.sql            # Full 8-table join with CTEs
│   └── 03_business_kpis.sql          # Business metric queries
│
├── api/
│   ├── app.py                        # Flask REST API with /docs, /predict, /predict/batch
│   └── templates/dashboard.html      # Lightweight HTML interface
│
├── app/
│   └── dashboard.py                  # Streamlit interactive dashboard (4 pages)
│
├── models/                           # Saved model + feature names (generated)
├── reports/
│   ├── figures/                      # All generated charts
│   └── final_report.md               # Executive summary with recommendations
│
├── Dockerfile
├── docker-compose.yml
├── environment.yml                   # Conda environment
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.12
- Anaconda/Miniconda (recommended)
- [Olist dataset from Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — download and place all CSVs in `data/raw/`

### Setup
```bash
git clone https://github.com/YOUR_USERNAME/olist-ecommerce-intelligence.git
cd olist-ecommerce-intelligence

# Create environment
conda env create -f environment.yml
conda activate olist_ds
pip install streamlit
```

### Run the Pipeline
```bash
# Step 1: Build master table from 8 raw CSVs
python src/data_loader.py

# Step 2: Train and compare 4 models
python src/train.py

# Step 3: Launch the interactive dashboard
streamlit run app/dashboard.py

# Step 4: Launch the REST API
python api/app.py
# Visit http://localhost:5000/docs for interactive API documentation
```

---

## How It Works

### Data Pipeline
The project starts with 8 normalized CSV tables from a real e-commerce database. The data loader joins them into a single order-level table, handling tricky parts like multi-item orders (aggregated by summing prices and counting items), multiple payment methods per order, and duplicate geolocation entries (averaged by zip code). The result is one row per order with customer, product, seller, payment, review, and geographic information.

### Feature Engineering
Raw columns aren't enough for good predictions. The pipeline creates 20+ features across 7 groups:

- **Delivery features** — actual delivery time, how late/early vs the estimate, binary late flag
- **Order features** — item count, total value, freight cost, shipping-to-price ratio
- **Seller features** — historical average review, order count, typical delivery speed, negative review percentage
- **Geographic features** — haversine distance between seller and customer, same-state flag
- **Product features** — weight, dimensions, photo count, description length
- **Payment features** — installment count, payment method
- **Time features** — hour, day of week, month, weekend flag

High-cardinality categoricals (product category, state) use target encoding with smoothing to prevent leakage. The 88/12 class imbalance is handled with SMOTE oversampling on the training set only.

### Models
Four algorithms are compared using 5-fold stratified cross-validation with randomized hyperparameter search:

| Model | Description |
|-------|-------------|
| Logistic Regression | Interpretable baseline |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosting (industry standard) |
| LightGBM | Fast gradient boosting |

The best performing model was **LightGBM**, selected automatically after comparing all four:

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.8029 |
| PR-AUC | 0.5032 |
| F1 Score | 0.4422 |
| Precision | 0.7044 |
| Recall | 0.3223 |
| Top-20% Capture | 59.0% |

**What these numbers mean:** The model correctly identifies high-risk orders with 70% precision — when it flags an order, it's right 7 out of 10 times. By flagging just the top 20% riskiest orders, it catches 59% of all orders that will eventually receive negative reviews. This means the support team can focus on a manageable subset of orders while still catching the majority of problems.

All experiments are tracked with MLflow for reproducibility.

### Model Interpretation
SHAP (SHapley Additive exPlanations) values reveal what drives each prediction. The top predictive features are delivery-related — how late or early the order arrived, the seller's track record, and the geographic distance between seller and customer. These findings directly inform the business recommendations.

### NLP Analysis
Customer reviews are in Brazilian Portuguese. Using language-agnostic techniques (TF-IDF vectorization, NMF topic modeling), the analysis identifies what specifically makes customers unhappy. Delivery complaints dominate negative reviews, confirming the findings from the structured data analysis. A text-only classifier demonstrates that review text contains meaningful predictive signal.

---

## Interactive Dashboard

The Streamlit dashboard has 4 pages:

**📊 EDA Explorer** — Filter by state, category, and date range. Interactive charts for monthly trends, review distribution, delivery impact, payment methods, and time patterns.

**🤖 Model Performance** — Live ROC and Precision-Recall curves, confusion matrix, feature importance ranking, and a business metric showing how many negative reviews the model catches at different flagging thresholds.

**🏪 Seller & Geography** — Interactive map of Brazil showing order volume, average reviews, and negative rates by state. Seller leaderboard with sorting and filtering. Same-state vs cross-state delivery comparison.

**🔮 Risk Predictor** — Input order details (delivery time, seller rating, distance, etc.) and get a live risk prediction with a gauge visualization, risk tier, and recommended action.

```bash
streamlit run app/dashboard.py
```

---

## REST API

The Flask API serves the model as a production-ready service with interactive documentation.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/docs` | Interactive API documentation with "Try it" buttons |
| GET | `/health` | Model status, uptime, metadata |
| GET | `/features` | All features with descriptions and examples |
| GET | `/stats` | Summary statistics from training data |
| POST | `/predict` | Risk prediction for a single order |
| POST | `/predict/batch` | Batch predictions (up to 100 orders) |

**Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "delivery_time_days": 20,
    "delivery_delta_days": 8,
    "is_late": 1,
    "seller_avg_review": 3.0,
    "seller_customer_distance_km": 1200
  }'
```

**Response:**
```json
{
  "risk_score": 72.3,
  "risk_tier": "HIGH",
  "recommended_action": "Proactive outreach needed",
  "features_provided": 5,
  "features_defaulted": 21
}
```

---

## Recommendations

Based on the analysis, four actions would have the highest impact:

1. **Proactive delivery alerts** — Flag orders running late and reach out to customers before they leave a review. Late orders have a dramatically higher negative review rate.

2. **Seller quality program** — The worst 10% of sellers generate a disproportionate share of complaints. Requiring quality improvement before allowing more listings would reduce negative reviews significantly.

3. **Regional fulfillment** — Same-state orders perform much better. Establishing distribution centers in high-volume states (São Paulo, Rio de Janeiro, Minas Gerais) would cut delivery times for the busiest routes.

4. **Real-time risk scoring** — Deploy the model to score orders at checkout and route high-risk orders to a dedicated support queue, enabling intervention before problems escalate.

---

## Tools & Technologies

**Data & ML:** Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP, SMOTE

**NLP:** NLTK, TF-IDF, NMF Topic Modeling, WordCloud

**Visualization:** Matplotlib, Seaborn, Plotly

**Dashboard:** Streamlit

**API & Deployment:** Flask, Waitress, Docker

**Tracking:** MLflow

**Database:** SQL (complex 8-table joins, CTEs, window functions)

---

## License

MIT

---

*Built as a comprehensive data science portfolio project demonstrating end-to-end ML capabilities — from raw relational data through deployed prediction service.*

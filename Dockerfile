FROM python:3.12-slim

WORKDIR /app

# Install dependencies for both Flask API and Streamlit dashboard
RUN pip install --no-cache-dir \
    flask==3.0.2 \
    waitress==3.0.0 \
    streamlit==1.41.0 \
    scikit-learn==1.4.1 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    numpy==1.26.4 \
    pandas==2.2.1 \
    plotly==5.19.0 \
    joblib==1.3.2

# Copy project files
COPY models/ models/
COPY data/processed/ data/processed/
COPY src/ src/
COPY api/ api/
COPY app/ app/

# Expose ports: 5000 for Flask API, 8501 for Streamlit
EXPOSE 5000 8501

# Default: run Flask API (override in docker-compose for Streamlit)
CMD ["python", "api/app.py"]
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY requirements/dashboard.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY dashboard ./dashboard
COPY config.yaml ./
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

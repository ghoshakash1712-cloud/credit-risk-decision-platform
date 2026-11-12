FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY requirements/api.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .
COPY api ./api
COPY config.yaml ./
CMD ["gunicorn", "-w", "2", "--preload", "-b", "0.0.0.0:8000", "api.app:create_app()"]

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src:/app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY broker ./broker
COPY config ./config
COPY storage ./storage
COPY ingestion ./ingestion
COPY configs ./configs
COPY .streamlit ./.streamlit
COPY main.py live_fire_test.py entrypoint.sh ./
RUN python -m pip install --no-cache-dir . && chmod +x /app/entrypoint.sh
EXPOSE 8501
ENTRYPOINT ["/app/entrypoint.sh"]

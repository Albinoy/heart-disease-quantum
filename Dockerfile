# CardiaQ container: works on Hugging Face Spaces, Render, Railway, Fly.io, Google Cloud Run, etc.
#   docker build -t cardiaq .
#   docker run -p 8501:8501 cardiaq        -> http://localhost:8501
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Most hosts pass the port to listen on in $PORT (Hugging Face Spaces uses 7860).
ENV PORT=8501
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/_stcore/health')"

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]

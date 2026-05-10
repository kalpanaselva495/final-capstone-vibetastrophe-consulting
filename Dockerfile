FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /stapp

# Install Python dependencies first to maximize Docker layer cache reuse.
COPY requirements.txt ./requirements.txt
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch && \
    python -m pip install -r requirements.txt

# Copy local project files into the image.
COPY . .

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)"

ENTRYPOINT ["streamlit", "run", "webapp/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

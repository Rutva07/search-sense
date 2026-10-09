FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir .
COPY artifacts ./artifacts
ENV REDIS_URL=redis://redis:6379/0
EXPOSE 8000
CMD ["searchsense", "serve", "--host", "0.0.0.0", "--port", "8000"]

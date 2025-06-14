# Build stage
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY Pipfile Pipfile.lock ./

# Install dependencies using pipenv
RUN pip install pipenv && \
    pipenv install --deploy --system

# Final stage
FROM python:3.11-slim AS runner

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /usr/local/include/ /usr/local/include/
COPY .env /app/.env

# Copy application code
COPY airtable_sync.py .

# Create directory for CSV output
RUN mkdir -p /data/csv_output

# Set environment variables
ENV AIRTABLE_OUTPUT_DIR=/data/csv_output

# Create a volume for CSV files
VOLUME ["/data/csv_output"]

CMD ["python", "airtable_sync.py"]
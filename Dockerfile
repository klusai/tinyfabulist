# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y gcc libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage build cache
COPY requirements.txt ./

# Install Python dependencies into the user's local directory
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Create necessary directories
RUN mkdir -p data

# Declare /app/data as a volume so that modifications persist
VOLUME ["/app/data"]

# Copy the installed Python packages from the builder stage
COPY --from=builder /root/.local /root/.local

# Update PATH to include the user-level Python binaries
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy the rest of the application code
COPY . .

# Ensure the entrypoint script has executable permissions
RUN chmod +x /app/scripts/entrypoint.sh

# Use bash to run the entrypoint, passing along default arguments
ENTRYPOINT ["/bin/bash", "/app/scripts/entrypoint.sh"]
CMD ["--help"]
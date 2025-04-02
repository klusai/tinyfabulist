FROM fedora:38

# Install Python and build dependencies
RUN dnf -y update && dnf install -y \
    python3 \
    python3-pip \
    python3-devel \
    bash \
    curl \
    git \
    gcc \
    glibc-devel \
    libffi-devel \
    openssl-devel \
    make \
    && dnf clean all
    
WORKDIR /app

RUN mkdir -p data

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# After copying requirements.txt, list its details for debugging
COPY requirements.txt ./
RUN ls -la /app/requirements.txt

# Then install the requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code, including entrypoint.sh
COPY . .

# Ensure the entrypoint script has executable permissions
RUN chmod +x /app/scripts/entrypoint.sh

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["--help"]
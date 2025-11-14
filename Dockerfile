# Dockerfile for beerbotplus
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps (if needed) and pip packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Use a non-root user in production if desired
# RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
# USER appuser

CMD ["python", "main.py"]

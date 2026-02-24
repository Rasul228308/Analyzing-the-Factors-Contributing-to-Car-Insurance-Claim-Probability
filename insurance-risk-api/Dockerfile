# Stage 1: Base image
# python:3.11-slim is a minimal Python image (~50MB vs ~900MB for full Python)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /insurance-api

# Copy requirements first (Docker caches layers — if requirements don't change,
# this layer is reused on rebuild, making rebuilds much faster)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project
COPY app/ ./app/
COPY model/ ./model/

# Tell Docker this container listens on port 8000
EXPOSE 8000

# Command to run when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

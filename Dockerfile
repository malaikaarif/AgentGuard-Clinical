# AgentGuard-Clinical — containerized FastAPI dashboard + agent pipeline

FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (separate layer so this is cached and
# doesn't reinstall on every code change, only when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project.
# Note: the model weights file (model/mobilenetv2_paper_exact.keras) is
# gitignored but NOT dockerignored — it must exist on your local disk
# before running "docker build", since Docker's build context reads
# from disk, not from git.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
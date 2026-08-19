# 🚀 CropCopilot Deployment Guide

This guide provides end-to-end instructions for deploying **CropCopilot** across multiple cloud providers, container runtimes, and PaaS platforms.

---

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Option 1: Deploy on Render (Recommended / Free Tier)](#option-1-deploy-on-render-recommended--free-tier)
- [Option 2: Deploy on Railway (1-Click Docker)](#option-2-deploy-on-railway-1-click-docker)
- [Option 3: Deploy on Vercel (Serverless)](#option-3-deploy-on-vercel-serverless)
- [Option 4: Deploy with Docker & Docker Compose (VPS / Self-Hosted)](#option-4-deploy-with-docker--docker-compose-vps--self-hosted)
- [Option 5: Deploy on Fly.io (Persistent Disk Edge)](#option-5-deploy-on-flyio-persistent-disk-edge)
- [Option 6: Deploy on Hugging Face Spaces](#option-6-deploy-on-hugging-face-spaces)
- [Option 7: Google Cloud Run / AWS App Runner](#option-7-google-cloud-run--aws-app-runner)
- [⚙️ Environment Variables Reference](#️-environment-variables-reference)
- [🩺 Health Checks & Verification](#-health-checks--verification)
- [🛠 Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🔑 Prerequisites

Before deploying to any platform, make sure you have:

1. **GitHub Repository**: Push this project to your GitHub account (`https://github.com/<your-username>/CropCopilot`).
2. **NVIDIA NIM API Key**: Free registration at [build.nvidia.com](https://build.nvidia.com). You need this key to power the LLaMA 3.1 70B LLM and NVIDIA E5 embedding model.

---

## Option 1: Deploy on Render (Recommended / Free Tier)

Render is the simplest and fastest platform to host the CropCopilot web application.

### Method A: Using the `render.yaml` Blueprint (Automatic)

1. Go to [dashboard.render.com](https://dashboard.render.com/) and log in.
2. Click **New +** → **Blueprint**.
3. Connect your GitHub repository `CropCopilot`.
4. Render will detect `render.yaml` automatically.
5. In the configuration screen, enter your `NVIDIA_API_KEY`.
6. Click **Apply**. Render will install dependencies, ingest the agriculture dataset, and start the app.

### Method B: Manual Web Service Setup

1. On the Render Dashboard, click **New +** → **Web Service**.
2. Connect your `CropCopilot` repository.
3. Configure the service settings:
   - **Name**: `cropcopilot`
   - **Language**: `Python`
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt && python data_ingestion.py`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Under **Environment Variables**, add:
   - `NVIDIA_API_KEY`: `nvapi-your-key-here`
   - `PYTHON_VERSION`: `3.11.9`
5. Click **Create Web Service**.

> [!TIP]
> Render sets the `$PORT` environment variable automatically, and the application will dynamically bind to it.

---

## Option 2: Deploy on Railway (1-Click Docker)

Railway automatically detects the `Dockerfile` and builds a production container.

1. Go to [railway.app](https://railway.app) and sign in with GitHub.
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select your `CropCopilot` repository.
4. Click on the newly created service → go to the **Variables** tab.
5. Add the environment variable:
   - `NVIDIA_API_KEY` = `nvapi-your-actual-api-key`
6. Go to the **Settings** tab:
   - Under **Networking**, click **Generate Domain** (gives you a public `*.up.railway.app` URL).
7. Railway will build the container, perform health checks on `/health`, and bring the app live!

---

## Option 3: Deploy on Vercel (Serverless)

CropCopilot includes [`vercel.json`](vercel.json) and [`api/index.py`](api/index.py) for Vercel Python serverless deployment.

### Method A: Deploy via Vercel Dashboard (Fastest)

1. Go to **[vercel.com](https://vercel.com)** and log in with your GitHub account.
2. Click **"Add New..."** → **"Project"**.
3. Import your **`CropCopilot`** repository.
4. In the Project Configuration:
   - **Framework Preset**: Other (automatically detected from `vercel.json`).
   - **Root Directory**: `./` (leave default).
5. Expand **"Environment Variables"**:
   - Name: `NVIDIA_API_KEY`
   - Value: `nvapi-your-nvidia-nim-api-key`
6. Click **"Deploy"**.
7. Vercel will bundle the FastAPI application and deploy it globally on serverless edge functions.

### Method B: Deploy using Vercel CLI

```bash
# 1. Install Vercel CLI
npm install -g vercel

# 2. Log in and deploy
vercel

# 3. Add NVIDIA_API_KEY when prompted or via CLI
vercel env add NVIDIA_API_KEY
vercel --prod
```

> [!NOTE]
> Vercel functions have a default 10-second timeout on the free (Hobby) tier. Complex CrewAI reasoning runs that execute both SQL + RAG queries may take 10-15s. For production AI agents, long-running web services like Render, Railway, or Docker containers are recommended if you exceed the serverless timeout.

---

## Option 4: Deploy with Docker & Docker Compose (VPS / Self-Hosted)

For deployment on your own Linux server (Ubuntu/Debian on AWS EC2, DigitalOcean Droplet, Linode, Hetzner, etc.):

### 1. Install Docker & Docker Compose

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker
```

### 2. Clone Repository & Setup Environment

```bash
git clone https://github.com/<your-username>/CropCopilot.git
cd CropCopilot

# Create your .env file
cp .env.example .env
nano .env   # Paste your NVIDIA_API_KEY
```

### 3. Run with Docker Compose

```bash
docker compose up -d --build
```

### 4. Check Logs & Status

```bash
docker compose logs -f
```

The application will be live at `http://<your-server-ip>:8000`.

### 5. (Optional) Run with Pure Docker CLI

```bash
# Build image
docker build -t cropcopilot:latest .

# Run container with mounted data volume
docker run -d \
  --name cropcopilot \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  cropcopilot:latest
```

---

## Option 4: Deploy on Fly.io (Persistent Disk Edge)

Fly.io runs Docker containers near your users and supports persistent volumes.

### 1. Install Fly CLI & Authenticate

```bash
# Windows (PowerShell)
pwsh -c "iwr https://fly.io/install.ps1 -useb | iex"

# macOS / Linux
curl -L https://fly.io/install.sh | sh

# Login
fly auth login
```

### 2. Launch Application

```bash
fly launch --no-deploy
```

### 3. Set API Secret and Deploy

```bash
fly secrets set NVIDIA_API_KEY=nvapi-your-actual-key-here
fly deploy
```

Fly.io will build the container and deploy the app to your `*.fly.dev` domain.

---

## Option 5: Deploy on Hugging Face Spaces

Hugging Face Spaces is great for hosting AI applications:

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Give your space a name (e.g., `CropCopilot`).
3. Select **Space SDK**: `Docker` → `Blank`.
4. Clone the space repo or connect your GitHub repo.
5. In your Space's **Settings** → **Variables and secrets**:
   - Add a new **Secret**: `NVIDIA_API_KEY` = your API key.
6. Push this codebase to the Hugging Face Space repository.
7. Hugging Face will build the Docker container and serve the web UI automatically.

---

## Option 6: Google Cloud Run / AWS App Runner

### Google Cloud Run (Serverless Container)

```bash
# 1. Build & Push to Google Container Registry / Artifact Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/cropcopilot

# 2. Deploy to Cloud Run
gcloud run deploy cropcopilot \
  --image gcr.io/PROJECT_ID/cropcopilot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars NVIDIA_API_KEY="your-key-here" \
  --memory 2Gi \
  --port 8000
```

### AWS App Runner

1. Push your Docker image to **AWS ECR** (Elastic Container Registry).
2. Go to **AWS App Runner** → **Create service**.
3. Select **Container registry** → your ECR image.
4. Set port to `8000` and add `NVIDIA_API_KEY` in environment variables.
5. Click **Deploy**.

---

## ⚙️ Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `NVIDIA_API_KEY` | **Yes** | — | NVIDIA NIM API key for LLM and embeddings (`meta/llama-3.1-70b-instruct` & `nvidia/nv-embedqa-e5-v5`). |
| `PORT` | No | `8000` | Port for the Uvicorn ASGI web server. Automatically populated by Render/Railway/Fly.io. |
| `HOST` | No | `0.0.0.0` | Host IP binding. Set to `0.0.0.0` in all cloud environments. |

---

## 🩺 Health Checks & Verification

CropCopilot exposes a dedicated health check endpoint:

```http
GET /health
```

**Sample Response:**
```json
{
  "status": "healthy",
  "service": "CropCopilot",
  "environment": {
    "sqlite_database_ready": true,
    "rag_documents_ready": true,
    "nvidia_api_key_configured": true
  }
}
```

You can test your deployed application using curl:

```bash
# Verify Health
curl -s https://your-deployed-domain.com/health

# Test Query API
curl -X POST https://your-deployed-domain.com/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which crops grow best in high rainfall?"}'
```

---

## 🛠 Troubleshooting & FAQ

### 1. Error: `NVIDIA_API_KEY is not set`
- **Cause**: The application cannot find `NVIDIA_API_KEY` in the environment.
- **Solution**: Check your hosting provider's Environment Variables / Secrets dashboard and ensure `NVIDIA_API_KEY` is defined without quotes or trailing spaces.

### 2. First Query Takes 20-30 Seconds
- **Cause**: On initial launch, ChromaDB creates the embedding collection and indexes the agricultural dataset into vectors.
- **Solution**: Subsequent queries will be fast as the collection is cached in memory and disk.

### 3. Container Out of Memory (OOM)
- **Cause**: CrewAI + ChromaDB embedding models require at least 1 GB RAM.
- **Solution**: Choose a cloud plan with minimum **1 GB RAM** (2 GB recommended).

### 4. Database Missing Errors
- **Cause**: The SQLite database was not found in `data/agriculture.db`.
- **Solution**: `main.py` will automatically run `data_ingestion.py` on startup if the database file is missing. Ensure the container has write permissions to `./data`.

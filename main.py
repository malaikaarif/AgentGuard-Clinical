"""
AgentGuard-Clinical — FastAPI dashboard.

Upload a brain MRI image, run it through the full 6-node pipeline,
and see diagnosis, reasoning, Grad-CAM heatmap, audit verdict, and
escalation status rendered on one page.
"""

import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from graph import build_graph

app = FastAPI(title="AgentGuard-Clinical")

UPLOAD_DIR = "uploaded_images"
HEATMAP_DIR = "explainability_output"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB, same cap as MedTrust-Audit
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

app.mount("/uploaded_images", StaticFiles(directory=UPLOAD_DIR), name="uploaded_images")
app.mount("/explainability_output", StaticFiles(directory=HEATMAP_DIR), name="explainability_output")

_graph_app = None


def _get_graph():
    """Lazy-build the graph once, reused across requests."""
    global _graph_app
    if _graph_app is None:
        _graph_app = build_graph()
    return _graph_app


def _safe_save_upload(upload_bytes: bytes, original_filename: str) -> str:
    """
    Saves an uploaded file under a random UUID name, never trusting the
    client-provided filename for the actual path — avoids path traversal
    and filename collisions between concurrent uploads.
    """
    ext = os.path.splitext(original_filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    safe_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(save_path, "wb") as f:
        f.write(upload_bytes)

    return save_path


def run_pipeline(image_path: str) -> dict:
    app_graph = _get_graph()

    initial_state = {
        "image_path": image_path,
        "diagnosis": None,
        "confidence": None,
        "logits": None,
        "class_names": None,
        "reasoning_text": None,
        "heatmap_path": None,
        "heatmap_array": None,
        "region_label": None,
        "audit_verdict": None,
        "audit_explanation": None,
        "needs_human_review": None,
        "review_reason": None,
        "error": None,
    }

    final_state = app_graph.invoke(initial_state)
    # heatmap_array is a raw numpy array — not JSON-serializable, drop it
    return {k: v for k, v in final_state.items() if k != "heatmap_array"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    upload_bytes = await file.read()

    if len(upload_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(upload_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds the 10MB size limit.")

    save_path = _safe_save_upload(upload_bytes, file.filename)

    try:
        result = run_pipeline(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    # Normalize paths for the browser to load as URLs
    result["uploaded_image_url"] = "/" + save_path.replace("\\", "/")
    if result.get("heatmap_path"):
        result["heatmap_url"] = "/" + result["heatmap_path"].replace("\\", "/")

    return result


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AgentGuard-Clinical</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            :root { --ink:#1a1a1a; --muted:#6b7280; --border:#e5e7eb; --bg:#fafafa; }
            * { box-sizing: border-box; }
            body {
                font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
                background: var(--bg);
                color: var(--ink);
                max-width: 760px;
                margin: 48px auto;
                padding: 0 24px;
                line-height: 1.5;
            }
            h1 { font-size: 28px; margin-bottom: 4px; }
            .subtitle { color: var(--muted); margin-top: 0; margin-bottom: 28px; }
            .upload-box {
                border: 2px dashed var(--border);
                border-radius: 12px;
                padding: 32px;
                text-align: center;
                margin-bottom: 24px;
                background: white;
            }
            button {
                background: #2563eb; color: white; border: none;
                padding: 10px 20px; border-radius: 8px; font-size: 14px;
                cursor: pointer; margin-top: 12px;
            }
            button:disabled { background: #9ca3af; cursor: not-allowed; }
            .pillar {
                background: white; border: 1px solid var(--border);
                border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;
            }
            .pillar h2 { font-size: 15px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin: 0 0 14px 0; }
            .metric-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f3f4f6; }
            .metric-row:last-child { border-bottom: none; }
            .metric-label { color: var(--muted); }
            .metric-value { font-weight: 600; }
            .verdict-badge {
                display: inline-block; padding: 4px 12px; border-radius: 999px;
                font-size: 13px; font-weight: 700; letter-spacing: 0.5px;
            }
            .verdict-consistent { background: #dcfce7; color: #16a34a; }
            .verdict-inconsistent { background: #fee2e2; color: #dc2626; }
            .verdict-uncertain { background: #fef9c3; color: #ca8a04; }
            .review-flag {
                background: #fef2f2; border: 1px solid #fecaca; color: #991b1b;
                padding: 12px 16px; border-radius: 8px; font-weight: 600; margin-bottom: 16px;
            }
            .no-review-flag {
                background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534;
                padding: 12px 16px; border-radius: 8px; font-weight: 600; margin-bottom: 16px;
            }
            img.result-img { max-width: 100%; border-radius: 8px; margin-top: 8px; }
            .error-box {
                background: #fef2f2; border: 1px solid #fecaca; color: #991b1b;
                padding: 16px; border-radius: 8px;
            }
            #loading { color: var(--muted); margin-top: 12px; display: none; }
        </style>
    </head>
    <body>
        <h1>AgentGuard-Clinical</h1>
        <p class="subtitle">Multi-agent clinical diagnosis pipeline with reasoning/heatmap consistency audit</p>

        <div class="upload-box">
            <input type="file" id="fileInput" accept=".jpg,.jpeg,.png" />
            <br>
            <button id="analyzeBtn" onclick="analyze()">Analyze</button>
            <div id="loading">Running full pipeline (classifier, reasoning, Grad-CAM, audit)... this can take 10-20s.</div>
        </div>

        <div id="results"></div>

        <script>
        async function analyze() {
            const fileInput = document.getElementById('fileInput');
            const btn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');

            if (!fileInput.files.length) {
                alert('Please choose an image first.');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            btn.disabled = true;
            loading.style.display = 'block';
            results.innerHTML = '';

            try {
                const resp = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await resp.json();

                if (!resp.ok) {
                    results.innerHTML = `<div class="error-box"><strong>Error:</strong> ${data.detail}</div>`;
                    return;
                }

                const verdictClass = data.audit_verdict === 'consistent' ? 'verdict-consistent'
                    : data.audit_verdict === 'inconsistent' ? 'verdict-inconsistent'
                    : 'verdict-uncertain';

                const reviewHtml = data.needs_human_review
                    ? `<div class="review-flag">⚠ FLAGGED FOR HUMAN REVIEW — ${data.review_reason}</div>`
                    : `<div class="no-review-flag">✓ No review needed — passed all checks</div>`;

                results.innerHTML = `
                    ${reviewHtml}
                    <div class="pillar">
                        <h2>Classification</h2>
                        <div class="metric-row"><span class="metric-label">Diagnosis</span><span class="metric-value">${data.diagnosis}</span></div>
                        <div class="metric-row"><span class="metric-label">Confidence</span><span class="metric-value">${(data.confidence*100).toFixed(2)}%</span></div>
                        <img class="result-img" src="${data.uploaded_image_url}" alt="Uploaded scan" style="max-width:300px;">
                    </div>
                    <div class="pillar">
                        <h2>Reasoning Agent</h2>
                        <p>${data.reasoning_text}</p>
                    </div>
                    <div class="pillar">
                        <h2>Explainability (Grad-CAM)</h2>
                        <img class="result-img" src="${data.heatmap_url}" alt="Grad-CAM heatmap" style="max-width:300px;">
                    </div>
                    <div class="pillar">
                        <h2>Audit — Reasoning/Heatmap Consistency</h2>
                        <div class="metric-row">
                            <span class="metric-label">Verdict</span>
                            <span class="verdict-badge ${verdictClass}">${data.audit_verdict.toUpperCase()}</span>
                        </div>
                        <p style="margin-top:12px;">${data.audit_explanation}</p>
                    </div>
                `;
            } catch (err) {
                results.innerHTML = `<div class="error-box"><strong>Request failed:</strong> ${err.message}</div>`;
            } finally {
                btn.disabled = false;
                loading.style.display = 'none';
            }
        }
        </script>
    </body>
    </html>
    """
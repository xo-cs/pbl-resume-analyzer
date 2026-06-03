from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from openai import AzureOpenAI, APIError
import os, uuid, json, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resume_analyzer")

# ── Azure Monitor / App Insights ─────────────────────────────────────────────
# Activates automatically when APPLICATIONINSIGHTS_CONNECTION_STRING is set.
if os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor()
        logger.info("Azure Monitor configured — telemetry active.")
    except ImportError:
        logger.warning("azure-monitor-opentelemetry not installed; skipping App Insights.")

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB hard limit
MIN_TEXT_CHARS      = 50                  # Minimum extracted text for analysis

ANALYSIS_PROMPT = """You are an expert resume coach and ATS (Applicant Tracking System) specialist.
Analyze the resume text below and return a JSON object with EXACTLY these fields:

- professional_summary (string): 2–3 sentence summary of the candidate's background and strengths
- skill_gaps (list of strings): 5–8 skills the candidate is missing for modern roles in their field
- ats_score (integer 0–100): ATS compatibility score based on formatting, keywords, and structure
- role_recommendations (list of strings): 4–6 specific job titles that best match this candidate
- improvement_suggestions (list of strings): 5–7 specific, actionable improvements

Be concrete and specific — generic advice is not helpful.
Return ONLY valid JSON. No markdown fences, no preamble, no trailing text.

Resume text:
{text}
"""


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Resume Analyzer", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.1.0", "message": "Resume Analyzer API is running"}


# ── Helper: initialize Azure clients ─────────────────────────────────────────
def _init_clients():
    """Build all Azure SDK clients. Raises 500 HTTPException on missing config."""
    missing = [
        k for k in [
            "AZURE_STORAGE_CONNECTION_STRING",
            "DOC_INTEL_ENDPOINT", "DOC_INTEL_KEY",
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_DEPLOYMENT",
        ]
        if not os.environ.get(k)
    ]
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        raise HTTPException(status_code=500, detail="Server configuration error. Contact the administrator.")

    try:
        blob_service = BlobServiceClient.from_connection_string(
            os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        )
    except Exception as e:
        logger.error(f"BlobServiceClient init failed: {e}")
        raise HTTPException(status_code=500, detail="Storage service could not be initialized.")

    doc_client = DocumentIntelligenceClient(
        endpoint=os.environ["DOC_INTEL_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["DOC_INTEL_KEY"])
    )

    openai_client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version="2024-02-01"
    )

    return blob_service, doc_client, openai_client


# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...)):
    # ── Step 1: Validate file ─────────────────────────────────────────────────
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file."
        )

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE_BYTES:
        mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File is too large ({mb:.1f} MB). Maximum allowed size is 10 MB."
        )

    # Magic bytes check — PDFs always start with %PDF
    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not appear to be a valid PDF."
        )

    blob_service, doc_client, openai_client = _init_clients()

    # ── Step 2: Upload to Azure Blob Storage ──────────────────────────────────
    blob_name = f"{uuid.uuid4()}-{file.filename}"
    try:
        container = blob_service.get_container_client("resumes")
        container.upload_blob(name=blob_name, data=content, overwrite=True)
        logger.info(f"Blob uploaded: {blob_name} ({len(content)} bytes)")
    except AzureError as e:
        logger.error(f"Blob upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to store the uploaded file. Please try again in a moment."
        )

    # ── Step 3: OCR via Azure AI Document Intelligence ────────────────────────
    try:
        poller = doc_client.begin_analyze_document(
            "prebuilt-read",
            AnalyzeDocumentRequest(bytes_source=content)
        )
        result = poller.result()
    except AzureError as e:
        logger.error(f"Document Intelligence error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Text extraction failed. The PDF may be corrupted, password-protected, or contain only images with no readable text."
        )

    extracted_text = " ".join(
        line.content
        for page in result.pages
        for line in page.lines
    ).strip()

    if len(extracted_text) < MIN_TEXT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Only {len(extracted_text)} characters were extracted — not enough to analyze. "
                "Please upload a text-based PDF (not a scanned image) with at least a few paragraphs of content."
            )
        )

    logger.info(f"OCR complete: {len(extracted_text)} chars from {blob_name}")

    # ── Step 4: Azure OpenAI analysis ─────────────────────────────────────────
    try:
        response = openai_client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages=[{
                "role": "user",
                "content": ANALYSIS_PROMPT.format(text=extracted_text[:4000])
            }],
            temperature=0.3,
            response_format={"type": "json_object"}   # Forces JSON mode
        )
        raw_output = response.choices[0].message.content
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(
            status_code=502,
            detail="The AI analysis service is temporarily unavailable. Please try again in a moment."
        )

    # ── Step 5: Parse and validate JSON response ──────────────────────────────
    try:
        feedback = json.loads(raw_output)
    except json.JSONDecodeError:
        # Strip markdown fences if the model added them despite instructions
        cleaned = raw_output.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            feedback = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(f"Unparseable OpenAI response (first 300 chars): {raw_output[:300]}")
            raise HTTPException(
                status_code=500,
                detail="The AI returned an unexpected response format. Please try again."
            )

    # ── Step 6: Ensure all required fields exist with safe defaults ───────────
    DEFAULTS = {
        "professional_summary":   "Summary not available.",
        "skill_gaps":             [],
        "ats_score":              0,
        "role_recommendations":   [],
        "improvement_suggestions": [],
    }
    for key, default in DEFAULTS.items():
        feedback.setdefault(key, default)

    # Clamp ATS score to [0, 100]
    try:
        feedback["ats_score"] = max(0, min(100, int(feedback["ats_score"])))
    except (ValueError, TypeError):
        feedback["ats_score"] = 0

    logger.info(f"Analysis complete — blob: {blob_name}, ATS: {feedback['ats_score']}")
    return {"blob_name": blob_name, "feedback": feedback}
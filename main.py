from fastapi import FastAPI, UploadFile, File, HTTPException, Request
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
import httpx
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resume_analyzer")

if os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor()
        logger.info("Azure Monitor configured.")
    except ImportError:
        logger.warning("azure-monitor-opentelemetry not installed.")

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
MIN_TEXT_CHARS = 50

CV_CHECK_PROMPT = """Does the following text appear to be a résumé or CV — a document listing someone's work experience, education, skills, and contact details?

Answer with exactly ONE word: RESUME or NOT_RESUME

Text:
{text}"""

ANALYSIS_PROMPT = """You are an expert resume coach and ATS specialist.
Analyze the resume text below and return a JSON object with EXACTLY these fields:

- professional_summary (string): 2-3 sentence summary of the candidate's background and strengths
- skill_gaps (list of strings): 5-8 skills the candidate is missing for modern roles in their field
- ats_score (integer 0-100): ATS compatibility score based on formatting, keywords, and structure
- role_recommendations (list of strings): 4-6 specific job titles that best match this candidate
- improvement_suggestions (list of strings): 5-7 specific, actionable improvements

Be concrete and specific. Return ONLY valid JSON. No markdown, no preamble.

Resume text:
{text}"""

COVER_LETTER_PROMPT = """Write a compelling, genuine cover letter for this application:

Position: {job_title}
Company: {company}

Candidate profile (from resume analysis):
{resume_summary}

Rules:
- 3 tight paragraphs: strong opener / value proposition / confident close
- Under 280 words total
- Do NOT start with "I am writing to" or "I am excited to apply"
- Be specific to the role and company
- Sound like a real person, not a template

Return only the cover letter text. No subject line, no labels."""

app = FastAPI(title="AI Resume Analyzer", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


def _init_clients():
    missing = [k for k in [
        "AZURE_STORAGE_CONNECTION_STRING", "DOC_INTEL_ENDPOINT", "DOC_INTEL_KEY",
        "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_DEPLOYMENT",
    ] if not os.environ.get(k)]
    if missing:
        logger.error(f"Missing env vars: {missing}")
        raise HTTPException(status_code=500, detail="Server configuration error.")

    blob_service = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
    doc_client   = DocumentIntelligenceClient(
        endpoint=os.environ["DOC_INTEL_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["DOC_INTEL_KEY"])
    )
    openai_client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version="2024-02-01"
    )
    return blob_service, doc_client, openai_client


@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...)):
    # ── Validate ────────────────────────────────────────────────────────────
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large ({len(content)/1024/1024:.1f} MB). Max 10 MB.")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File does not appear to be a valid PDF.")

    blob_service, doc_client, openai_client = _init_clients()

    # ── Blob upload ──────────────────────────────────────────────────────────
    blob_name = f"{uuid.uuid4()}-{file.filename}"
    try:
        blob_service.get_container_client("resumes").upload_blob(name=blob_name, data=content, overwrite=True)
        logger.info(f"Blob uploaded: {blob_name}")
    except AzureError as e:
        logger.error(f"Blob upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to store the uploaded file. Please try again.")

    # ── OCR ──────────────────────────────────────────────────────────────────
    try:
        poller = doc_client.begin_analyze_document("prebuilt-read", AnalyzeDocumentRequest(bytes_source=content))
        result = poller.result()
    except AzureError as e:
        logger.error(f"Document Intelligence error: {e}")
        raise HTTPException(status_code=500, detail="Text extraction failed. PDF may be corrupted or password-protected.")

    extracted_text = " ".join(line.content for page in result.pages for line in page.lines).strip()

    if len(extracted_text) < MIN_TEXT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(extracted_text)} characters extracted. Please upload a text-based PDF, not a scanned image."
        )

    # ── CV Validation ─────────────────────────────────────────────────────────
    try:
        cv_check = openai_client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages=[{"role": "user", "content": CV_CHECK_PROMPT.format(text=extracted_text[:1500])}],
            temperature=0,
            max_tokens=10
        )
        verdict = cv_check.choices[0].message.content.strip().upper()
        logger.info(f"CV validation verdict: {verdict}")
        if "NOT_RESUME" in verdict:
            raise HTTPException(
                status_code=422,
                detail="This document doesn't look like a résumé or CV. Please upload your actual résumé PDF."
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"CV validation failed (non-blocking): {e}")

    # ── OpenAI Analysis ───────────────────────────────────────────────────────
    try:
        response = openai_client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages=[{"role": "user", "content": ANALYSIS_PROMPT.format(text=extracted_text[:4000])}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        raw = response.choices[0].message.content
    except APIError as e:
        logger.error(f"OpenAI error: {e}")
        raise HTTPException(status_code=502, detail="AI analysis temporarily unavailable. Please try again.")

    # ── Parse ────────────────────────────────────────────────────────────────
    try:
        feedback = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            feedback = json.loads(cleaned)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="AI returned unexpected format. Please try again.")

    for k, v in {
        "professional_summary": "Summary not available.",
        "skill_gaps": [], "ats_score": 0,
        "role_recommendations": [], "improvement_suggestions": [],
    }.items():
        feedback.setdefault(k, v)

    try:
        feedback["ats_score"] = max(0, min(100, int(feedback["ats_score"])))
    except (ValueError, TypeError):
        feedback["ats_score"] = 0

    logger.info(f"Analysis complete — ATS: {feedback['ats_score']}")
    return {"blob_name": blob_name, "feedback": feedback}


MOCK_JOBS = [
    {"id":"m1","title":"AI / ML Engineer",          "company":"Samsung SDS",  "location":"Seoul, Korea",    "job_type":"Full-time","salary":"Competitive","url":"https://www.samsungsds.com/en/careers/"},
    {"id":"m2","title":"Machine Learning Engineer",  "company":"Kakao",        "location":"Pangyo, Korea",   "job_type":"Full-time","salary":"Competitive","url":"https://careers.kakao.com/"},
    {"id":"m3","title":"Data Scientist",             "company":"Naver",        "location":"Seongnam, Korea", "job_type":"Full-time","salary":"Competitive","url":"https://recruit.navercorp.com/"},
    {"id":"m4","title":"Cloud & AI Engineer",        "company":"LG CNS",       "location":"Seoul, Korea",    "job_type":"Full-time","salary":"Competitive","url":"https://www.lgcns.com/en/careers/"},
    {"id":"m5","title":"Backend Software Engineer",  "company":"SK Telecom",   "location":"Seoul, Korea",    "job_type":"Full-time","salary":"Competitive","url":"https://careers.sktelecom.com/"},
]

@app.get("/jobs")
async def search_jobs(role: str, count: int = 5):
    linkedin_url = (
        f"https://www.linkedin.com/jobs/search/?"
        f"keywords={urllib.parse.quote(role)}&location=South%20Korea"
    )
    api_key = os.environ.get("JSEARCH_API_KEY")

    # ── No key → serve mock jobs (demo-safe) ──────────────────────────────
    if not api_key:
        return {"jobs": MOCK_JOBS[:count], "linkedin_url": linkedin_url,
                "keyword": role, "is_mock": True}

    # ── JSearch (RapidAPI) ─────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://jsearch.p.rapidapi.com/search",
                params={"query": f"{role} in South Korea", "num_pages": "1", "page": "1"},
                headers={"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"},
                timeout=10.0
            )
        if resp.status_code != 200:
            logger.warning(f"JSearch {resp.status_code} — falling back to mock")
            return {"jobs": MOCK_JOBS[:count], "linkedin_url": linkedin_url,
                    "keyword": role, "is_mock": True}

        data = resp.json().get("data", [])
        jobs = [{
            "id":       j.get("job_id", ""),
            "title":    j.get("job_title", ""),
            "company":  j.get("employer_name", ""),
            "location": ", ".join(filter(None, [j.get("job_city",""), j.get("job_country","")])),
            "job_type": j.get("job_employment_type", ""),
            "salary":   "",
            "url":      j.get("job_apply_link", ""),
            "closing_date": "",
        } for j in data[:count]]

        if not jobs:
            return {"jobs": MOCK_JOBS[:count], "linkedin_url": linkedin_url,
                    "keyword": role, "is_mock": True}

        return {"jobs": jobs, "linkedin_url": linkedin_url, "keyword": role, "is_mock": False}

    except Exception as e:
        logger.error(f"JSearch error: {e} — falling back to mock")
        return {"jobs": MOCK_JOBS[:count], "linkedin_url": linkedin_url,
                "keyword": role, "is_mock": True}


@app.post("/cover-letter")
async def generate_cover_letter(req: Request):
    body         = await req.json()
    job_title    = body.get("job_title", "the role")
    company      = body.get("company", "the company")
    resume_summary = body.get("resume_summary", "")

    _, _, openai_client = _init_clients()
    try:
        response = openai_client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages=[{"role": "user", "content": COVER_LETTER_PROMPT.format(
                job_title=job_title, company=company, resume_summary=resume_summary
            )}],
            temperature=0.75,
            max_tokens=500
        )
        return {"cover_letter": response.choices[0].message.content.strip()}
    except Exception as e:
        logger.error(f"Cover letter error: {e}")
        raise HTTPException(status_code=500, detail="Cover letter generation failed. Please try again.")
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import os, uuid, json

app = FastAPI()

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
    return {"status": "ok", "message": "Resume Analyzer API is running"}

@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...)):
    blob_service = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
    doc_client = DocumentIntelligenceClient(
        endpoint=os.environ["DOC_INTEL_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["DOC_INTEL_KEY"])
    )
    openai_client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version="2024-02-01"
    )

    # 1. Read file content
    content = await file.read()

    # 2. Upload to Blob Storage
    blob_name = f"{uuid.uuid4()}-{file.filename}"
    container = blob_service.get_container_client("resumes")
    container.upload_blob(name=blob_name, data=content, overwrite=True)

    # 3. OCR with Document Intelligence
    poller = doc_client.begin_analyze_document(
        "prebuilt-read",
        AnalyzeDocumentRequest(bytes_source=content)
    )
    result = poller.result()
    extracted_text = " ".join([
        line.content
        for page in result.pages
        for line in page.lines
    ])

    # 4. Azure OpenAI structured feedback
    prompt = f"""
You are an expert resume coach. Analyze the resume text below and return a JSON object with exactly these fields:
- professional_summary (string)
- skill_gaps (list of strings)
- ats_score (integer 0-100)
- role_recommendations (list of strings)
- improvement_suggestions (list of strings)

Resume text:
{extracted_text[:4000]}

Return only valid JSON, no extra text, no markdown.
"""
    response = openai_client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    feedback = json.loads(response.choices[0].message.content)
    return {"blob_name": blob_name, "feedback": feedback}
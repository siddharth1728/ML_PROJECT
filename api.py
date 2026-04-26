"""
api.py - FastAPI REST API Wrapper

Provides a RESTful API for the ATS Resume Parser, allowing users to
upload resume files via HTTP and receive parsed JSON responses.

Endpoints:
    GET  /        -> Welcome page
    GET  /health  -> Health check
    POST /parse   -> Upload and parse a resume file
    POST /match   -> Compare resume against a job description
    POST /score   -> Get ATS score only

Start the server:
    uvicorn api:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

from parser import ResumeParser
from skill_extractor import SEMANTIC_AVAILABLE

# ─────────────────────────────────────────────────────────────
# APP INITIALIZATION
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="ATS Resume Parser API",
    description=(
        "An intelligent resume parsing engine that extracts structured data "
        "from PDF/DOCX resumes using NLP (spaCy NER), Regex, and "
        "Sentence Transformer-based skill matching."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize the parser once at startup
parser = ResumeParser(use_semantic=True)

# ─────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    semantic_matching: bool

class ParseResponse(BaseModel):
    metadata: Dict[str, Any]
    candidate: Dict[str, Any]
    summary: Optional[str] = None
    experience: List[Dict[str, Any]] = []
    education: List[Dict[str, Any]] = []
    skills: Dict[str, List[str]]
    projects: Optional[str] = None
    certifications: Optional[str] = None
    organizations: List[str] = []
    ats_score: int

class ScoreResponse(BaseModel):
    file_name: str
    ats_score: int
    skills_count: int
    experience_count: int
    education_count: int

class MatchResponse(BaseModel):
    file_name: str
    match_percentage: float
    matched_skills: List[str]
    missing_skills: List[str]
    resume_skills: List[str]

# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
async def root():
    """Welcome page with API information."""
    return {
        "message": "Welcome to the ATS Resume Parser API",
        "version": "2.0.0",
        "endpoints": {
            "GET /": "This welcome page",
            "GET /health": "Health check",
            "GET /docs": "Interactive Swagger documentation",
            "POST /parse": "Upload and parse a resume (PDF/DOCX)",
            "POST /match": "Match resume against job description",
            "POST /score": "Get ATS score for a resume"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        semantic_matching=SEMANTIC_AVAILABLE and getattr(parser, 'semantic_matcher', None) is not None
    )

@app.post("/parse", response_model=ParseResponse, tags=["Parser"])
async def parse_resume(file: UploadFile = File(...)):
    """
    Upload and parse a resume file.

    Accepts PDF or DOCX files and returns structured JSON data including:
    - Candidate name and contact info
    - Work experience entries
    - Education entries
    - Skills (technical + soft)
    - Projects, certifications, and more
    - ATS compatibility score (0-100)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.lower().split('.')[-1]
    if ext not in ('pdf', 'docx', 'doc'):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: .{ext}. Supported: .pdf, .docx"
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Parse the resume
    result = parser.parse_file(
        file_bytes=file_bytes,
        filename=file.filename
    )

    if "error" in result.get("metadata", {}):
        raise HTTPException(
            status_code=422,
            detail=result["metadata"]["error"]
        )

    return result

@app.post("/score", response_model=ScoreResponse, tags=["Parser"])
async def get_ats_score(file: UploadFile = File(...)):
    """
    Get only the ATS compatibility score for a resume.
    Returns a simplified response with the score and basic counts.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.lower().split('.')[-1]
    if ext not in ('pdf', 'docx', 'doc'):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: .{ext}. Supported: .pdf, .docx"
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    result = parser.parse_file(file_bytes=file_bytes, filename=file.filename)

    if "error" in result.get("metadata", {}):
        raise HTTPException(status_code=422, detail=result["metadata"]["error"])

    return ScoreResponse(
        file_name=file.filename,
        ats_score=result.get("ats_score", 0),
        skills_count=len(result.get("skills", {}).get("all", [])),
        experience_count=len(result.get("experience", [])),
        education_count=len(result.get("education", []))
    )

@app.post("/match", response_model=MatchResponse, tags=["Matching"])
async def match_job_description(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    """
    Match a resume against a job description.

    Extracts skills from both the resume and job description,
    then calculates a match percentage based on skill overlap.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.lower().split('.')[-1]
    if ext not in ('pdf', 'docx', 'doc'):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: .{ext}. Supported: .pdf, .docx"
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description is empty")

    result = parser.parse_file(file_bytes=file_bytes, filename=file.filename)

    if "error" in result.get("metadata", {}):
        raise HTTPException(status_code=422, detail=result["metadata"]["error"])

    # Extract skills from job description
    from skill_extractor import extract_skills_keyword
    jd_skills = set(extract_skills_keyword(job_description, getattr(parser, 'skills_db', [])))

    # Get resume skills
    resume_skills = set(result.get("skills", {}).get("all", []))

    # Calculate match
    if jd_skills:
        matched = resume_skills.intersection(jd_skills)
        missing = jd_skills - resume_skills
        match_pct = round((len(matched) / len(jd_skills)) * 100, 1)
    else:
        matched = set()
        missing = set()
        match_pct = 0.0

    return MatchResponse(
        file_name=file.filename,
        match_percentage=match_pct,
        matched_skills=sorted(list(matched)),
        missing_skills=sorted(list(missing)),
        resume_skills=sorted(list(resume_skills))
    )

# ─────────────────────────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("Starting ATS Resume Parser API...")
    print("Docs available at: http://localhost:8000/docs")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
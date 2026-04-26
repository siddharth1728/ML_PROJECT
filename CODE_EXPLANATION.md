# ATS Resume Parser — Code Explanation & Walkthrough

## Overview

This document provides a detailed walkthrough of the ATS Resume Parser codebase,
explaining the logic, NLP/ML concepts, and design decisions behind each module.

---

## Module Architecture

```
Input (PDF/DOCX)
    │
    ▼
text_extractor.py ──── pdfminer.six / python-docx
    │                   + cleaning pipeline
    ▼
section_segmenter.py ── keyword heuristics
    │                    + structural analysis
    ▼
┌──────────────────────────────────────┐
│         Parallel Extraction          │
├──────────────────────────────────────┤
│ contact_extractor.py  (Regex)        │
│ entity_extractor.py   (spaCy NER)    │
│ skill_extractor.py    (KB + Semantic)│
└──────────────────────────────────────┘
    │
    ▼
parser.py ──── Orchestrator + ATS Scoring
    │
    ▼
Output (Structured JSON)
    │
    ├──▶ api.py (FastAPI REST)
    └──▶ CLI (parser.py __main__)
```

---

## 1. Text Extraction (`text_extractor.py`)

### PDF Extraction
- Uses `pdfminer.six` library with `extract_text()` function
- LAParams tuning: `line_margin=0.5`, `word_margin=0.1` optimized for dense resume layouts
- Handles both file paths and byte streams (for web uploads)

### DOCX Extraction
- Uses `python-docx` library to iterate through `Document.paragraphs` and `Document.tables`
- Tables are joined with ` | ` separators (many resumes use tables for layout)

### Cleaning Pipeline
The cleaning pipeline runs 6 sequential steps:

1. **CID Removal**: `(cid:123)` → removed (PDF encoding artifacts)
2. **Unicode Normalization**: NFKD decomposition (handles ligatures, accented chars)
3. **Control Char Removal**: Strip non-printable chars except `\n` and `\t`
4. **Bullet Standardization**: Convert `-`, `*`, `>`, `~` line starters to `  * `
5. **Whitespace Normalization**: Collapse multiple spaces/newlines
6. **Line Stripping**: Remove leading/trailing spaces from each line

---

## 2. Section Segmentation (`section_segmenter.py`)

### How It Works
The segmenter uses a **dictionary of regex patterns** mapped to section types:

```python
SECTION_PATTERNS = {
    "experience": [r"experience", r"work\s+experience", r"employment\s+history", ...],
    "education": [r"education", r"academic\s+background", ...],
    "skills": [r"skills", r"technical\s+skills", r"core\s+competencies", ...],
    ...
}
```

### Header Detection Rules
A line is considered a section header if:
- It matches a known pattern (case-insensitive)
- It's short (< 80 characters)
- It has few words (≤ 6 words)
- After stripping decorators (`===`, `---`, `:`, `#`, etc.)

### Section Assignment
- Text **before** the first detected header → `contact_info`
- Text **between** headers → assigned to the detected section type
- If a section type appears twice, content is concatenated

---

## 3. Contact Extraction (`contact_extractor.py`)

### Email Regex
```
[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
```
- Matches standard email formats
- Results are lowercased and deduplicated

### Phone Regex
```
(?:\+?\d{1,3}[\s\-.]?)?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}
```
Handles:
- `+1-555-123-4567` (US international)
- `(555) 123-4567` (US domestic)
- `+91 98765 43210` (India)
- `555.123.4567` (dot-separated)
- Extensions: `ext. 1234`

### Phone Normalization
All phone numbers are normalized to international format:
- 10 digits → `+1-XXX-XXX-XXXX` (assumed US)
- 11 digits starting with 1 → `+1-XXX-XXX-XXXX`
- 12 digits starting with 91 → `+91-XXXXX-XXXXX` (India)

### URL Extraction
- **LinkedIn**: `linkedin.com/in/username` pattern
- **GitHub**: `github.com/username` pattern
- **General URLs**: Any `https://...` not matching LinkedIn/GitHub

### Location Detection
Uses pattern matching for:
- US format: `City, ST 12345`
- General: `City, State` or `City, Country`
- Only searches first 500 characters (location is at top of resume)

---

## 4. Entity Extraction (`entity_extractor.py`)

### Named Entity Recognition (NER)
Uses spaCy's `en_core_web_sm` model which can detect:
- **PERSON**: Human names
- **ORG**: Companies, universities, organizations
- **DATE**: Dates and time periods
- **GPE**: Geopolitical entities (countries, cities)

### Name Extraction Strategy
1. Only process the first 500 characters (name is at the top)
2. Find all PERSON entities
3. Filter to 2-4 word names (skip single words, skip long phrases)
4. Exclude false positives (university names, company names)
5. Fallback: first short line without `@` or digits

### Experience Parsing
For each experience entry, the parser:
1. Detects date ranges using regex
2. Splits company/title using separators (`—`, `|`, `-`)
3. Uses role keywords to classify lines as job titles
4. Remaining lines become the description
5. Falls back to NER for missing company names

### Education Parsing
Two-pass approach:
1. **First pass**: Detect degrees (B.S., M.S., Ph.D., etc.), dates, and GPA
2. **Second pass**: Match institution names via keywords (`university`, `college`, `institute`)
3. **NER fallback**: Use spaCy ORG entities for undetected institutions

---

## 5. Skill Extraction (`skill_extractor.py`)

### Dual-Mode Approach

#### Mode 1: Keyword Matching
- Word-boundary-aware regex matching: `(?<![a-zA-Z0-9])python(?![a-zA-Z0-9])`
- Prevents false matches (e.g., "Java" inside "JavaScript")
- Alias resolution: "ML" → "Machine Learning", "K8s" → "Kubernetes"
- 500+ skills across 11 categories

#### Mode 2: Semantic Matching (Sentence Transformers)
- Model: `all-MiniLM-L6-v2` (~22M params, ~80MB)
- Pre-computes embeddings for all skills at init time
- Splits resume text into sentence chunks
- Computes cosine similarity between chunks and skill embeddings
- Threshold: 0.55 (balances precision/recall)
- Handles: "ML" ≈ "Machine Learning", "cloud computing" ≈ "AWS"

#### Combined Result
Skills from both modes are unioned, deduplicated, and categorized by their
knowledge base category (programming_languages, web_frameworks, databases, etc.)

### Skills Knowledge Base (`skills_knowledge_base.json`)
- **11 categories**: Programming, Frameworks, Databases, Cloud, Data Science,
  DevOps, Tools, Testing, Soft Skills, Certifications, Domains
- **500+ canonical skills**
- **80+ aliases** for variation handling

---

## 6. ATS Scoring (`parser.py`)

### Scoring Formula (0-100 points)

| Component | Weight | Max Points | Criteria |
|-----------|--------|------------|----------|
| Skills | 30% | 30 | 15+ skills = 30, 10+ = 25, 5+ = 18, 1+ = 10 |
| Experience | 25% | 25 | Completeness of each entry (company, title, dates, desc) |
| Education | 20% | 20 | Institution, degree, dates, GPA per entry |
| Contact | 10% | 10 | Name, email, phone, LinkedIn, GitHub |
| Quality | 15% | 15 | Summary length, section structure, key sections present |

---

## 7. FastAPI Backend (`api.py`)

### Endpoints

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/` | GET | - | Welcome + endpoint list |
| `/health` | GET | - | Status, version, semantic flag |
| `/parse` | POST | `file` (PDF/DOCX) | Full parsed JSON |
| `/score` | POST | `file` (PDF/DOCX) | ATS score + counts |
| `/match` | POST | `file` + `job_description` | Match %, matched/missing skills |

### Features
- CORS middleware enabled for cross-origin access
- Pydantic response models for type safety
- Auto-generated Swagger docs at `/docs`
- File validation (format, empty file checks)

---

## Design Decisions

1. **Modular Architecture**: Each extraction concern is isolated in its own module
   for testability and maintainability.

2. **Graceful Degradation**: If sentence-transformers is not installed, the parser
   falls back to keyword-only matching. If spaCy model is missing, it uses
   heuristic-based name extraction.

3. **Dual Input Modes**: Every function accepts both file paths and byte streams,
   supporting both CLI and web upload use cases.

4. **Small Model First**: Using `en_core_web_sm` (12MB) instead of transformer models
   (500MB+) keeps the project lightweight while still achieving good NER accuracy.

5. **Pre-computed Embeddings**: Skill embeddings are computed once at initialization,
   making subsequent parsing calls fast (~O(n) where n = text chunks).

"""
entity_extractor.py - Regex-Based Entity Extraction (No spaCy)

Extracts structured data from resume text using pure Python + regex.
Replaces spaCy NER to avoid DLL/Rust dependency issues on restricted
Windows environments.

Extracts:
    - Candidate Name
    - Organization / Institution Names
    - Date Ranges
    - Experience entries (Company, Role, Dates, Description)
    - Education entries (Institution, Degree, Dates, GPA)
"""

import re
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# COMPILED REGEX PATTERNS
# ─────────────────────────────────────────────────────────────

# Date range: "Jan 2020 - Present", "2018–2022", "March 2019 to Dec 2021"
DATE_RANGE_PATTERN = re.compile(
    r'(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|'
    r'Dec(?:ember)?)\s+)?'
    r'\d{4}'
    r'\s*[\-\u2013\u2014to]+\s*'
    r'(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|'
    r'Dec(?:ember)?)\s+)?'
    r'(?:\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow|[Oo]ngoing)',
    re.IGNORECASE
)

# GPA: "GPA: 3.8/4.0", "CGPA 8.5"
GPA_PATTERN = re.compile(
    r'(?:GPA|CGPA|Grade|Score)\s*:?\s*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?',
    re.IGNORECASE
)

# Degree patterns
DEGREE_PATTERNS = [
    re.compile(r'(?:Bachelor|B\.?S\.?|B\.?A\.?|B\.?E\.?|B\.?Tech|B\.?Sc)\s*(?:of|in|\.?\s*)?\s*[A-Z][A-Za-z\s&,]+', re.IGNORECASE),
    re.compile(r'(?:Master|M\.?S\.?|M\.?A\.?|M\.?E\.?|M\.?Tech|M\.?Sc|MBA)\s*(?:of|in|\.?\s*)?\s*[A-Z][A-Za-z\s&,]+', re.IGNORECASE),
    re.compile(r'(?:Ph\.?D\.?|Doctorate|Doctor)\s*(?:of|in|\.?\s*)?\s*[A-Z][A-Za-z\s&,]+', re.IGNORECASE),
    re.compile(r'(?:Associate|A\.?S\.?|A\.?A\.?)\s*(?:of|in|\.?\s*)?\s*[A-Z][A-Za-z\s&,]+', re.IGNORECASE),
    re.compile(r'(?:Diploma|Certificate|Certification)\s*(?:of|in|\.?\s*)?\s*[A-Z][A-Za-z\s&,]+', re.IGNORECASE),
]

# Email and phone (used to filter out false-positive names)
EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_PATTERN = re.compile(r'[\+\(]?[\d\s\-\(\)]{7,15}')

# Words that disqualify a line from being a person's name
NAME_BLACKLIST = {
    'university', 'college', 'institute', 'school', 'academy',
    'inc', 'corp', 'ltd', 'llc', 'pvt', 'technologies', 'solutions',
    'resume', 'curriculum', 'vitae', 'profile', 'summary', 'objective',
    'experience', 'education', 'skills', 'projects', 'certifications',
    'email', 'phone', 'mobile', 'address', 'linkedin', 'github'
}

# Keywords that suggest a line is a job title/role
ROLE_KEYWORDS = [
    'engineer', 'developer', 'architect', 'manager', 'director',
    'analyst', 'scientist', 'designer', 'lead', 'senior', 'junior',
    'intern', 'consultant', 'administrator', 'specialist', 'coordinator',
    'associate', 'president', 'vice president', 'cto', 'ceo', 'cfo',
    'head', 'chief', 'principal', 'staff', 'fellow', 'technician',
    'researcher', 'professor', 'instructor', 'teacher', 'trainer'
]

# Keywords that suggest a line is an institution name
INSTITUTION_KEYWORDS = [
    'university', 'college', 'institute', 'school', 'academy',
    'polytechnic', 'iit', 'nit', 'iiit', 'bits', 'faculty', 'department'
]


# ─────────────────────────────────────────────────────────────
# NAME EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_name(text: str) -> Optional[str]:
    """
    Extract candidate name from the top of the resume.

    Strategy (no NER):
        1. Only scan the first ~500 characters (name is at the top).
        2. Skip lines that look like emails, phones, URLs, or section headers.
        3. Return the first short line (2–4 words) made of capitalized words.
        4. Fallback: return the very first non-empty line.

    Args:
        text: Full resume text.

    Returns:
        Candidate name string or None.
    """
    header = text[:500]
    lines = [l.strip() for l in header.split('\n') if l.strip()]

    for line in lines:
        # Skip lines with emails, phones, or URLs
        if EMAIL_PATTERN.search(line):
            continue
        if PHONE_PATTERN.search(line):
            continue
        if 'http' in line.lower() or 'www.' in line.lower():
            continue
        # Skip lines that are clearly section headers or contact info
        lower = line.lower()
        if any(kw in lower for kw in NAME_BLACKLIST):
            continue
        # Skip lines with special characters typical of contact lines
        if any(ch in line for ch in ['@', '|', '/', '\\', '#']):
            continue

        words = line.split()
        # A person's name is typically 2–4 words, all starting capitalized
        if 2 <= len(words) <= 4:
            if all(w[0].isupper() for w in words if w.isalpha()):
                return line

    # Fallback: first non-empty line that isn't too long
    for line in lines:
        if line and len(line.split()) <= 4 and len(line) < 50:
            if '@' not in line and not PHONE_PATTERN.search(line):
                return line

    return None


# ─────────────────────────────────────────────────────────────
# ORGANIZATION EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_organizations(text: str) -> List[str]:
    """
    Extract organization names from text using heuristics.

    Looks for lines/phrases containing known org keywords
    (Inc, Corp, Ltd, Technologies, University, etc.)

    Args:
        text: Resume text.

    Returns:
        List of unique organization name strings.
    """
    ORG_SUFFIXES = re.compile(
        r'\b(?:Inc\.?|Corp\.?|Ltd\.?|LLC|LLP|Pvt\.?|Technologies|Solutions|'
        r'Systems|Services|Group|Consulting|Labs|University|College|Institute|'
        r'School|Academy|Foundation|Agency|Studio|Works|Partners|Ventures)\b',
        re.IGNORECASE
    )

    orgs = []
    seen = set()

    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) > 80:
            continue
        if ORG_SUFFIXES.search(line):
            key = line.lower()
            if key not in seen:
                seen.add(key)
                orgs.append(line)

    return orgs


# ─────────────────────────────────────────────────────────────
# DATE EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_dates(text: str) -> List[str]:
    """
    Extract date ranges from text.

    Args:
        text: Text to search.

    Returns:
        List of date range strings.
    """
    return DATE_RANGE_PATTERN.findall(text)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _detect_role(line: str) -> bool:
    """Return True if the line likely contains a job title."""
    lower = line.lower()
    return any(kw in lower for kw in ROLE_KEYWORDS)


def _detect_institution(line: str) -> bool:
    """Return True if the line likely contains an institution name."""
    lower = line.lower()
    return any(kw in lower for kw in INSTITUTION_KEYWORDS)


# ─────────────────────────────────────────────────────────────
# EXPERIENCE PARSING
# ─────────────────────────────────────────────────────────────

def parse_experience(experience_text: str) -> List[Dict[str, str]]:
    """
    Parse the experience section into structured entries.

    Each entry contains:
        - company:     Organization name
        - title:       Job title / role
        - dates:       Employment date range
        - description: Bullet points / responsibilities

    Strategy:
        1. Walk through lines; detect date-range lines as entry boundaries.
        2. Use role-keyword matching to identify job titles.
        3. Use separator patterns (" | ", " — ") to split company/title.
        4. Remaining lines become the description.

    Args:
        experience_text: Text of the Experience section.

    Returns:
        List of experience entry dicts.
    """
    if not experience_text:
        return []

    entries = []
    lines = experience_text.split('\n')

    current_entry = {"company": "", "title": "", "dates": "", "description": ""}
    description_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        date_match = DATE_RANGE_PATTERN.search(line)

        if date_match:
            # Save previous entry if it has data
            if current_entry["company"] or current_entry["title"]:
                current_entry["description"] = '\n'.join(description_lines).strip()
                entries.append(current_entry)
                current_entry = {"company": "", "title": "", "dates": "", "description": ""}
                description_lines = []

            current_entry["dates"] = date_match.group(0).strip()

            # Anything left on the same line after stripping the date
            remaining = DATE_RANGE_PATTERN.sub('', line).strip()
            remaining = re.sub(r'^[\s|,\-\u2013\u2014]+|[\s|,\-\u2013\u2014]+$', '', remaining)

            if remaining:
                if _detect_role(remaining):
                    current_entry["title"] = remaining
                else:
                    current_entry["company"] = remaining

        elif not current_entry["company"] and not current_entry["title"]:
            # Try to split on common separators: " — ", " | ", " - "
            parts = re.split(r'\s*[\u2013\u2014|]\s*|\s+-\s+', line)

            if len(parts) >= 2:
                if _detect_role(parts[0]):
                    current_entry["title"] = parts[0].strip()
                    current_entry["company"] = parts[1].strip()
                elif _detect_role(parts[1]):
                    current_entry["company"] = parts[0].strip()
                    current_entry["title"] = parts[1].strip()
                else:
                    current_entry["company"] = parts[0].strip()
                    current_entry["title"] = parts[1].strip()
            else:
                if _detect_role(line):
                    current_entry["title"] = line
                else:
                    current_entry["company"] = line

        elif not current_entry["title"] and _detect_role(line):
            current_entry["title"] = line

        elif not current_entry["company"] and not _detect_role(line):
            if len(line.split()) <= 6:
                current_entry["company"] = line
            else:
                description_lines.append(line)
        else:
            description_lines.append(line)

    # Save the last entry
    if current_entry["company"] or current_entry["title"]:
        current_entry["description"] = '\n'.join(description_lines).strip()
        entries.append(current_entry)

    # Fill in missing company names using org heuristics from description
    for entry in entries:
        if not entry["company"] and entry["description"]:
            orgs = extract_organizations(entry["description"][:300])
            if orgs:
                entry["company"] = orgs[0]

    return entries


# ─────────────────────────────────────────────────────────────
# EDUCATION PARSING
# ─────────────────────────────────────────────────────────────

def parse_education(education_text: str) -> List[Dict[str, str]]:
    """
    Parse the education section into structured entries.

    Each entry contains:
        - institution: University / college name
        - degree:      Degree title
        - dates:       Attendance date range
        - gpa:         GPA score (if found)

    Strategy:
        1. Scan each line for degree patterns → marks start of an entry.
        2. Detect date ranges and GPA via regex.
        3. Detect institution names by keyword (university, college, etc.)
           or by capitalized short lines near degree lines.

    Args:
        education_text: Text of the Education section.

    Returns:
        List of education entry dicts.
    """
    if not education_text:
        return []

    entries = []
    lines = education_text.split('\n')
    current_entry = {"institution": "", "degree": "", "dates": "", "gpa": ""}
    used_lines = set()

    # First pass: degrees, dates, GPA
    for i, line in enumerate(lines):
        line_s = line.strip()
        if not line_s:
            continue

        for pattern in DEGREE_PATTERNS:
            degree_match = pattern.search(line_s)
            if degree_match:
                if current_entry["degree"] and current_entry["institution"]:
                    entries.append(current_entry)
                    current_entry = {"institution": "", "degree": "", "dates": "", "gpa": ""}
                current_entry["degree"] = re.sub(
                    r'[\s,]+$', '', degree_match.group(0).strip()
                )
                used_lines.add(i)
                break

        date_match = DATE_RANGE_PATTERN.search(line_s)
        if date_match:
            current_entry["dates"] = date_match.group(0).strip()
            used_lines.add(i)

        gpa_match = GPA_PATTERN.search(line_s)
        if gpa_match:
            gpa_val = gpa_match.group(1)
            scale = gpa_match.group(2) or ""
            current_entry["gpa"] = f"{gpa_val}/{scale}" if scale else gpa_val
            used_lines.add(i)

    # Second pass: institution names
    for i, line in enumerate(lines):
        line_s = line.strip()
        if not line_s or i in used_lines:
            continue

        if _detect_institution(line_s):
            if current_entry["institution"] and (current_entry["degree"] or current_entry["dates"]):
                entries.append(current_entry)
                current_entry = {"institution": "", "degree": "", "dates": "", "gpa": ""}
            current_entry["institution"] = line_s
            used_lines.add(i)
        elif not current_entry["institution"] and len(line_s.split()) <= 8:
            # Short, unused line near degree — likely institution name
            current_entry["institution"] = line_s
            used_lines.add(i)

    # Save last entry
    if current_entry["institution"] or current_entry["degree"]:
        entries.append(current_entry)

    return entries


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    sample_exp = """
Google — Senior Software Engineer
January 2022 - Present
Led a team of 8 engineers building microservices.
Designed and implemented a distributed caching layer reducing latency by 40%.

Microsoft — Software Developer
June 2019 - December 2021
Built front-end components using React and TypeScript.
Developed RESTful APIs serving 500K+ daily requests.
"""
    print("=== EXPERIENCE ===")
    print(json.dumps(parse_experience(sample_exp), indent=2))

    sample_edu = """
Massachusetts Institute of Technology
Bachelor of Science in Computer Science
2015 - 2019
GPA: 3.8/4.0

Stanford University
Master of Science in Artificial Intelligence
2019 - 2021
"""
    print("\n=== EDUCATION ===")
    print(json.dumps(parse_education(sample_edu), indent=2))

    print("\n=== NAME ===")
    sample_name = "John Alexander Smith\njohn@email.com\nSan Francisco, CA"
    print(f"Extracted name: {extract_name(sample_name)}")
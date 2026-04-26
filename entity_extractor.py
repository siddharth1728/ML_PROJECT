"""
entity_extractor.py - spaCy NER-Based Entity Extraction

Uses spaCy's Named Entity Recognition (NER) pipeline to extract:
    - Candidate Name (PERSON entities)
    - University/Institution Names (ORG entities)
    - Degree Titles (pattern matching + NER)
    - Date Ranges (DATE entities + regex)
    - Experience entries (Company, Role, Dates, Description)
    - Education entries (Institution, Degree, Dates, GPA)
"""

import re
import spacy
from typing import Dict, List, Optional


# Load spaCy model globally for efficiency
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("WARNING: spaCy model 'en_core_web_sm' not found.")
    print("Install it with: python -m spacy download en_core_web_sm")
    nlp = None


# ─────────────────────────────────────────────────────────────
# PATTERNS FOR EXPERIENCE & EDUCATION PARSING
# ─────────────────────────────────────────────────────────────

# Date range patterns (e.g., "Jan 2020 - Present", "2018-2022", "March 2019 - Dec 2021")
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

# GPA pattern
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

# Job title patterns (common role keywords)
ROLE_KEYWORDS = [
    'engineer', 'developer', 'architect', 'manager', 'director',
    'analyst', 'scientist', 'designer', 'lead', 'senior', 'junior',
    'intern', 'consultant', 'administrator', 'specialist', 'coordinator',
    'associate', 'president', 'vice president', 'cto', 'ceo', 'cfo',
    'head', 'chief', 'principal', 'staff', 'fellow', 'technician',
    'researcher', 'professor', 'instructor', 'teacher', 'trainer'
]


def extract_name(text: str) -> Optional[str]:
    """
    Extract candidate name from resume text using spaCy NER.

    Strategy:
        1. Process the first ~500 chars (name is typically at the top)
        2. Find PERSON entities
        3. Return the first PERSON entity with 2-4 words (likely a name)
        4. Fallback: return the first non-empty line if no NER match

    Args:
        text: Full resume text.

    Returns:
        Candidate name string or None.
    """
    if nlp is None:
        return _fallback_name(text)

    # Only look at the header area (first ~500 chars)
    header = text[:500]
    doc = nlp(header)

    # Find PERSON entities
    person_entities = [
        ent.text.strip() for ent in doc.ents
        if ent.label_ == "PERSON"
    ]

    for name in person_entities:
        words = name.split()
        # A real name typically has 2-4 words
        if 2 <= len(words) <= 4:
            # Filter out common false positives
            lower = name.lower()
            if not any(kw in lower for kw in ['university', 'college', 'inc', 'corp', 'ltd']):
                return name

    # If NER found PERSON entities but none passed validation, return first
    if person_entities:
        return person_entities[0]

    # Fallback: first non-empty line
    return _fallback_name(text)


def _fallback_name(text: str) -> Optional[str]:
    """Fallback name extraction: return the first non-empty short line."""
    for line in text.split('\n'):
        line = line.strip()
        if line and len(line.split()) <= 4 and len(line) < 50:
            # Check it's not an email or phone
            if '@' not in line and not re.search(r'\d{3}', line):
                return line
    return None


def extract_organizations(text: str) -> List[str]:
    """
    Extract organization names from text using spaCy NER.

    Args:
        text: Resume text to analyze.

    Returns:
        List of unique organization names (companies, universities).
    """
    if nlp is None:
        return []

    doc = nlp(text)
    orgs = []
    seen = set()

    for ent in doc.ents:
        if ent.label_ == "ORG":
            org_name = ent.text.strip()
            org_lower = org_name.lower()
            # Filter out very short or very long org names
            if 2 <= len(org_name) <= 60 and org_lower not in seen:
                seen.add(org_lower)
                orgs.append(org_name)

    return orgs


def extract_dates(text: str) -> List[str]:
    """
    Extract date ranges from text.

    Args:
        text: Text to search for date ranges.

    Returns:
        List of date range strings found.
    """
    return DATE_RANGE_PATTERN.findall(text)


def _detect_role(line: str) -> bool:
    """Check if a line likely contains a job title."""
    lower = line.lower()
    return any(kw in lower for kw in ROLE_KEYWORDS)


def parse_experience(experience_text: str) -> List[Dict[str, str]]:
    """
    Parse the experience section into structured entries.

    Each entry contains:
        - company: Organization name
        - title: Job title/role
        - dates: Employment date range
        - description: Bullet points / description text

    Strategy:
        1. Split text into blocks separated by blank lines or date-containing lines
        2. Use NER to identify company names (ORG entities)
        3. Use keyword matching to identify job titles
        4. Use regex to extract date ranges
        5. Remaining text becomes the description

    Args:
        experience_text: Text from the Experience section.

    Returns:
        List of experience entry dictionaries.
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

        # Check for date range
        date_match = DATE_RANGE_PATTERN.search(line)

        # Check if this line starts a new entry
        # (has a date range or looks like a company/title header)
        if date_match:
            # If we already have accumulated data, save the previous entry
            if current_entry["company"] or current_entry["title"]:
                current_entry["description"] = '\n'.join(description_lines).strip()
                entries.append(current_entry)
                current_entry = {"company": "", "title": "", "dates": "", "description": ""}
                description_lines = []

            current_entry["dates"] = date_match.group(0).strip()

            # The rest of the line might contain company/title info
            remaining = DATE_RANGE_PATTERN.sub('', line).strip()
            remaining = re.sub(r'^[\s|,\-\u2013\u2014]+|[\s|,\-\u2013\u2014]+$', '', remaining)

            if remaining:
                if _detect_role(remaining):
                    current_entry["title"] = remaining
                else:
                    current_entry["company"] = remaining

        elif not current_entry["company"] and not current_entry["title"]:
            # First lines after a new entry starts — look for company/title
            # Lines with separators like " — " or " | " or " - " often split company/title
            parts = re.split(r'\s*[\u2013\u2014|]\s*|\s+-\s+', line)

            if len(parts) >= 2:
                # e.g., "Google — Software Engineer" or "Software Engineer | Google"
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
            # If we don't have a company yet, and this isn't a role line
            if not date_match and len(line.split()) <= 6:
                current_entry["company"] = line
            else:
                description_lines.append(line)
        else:
            description_lines.append(line)

    # Save the last entry
    if current_entry["company"] or current_entry["title"]:
        current_entry["description"] = '\n'.join(description_lines).strip()
        entries.append(current_entry)

    # Use NER to fill in missing company names
    if nlp is not None:
        for entry in entries:
            if not entry["company"] and entry["description"]:
                doc = nlp(entry["description"][:200])
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        entry["company"] = ent.text
                        break

    return entries


def parse_education(education_text: str) -> List[Dict[str, str]]:
    """
    Parse the education section into structured entries.

    Each entry contains:
        - institution: University/college name
        - degree: Degree title
        - dates: Attendance date range
        - gpa: GPA score (if found)

    Args:
        education_text: Text from the Education section.

    Returns:
        List of education entry dictionaries.
    """
    if not education_text:
        return []

    entries = []
    lines = education_text.split('\n')
    current_entry = {"institution": "", "degree": "", "dates": "", "gpa": ""}
    used_lines = set()

    # First pass: extract dates and GPA
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Check for degree patterns
        for pattern in DEGREE_PATTERNS:
            degree_match = pattern.search(line)
            if degree_match:
                # If we already have a degree, this might be a new entry
                if current_entry["degree"] and current_entry["institution"]:
                    entries.append(current_entry)
                    current_entry = {"institution": "", "degree": "", "dates": "", "gpa": ""}

                current_entry["degree"] = degree_match.group(0).strip()
                # Clean trailing punctuation
                current_entry["degree"] = re.sub(r'[\s,]+$', '', current_entry["degree"])
                used_lines.add(i)
                break

        # Check for dates
        date_match = DATE_RANGE_PATTERN.search(line)
        if date_match:
            current_entry["dates"] = date_match.group(0).strip()
            used_lines.add(i)

        # Check for GPA
        gpa_match = GPA_PATTERN.search(line)
        if gpa_match:
            gpa_val = gpa_match.group(1)
            scale = gpa_match.group(2) if gpa_match.group(2) else ""
            current_entry["gpa"] = f"{gpa_val}/{scale}" if scale else gpa_val
            used_lines.add(i)

    # Second pass: find institution names using NER and remaining lines
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or i in used_lines:
            continue

        # If this looks like an institution name (use NER or heuristics)
        institution_keywords = [
            'university', 'college', 'institute', 'school',
            'academy', 'polytechnic', 'iit', 'nit', 'iiit', 'bits'
        ]
        lower = line.lower()

        if any(kw in lower for kw in institution_keywords):
            if current_entry["institution"] and (current_entry["degree"] or current_entry["dates"]):
                entries.append(current_entry)
                current_entry = {"institution": "", "degree": "", "dates": "", "gpa": ""}
            current_entry["institution"] = line
            used_lines.add(i)
        elif not current_entry["institution"] and len(line.split()) <= 8:
            # Short line without a degree/date — might be institution name
            current_entry["institution"] = line
            used_lines.add(i)

    # Try NER for institution names if still missing
    if nlp is not None and not current_entry["institution"]:
        doc = nlp(education_text[:500])
        for ent in doc.ents:
            if ent.label_ == "ORG":
                org = ent.text.strip()
                lower_org = org.lower()
                if any(kw in lower_org for kw in ['university', 'college', 'institute', 'school']):
                    current_entry["institution"] = org
                    break

    # Save last entry
    if current_entry["institution"] or current_entry["degree"]:
        entries.append(current_entry)

    return entries


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
    exp_result = parse_experience(sample_exp)
    print(json.dumps(exp_result, indent=2))

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
    edu_result = parse_education(sample_edu)
    print(json.dumps(edu_result, indent=2))

    print("\n=== NAME ===")
    sample_name = "John Alexander Smith\njohn@email.com\nSan Francisco, CA"
    print(f"Extracted name: {extract_name(sample_name)}")
"""
section_segmenter.py - Resume Section Detection & Segmentation

Uses keyword-based heuristics to identify and split resume text into
logical sections: Contact Info, Summary, Experience, Education, Skills,
Projects, Certifications, Awards, Publications, Languages, etc.

Strategy:
    1. Each line is checked against a dictionary of known section header keywords
    2. Headers are typically short (<80 chars), in ALL CAPS or Title Case
    3. Lines matching keywords mark section boundaries
    4. Text before the first header is assumed to be contact information
    5. Handles variations (e.g., "Work Experience" = "Professional Experience")
"""

import re
from typing import Dict, List

# ─────────────────────────────────────────────────────────────
# SECTION HEADER PATTERNS
# ─────────────────────────────────────────────────────────────

# Maps section type -> list of keyword patterns (case-insensitive)
SECTION_PATTERNS: Dict[str, List[str]] = {
    "summary": [
        r"summary", r"objective", r"profile", r"about\s*me",
        r"professional\s+summary", r"career\s+summary",
        r"executive\s+summary", r"personal\s+statement",
        r"career\s+objective", r"professional\s+profile",
        r"overview"
    ],
    "experience": [
        r"experience", r"work\s+experience", r"professional\s+experience",
        r"employment\s+history", r"work\s+history", r"career\s+history",
        r"professional\s+background", r"relevant\s+experience",
        r"industry\s+experience", r"internship(?:s)?",
        r"positions?\s+held"
    ],
    "education": [
        r"education", r"academic\s+background", r"academic\s+qualifications",
        r"educational\s+qualifications", r"academic\s+history",
        r"qualifications", r"degrees?", r"schooling",
        r"academic\s+credentials"
    ],
    "skills": [
        r"skills", r"technical\s+skills", r"core\s+competencies",
        r"competencies", r"expertise", r"proficiencies",
        r"technical\s+proficiencies", r"areas?\s+of\s+expertise",
        r"technologies", r"tools?\s+(?:and|&)\s+technologies",
        r"programming\s+languages", r"tech\s+stack",
        r"key\s+skills", r"relevant\s+skills",
        r"professional\s+skills", r"soft\s+skills"
    ],
    "projects": [
        r"projects", r"personal\s+projects", r"academic\s+projects",
        r"key\s+projects", r"notable\s+projects",
        r"selected\s+projects", r"portfolio"
    ],
    "certifications": [
        r"certifications?", r"licenses?\s+(?:and|&)\s+certifications?",
        r"professional\s+certifications?", r"credentials",
        r"accreditations?"
    ],
    "awards": [
        r"awards?", r"honors?", r"achievements?",
        r"awards?\s+(?:and|&)\s+honors?",
        r"recognition", r"accomplishments?"
    ],
    "publications": [
        r"publications?", r"research\s+papers?",
        r"papers?\s+(?:and|&)\s+publications?",
        r"published\s+works?", r"journal\s+articles?"
    ],
    "languages": [
        r"languages?", r"language\s+proficiency",
        r"linguistic\s+skills"
    ],
    "volunteer": [
        r"volunteer(?:ing)?", r"community\s+service",
        r"social\s+work", r"volunteer\s+experience",
        r"extracurricular"
    ],
    "references": [
        r"references?", r"referees?"
    ],
    "interests": [
        r"interests?", r"hobbies", r"hobbies?\s+(?:and|&)\s+interests?"
    ]
}


def _is_section_header(line: str) -> str:
    """
    Check if a line is a section header.

    A line is considered a section header if:
        1. It matches one of the known keyword patterns
        2. It is reasonably short (< 80 characters)
        3. It doesn't contain too many words (likely a sentence, not a header)

    Args:
        line: A single line of text from the resume.

    Returns:
        The section type string if matched, or empty string if not a header.
    """
    cleaned = line.strip()

    # Skip empty or very long lines (headers are short)
    if not cleaned or len(cleaned) > 80:
        return ""

    # Skip lines with too many words (sentences, not headers)
    word_count = len(cleaned.split())
    if word_count > 6:
        return ""

    # Remove common decorators from headers
    # e.g., "=== EXPERIENCE ===" or "--- Skills ---" or "EXPERIENCE:"
    cleaned_for_match = re.sub(r'[=\-_*#|:]+', ' ', cleaned).strip()

    for section_type, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            # Match the entire cleaned line against the pattern
            full_pattern = rf'^\s*{pattern}\s*$'
            if re.match(full_pattern, cleaned_for_match, re.IGNORECASE):
                return section_type

    return ""


def segment_sections(text: str) -> Dict[str, str]:
    """
    Segment resume text into logical sections.

    Splits the full resume text into named sections by detecting
    section headers using keyword-based heuristics.

    Args:
        text: The full cleaned text of the resume.

    Returns:
        Dictionary mapping section names to their text content.
        Always includes "contact_info" for text before the first header.
        Example: {
            "contact_info": "John Doe\\njohn@email.com\\n...",
            "summary": "Experienced software engineer...",
            "experience": "Google — Software Engineer\\n...",
            "education": "MIT — BS Computer Science\\n...",
            "skills": "Python, Java, Docker, AWS...",
        }
    """
    if not text:
        return {"raw_text": ""}

    lines = text.split('\n')
    sections: Dict[str, str] = {}
    current_section = "contact_info"  # Default for text before first header
    current_lines: List[str] = []

    for line in lines:
        detected_section = _is_section_header(line)

        if detected_section:
            # Save the accumulated lines for the previous section
            section_text = '\n'.join(current_lines).strip()
            if section_text:
                # If section already exists, append to it
                if current_section in sections:
                    sections[current_section] += '\n' + section_text
                else:
                    sections[current_section] = section_text

            # Start a new section
            current_section = detected_section
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last section
    section_text = '\n'.join(current_lines).strip()
    if section_text:
        if current_section in sections:
            sections[current_section] += '\n' + section_text
        else:
            sections[current_section] = section_text

    return sections


def get_section_names() -> List[str]:
    """Return all supported section type names."""
    return list(SECTION_PATTERNS.keys())


if __name__ == "__main__":
    # Quick test with sample resume text
    sample = """
John Doe
john.doe@email.com | (555) 123-4567
linkedin.com/in/johndoe | github.com/johndoe

SUMMARY
Experienced software engineer with 5+ years building scalable web apps.

WORK EXPERIENCE
Google — Software Engineer
Jan 2022 - Present
Built microservices handling 1M+ requests/day using Python and Go.

Microsoft — Junior Developer
Jun 2019 - Dec 2021
Developed front-end components using React and TypeScript.

EDUCATION
Massachusetts Institute of Technology
B.S. Computer Science, GPA: 3.8
2015 - 2019

SKILLS
Python, Java, Go, JavaScript, TypeScript, React, Django, Docker,
Kubernetes, AWS, PostgreSQL, MongoDB, Git, CI/CD

PROJECTS
Resume Parser — Built an NLP-based resume parser using spaCy.
ChatBot — Created a customer service chatbot with GPT-4.

CERTIFICATIONS
AWS Certified Solutions Architect — Associate (2023)
Google Cloud Professional Data Engineer (2022)
"""
    sections = segment_sections(sample.strip())
    for section_name, content in sections.items():
        print(f"\n{'='*50}")
        print(f"[{section_name.upper()}]")
        print(f"{'='*50}")
        print(content[:300])
        

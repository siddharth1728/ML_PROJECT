"""
contact_extractor.py - Regex-Based Contact Information Extraction

Extracts structured contact information from resume text using
complex Regular Expressions:
    - Email addresses
    - Phone numbers (with international format normalization)
    - LinkedIn profile URLs
    - GitHub profile URLs
    - General website/portfolio URLs
    - Physical location/address
"""

import re
from typing import Dict, List, Optional, Any


# ─────────────────────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────────────────────

# Email: standard email pattern
EMAIL_PATTERN = re.compile(
    r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
    re.IGNORECASE
)

# Phone: handles multiple international formats
# Matches: +1-555-123-4567, (555) 123-4567, 555.123.4567,
#          +91 98765 43210, +44 20 7946 0958, etc.
PHONE_PATTERN = re.compile(
    r'(?:\+?\d{1,3}[\s\-.]?)?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}(?:\s*(?:ext|x)\s*\d{1,5})?',
    re.IGNORECASE
)

# LinkedIn: various URL formats
LINKEDIN_PATTERN = re.compile(
    r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+/?',
    re.IGNORECASE
)

# GitHub: various URL formats
GITHUB_PATTERN = re.compile(
    r'(?:https?://)?(?:www\.)?github\.com/[\w\-]+/?',
    re.IGNORECASE
)

# General URL: catches portfolio/website links
URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+',
    re.IGNORECASE
)

# Location: common city/state/country patterns
LOCATION_PATTERNS = [
    # City, State ZIP (US)
    re.compile(r'([A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s*\d{5}(?:\-\d{4})?)'),
    # City, State (US)
    re.compile(r'([A-Z][a-zA-Z\s]+,\s*[A-Z]{2})\b'),
    # City, Country
    re.compile(r'([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)\b'),
]


def extract_emails(text: str) -> List[str]:
    """
    Extract all email addresses from text.

    Args:
        text: Resume text to search.

    Returns:
        List of unique email addresses found, lowercased.
    """
    emails = EMAIL_PATTERN.findall(text)
    # Deduplicate and lowercase
    seen = set()
    result = []
    for email in emails:
        email_lower = email.lower()
        if email_lower not in seen:
            seen.add(email_lower)
            result.append(email_lower)
    return result


def _normalize_phone(raw_phone: str) -> str:
    """
    Normalize a phone number to a standard format.

    Strips all non-digit characters except leading '+',
    then formats based on digit count.

    Args:
        raw_phone: Raw phone string from regex match.

    Returns:
        Formatted phone number string.
    """
    # Extract just digits
    digits = re.sub(r'[^\d]', '', raw_phone)

    if not digits or len(digits) < 7:
        return ""

    # Handle extension
    ext_match = re.search(r'(?:ext|x)\s*(\d+)', raw_phone, re.IGNORECASE)
    ext = f" ext. {ext_match.group(1)}" if ext_match else ""

    # Format based on length
    if len(digits) == 10:
        # US domestic: (XXX) XXX-XXXX
        formatted = f"+1-{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits[0] == '1':
        # US with country code: +1-XXX-XXX-XXXX
        formatted = f"+{digits[0]}-{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
    elif len(digits) == 12 and digits[:2] == '91':
        # India: +91-XXXXX-XXXXX
        formatted = f"+{digits[:2]}-{digits[2:7]}-{digits[7:]}"
    elif len(digits) >= 11:
        # International: +CC-REST
        cc_len = 1 if digits[0] == '1' else (2 if len(digits) <= 12 else 3)
        formatted = f"+{digits[:cc_len]}-{digits[cc_len:]}"
    else:
        # Short format: just join with dashes
        formatted = f"+{digits}"

    return formatted + ext


def extract_phones(text: str) -> List[str]:
    """
    Extract and normalize phone numbers from text.

    Args:
        text: Resume text to search.

    Returns:
        List of unique, normalized phone number strings.
    """
    raw_phones = PHONE_PATTERN.findall(text)
    seen = set()
    result = []

    for raw in raw_phones:
        normalized = _normalize_phone(raw)
        if normalized and normalized not in seen:
            # Verify it has enough digits to be a real phone number
            digit_count = len(re.sub(r'[^\d]', '', normalized))
            if digit_count >= 10:
                seen.add(normalized)
                result.append(normalized)

    return result


def extract_linkedin(text: str) -> Optional[str]:
    """
    Extract LinkedIn profile URL from text.

    Args:
        text: Resume text to search.

    Returns:
        LinkedIn URL string or None if not found.
    """
    match = LINKEDIN_PATTERN.search(text)
    if match:
        url = match.group(0)
        # Ensure https:// prefix
        if not url.startswith('http'):
            url = 'https://' + url
        return url.rstrip('/')
    return None


def extract_github(text: str) -> Optional[str]:
    """
    Extract GitHub profile URL from text.

    Args:
        text: Resume text to search.

    Returns:
        GitHub URL string or None if not found.
    """
    match = GITHUB_PATTERN.search(text)
    if match:
        url = match.group(0)
        if not url.startswith('http'):
            url = 'https://' + url
        return url.rstrip('/')
    return None


def extract_websites(text: str) -> List[str]:
    """
    Extract general website/portfolio URLs from text.
    Excludes LinkedIn and GitHub URLs (handled separately).

    Args:
        text: Resume text to search.

    Returns:
        List of unique website URL strings.
    """
    all_urls = URL_PATTERN.findall(text)
    # Filter out LinkedIn and GitHub (already extracted separately)
    excluded = {'linkedin.com', 'github.com'}
    result = []
    seen = set()

    for url in all_urls:
        url_lower = url.lower()
        if not any(excl in url_lower for excl in excluded):
            if url_lower not in seen:
                seen.add(url_lower)
                result.append(url)

    return result


def extract_location(text: str) -> Optional[str]:
    """
    Attempt to extract a location/address from text.
    Uses pattern matching for common city/state/country formats.

    Args:
        text: Resume text (preferably from the contact/header section).

    Returns:
        Location string or None if not found.
    """
    # Only search the first ~500 chars (location is usually at the top)
    header_text = text[:500]

    for pattern in LOCATION_PATTERNS:
        match = pattern.search(header_text)
        if match:
            location = match.group(1).strip()
            # Validate: skip if it looks like a section header
            if len(location) < 50 and not any(
                kw in location.lower()
                for kw in ['experience', 'education', 'skills', 'summary']
            ):
                return location

    return None


def extract_contact_info(text: str) -> Dict[str, Any]:
    """
    Extract all contact information from resume text.

    This is the main entry point that combines all individual
    extraction functions into a single structured result.

    Args:
        text: Full resume text.

    Returns:
        Dictionary with contact fields:
        {
            "emails": [...],
            "phones": [...],
            "linkedin": "..." or None,
            "github": "..." or None,
            "websites": [...],
            "location": "..." or None
        }
    """
    return {
        "emails": extract_emails(text),
        "phones": extract_phones(text),
        "linkedin": extract_linkedin(text),
        "github": extract_github(text),
        "websites": extract_websites(text),
        "location": extract_location(text)
    }


if __name__ == "__main__":
    sample = """
    John Smith
    San Francisco, CA 94102
    Email: john.smith@gmail.com | Phone: +1 (555) 123-4567
    LinkedIn: https://linkedin.com/in/johnsmith
    GitHub: https://github.com/johnsmith
    Portfolio: https://johnsmith.dev
    """
    result = extract_contact_info(sample)
    import json
    print(json.dumps(result, indent=2))
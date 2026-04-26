"""
parser.py - Main ResumeParser Class (Orchestrator)

The core engine of the ATS Resume Parser. Orchestrates all extraction
modules to produce a standardized JSON output from PDF/DOCX resumes.

Pipeline:
    1. Text Extraction (text_extractor.py)
    2. Section Segmentation (section_segmenter.py)
    3. Contact Info Extraction via Regex (contact_extractor.py)
    4. Named Entity Recognition via spaCy (entity_extractor.py)
    5. Skill Extraction via keyword + semantic matching (skill_extractor.py)
    6. ATS Scoring
    7. JSON Output Assembly

Usage:
    # Python API
    from parser import ResumeParser
    parser = ResumeParser()
    result = parser.parse_file("resume.pdf")

    # CLI
    python parser.py resume.pdf [output.json]
"""

import os
import sys
import json
import datetime
from typing import Dict, List, Optional, Any

# Import all sub-modules
from text_extractor import extract_text_from_file
from section_segmenter import segment_sections
from contact_extractor import extract_contact_info
from entity_extractor import (
    extract_name, extract_organizations,
    parse_experience, parse_education
)
from skill_extractor import (
    load_skills_database, extract_skills,
    SemanticSkillMatcher, SEMANTIC_AVAILABLE
)


class ResumeParser:
    """
    Main ATS Resume Parser class.

    Orchestrates text extraction, section segmentation, contact info extraction,
    NER-based entity extraction, skill matching, and ATS scoring to produce
    a comprehensive structured JSON output from resume files.

    Attributes:
        skills_db: Loaded skills knowledge base dictionary.
        semantic_matcher: SemanticSkillMatcher instance (if available).
        use_semantic: Whether to use semantic skill matching.
    """

    def __init__(self, use_semantic: bool = True):
        """
        Initialize the ResumeParser.

        Args:
            use_semantic: Whether to enable semantic skill matching.
                          Set to False for faster processing without
                          Sentence Transformer overhead.
        """
        # Load skills knowledge base
        self.skills_db = load_skills_database()
        self.use_semantic = use_semantic

        # Initialize semantic matcher if available and enabled
        self.semantic_matcher = None
        if use_semantic and SEMANTIC_AVAILABLE:
            try:
                self.semantic_matcher = SemanticSkillMatcher(self.skills_db)
                print("[ResumeParser] Semantic skill matching: ENABLED")
            except Exception as e:
                print(f"[ResumeParser] Semantic matching failed to init: {e}")
                self.semantic_matcher = None
        else:
            reason = "disabled by user" if not use_semantic else "sentence-transformers not installed"
            print(f"[ResumeParser] Semantic skill matching: DISABLED ({reason})")

        print("[ResumeParser] Initialized successfully.")

    def parse_file(
        self,
        file_path: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse a resume file and return structured data.

        This is the main entry point. Accepts either a file path
        or raw bytes (for web uploads).

        Args:
            file_path: Path to the resume file on disk.
            file_bytes: Raw file bytes (e.g., from web upload).
            filename: Original filename (required with file_bytes).

        Returns:
            Dictionary containing all extracted resume data.
            See _assemble_output() for the full schema.
        """
        # Determine the filename for metadata
        if file_path:
            actual_filename = os.path.basename(file_path)
        elif filename:
            actual_filename = filename
        else:
            actual_filename = "unknown"

        try:
            # ── Step 1: Extract text ──────────────────────────────────
            raw_text = extract_text_from_file(
                file_path=file_path,
                file_bytes=file_bytes,
                filename=filename
            )

            if not raw_text or len(raw_text.strip()) < 10:
                return self._error_output(
                    actual_filename,
                    "No text could be extracted from the file."
                )

            # ── Step 2: Segment into sections ─────────────────────────
            sections = segment_sections(raw_text)

            # ── Step 3: Extract contact info (Regex) ──────────────────
            contact = extract_contact_info(raw_text)

            # ── Step 4: Extract name (spaCy NER) ──────────────────────
            # Use contact_info section if available, otherwise full text
            name_text = sections.get("contact_info", raw_text[:500])
            name = extract_name(name_text)

            # ── Step 5: Extract organizations (spaCy NER) ─────────────
            organizations = extract_organizations(raw_text)

            # ── Step 6: Parse experience section ──────────────────────
            experience_text = sections.get("experience", "")
            experience = parse_experience(experience_text)

            # ── Step 7: Parse education section ───────────────────────
            education_text = sections.get("education", "")
            education = parse_education(education_text)

            # ── Step 8: Extract skills (keyword + semantic) ───────────
            skills_result = extract_skills(
                raw_text,
                self.skills_db,
                semantic_matcher=self.semantic_matcher,
                use_semantic=self.use_semantic
            )

            # ── Step 9: Extract other sections ────────────────────────
            summary = sections.get("summary", "")
            projects = sections.get("projects", "")
            certifications = sections.get("certifications", "")
            awards = sections.get("awards", "")
            languages = sections.get("languages", "")
            volunteer = sections.get("volunteer", "")
            interests = sections.get("interests", "")
            publications = sections.get("publications", "")

            # ── Step 10: Calculate ATS score ──────────────────────────
            ats_score = self._calculate_ats_score(
                contact=contact,
                name=name,
                experience=experience,
                education=education,
                skills=skills_result,
                summary=summary,
                sections_found=list(sections.keys())
            )

            # ── Step 11: Assemble output ──────────────────────────────
            return self._assemble_output(
                filename=actual_filename,
                name=name,
                contact=contact,
                summary=summary,
                experience=experience,
                education=education,
                skills=skills_result,
                projects=projects,
                certifications=certifications,
                awards=awards,
                languages=languages,
                volunteer=volunteer,
                interests=interests,
                publications=publications,
                organizations=organizations,
                sections_found=list(sections.keys()),
                ats_score=ats_score,
                raw_text_length=len(raw_text)
            )

        except FileNotFoundError as e:
            return self._error_output(actual_filename, f"File not found: {e}")
        except ValueError as e:
            return self._error_output(actual_filename, f"Invalid input: {e}")
        except Exception as e:
            return self._error_output(actual_filename, f"Parsing error: {e}")

    def parse_file_to_json(
        self,
        file_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Parse a resume file and save the result as a JSON file.

        Args:
            file_path: Path to the resume file.
            output_path: Path for the output JSON file.
                         Defaults to <filename>_parsed.json.

        Returns:
            Path to the saved JSON file.
        """
        result = self.parse_file(file_path=file_path)

        if output_path is None:
            base = os.path.splitext(file_path)[0]
            output_path = f"{base}_parsed.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[ResumeParser] Parsed output saved to: {output_path}")
        return output_path

    def _calculate_ats_score(
        self,
        contact: Dict[str, Any],
        name: Optional[str],
        experience: List[Dict[str, Any]],
        education: List[Dict[str, Any]],
        skills: Dict[str, Any],
        summary: str,
        sections_found: List[str]
    ) -> int:
        """
        Calculate an ATS compatibility score (0-100).

        Scoring Formula:
            - Skills (30%): Based on number of skills found
            - Experience (25%): Based on completeness of experience entries
            - Education (20%): Based on completeness of education entries
            - Contact Info (10%): Based on completeness of contact details
            - Content Quality (15%): Summary, sections, and overall structure

        Returns:
            Integer score between 0 and 100.
        """
        score = 0.0

        # ── Skills Score (30 points max) ──────────────────────────────
        num_skills = len(skills.get("all", []))
        if num_skills >= 15:
            score += 30
        elif num_skills >= 10:
            score += 25
        elif num_skills >= 5:
            score += 18
        elif num_skills >= 1:
            score += 10

        # ── Experience Score (25 points max) ──────────────────────────
        if experience:
            exp_score = 0
            for entry in experience:
                entry_score = 0
                if entry.get("company"):
                    entry_score += 2
                if entry.get("title"):
                    entry_score += 2
                if entry.get("dates"):
                    entry_score += 1.5
                if entry.get("description"):
                    entry_score += 1.5
                exp_score += min(entry_score, 7)
            score += min(exp_score, 25)

        # ── Education Score (20 points max) ───────────────────────────
        if education:
            edu_score = 0
            for entry in education:
                entry_score = 0
                if entry.get("institution"):
                    entry_score += 4
                if entry.get("degree"):
                    entry_score += 4
                if entry.get("dates"):
                    entry_score += 1
                if entry.get("gpa"):
                    entry_score += 1
                edu_score += min(entry_score, 10)
            score += min(edu_score, 20)

        # ── Contact Info Score (10 points max) ────────────────────────
        contact_score = 0
        if name:
            contact_score += 2
        if contact.get("emails"):
            contact_score += 3
        if contact.get("phones"):
            contact_score += 2
        if contact.get("linkedin"):
            contact_score += 1.5
        if contact.get("github"):
            contact_score += 1.5
        score += min(contact_score, 10)

        # ── Content Quality Score (15 points max) ─────────────────────
        quality_score = 0
        if summary and len(summary) > 20:
            quality_score += 5
        # Bonus for having well-structured sections
        key_sections = {"experience", "education", "skills"}
        found_key = key_sections.intersection(set(sections_found))
        quality_score += len(found_key) * 2.5  # Up to 7.5
        if len(sections_found) >= 5:
            quality_score += 2.5
        score += min(quality_score, 15)

        return min(int(round(score)), 100)

    def _assemble_output(self, **kwargs: Any) -> Dict[str, Any]:
        """Assemble the final structured JSON output."""
        return {
            "metadata": {
                "file_name": kwargs.get("filename", "unknown"),
                "parsed_at": datetime.datetime.now().isoformat(),
                "sections_found": kwargs.get("sections_found", []),
                "raw_text_length": kwargs.get("raw_text_length", 0),
                "parser_version": "2.0.0"
            },
            "candidate": {
                "name": kwargs.get("name", ""),
                "contact": kwargs.get("contact", {})
            },
            "summary": kwargs.get("summary", ""),
            "experience": kwargs.get("experience", []),
            "education": kwargs.get("education", []),
            "skills": {
                "all": kwargs.get("skills", {}).get("all", []),
                "technical": kwargs.get("skills", {}).get("technical", []),
                "soft": kwargs.get("skills", {}).get("soft", []),
                "by_category": kwargs.get("skills", {}).get("by_category", {})
            },
            "projects": kwargs.get("projects", ""),
            "certifications": kwargs.get("certifications", ""),
            "awards": kwargs.get("awards", ""),
            "languages": kwargs.get("languages", ""),
            "volunteer": kwargs.get("volunteer", ""),
            "interests": kwargs.get("interests", ""),
            "publications": kwargs.get("publications", ""),
            "organizations": kwargs.get("organizations", []),
            "ats_score": kwargs.get("ats_score", 0)
        }

    def _error_output(self, filename: str, error_msg: str) -> Dict[str, Any]:
        """Create an error output when parsing fails."""
        return {
            "metadata": {
                "file_name": filename,
                "parsed_at": datetime.datetime.now().isoformat(),
                "parser_version": "2.0.0",
                "error": error_msg
            },
            "candidate": {"name": "", "contact": {}},
            "summary": "",
            "experience": [],
            "education": [],
            "skills": {"all": [], "technical": [], "soft": [], "by_category": {}},
            "projects": "",
            "certifications": "",
            "organizations": [],
            "ats_score": 0
        }


# ─────────────────────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ATS Resume Parser v2.0")
        print("=" * 40)
        print("Usage: python parser.py <resume_file> [output.json]")
        print("Supported formats: .pdf, .docx")
        print()
        print("Examples:")
        print("  python parser.py resume.pdf")
        print("  python parser.py resume.docx output.json")
        sys.exit(0)

    file_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Parsing: {file_path}")
    print("Initializing parser...")

    parser = ResumeParser(use_semantic=True)

    if output_path:
        parser.parse_file_to_json(file_path, output_path)
    else:
        result = parser.parse_file(file_path=file_path)
        print("\n" + "=" * 60)
        print("PARSED RESULT")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
"""
skill_extractor.py - Skill Matching Against Knowledge Base

Implements dual-mode skill extraction:
    1. Keyword Matching: Direct string matching against the skill ontology
       with alias resolution (e.g., "React.js" -> "React")
    2. Semantic Matching: Uses Sentence Transformers to find skills via
       embedding cosine similarity (handles variations like "ML" ≈ "Machine Learning")

The knowledge base is loaded from skills_knowledge_base.json.
"""

import os
import re
import json
from typing import Dict, List, Set, Tuple, Optional, Any

# Sentence Transformers for semantic matching (optional, degrades gracefully)
try:
    from sentence_transformers import SentenceTransformer, util
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("WARNING: sentence-transformers not installed. Semantic matching disabled.")
    print("Install with: pip install sentence-transformers")


# ─────────────────────────────────────────────────────────────
# KNOWLEDGE BASE LOADER
# ─────────────────────────────────────────────────────────────

def load_skills_database(json_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the skills knowledge base from JSON file.

    Args:
        json_path: Path to skills_knowledge_base.json.
                   Defaults to same directory as this script.

    Returns:
        Dictionary with 'categories' and 'aliases' keys.
    """
    if json_path is None:
        # Default: same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "skills_knowledge_base.json")

    if not os.path.exists(json_path):
        print(f"WARNING: Skills database not found at {json_path}")
        return {"categories": {}, "aliases": {}}

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _build_skill_lookup(skills_db: Dict[str, Any]) -> Tuple[Set[str], Dict[str, str], Dict[str, str]]:
    """
    Build efficient lookup structures from the skills database.

    Returns:
        Tuple of:
        - all_skills_lower: Set of all canonical skill names (lowercased)
        - lower_to_canonical: Map from lowercase to original casing
        - alias_to_canonical: Map from lowercase alias to canonical skill name
    """
    all_skills_lower = set()
    lower_to_canonical = {}
    alias_to_canonical = {}

    # Process all categories
    for category, skills in skills_db.get("categories", {}).items():
        for skill in skills:
            skill_lower = skill.lower()
            all_skills_lower.add(skill_lower)
            lower_to_canonical[skill_lower] = skill

    # Process aliases
    for alias, canonical in skills_db.get("aliases", {}).items():
        alias_to_canonical[alias.lower()] = canonical

    return all_skills_lower, lower_to_canonical, alias_to_canonical


# ─────────────────────────────────────────────────────────────
# KEYWORD-BASED SKILL MATCHING
# ─────────────────────────────────────────────────────────────

def extract_skills_keyword(
    text: str,
    skills_db: Dict[str, Any]
) -> List[str]:
    """
    Extract skills from text using direct keyword matching.

    Strategy:
        1. Build a lookup set of all canonical skill names + aliases
        2. For each skill, check if it appears in the text (case-insensitive)
        3. Use word boundary matching to avoid partial matches
           (e.g., "Java" shouldn't match inside "JavaScript")
        4. Resolve aliases to canonical names

    Args:
        text: Resume text to search for skills.
        skills_db: The loaded skills knowledge base dictionary.

    Returns:
        List of unique canonical skill names found in the text.
    """
    all_skills_lower, lower_to_canonical, alias_to_canonical = _build_skill_lookup(skills_db)

    found_skills = set()
    text_lower = text.lower()

    # Match canonical skill names
    for skill_lower, canonical in lower_to_canonical.items():
        # Use word boundary regex for precise matching
        # Escape special regex characters in the skill name
        escaped = re.escape(skill_lower)
        pattern = rf'(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])'
        if re.search(pattern, text_lower):
            found_skills.add(canonical)

    # Match aliases and resolve to canonical names
    for alias_lower, canonical in alias_to_canonical.items():
        escaped = re.escape(alias_lower)
        pattern = rf'(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])'
        if re.search(pattern, text_lower):
            found_skills.add(canonical)

    return sorted(list(found_skills))


# ─────────────────────────────────────────────────────────────
# SEMANTIC SKILL MATCHING
# ─────────────────────────────────────────────────────────────

class SemanticSkillMatcher:
    """
    Matches skills using Sentence Transformer embeddings.

    Pre-computes embeddings for all skills in the knowledge base,
    then compares resume text chunks against these embeddings
    using cosine similarity.
    """

    def __init__(
        self,
        skills_db: Dict[str, Any],
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.55
    ):
        """
        Initialize the semantic matcher.

        Args:
            skills_db: Loaded skills knowledge base.
            model_name: Sentence transformer model to use.
            threshold: Minimum cosine similarity score to consider a match.
        """
        self.threshold = threshold
        self.skills_db = skills_db

        if not SEMANTIC_AVAILABLE:
            self.model = None
            self.skill_embeddings = None
            self.skill_names = []
            return

        # Load the sentence transformer model
        self.model = SentenceTransformer(model_name)

        # Collect all canonical skill names
        self.skill_names = []
        for category, skills in skills_db.get("categories", {}).items():
            self.skill_names.extend(skills)
        self.skill_names = list(set(self.skill_names))

        # Pre-compute embeddings for all skills
        if self.skill_names:
            self.skill_embeddings = self.model.encode(
                self.skill_names,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        else:
            self.skill_embeddings = None

    def match(self, text: str) -> List[str]:
        """
        Find skills in text using semantic similarity.

        Strategy:
            1. Split text into sentences/chunks
            2. Encode each chunk
            3. Compare against pre-computed skill embeddings
            4. Return skills with similarity above threshold

        Args:
            text: Resume text to analyze.

        Returns:
            List of matched canonical skill names.
        """
        if self.model is None or self.skill_embeddings is None:
            return []

        # Split text into manageable chunks (by sentence/line)
        chunks = [
            chunk.strip()
            for chunk in re.split(r'[.\n;,|]', text)
            if chunk.strip() and len(chunk.strip()) > 3
        ]

        if not chunks:
            return []

        # Encode text chunks
        chunk_embeddings = self.model.encode(
            chunks,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # Compute cosine similarity between chunks and skills
        cosine_scores = util.cos_sim(chunk_embeddings, self.skill_embeddings)

        # Find skills above threshold
        matched_skills = set()
        for i in range(len(chunks)):
            for j in range(len(self.skill_names)):
                if cosine_scores[i][j].item() >= self.threshold:
                    matched_skills.add(self.skill_names[j])

        return sorted(list(matched_skills))


# ─────────────────────────────────────────────────────────────
# COMBINED SKILL EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_skills(
    text: str,
    skills_db: Dict[str, Any],
    semantic_matcher: Optional[SemanticSkillMatcher] = None,
    use_semantic: bool = True
) -> Dict[str, Any]:
    """
    Extract skills using both keyword and semantic matching.

    Combines results from both approaches, deduplicates, and
    categorizes skills by their knowledge base categories.

    Args:
        text: Resume text to analyze.
        skills_db: Loaded skills knowledge base.
        semantic_matcher: Pre-initialized SemanticSkillMatcher instance.
        use_semantic: Whether to use semantic matching (default True).

    Returns:
        Dictionary with:
        - "all": Complete deduplicated list of matched skills
        - "technical": Technical skills (programming, frameworks, etc.)
        - "soft": Soft skills
        - "by_category": Skills grouped by their KB category
    """
    # Keyword matching
    keyword_skills = set(extract_skills_keyword(text, skills_db))

    # Semantic matching
    semantic_skills = set()
    if use_semantic and semantic_matcher is not None:
        semantic_skills = set(semantic_matcher.match(text))

    # Union of both approaches
    all_skills = keyword_skills | semantic_skills

    # Categorize skills
    categories = skills_db.get("categories", {})
    by_category = {}
    technical = set()
    soft = set()

    # Build reverse lookup: skill -> category
    skill_to_category = {}
    for category, cat_skills in categories.items():
        for skill in cat_skills:
            skill_to_category[skill.lower()] = category

    for skill in all_skills:
        category = skill_to_category.get(skill.lower(), "other")
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(skill)

        if category == "soft_skills":
            soft.add(skill)
        else:
            technical.add(skill)

    # Sort within each category
    for cat in by_category:
        by_category[cat].sort()

    return {
        "all": sorted(list(all_skills)),
        "technical": sorted(list(technical)),
        "soft": sorted(list(soft)),
        "by_category": by_category
    }


if __name__ == "__main__":
    # Quick test
    skills_db = load_skills_database()

    sample_text = """
    Experienced software engineer proficient in Python, JavaScript, and TypeScript.
    Built web applications using React.js and Node.js with MongoDB backend.
    Deployed services on AWS using Docker and K8s (Kubernetes).
    Strong skills in ML, deep learning, and natural language processing.
    Excellent leadership and team management abilities.
    Certified AWS Solutions Architect and Scrum Master.
    """

    print("=== KEYWORD MATCHING ===")
    kw_skills = extract_skills_keyword(sample_text, skills_db)
    print(f"Found {len(kw_skills)} skills: {kw_skills}")

    print("\n=== COMBINED EXTRACTION ===")
    # Initialize semantic matcher (if available)
    matcher = SemanticSkillMatcher(skills_db) if SEMANTIC_AVAILABLE else None
    result = extract_skills(sample_text, skills_db, matcher)
    print(f"Total skills: {len(result['all'])}")
    print(f"Technical: {result['technical']}")
    print(f"Soft: {result['soft']}")
    print(f"\nBy Category:")
    for cat, skills in result['by_category'].items():
        print(f"  {cat}: {skills}")
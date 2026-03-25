"""
nlp_extractor.py
----------------
Local NLP-based skill extraction, section detection, and keyword density
analysis using spaCy. Runs entirely offline — no API calls needed.

Install spaCy model once with:
    python -m spacy download en_core_web_sm
"""

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated skills taxonomy
# (extend this list with domain-specific terms as needed)
# ---------------------------------------------------------------------------

SKILLS_TAXONOMY = {
    # ---------- Programming Languages ----------
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go",
    "rust", "kotlin", "swift", "scala", "r", "matlab", "julia", "ruby",
    "php", "bash", "shell", "powershell", "perl",

    # ---------- AI / ML / Data Science ----------
    "machine learning", "deep learning", "reinforcement learning",
    "natural language processing", "nlp", "computer vision", "llm",
    "large language model", "generative ai", "genai", "rag",
    "retrieval augmented generation", "fine-tuning", "prompt engineering",
    "transfer learning", "transformer", "bert", "gpt", "llama",
    "langchain", "llamaindex", "hugging face", "huggingface",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "xgboost", "lightgbm", "catboost", "spacy", "nltk",
    "opencv", "pillow", "pandas", "numpy", "scipy", "matplotlib",
    "seaborn", "plotly", "statsmodels",

    # ---------- Data Engineering ----------
    "apache spark", "spark", "hadoop", "kafka", "airflow",
    "dbt", "etl", "data pipeline", "data warehouse", "data lake",
    "snowflake", "databricks", "bigquery", "redshift", "hive",
    "flink", "nifi", "prefect", "luigi",

    # ---------- Databases ----------
    "sql", "mysql", "postgresql", "postgres", "sqlite",
    "mongodb", "redis", "elasticsearch", "cassandra", "dynamodb",
    "neo4j", "firebase", "supabase", "pinecone", "weaviate",
    "chroma", "faiss", "vector database", "milvus",

    # ---------- Cloud & DevOps ----------
    "aws", "gcp", "azure", "google cloud", "s3", "ec2", "lambda",
    "sagemaker", "vertex ai", "azure ml",
    "docker", "kubernetes", "k8s", "terraform", "ansible",
    "jenkins", "github actions", "gitlab ci", "ci/cd",
    "helm", "argocd", "istio",

    # ---------- Web & API ----------
    "rest", "rest api", "graphql", "fastapi", "flask", "django",
    "node.js", "express", "react", "vue", "angular", "next.js",
    "html", "css", "tailwind", "spring boot", "grpc",

    # ---------- Tools & Methodologies ----------
    "git", "github", "gitlab", "jira", "confluence", "agile",
    "scrum", "kanban", "devops", "mlops", "llmops",
    "unit testing", "pytest", "tdd", "bdd",
    "linux", "unix", "windows server",

    # ---------- Soft Skills ----------
    "communication", "leadership", "teamwork", "problem solving",
    "critical thinking", "project management", "mentoring",
    "collaboration", "time management", "presentation",

    # ---------- Analytics & BI ----------
    "tableau", "power bi", "looker", "metabase", "excel",
    "google analytics", "mixpanel", "amplitude", "a/b testing",
    "statistical analysis", "data visualization",
}

# ---------------------------------------------------------------------------
# Section heading patterns
# ---------------------------------------------------------------------------

SECTION_PATTERNS = {
    "summary": r"\b(summary|profile|objective|about me|about)\b",
    "experience": r"\b(experience|work experience|employment|career|professional experience)\b",
    "education": r"\b(education|academic|qualification|degree|university|college|school)\b",
    "skills": r"\b(skills|technical skills|competencies|expertise|technologies)\b",
    "projects": r"\b(projects|personal projects|portfolio|side projects)\b",
    "certifications": r"\b(certifications?|certificates?|credentials?|licenses?)\b",
    "achievements": r"\b(achievements?|awards?|honors?|accomplishments?)\b",
    "publications": r"\b(publications?|papers?|research|journals?)\b",
}


# ---------------------------------------------------------------------------
# spaCy loader (lazy — only loads model if spaCy is installed)
# ---------------------------------------------------------------------------

def _load_spacy():
    """Lazily load spaCy model. Returns None if not available."""
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            return None
    except ImportError:
        logger.warning("spaCy not installed. NLP extraction will be skipped.")
        return None


_nlp_model = None  # module-level singleton


def _get_nlp():
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = _load_spacy()
    return _nlp_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_skills_nlp(text: str) -> Dict[str, List[str]]:
    """
    Extract skills from resume text using spaCy + taxonomy matching.

    Returns a dict with:
        {
            "taxonomy_matched": [...],   # Skills from SKILLS_TAXONOMY found in text
            "nlp_entities":     [...],   # Named entities (ORG, PRODUCT, NORP) from spaCy
            "all_skills":       [...],   # Union of both, deduplicated & sorted
        }
    """
    text_lower = text.lower()

    # 1. Taxonomy matching (fast, deterministic)
    taxonomy_matched = []
    for skill in SKILLS_TAXONOMY:
        # Use word-boundary regex for multi-word skills
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            taxonomy_matched.append(skill.title() if len(skill.split()) == 1 else skill.title())

    # 2. spaCy entity / noun chunk extraction
    nlp_entities: List[str] = []
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text[:5000])  # limit for performance
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "NORP", "GPE"} and len(ent.text) > 2:
                candidate = ent.text.strip()
                if candidate not in nlp_entities:
                    nlp_entities.append(candidate)

    # 3. Merge and deduplicate
    all_skills = sorted(set(taxonomy_matched + nlp_entities))

    return {
        "taxonomy_matched": sorted(set(taxonomy_matched)),
        "nlp_entities": sorted(set(nlp_entities)),
        "all_skills": all_skills,
    }


def extract_sections(text: str) -> Dict[str, str]:
    """
    Detect resume sections using heuristic heading patterns.

    Returns:
        Dict mapping section name → content text snippet (first 500 chars).
    """
    detected: Dict[str, str] = {}
    lines = text.splitlines()
    current_section = "preamble"
    section_content: Dict[str, List[str]] = {"preamble": []}

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Check if this line is a section heading
        matched_section = None
        for section_name, pattern in SECTION_PATTERNS.items():
            if re.search(pattern, line_stripped, re.IGNORECASE):
                # Likely a heading if it's short (< 60 chars) and not a sentence
                if len(line_stripped) < 60 and not line_stripped.endswith("."):
                    matched_section = section_name
                    break

        if matched_section:
            current_section = matched_section
            if matched_section not in section_content:
                section_content[matched_section] = []
        else:
            if current_section not in section_content:
                section_content[current_section] = []
            section_content[current_section].append(line_stripped)

    for section, lines_list in section_content.items():
        content = " ".join(lines_list).strip()
        if content:
            detected[section] = content[:500]

    return detected


def compute_keyword_density(text: str, keywords: List[str]) -> Dict[str, int]:
    """
    Count occurrences of each keyword (case-insensitive) in the resume text.

    Args:
        text (str): Resume full text.
        keywords (list): List of keywords to search for.

    Returns:
        Dict mapping keyword → count (only includes keywords with count > 0).
    """
    text_lower = text.lower()
    density: Dict[str, int] = {}

    for kw in keywords:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        count = len(re.findall(pattern, text_lower))
        if count > 0:
            density[kw] = count

    return dict(sorted(density.items(), key=lambda x: x[1], reverse=True))


def compute_skill_overlap(
    detected_skills: List[str],
    required_skills: List[str],
) -> Tuple[List[str], List[str], float]:
    """
    Compare detected skills against a required skills list.

    Returns:
        ( matched_skills, missing_skills, match_percentage )
    """
    detected_lower = {s.lower() for s in detected_skills}
    matched = []
    missing = []

    for skill in required_skills:
        if skill.lower() in detected_lower:
            matched.append(skill)
        else:
            missing.append(skill)

    total = len(required_skills)
    pct = (len(matched) / total * 100) if total > 0 else 0.0

    return matched, missing, round(pct, 1)

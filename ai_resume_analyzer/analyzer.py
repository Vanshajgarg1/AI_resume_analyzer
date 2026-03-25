"""
analyzer.py
-----------
Core AI analysis logic using LangChain + Groq.

Two modes:
  1. analyze(resume_text, job_role)         — standard resume analysis
  2. analyze_with_jd(resume_text, job_role, job_description) — JD comparison mode
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt: Standard Analysis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert HR consultant and ATS (Applicant Tracking System) specialist
with 15+ years of experience reviewing resumes across all industries.

Your job is to analyze resumes and provide honest, actionable, structured feedback.
Always respond ONLY with a valid JSON object — no prose, no markdown fences, just raw JSON.
"""

ANALYSIS_PROMPT = """
Analyze the following resume for the target job role: **{job_role}**

--- RESUME START ---
{resume_text}
--- RESUME END ---

Return a JSON object with EXACTLY these keys (all are required):

{{
  "candidate_name": "<name if found, else 'Not found'>",
  "years_of_experience": "<estimated experience string, e.g. '3-5 years'>",
  "overall_evaluation": "<2-4 sentence high-level assessment>",

  "ats_score": <integer 0-100>,
  "ats_score_explanation": "<2-3 sentences explaining this score>",
  "ats_breakdown": {{
    "keyword_match": <integer 0-25>,
    "formatting": <integer 0-25>,
    "section_completeness": <integer 0-25>,
    "contact_info": <integer 0-25>
  }},

  "skills_detected": ["<skill>"],
  "missing_skills": ["<skill missing for {job_role}>"],

  "strengths": ["<strength>"],

  "section_feedback": {{
    "summary":    "<one-sentence tip or 'Not found'>",
    "experience": "<one-sentence tip or 'Not found'>",
    "education":  "<one-sentence tip or 'Not found'>",
    "skills":     "<one-sentence tip or 'Not found'>"
  }},

  "suggested_improvements": ["<specific actionable improvement>"],

  "keyword_optimization_tips": [
    "<tip: replace 'X' with 'Y' to better match ATS for {job_role}>"
  ],

  "keywords_to_add": ["<ATS keyword absent from resume>"],

  "recommended_job_titles": ["<alternative job title>"]
}}

Rules:
- ats_breakdown sub-scores must sum to ats_score (approximately).
- missing_skills: only skills relevant to {job_role}.
- keyword_optimization_tips: give 3-5 concrete before/after phrasing examples.
- suggested_improvements: be specific (not generic advice).
- Return valid JSON only.
"""

# ---------------------------------------------------------------------------
# Prompt: JD-Comparison Mode (adds jd_match_score, jd_matched_keywords, jd_missing_keywords)
# ---------------------------------------------------------------------------

JD_ANALYSIS_PROMPT = """
Analyze the following resume against both the target job role AND the provided job description.

**Target Job Role:** {job_role}

--- RESUME START ---
{resume_text}
--- RESUME END ---

--- JOB DESCRIPTION START ---
{job_description}
--- JOB DESCRIPTION END ---

Return a JSON object with EXACTLY these keys:

{{
  "candidate_name": "<name if found, else 'Not found'>",
  "years_of_experience": "<estimated experience string>",
  "overall_evaluation": "<2-4 sentence high-level assessment>",

  "ats_score": <integer 0-100>,
  "ats_score_explanation": "<2-3 sentences explaining score>",
  "ats_breakdown": {{
    "keyword_match": <integer 0-25>,
    "formatting": <integer 0-25>,
    "section_completeness": <integer 0-25>,
    "contact_info": <integer 0-25>
  }},

  "jd_match_score": <integer 0-100, how well resume matches the job description>,
  "jd_matched_keywords": ["<keyword from JD found in resume>"],
  "jd_missing_keywords": ["<important keyword in JD but absent from resume>"],

  "skills_detected": ["<skill>"],
  "missing_skills": ["<skill missing for {job_role}>"],

  "strengths": ["<strength>"],

  "section_feedback": {{
    "summary":    "<tip or 'Not found'>",
    "experience": "<tip or 'Not found'>",
    "education":  "<tip or 'Not found'>",
    "skills":     "<tip or 'Not found'>"
  }},

  "suggested_improvements": ["<specific actionable improvement>"],

  "keyword_optimization_tips": [
    "<concrete before/after phrasing suggestion>"
  ],

  "keywords_to_add": ["<ATS keyword absent from resume>"],

  "recommended_job_titles": ["<alternative job title>"]
}}

Rules:
- jd_match_score: base this strictly on the supplied job description text.
- jd_matched_keywords: important role-relevant terms from the JD present in the resume.
- jd_missing_keywords: important terms from the JD completely absent from the resume.
- keyword_optimization_tips: 3-5 concrete examples.
- Return valid JSON only.
"""


# ---------------------------------------------------------------------------
# ResumeAnalyzer
# ---------------------------------------------------------------------------

class ResumeAnalyzer:
    """
    Orchestrates resume analysis using LangChain LCEL chains and OpenAI GPT.

    Supports two modes:
    - Standard: analyze(resume_text, job_role)
    - JD Comparison: analyze_with_jd(resume_text, job_role, job_description)
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2):
        """
        Initialise the analyser.

        Args:
            model_name (str): OpenAI model. Defaults to gpt-4o-mini.
            temperature (float): Lower = more deterministic JSON output.

        Raises:
            EnvironmentError: If OPENAI_API_KEY is missing.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Please add it to your .env file or enter it in the sidebar."
            )

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

        # Standard chain
        self._standard_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", ANALYSIS_PROMPT),
        ])
        self._standard_chain = self._standard_prompt | self.llm | JsonOutputParser()

        # JD-comparison chain
        self._jd_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", JD_ANALYSIS_PROMPT),
        ])
        self._jd_chain = self._jd_prompt | self.llm | JsonOutputParser()

        logger.info(f"ResumeAnalyzer ready — model: {model_name}")

    def analyze(self, resume_text: str, job_role: str) -> dict:
        """
        Standard resume analysis (no job description).

        Args:
            resume_text (str): Cleaned resume text.
            job_role (str): Target job role.

        Returns:
            dict: Structured analysis result.
        """
        logger.info(f"Standard analysis — role: '{job_role}'")
        return self._invoke(
            self._standard_chain,
            {"resume_text": resume_text, "job_role": job_role},
        )

    def analyze_with_jd(
        self,
        resume_text: str,
        job_role: str,
        job_description: str,
    ) -> dict:
        """
        JD-comparison analysis — scores resume against a pasted job description.

        Args:
            resume_text (str): Cleaned resume text.
            job_role (str): Target job role.
            job_description (str): Full job description text pasted by the user.

        Returns:
            dict: Structured analysis result including jd_match_score and JD keywords.
        """
        logger.info(f"JD-comparison analysis — role: '{job_role}'")
        return self._invoke(
            self._jd_chain,
            {
                "resume_text": resume_text,
                "job_role": job_role,
                "job_description": job_description[:4000],  # Truncate very long JDs
            },
        )

    def _invoke(self, chain, inputs: dict) -> dict:
        """Internal helper to invoke a chain with error handling."""
        try:
            result = chain.invoke(inputs)
        except Exception as e:
            logger.error(f"LLM chain failed: {e}")
            raise RuntimeError(
                f"AI analysis failed: {e}\n\nCheck your API key and internet connection."
            ) from e

        required_keys = ["overall_evaluation", "ats_score", "skills_detected", "missing_skills"]
        missing = [k for k in required_keys if k not in result]
        if missing:
            logger.warning(f"LLM response missing keys: {missing}")

        logger.info(f"Analysis complete. ATS={result.get('ats_score', '?')}")
        return result


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def analyze_resume(
    resume_text: str,
    job_role: str,
    job_description: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    One-call convenience wrapper.

    Args:
        resume_text: Extracted resume text.
        job_role: Target job role.
        job_description: Optional JD text (enables JD-comparison mode).
        model: OpenAI model name.
    """
    analyzer = ResumeAnalyzer(model_name=model)
    if job_description and job_description.strip():
        return analyzer.analyze_with_jd(resume_text, job_role, job_description)
    return analyzer.analyze(resume_text, job_role)

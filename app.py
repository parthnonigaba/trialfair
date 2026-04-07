#!/usr/bin/env python3
"""
TrialFair API Server

FastAPI backend that provides:
- POST /api/analyze - Analyze eligibility text (gates + LLM explanation)
- GET /api/lookup/{nct_id} - Lookup trial from AACT data
- GET /api/stats - Aggregate statistics
- GET / - Serve frontend

Environment variables:
- OPENAI_API_KEY: Required for LLM explanations
"""

import os
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

# ============================================================
# CONFIGURATION
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Paths - use resolve() to get absolute paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "final_data"
STATIC_DIR = BASE_DIR / "static"

# Data files
SQLITE_DB_FILE = DATA_DIR / "trialfair.db"  # SQLite database (preferred, fast)
AACT_MASTER_FILE = DATA_DIR / "aact_master.csv"  # Fallback CSV
EU_MASTER_FILE = DATA_DIR / "eu_master.csv"  # Fallback CSV
EXCEPTIONS_FILE = DATA_DIR / "trialfair_exceptions.json"

# Debug: Print paths on startup
print(f"[TrialFair] BASE_DIR: {BASE_DIR}")
print(f"[TrialFair] DATA_DIR: {DATA_DIR}")
print(f"[TrialFair] SQLITE_DB exists: {SQLITE_DB_FILE.exists()}")
print(f"[TrialFair] AACT_CSV exists: {AACT_MASTER_FILE.exists()}")

# ============================================================
# GATE DETECTION (from your trialfair_gates.py)
# ============================================================

GATE_PATTERNS = {
    "pregnancy_test_required": {
        # Tightened - require explicit "test required" or "negative test" context
        # Avoid matching simple mentions of pregnancy tests in exclusion criteria
        "pattern": re.compile(
            r'\b(negative\s+)(serum|urine|blood)?\s*(β[-\s]?hcg|hcg|pregnancy)\s+test\b'
            r'|\bpregnancy\s+test(ing)?\s+(is\s+)?(required|must|needed|necessary)\b'
            r'|\bmust\s+have\s+(a\s+)?(negative\s+)?pregnancy\s+test\b'
            r'|\brequire[sd]?\s+(a\s+)?(negative\s+)?pregnancy\s+test\b',
            re.IGNORECASE
        ),
        "severity": 2,
        "label": "Pregnancy Test Required",
    },
    "pregnancy_exclusion": {
        # Expanded - catch more variations of pregnancy exclusion
        "pattern": re.compile(
            r'\b(pregnan(t|cy)|breastfeeding|nursing|lactating)\b[^.]{0,50}\b(exclude[ds]?|not\s+eligible|ineligible|cannot\s+participate|must\s+not\s+be|will\s+be\s+excluded|exclusion)\b'
            r'|\b(exclude[ds]?|excluded?|ineligible|exclusion)[^.]{0,50}\b(pregnan|breastfeeding|nursing|lactating)\b'
            r'|\bwomen\s+who\s+are\s+pregnant\b'
            r'|\bpregnant\s+or\s+(breastfeeding|nursing|lactating)\b'
            r'|\bif\s+pregnant\b'
            r'|\bno\s+pregnan(t|cy)\b'
            r'|\bpregnancy\s+(is\s+)?(an?\s+)?(exclusion|contraindication)\b',
            re.IGNORECASE
        ),
        "severity": 3,
        "label": "Pregnancy Exclusion",
    },
    "wocbp_contraception_requirement": {
        # Expanded - catch more contraception requirements
        "pattern": re.compile(
            r'\b(wocbp|women\s+of\s+child[-\s]?bearing\s+potential|female[s]?\s+of\s+child[-\s]?bearing\s+potential|wcbp)\b'
            r'|\b(contracepti|birth\s+control)\b[^.]{0,80}\b(require|must|need|use|agree|practice)\b'
            r'|\b(require|must|need|agree)[^.]{0,80}\b(contracepti|birth\s+control)\b'
            r'|\b(adequate|effective|acceptable|reliable)\s+(method\s+of\s+)?(contraception|birth\s+control)\b'
            r'|\bpractice[sd]?\s+(effective\s+)?contraception\b'
            r'|\b(two|2)\s+(forms?|methods?)\s+of\s+(contraception|birth\s+control)\b'
            r'|\bsurgically\s+sterile\s+or\s+(agree|willing)\b'
            r'|\bwilling\s+to\s+use\s+(contraception|birth\s+control)\b'
            r'|\bchild[-\s]?bearing\s+potential\s+must\b',
            re.IGNORECASE
        ),
        "severity": 2,
        "label": "WOCBP Contraception Requirement",
    },
    "consent_capacity_restriction": {
        # Expanded - catch cognitive/capacity restrictions
        "pattern": re.compile(
            r'\b(unable|not\s+able|inability|incapable)\s+(to\s+)?(provide|give|sign|understand)[^.]{0,30}(informed\s+)?consent\b'
            r'|\b(capacity|ability)\s+to\s+(provide|give|understand)[^.]{0,20}consent\b'
            r'|\b(cognitive\s+impairment|cognitively\s+impaired|dementia|mental\s+impairment|mentally\s+impaired)\b'
            r'|\b(lack[s]?\s+(the\s+)?capacity|lacks?\s+capacity)\b'
            r'|\blegal(ly)?\s+authorized\s+representative\b'
            r'|\bguardian\s+(consent|required)\b'
            r'|\b(able|ability)\s+to\s+(understand|comprehend)[^.]{0,40}(study|trial|consent|procedure)\b'
            r'|\bmini[-\s]?mental|mmse|moca\s+score\b'
            r'|\bintellectual\s+(capacity|incapacity|disability)\b'
            r'|\bpsychiatric\s+(illness|condition|disorder)[^.]{0,30}(prevent|preclude|interfere)\b',
            re.IGNORECASE
        ),
        "severity": 3,
        "label": "Consent/Capacity Restriction",
    },
    "language_requirement": {
        # Expanded - catch more language requirements
        "pattern": re.compile(
            r'\b(must|able|ability)\s+(to\s+)?(speak|read|write|understand|communicate)[^.]{0,30}(english|spanish|french|german|chinese|mandarin|japanese|korean|urdu|hindi)\b'
            r'|\b(english|spanish|french|german|chinese|mandarin|urdu)[-\s]?(speaking|proficien|fluent|literate)\b'
            r'|\b(fluent|proficient)\s+(in\s+)?(english|spanish|french|german)\b'
            r'|\b(speak[s]?|read[s]?|understand[s]?)\s+(english|spanish|french|german)\b'
            r'|\blanguage\s+(barrier|requirement|skill)\b'
            r'|\b(complete|fill\s+out)[^.]{0,30}(questionnaire|diary|form)[^.]{0,20}(english|spanish)\b'
            r'|\bnon[-\s]?english\s+speaking\s+(excluded?|not\s+eligible)\b'
            r'|\b(english|spanish)\s+is\s+(spoken|required)\b'
            r'|\binsufficient\s+language\b',
            re.IGNORECASE
        ),
        "severity": 2,
        "label": "Language Requirement",
    },
    "technology_access_requirement": {
        # Kept similar but added context requirements
        "pattern": re.compile(
            r'\b(smartphone|mobile\s+phone|cell\s*phone|internet\s+access|email\s+(address|access)|computer\s+access|mobile\s+app|tablet)\b[^.]{0,50}\b(require|access|have|own|use)\b'
            r'|\b(require|must\s+have|need)[^.]{0,50}\b(smartphone|mobile\s+phone|internet|email|computer)\b'
            r'|\b(access\s+to|own\s+a)\s+(smartphone|computer|internet|tablet)\b'
            r'|\btelehealth\s+(visit|capable|platform)\b'
            r'|\bvideo\s+(visit|call|conferenc)\b[^.]{0,30}(require|capable|access)\b'
            r'|\bzoom\s+(access|capable|visit)\b',
            re.IGNORECASE
        ),
        "severity": 2,
        "label": "Technology Access Requirement",
    },
    "insurance_requirement": {
        # Expanded - catch health system affiliations
        "pattern": re.compile(
            r'\b(health\s+)?insurance\b[^.]{0,40}\b(require|must|need|covered)\b'
            r'|\b(require|must|need)[^.]{0,40}\b(health\s+)?insurance\b'
            r'|\baffiliated\s+with\s+(the\s+)?(health|social)\s+(insurance|security)\b'
            r'|\b(insured|coverage)\s+(require|must|need)\b'
            r'|\bmedicaid|medicare\s+(eligible|enrolled|beneficiar)\b',
            re.IGNORECASE
        ),
        "severity": 3,
        "label": "Insurance Requirement",
    },
    "travel_transport_requirement": {
        # Expanded patterns
        "pattern": re.compile(
            r'\b(able|willing|ability)\s+to\s+travel\b'
            r'|\btravel\s+to\s+(the\s+)?(study\s+)?(site|center|clinic)\b'
            r'|\btransportation\s+(to|access|available)\b'
            r'|\bmust\s+live\s+within\b'
            r'|\b(reside|live)[^.]{0,30}(miles?|km|kilometers?|minutes?)\s+(of|from)\b'
            r'|\bgeographic(al)?\s+(proximity|location)\b'
            r'|\bcommut(e|ing)\s+(distance|time)\b'
            r'|\bable\s+to\s+attend\s+(all\s+)?(study\s+)?visits\b',
            re.IGNORECASE
        ),
        "severity": 2,
        "label": "Travel/Transport Requirement",
    },
    "residency_citizenship_requirement": {
        # Expanded patterns
        "pattern": re.compile(
            r'\b(resident|reside)\s+(of|in|within)\b'
            r'|\bresidency\s+(require|status)\b'
            r'|\b(citizen(ship)?|permanent\s+resident|legal\s+resident)\b'
            r'|\b(US|U\.S\.|united\s+states)\s+citizen\b'
            r'|\bcitizen[s]?\s+only\b'
            r'|\bresident[s]?\s+only\b'
            r'|\bmust\s+reside\s+(in|within)\b'
            r'|\bIP\s+address[^.]{0,30}(united\s+states|US|U\.S\.)\b'
            r'|\blegal(ly)?\s+(authorized|permitted)\s+to\s+(work|reside)\b'
            r'|\bnational[s]?\s+only\b',
            re.IGNORECASE
        ),
        "severity": 2,
        "label": "Residency/Citizenship Requirement",
    },
    "substance_use_exclusion": {
        # Tightened - require exclusion context, not just mentions
        "pattern": re.compile(
            r'\b(alcohol|substance|drug)\s+(abuse|dependence|addict|disorder)\b[^.]{0,50}\b(exclude|history|current|active)\b'
            r'|\b(exclude|history\s+of|current|active)[^.]{0,50}\b(alcohol|substance|drug)\s+(abuse|dependence|addict)\b'
            r'|\b(illicit|illegal)\s+drug\s+(use|abuse)\b'
            r'|\bpositive\s+(drug|urine|toxicology)\s+(screen|test)\b[^.]{0,30}(exclude|ineligible)\b'
            r'|\b(chronic|current)\s+alcohol(ism)?\b'
            r'|\bsubstance\s+use\s+disorder\b'
            r'|\balcohol\s+(misuse|abuse|dependence)\b',
            re.IGNORECASE
        ),
        "severity": 2,
        "label": "Substance Use Exclusion",
    },
    "health_status_exclusion": {
        # Tightened - require "healthy volunteer" context more strictly
        "pattern": re.compile(
            r'\bhealthy\s+(volunteer[s]?|subject[s]?|participant[s]?|individual[s]?|adult[s]?)\b'
            r'|\b(good|excellent)\s+general\s+health\b'
            r'|\bgenerally\s+healthy\b'
            r'|\bno\s+(significant|major|chronic|serious)\s+(illness|disease|medical\s+condition)\b'
            r'|\bfree\s+(of|from)\s+(significant\s+)?(illness|disease)\b',
            re.IGNORECASE
        ),
        "severity": 1,
        "label": "Health Status Requirement",
    },
}

DESIGN_PATTERNS = {
    "healthy_volunteer_design": {
        "pattern": re.compile(r'\bhealthy\s+(volunteers?|subjects?)\b', re.IGNORECASE),
        "label": "Healthy Volunteer Design",
    },
}


def extract_evidence(text: str, pattern: re.Pattern, window: int = 100) -> Optional[str]:
    """Extract evidence snippet around pattern match."""
    match = pattern.search(text)
    if not match:
        return None
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return re.sub(r'\s+', ' ', snippet)


def detect_gates(eligibility_text: str) -> Dict[str, Any]:
    """Detect all gates in eligibility text."""
    normalized = re.sub(r'\s+', ' ', eligibility_text)
    
    gates = []
    flags = {}
    
    # Check gate patterns
    for gate_id, config in GATE_PATTERNS.items():
        if config["pattern"].search(normalized):
            gates.append({
                "type": gate_id,
                "label": config["label"],
                "severity": config["severity"],
                "evidence": extract_evidence(normalized, config["pattern"]),
            })
            flags[f"gate_{gate_id}"] = 1
        else:
            flags[f"gate_{gate_id}"] = 0
    
    # Check design patterns (not penalized)
    for design_id, config in DESIGN_PATTERNS.items():
        if config["pattern"].search(normalized):
            flags[f"design_{design_id}"] = 1
        else:
            flags[f"design_{design_id}"] = 0
    
    # Compute gating score (sum of severities for gates, not design)
    gating_score = sum(g["severity"] for g in gates)
    flags["gating_score"] = gating_score
    
    return {
        "gates": gates,
        "flags": flags,
        "gating_score": gating_score,
    }


# ============================================================
# LLM-ENHANCED GATE DETECTION (HYBRID APPROACH)
# ============================================================

LLM_GATE_VERIFICATION_PROMPT = """You are analyzing clinical trial eligibility criteria for participation barriers (gates).

Given the eligibility text below, identify ALL gates that apply. Be precise - only include gates that are EXPLICITLY stated or strongly implied.

GATE TYPES (use these exact names):
- pregnancy_test_required: Explicitly requires pregnancy test (not just mentions pregnancy)
- pregnancy_exclusion: Explicitly excludes pregnant/breastfeeding women
- wocbp_contraception_requirement: Requires contraception for women of childbearing potential
- consent_capacity_restriction: Requires cognitive ability to consent (excludes dementia, cognitive impairment)
- language_requirement: Requires specific language proficiency (e.g., "must speak English")
- technology_access_requirement: Requires smartphone, internet, computer, email access
- insurance_requirement: Requires health insurance coverage
- travel_transport_requirement: Requires ability to travel to site, transportation, proximity
- residency_citizenship_requirement: Requires citizenship, residency, or legal status
- substance_use_exclusion: Excludes based on drug/alcohol abuse history
- health_status_exclusion: Requires "healthy volunteers" or excludes based on general health

RULES:
1. Only include gates EXPLICITLY stated or strongly implied
2. Standard informed consent is NOT consent_capacity_restriction unless cognitive ability is mentioned
3. Medical exclusions for safety (e.g., "no liver disease" in a drug trial) are NOT health_status_exclusion
4. Age restrictions alone are NOT gates
5. Be conservative - when in doubt, don't include it

ELIGIBILITY TEXT:
{eligibility_text}

Respond with ONLY a JSON array of gate type strings, e.g.: ["pregnancy_exclusion", "language_requirement"]
If no gates apply, respond: []"""


async def detect_gates_with_llm(eligibility_text: str, openai_client=None) -> Dict[str, Any]:
    """
    Hybrid gate detection: regex first, then LLM verification.
    Falls back to regex-only if no OpenAI client available.
    """
    # Step 1: Regex detection (fast, high recall)
    regex_result = detect_gates(eligibility_text)
    
    # If no OpenAI client, return regex results
    if not openai_client:
        return regex_result
    
    # Step 2: LLM verification (slower, higher precision)
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": LLM_GATE_VERIFICATION_PROMPT.format(eligibility_text=eligibility_text[:8000])}
            ],
            temperature=0,
            max_tokens=200,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        import json
        llm_gates = json.loads(content)
        
        # Validate gate names
        valid_gate_types = set(GATE_PATTERNS.keys())
        llm_gates = [g for g in llm_gates if g in valid_gate_types]
        
        # Build gates list from LLM results
        gates = []
        flags = {}
        
        for gate_id in llm_gates:
            config = GATE_PATTERNS[gate_id]
            gates.append({
                "type": gate_id,
                "label": config["label"],
                "severity": config["severity"],
                "evidence": extract_evidence(eligibility_text, config["pattern"]) or "LLM detected",
            })
            flags[f"gate_{gate_id}"] = 1
        
        # Set flags for gates not detected
        for gate_id in GATE_PATTERNS:
            if f"gate_{gate_id}" not in flags:
                flags[f"gate_{gate_id}"] = 0
        
        # Design patterns (still use regex)
        normalized = re.sub(r'\s+', ' ', eligibility_text)
        for design_id, config in DESIGN_PATTERNS.items():
            if config["pattern"].search(normalized):
                flags[f"design_{design_id}"] = 1
            else:
                flags[f"design_{design_id}"] = 0
        
        gating_score = sum(g["severity"] for g in gates)
        flags["gating_score"] = gating_score
        
        return {
            "gates": gates,
            "flags": flags,
            "gating_score": gating_score,
            "detection_method": "hybrid_llm",
        }
        
    except Exception as e:
        # Fall back to regex on any error
        print(f"LLM gate detection failed, using regex: {e}")
        return regex_result


# ============================================================
# R-INDEX COMPUTATION
# ============================================================

# Population references by region
POP_REFERENCES = {
    "US": {
        "sex_male": 0.4948, "sex_female": 0.5052,
        "age_lt18": 0.2169, "age_18_44": 0.3600, "age_45_64": 0.2460, "age_65plus": 0.1771,
        "race_white": 0.5713, "race_black": 0.1181, "race_asian": 0.0590, "race_other": 0.0570, "race_hispanic": 0.1945,
    },
    "EU": {
        "sex_male": 0.4888, "sex_female": 0.5112,
        "age_lt18": 0.1801, "age_18_44": 0.3266, "age_45_64": 0.2802, "age_65plus": 0.2131,
        # No race data for EU
    },
    "India": {
        "sex_male": 0.5158, "sex_female": 0.4842,
        "age_lt18": 0.3011, "age_18_44": 0.4383, "age_45_64": 0.1903, "age_65plus": 0.0703,
        # No race data for India
    },
    "China": {
        "sex_male": 0.5096, "sex_female": 0.4904,
        "age_lt18": 0.1985, "age_18_44": 0.3670, "age_45_64": 0.2891, "age_65plus": 0.1454,
        # No race data for China
    },
}

# Default to US for backward compatibility
US_POP_REF = POP_REFERENCES["US"]


def compute_rindex(trial_dist: Dict[str, float], pop_ref: Dict[str, float], keys: List[str]) -> Optional[float]:
    """Compute R-index using TVD."""
    trial_vals = [trial_dist.get(k, 0) for k in keys]
    pop_vals = [pop_ref.get(k, 0) for k in keys]
    
    trial_sum = sum(trial_vals)
    pop_sum = sum(pop_vals)
    
    if trial_sum == 0 or pop_sum == 0:
        return None
    
    trial_norm = [v / trial_sum for v in trial_vals]
    pop_norm = [v / pop_sum for v in pop_vals]
    
    tvd = sum(abs(t - p) for t, p in zip(trial_norm, pop_norm)) / 2
    return max(0.0, min(1.0, 1.0 - tvd))


def compute_all_rindex(demographics: Dict[str, float], region: str = "US") -> Dict[str, Optional[float]]:
    """Compute R-index for all dimensions against specified region's population."""
    pop_ref = POP_REFERENCES.get(region, POP_REFERENCES["US"])
    result = {"r_sex": None, "r_age": None, "r_race": None, "r_overall": None, "region": region}
    
    # Sex
    if demographics.get("sex_female") is not None or demographics.get("sex_male") is not None:
        sex_dist = {
            "sex_female": demographics.get("sex_female", 0),
            "sex_male": demographics.get("sex_male", 0),
        }
        result["r_sex"] = compute_rindex(sex_dist, pop_ref, ["sex_female", "sex_male"])
    
    # Age
    age_keys = ["age_lt18", "age_18_44", "age_45_64", "age_65plus"]
    if any(demographics.get(k) is not None for k in age_keys):
        age_dist = {k: demographics.get(k, 0) for k in age_keys}
        result["r_age"] = compute_rindex(age_dist, pop_ref, age_keys)
    
    # Race (only for US - other regions don't track this)
    if region == "US":
        race_keys = ["race_white", "race_black", "race_asian", "race_other"]
        if any(demographics.get(k) is not None for k in race_keys):
            race_dist = {k: demographics.get(k, 0) for k in race_keys}
            result["r_race"] = compute_rindex(race_dist, pop_ref, race_keys)
    
    # Overall (average of available dimensions)
    valid = [v for v in [result["r_sex"], result["r_age"], result["r_race"]] if v is not None]
    result["r_overall"] = sum(valid) / len(valid) if valid else None
    
    return result


# ============================================================
# EXCEPTION HANDLING
# ============================================================

@lru_cache(maxsize=1)
def load_exceptions() -> Dict[str, Any]:
    """Load exception ontology."""
    if EXCEPTIONS_FILE.exists():
        return json.loads(EXCEPTIONS_FILE.read_text())
    return {}


def check_exceptions(gates: List[Dict], metadata: Dict[str, Any]) -> List[Dict]:
    """Check if any gates have exceptions that apply."""
    exceptions = load_exceptions()
    phase = str(metadata.get("phase", "")).lower()
    condition = str(metadata.get("condition", "")).lower()
    
    # Detect conditions
    detected_conditions = set()
    if "phase 1" in phase or "phase1" in phase:
        detected_conditions.add("phase_1")
    if any(k in condition for k in ["cancer", "carcinoma", "tumor", "oncolog"]):
        detected_conditions.add("oncology")
    
    # Enrich gates with exception info
    for gate in gates:
        gate_key = f"gate_{gate['type']}"
        exc_info = exceptions.get(gate_key, {})
        
        exception_matches = []
        for allowed in exc_info.get("allowed_if", []):
            if allowed.get("condition") in detected_conditions:
                exception_matches.append(allowed)
        
        gate["exception_applies"] = len(exception_matches) > 0
        gate["exception_matches"] = exception_matches
    
    return gates


# ============================================================
# LLM AUDITOR (OpenAI)
# ============================================================

def get_openai_client():
    """Get OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    if api_key == "YOUR_API_KEY_HERE":
        return None
    return openai.OpenAI(api_key=api_key)


def get_async_openai_client():
    """Get async OpenAI client for hybrid detection."""
    api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    if api_key == "YOUR_API_KEY_HERE":
        return None
    return openai.AsyncOpenAI(api_key=api_key)


# Initialize async OpenAI client at module level
OPENAI_CLIENT = get_async_openai_client()


LLM_SYSTEM_PROMPT = """You are a clinical trial eligibility criteria auditor. Your role is to:
1. EXPLAIN why specific eligibility criteria appear
2. ASSESS whether wording is broader than medically necessary
3. SUGGEST concrete rewording that maintains safety while improving access

You are given eligibility criteria text and detected "gates" (potential barriers to enrollment).

For each gate, provide:
- explanation: 1-2 sentences on why this criterion likely exists
- over_broad: true/false - is the wording broader than necessary?
- suggested_revision: concrete rewording, or null if appropriate as-is

NEVER use the word "bias". Focus on clarity, scope, and access.

Respond ONLY with valid JSON in this exact format:
{
  "gate_analyses": [
    {
      "gate_type": "...",
      "explanation": "...",
      "over_broad": true/false,
      "suggested_revision": "..." or null
    }
  ],
  "overall_summary": "1-2 sentence summary of the criteria's inclusivity"
}"""


def call_llm_auditor(eligibility_text: str, gates: List[Dict], metadata: Dict[str, Any]) -> Optional[Dict]:
    """Call OpenAI to get explanations and rewrites."""
    client = get_openai_client()
    if not client:
        return None
    
    # Build user prompt
    gate_list = "\n".join([
        f"- {g['label']}: {g.get('evidence', 'No evidence extracted')}"
        for g in gates
    ])
    
    user_prompt = f"""## Trial Information
- Phase: {metadata.get('phase', 'Unknown')}
- Condition: {metadata.get('condition', 'Not specified')}

## Eligibility Criteria
{eligibility_text[:4000]}

## Detected Gates
{gate_list}

Analyze each gate and provide your response as JSON."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None
        
    except Exception as e:
        print(f"LLM error: {e}")
        return None


# ============================================================
# DATA LOADING (SQLite preferred, CSV fallback)
# ============================================================

import sqlite3

def get_db_connection():
    """Get SQLite database connection if available."""
    if SQLITE_DB_FILE.exists():
        return sqlite3.connect(str(SQLITE_DB_FILE))
    return None


def lookup_trial_sqlite(nct_id: str) -> Optional[Dict[str, Any]]:
    """Lookup AACT trial by NCT ID using SQLite (fast)."""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM aact_trials WHERE UPPER(nct_id) = ? LIMIT 1", (nct_id.upper(),))
        row = cursor.fetchone()
        if row:
            return {k: row[k] for k in row.keys()}
        return None
    except Exception as e:
        print(f"SQLite AACT lookup error: {e}")
        return None
    finally:
        conn.close()


def lookup_eu_trial_sqlite(eudract_id: str) -> Optional[Dict[str, Any]]:
    """Lookup EU trial by EudraCT number using SQLite (fast)."""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM eu_trials WHERE eudract_number = ? LIMIT 1", (eudract_id.strip(),))
        row = cursor.fetchone()
        if row:
            result = {k: row[k] for k in row.keys()}
            # Map EU fields to common format
            result["trial_id"] = result.get("eudract_number")
            result["source"] = "EU"
            result["phase"] = result.get("phase_norm")
            result["brief_title"] = result.get("trial_title")
            result["condition"] = result.get("trial_title")
            return result
        return None
    except Exception as e:
        print(f"SQLite EU lookup error: {e}")
        return None
    finally:
        conn.close()


@lru_cache(maxsize=1)
def load_aact_data() -> Optional[pd.DataFrame]:
    """Load AACT master data from CSV (fallback if no SQLite)."""
    if not AACT_MASTER_FILE.exists():
        return None
    
    df = pd.read_csv(AACT_MASTER_FILE, low_memory=False)
    return df


@lru_cache(maxsize=1)
def load_eu_data() -> Optional[pd.DataFrame]:
    """Load EU (EudraCT) master data from CSV (fallback if no SQLite)."""
    if not EU_MASTER_FILE.exists():
        return None
    
    df = pd.read_csv(EU_MASTER_FILE, low_memory=False)
    return df


def lookup_trial(nct_id: str) -> Optional[Dict[str, Any]]:
    """Lookup trial by NCT ID. Uses SQLite if available, falls back to CSV."""
    # Try SQLite first (fast)
    if SQLITE_DB_FILE.exists():
        result = lookup_trial_sqlite(nct_id)
        if result is not None:
            return result
    
    # Fallback to CSV
    df = load_aact_data()
    if df is None:
        return None
    
    nct_id = nct_id.upper().strip()
    mask = df["nct_id"].str.upper() == nct_id
    
    if not mask.any():
        return None
    
    row = df[mask].iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def lookup_eu_trial(eudract_id: str) -> Optional[Dict[str, Any]]:
    """Lookup trial by EudraCT number. Uses SQLite if available, falls back to CSV."""
    # Try SQLite first (fast)
    if SQLITE_DB_FILE.exists():
        result = lookup_eu_trial_sqlite(eudract_id)
        if result is not None:
            return result
    
    # Fallback to CSV
    df = load_eu_data()
    if df is None:
        return None
    
    eudract_id = eudract_id.strip()
    mask = df["eudract_number"] == eudract_id
    
    if not mask.any():
        return None
    
    row = df[mask].iloc[0].to_dict()
    result = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    
    # Map EU fields to common format
    result["trial_id"] = result.get("eudract_number")
    result["source"] = "EU"
    result["phase"] = result.get("phase_norm")
    result["brief_title"] = result.get("trial_title")
    result["condition"] = result.get("trial_title")
    
    return result


def detect_trial_id_type(trial_id: str) -> str:
    """Detect if trial ID is NCT (US) or EudraCT (EU) format."""
    trial_id = trial_id.strip().upper()
    
    # NCT format: NCT followed by 8 digits
    if re.match(r'^NCT\d{8}$', trial_id):
        return "AACT"
    
    # EudraCT format: YYYY-NNNNNN-NN (e.g., 2004-000137-11)
    if re.match(r'^\d{4}-\d{6}-\d{2}$', trial_id):
        return "EU"
    
    return "UNKNOWN"


# ============================================================
# API MODELS
# ============================================================

class AnalyzeRequest(BaseModel):
    eligibility_text: str
    phase: Optional[str] = None
    condition: Optional[str] = None
    demographics: Optional[Dict[str, float]] = None
    region: Optional[str] = "US"  # US, EU, India, China
    use_llm: Optional[bool] = False  # Use LLM for gate detection (hybrid mode)


class AnalyzeResponse(BaseModel):
    gates: List[Dict[str, Any]]
    gating_score: int
    flags: Dict[str, int]
    rindex: Optional[Dict[str, Optional[float]]] = None
    llm_analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="TrialFair API",
    description="Clinical Trial Eligibility Criteria Auditor",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "TrialFair API is running. Frontend not found."}


@app.post("/api/analyze")
async def analyze_eligibility(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    Analyze eligibility criteria.
    
    Always returns: gates, gating_score, flags
    If demographics provided: also returns rindex
    If OpenAI key configured: also returns llm_analysis
    """
    # Handle empty text gracefully
    if not request.eligibility_text or not request.eligibility_text.strip():
        return {
            "gates": [],
            "gating_score": 0,
            "flags": {},
            "metadata": {"region": request.region or "US"},
            "rindex": None,
        }
    
    region = request.region or "US"
    if region not in POP_REFERENCES:
        region = "US"
    
    metadata = {
        "phase": request.phase,
        "condition": request.condition,
        "region": region,
    }
    
    # Detect gates (hybrid LLM if requested and available)
    if request.use_llm and OPENAI_CLIENT:
        detection = await detect_gates_with_llm(request.eligibility_text, OPENAI_CLIENT)
        metadata["detection_method"] = detection.get("detection_method", "hybrid_llm")
    else:
        detection = detect_gates(request.eligibility_text)
        metadata["detection_method"] = "regex"
    
    gates = detection["gates"]
    
    # Check exceptions
    gates = check_exceptions(gates, metadata)
    
    # Compute R-index only if demographics provided
    rindex = None
    if request.demographics:
        rindex = compute_all_rindex(request.demographics, region)
    
    # Call LLM if available and gates detected
    llm_analysis = None
    if gates:
        llm_analysis = call_llm_auditor(request.eligibility_text, gates, metadata)
    
    return {
        "gates": gates,
        "gating_score": detection["gating_score"],
        "flags": detection["flags"],
        "rindex": rindex,
        "llm_analysis": llm_analysis,
        "metadata": metadata,
    }


@app.get("/api/lookup/{trial_id}")
async def lookup_by_trial_id(trial_id: str, use_llm: bool = False) -> Dict[str, Any]:
    """
    Lookup trial by ID (supports both NCT and EudraCT formats).
    
    - NCT format: NCT12345678 (US/AACT)
    - EudraCT format: 2004-000137-11 (EU)
    
    Query params:
    - use_llm: If true, use hybrid LLM detection for gates (more accurate, slower)
    
    Returns full report including demographics and R-index if available.
    """
    id_type = detect_trial_id_type(trial_id)
    
    if id_type == "AACT":
        trial = lookup_trial(trial_id)
        source = "AACT"
        region = "US"
    elif id_type == "EU":
        trial = lookup_eu_trial(trial_id)
        source = "EU"
        region = "EU"
    else:
        # Try both
        trial = lookup_trial(trial_id)
        if trial:
            source = "AACT"
            region = "US"
        else:
            trial = lookup_eu_trial(trial_id)
            source = "EU"
            region = "EU"
    
    if not trial:
        raise HTTPException(status_code=404, detail=f"Trial {trial_id} not found in AACT or EU databases")
    
    # Get eligibility text
    eligibility_text = trial.get("eligibility_text") or trial.get("eligibility_criteria") or ""
    
    if not eligibility_text:
        raise HTTPException(status_code=404, detail=f"No eligibility text found for {trial_id}")
    
    metadata = {
        "phase": trial.get("phase") or trial.get("phase_norm"),
        "condition": trial.get("condition_text") or trial.get("disease_area") or trial.get("trial_title"),
        "trial_id": trial_id,
        "source": source,
        "region": region,
    }
    
    # Detect gates (hybrid LLM if requested and available)
    if use_llm and OPENAI_CLIENT:
        detection = await detect_gates_with_llm(eligibility_text, OPENAI_CLIENT)
        metadata["detection_method"] = detection.get("detection_method", "hybrid_llm")
    else:
        detection = detect_gates(eligibility_text)
        metadata["detection_method"] = "regex"
    
    gates = check_exceptions(detection["gates"], metadata)
    
    # Extract demographics from trial data
    demographics = {}
    
    if source == "AACT":
        # AACT demographics
        if trial.get("pct_female") is not None:
            pct = float(trial["pct_female"])
            demographics["sex_female"] = pct / 100 if pct > 1 else pct
            demographics["sex_male"] = 1 - demographics["sex_female"]
        
        for age_key in ["age_lt18_prop", "age_18_44_prop", "age_45_64_prop", "age_65plus_prop"]:
            if trial.get(age_key) is not None:
                demo_key = age_key.replace("_prop", "")
                demographics[demo_key] = float(trial[age_key])
        
        for race_key in ["pct_white", "pct_black", "pct_asian", "pct_other"]:
            if trial.get(race_key) is not None:
                demo_key = "race_" + race_key.replace("pct_", "")
                demographics[demo_key] = float(trial[race_key]) / 100
    
    elif source == "EU":
        # EU demographics (different structure)
        # Sex from sex_pattern or has_female/has_male
        sex_pattern = trial.get("sex_pattern")
        if sex_pattern == "female_only":
            demographics["sex_female"] = 1.0
            demographics["sex_male"] = 0.0
        elif sex_pattern == "male_only":
            demographics["sex_female"] = 0.0
            demographics["sex_male"] = 1.0
        elif sex_pattern == "both":
            demographics["sex_female"] = 0.5  # Estimate
            demographics["sex_male"] = 0.5
        
        # Age - EU has different buckets (lt18, 18-64, 65+)
        if trial.get("age_lt18_prop") is not None:
            demographics["age_lt18"] = float(trial["age_lt18_prop"])
        if trial.get("age_18_64_prop") is not None:
            # Split 18-64 into 18-44 and 45-64 (estimate 50/50)
            prop = float(trial["age_18_64_prop"])
            demographics["age_18_44"] = prop * 0.5
            demographics["age_45_64"] = prop * 0.5
        if trial.get("age_65plus_prop") is not None:
            demographics["age_65plus"] = float(trial["age_65plus_prop"])
        
        # No race data for EU
    
    # Use pre-computed R-index if available, otherwise compute
    if trial.get("r_overall") is not None:
        rindex = {
            "r_sex": trial.get("r_sex"),
            "r_age": trial.get("r_age"),
            "r_race": trial.get("r_race"),
            "r_overall": trial.get("r_overall"),
            "region": region,
        }
    elif demographics:
        rindex = compute_all_rindex(demographics, region)
    else:
        rindex = None
    
    # Call LLM
    llm_analysis = None
    if gates:
        llm_analysis = call_llm_auditor(eligibility_text, gates, metadata)
    
    return {
        "trial_id": trial_id,
        "nct_id": trial_id if source == "AACT" else None,
        "eudract_id": trial_id if source == "EU" else None,
        "source": source,
        "eligibility_text": eligibility_text,
        "gates": gates,
        "gating_score": detection["gating_score"],
        "flags": detection["flags"],
        "demographics": demographics,
        "rindex": rindex,
        "llm_analysis": llm_analysis,
        "metadata": metadata,
        "raw_trial_data": trial,
    }


@app.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    """Get aggregate statistics about the datasets."""
    
    stats = {
        "sqlite_enabled": SQLITE_DB_FILE.exists(),
        "aact": None,
        "eu": None,
    }
    
    # Check SQLite first
    if SQLITE_DB_FILE.exists():
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                # AACT stats from SQLite
                cursor.execute("SELECT COUNT(*) FROM aact_trials")
                aact_count = cursor.fetchone()[0]
                stats["aact"] = {
                    "loaded": True,
                    "source": "sqlite",
                    "total_trials": aact_count,
                }
                
                # EU stats from SQLite
                cursor.execute("SELECT COUNT(*) FROM eu_trials")
                eu_count = cursor.fetchone()[0]
                stats["eu"] = {
                    "loaded": True,
                    "source": "sqlite",
                    "total_trials": eu_count,
                }
                
                conn.close()
                return stats
            except Exception as e:
                print(f"SQLite stats error: {e}")
                conn.close()
    
    # Fallback to CSV stats
    aact_df = load_aact_data()
    eu_df = load_eu_data()
    
    if aact_df is not None:
        stats["aact"] = {
            "loaded": True,
            "source": "csv",
            "total_trials": len(aact_df),
            "trials_with_demographics": int(aact_df["pct_female"].notna().sum()) if "pct_female" in aact_df.columns else 0,
        }
        if "phase" in aact_df.columns:
            stats["aact"]["phase_distribution"] = aact_df["phase"].value_counts().head(10).to_dict()
    else:
        stats["aact"] = {"loaded": False, "message": "Place aact_master.csv in final_data/ or run convert_to_sqlite.py"}
    
    if eu_df is not None:
        stats["eu"] = {
            "loaded": True,
            "source": "csv",
            "total_trials": len(eu_df),
            "trials_with_eligibility": int(eu_df["eligibility_text"].notna().sum()) if "eligibility_text" in eu_df.columns else 0,
            "trials_with_rindex": int(eu_df["r_overall"].notna().sum()) if "r_overall" in eu_df.columns else 0,
        }
        if "phase_norm" in eu_df.columns:
            stats["eu"]["phase_distribution"] = eu_df["phase_norm"].value_counts().head(10).to_dict()
    else:
        stats["eu"] = {"loaded": False, "message": "Place eu_master.csv in final_data/ or run convert_to_sqlite.py"}
    
    return stats


@app.post("/api/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    """
    Extract trial information from uploaded PDF using LLM.
    Returns structured data: eligibility criteria, phase, condition, demographics if found.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Extract raw text from PDF
        raw_text = ""
        try:
            import fitz  # PyMuPDF
            content = await file.read()
            doc = fitz.open(stream=content, filetype="pdf")
            for page in doc:
                raw_text += page.get_text() + "\n"
            doc.close()
        except ImportError:
            try:
                import pdfplumber
                import io
                content = await file.read()
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    for page in pdf.pages:
                        raw_text += (page.extract_text() or "") + "\n"
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="PDF extraction not available. Install PyMuPDF: pip install pymupdf"
                )
        
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Use LLM to intelligently extract structured info
        client = get_openai_client()
        if client:
            extraction_prompt = f"""Extract clinical trial information from this document. Return ONLY valid JSON with these fields:

{{
  "nct_id": "NCT number if found, else null",
  "phase": "Phase 1, Phase 2, Phase 3, Phase 4, or null if not found",
  "condition": "Primary condition/disease being studied, or null",
  "sponsor": "Sponsor organization if found, or null",
  "region": "Detect the region/country where this trial is conducted. Return one of: 'US', 'EU', 'India', 'China', or null if unclear. Look for country mentions, site locations, regulatory references (FDA=US, EMA=EU, CDSCO=India, NMPA=China), language requirements, etc.",
  "eligibility_criteria": "Full inclusion and exclusion criteria text - extract ALL criteria found",
  "demographics": {{
    "pct_female": number or null,
    "pct_male": number or null,
    "age_range": "e.g. 18-65 years" or null,
    "pct_white": number or null,
    "pct_black": number or null,
    "pct_asian": number or null,
    "pct_other": number or null
  }},
  "missing_fields": ["list of important fields that could not be found"]
}}

IMPORTANT: 
- Extract eligibility criteria as completely as possible
- If demographics percentages are mentioned anywhere, extract them
- DETECT THE REGION: Look for mentions of countries, cities, regulatory bodies, or site locations
  - US indicators: FDA, United States, American cities, US states, English-only requirement
  - EU indicators: EMA, European Union, EU countries (Germany, France, Spain, Italy, etc.), CTFG, EU CTR
  - India indicators: CDSCO, India, Indian cities (Mumbai, Delhi, Chennai, Bangalore), AIIMS
  - China indicators: NMPA, China, Chinese cities (Beijing, Shanghai), CDE
- List any important missing information in missing_fields

Document text (truncated to 8000 chars):
{raw_text[:8000]}"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a clinical trial document parser. Extract structured information and return ONLY valid JSON."},
                        {"role": "user", "content": extraction_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                )
                
                content = response.choices[0].message.content
                
                # Parse JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    extracted = json.loads(json_match.group())
                    extracted["extraction_method"] = "llm"
                    extracted["raw_text_length"] = len(raw_text)
                    return extracted
                    
            except Exception as e:
                print(f"LLM extraction error: {e}")
                # Fall through to basic extraction
        
        # Fallback: Basic regex extraction without LLM
        extracted = {
            "nct_id": None,
            "phase": None,
            "condition": None,
            "sponsor": None,
            "eligibility_criteria": None,
            "demographics": {},
            "missing_fields": [],
            "extraction_method": "basic",
            "raw_text_length": len(raw_text),
        }
        
        # Try to find NCT ID
        nct_match = re.search(r'NCT\d{8}', raw_text)
        if nct_match:
            extracted["nct_id"] = nct_match.group()
        else:
            extracted["missing_fields"].append("NCT ID")
        
        # Try to find phase
        phase_match = re.search(r'Phase\s*([1-4]|I{1,3}|IV)', raw_text, re.IGNORECASE)
        if phase_match:
            phase_map = {"I": "1", "II": "2", "III": "3", "IV": "4"}
            p = phase_match.group(1).upper()
            extracted["phase"] = f"Phase {phase_map.get(p, p)}"
        else:
            extracted["missing_fields"].append("Phase")
        
        # Try to find eligibility criteria section
        elig_patterns = [
            r'(?i)(inclusion\s+criteria.*?)((?=\n\s*\d+\.\s*[A-Z])|(?=\nexclusion)|$)',
            r'(?i)(eligibility.*?criteria.*?)(?=\n\n[A-Z]|\Z)',
            r'(?i)(who\s+can\s+participate.*?)(?=\n\n[A-Z]|\Z)',
        ]
        
        for pattern in elig_patterns:
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                extracted["eligibility_criteria"] = match.group(1).strip()[:5000]
                break
        
        if not extracted["eligibility_criteria"]:
            # Just get a chunk that might contain criteria
            lower_text = raw_text.lower()
            for keyword in ["inclusion", "eligibility", "criteria", "exclusion"]:
                idx = lower_text.find(keyword)
                if idx != -1:
                    extracted["eligibility_criteria"] = raw_text[idx:idx+3000].strip()
                    break
        
        if not extracted["eligibility_criteria"]:
            extracted["missing_fields"].append("Eligibility Criteria")
            extracted["eligibility_criteria"] = raw_text[:2000]  # Just return first 2000 chars
        
        extracted["missing_fields"].append("Condition (please specify)")
        extracted["missing_fields"].append("Demographics (if available)")
        
        return extracted
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.get("/api/regions")
async def get_regions():
    """Get available regions for R-Index calculation."""
    regions = []
    for region, pop in POP_REFERENCES.items():
        has_race = any(k.startswith("race_") for k in pop.keys())
        regions.append({
            "id": region,
            "name": {
                "US": "United States (2023 Census)",
                "EU": "European Union (EU27, 2020)",
                "India": "India (UN WPP 2023)",
                "China": "China (UN WPP 2023)",
            }.get(region, region),
            "has_race_data": has_race,
            "demographics": list(pop.keys()),
        })
    return {"regions": regions}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "openai_configured": get_openai_client() is not None,
        "data_file_exists": AACT_MASTER_FILE.exists(),
    }


# Mount static files (after routes so routes take precedence)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
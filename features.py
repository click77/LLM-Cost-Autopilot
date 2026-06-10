import re
from typing import Dict, Any

def extract_prompt_features(prompt: str) -> Dict[str, Any]:
    """
    Transforms a raw textual prompt into numerical and boolean features
    directly mapped to our Tier 1, Tier 2, and Tier 3 definitions.
    """
    # Clean and normalize input text
    prompt_lower = prompt.lower()
    words = prompt_lower.split()
    token_estimate = int(len(words) * 1.3) # Fast token calculation heuristic
    
    # 1. Anchor Keyword Lists based on Tier Contracts
    tier1_keywords = ['extract', 'format', 'convert', 'find', 'get', 'json', 'csv', 'yaml', 'regex']
    tier2_keywords = ['summarize', 'classify', 'categorize', 'summary', 'overview', 'sentiment', 'analysis']
    tier3_keywords = ['think step', 'reason', 'prove', 'evaluate', 'write code', 'create a script', 'draft', 'compare and contrast']
    
    # Count occurrences
    t1_count = sum(1 for kw in tier1_keywords if kw in prompt_lower)
    t2_count = sum(1 for kw in tier2_keywords if kw in prompt_lower)
    t3_count = sum(1 for kw in tier3_keywords if kw in prompt_lower)
    
    # 2. Structural Constraint Counting Heuristic
    # Counts punctuation bullets, structural items like numbers, or conditional markers
    constraint_markers = len(re.findall(r'(\d\.|must|should|do not|never|always|format:)', prompt_lower))
    
    # 3. Code & Mathematical Character Density Check (Signals Tier 3 Complexity)
    has_code_syntax = 1.0 if any(char in prompt for char in ['{', '}', '[', ']', 'def ', 'import ', 'lambda']) else 0.0
    
    return {
        "token_count": token_estimate,
        "tier1_keyword_density": t1_count,
        "tier2_keyword_density": t2_count,
        "tier3_keyword_density": t3_count,
        "constraint_count": constraint_markers,
        "contains_code_syntax": has_code_syntax,
        "is_long_context": 1.0 if token_estimate > 2500 else 0.0
    }

# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path
import sys
import time
import re
import logging
import traceback
from dotenv import load_dotenv

# Ollama and its tools
from ollama_client import generate_text  # type: ignore
from config import MODEL_CONFIG  # type: ignore
from ag2_agents import AG2LiteratureReviewSystem  # Introduce the GroupChat system

import logging
import sys

# ================= Unified log configuration (Only use step3.out / step3.err) =================
def init_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger_name = "step3"
    logger = logging.getLogger(logger_name)
    if logger.handlers:  # Prevent duplicate additions (imported multiple times)
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")

    # info / all handler
    fh_all = logging.FileHandler(log_dir / "step3.out", mode="w", encoding="utf-8")
    fh_all.setLevel(logging.INFO)
    fh_all.setFormatter(fmt)

    # error handler
    fh_err = logging.FileHandler(log_dir / "step3.err", mode="w", encoding="utf-8")
    fh_err.setLevel(logging.ERROR)
    fh_err.setFormatter(fmt)

    # console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh_all)
    logger.addHandler(fh_err)
    logger.addHandler(ch)

    logger.propagate = False  # Avoid bubbling up to the root to prevent duplicate output
    # The root logger also appends and writes to capture the logs of third-party libraries
    root_logger = logging.getLogger()
    # Avoid adding the same path handler repeatedly
    need_root_out = True
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '').endswith('step3.out'):
            need_root_out = False
            break
    if need_root_out:
        root_out = logging.FileHandler(log_dir / "step3.out", mode="a", encoding="utf-8")
        root_out.setLevel(logging.INFO)
        root_out.setFormatter(fmt)
        root_logger.addHandler(root_out)
    need_root_err = True
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '').endswith('step3.err'):
            need_root_err = False
            break
    if need_root_err:
        root_err = logging.FileHandler(log_dir / "step3.err", mode="a", encoding="utf-8")
        root_err.setLevel(logging.ERROR)
        root_err.setFormatter(fmt)
        root_logger.addHandler(root_err)

    # Capture the log of the autogen library
    for name in ["autogen", "autogen.oai", "autogen.oai.client"]:
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = True  # Let it bubble up to root. Root has been written as step3.out.

    # Tee stdout/stderr -> Log + Original Console (to prevent recursive calls)
    class TeeStream:
        def __init__(self, logger_fn, original, log_file_path):
            self.logger_fn = logger_fn
            self.original = original
            self.log_file_path = log_file_path
            self._in_write = False  # Prevent recursion
        
        def write(self, msg):
            # Write directly to original stream
            result = self.original.write(msg)
            
            # Prevent recursion and empty messages
            if not self._in_write and msg and not msg.isspace():
                try:
                    self._in_write = True
                    # Write directly to log file, not through logger to avoid recursion
                    with open(str(self.log_file_path), 'a', encoding='utf-8') as f:
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                        for line in msg.rstrip().splitlines():
                            if line.strip():
                                f.write("{} [INFO] [console] {}\n".format(timestamp, line))
                except Exception:
                    pass  # Silent handling of write errors
                finally:
                    self._in_write = False
            
            return result
        
        def flush(self):
            try:
                self.original.flush()
            except Exception:
                pass

    # Only wrap on first setup
    if not isinstance(sys.stdout, TeeStream):
        sys.stdout = TeeStream(logger.info, sys.__stdout__, log_dir / "step3.out")
    if not isinstance(sys.stderr, TeeStream):
        sys.stderr = TeeStream(logger.error, sys.__stderr__, log_dir / "step3.err")

    logger.info("Logging initialized: logs/step3.out & logs/step3.err (stdout/stderr tee active)")
    return logger

logger = init_logging()

# Environment variables
load_dotenv()

THRESHOLD = 0.7  # Raised per new requirement
MAX_ITERATIONS = 4  # 1 baseline + up to 3 rewrites for better topic control
REMOVE_CHEM_TAGS = False  # Chemical tag feature fully disabled

def load_research_questions() -> List[str]:
    """Load research questions from all_questions.json for dynamic topic constraint."""
    try:
        questions_file = Path("initial_chapters") / "all_questions.json"
        if questions_file.exists():
            with open(questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                questions = data.get('questions', [])
                logger.info(f"Loaded {len(questions)} research questions for topic constraint")
                return questions
        else:
            logger.warning("all_questions.json not found, using fallback topic constraints")
            return []
    except Exception as e:
        logger.error(f"Failed to load research questions: {e}")
        return []

def get_chapter_topic_from_questions(chapter_num: int) -> str:
    """Get the specific topic for a chapter based on all_questions.json ordering."""
    questions = load_research_questions()
    if not questions or chapter_num < 1 or chapter_num > len(questions):
        return "general CO2 electroreduction research"
    
    # Chapter number maps directly to question index (1-based to 0-based)
    question = questions[chapter_num - 1]
    return identify_domain_from_question(question)

def identify_domain_from_question(question: str) -> str:
    """Extract the specific domain/topic from a research question dynamically."""
    question_lower = question.lower()
    
    # Extract key technical terms and phrases dynamically
    import re
    
    # Look for technical noun phrases (2-4 words)
    technical_phrases = re.findall(r'\b[a-z]+(?:\s+[a-z]+){1,3}\b', question_lower)
    
    # Look for specific material/device names
    material_patterns = [
        r'\bcu\b',  # Copper catalysts
        r'\btransition metal',  # Transition metal catalysts
        r'\bnano-?\w+',  # Nano materials
        r'\belectrode',  # Electrode materials
    ]
    
    key_materials = []
    for pattern in material_patterns:
        matches = re.findall(pattern, question)
        key_materials.extend(matches)
    
    # Build domain description from the question itself
    # Focus on the main subject matter
    words = question_lower.split()
    domain_indicators = []
    
    # Look for domain-indicating words
    for i, word in enumerate(words):
        if word in ['of', 'in', 'for', 'on', 'with']:
            # Get context around prepositions
            context = words[max(0, i-2):i+3]
            domain_indicators.extend(context)
    
    # Create a domain name from the most relevant terms
    if key_materials:
        primary_material = key_materials[0]
        return f"{primary_material} research and applications"
    
    # Fallback to extracting from technical phrases
    if technical_phrases:
        main_phrase = technical_phrases[0]
        return f"{main_phrase} research"
    
    return "general technical research"

def extract_key_terms_from_question(question: str) -> List[str]:
    """Extract key technical terms from research question dynamically."""
    if not question:
        return []
    
    import re
    
    # Technical terms that should be preserved in chapter content
    key_terms = set()
    
    # Extract chemical formulas and materials
    chemical_patterns = [
        r'\b[A-Z][a-z]*[₀-₉]+[A-Za-z]*\b',  # Chemical formulas with subscripts
        r'\b[A-Z][a-z]*[0-9]+[A-Za-z]*\b',   # Chemical formulas with numbers
        r'\b[A-Z]{2,}\b',  # Acronyms
    ]
    
    for pattern in chemical_patterns:
        matches = re.findall(pattern, question)
        key_terms.update(matches)
    
    # Extract technical compound terms
    compound_patterns = [
        r'\b[a-z]+(?:-[a-z]+)+\b',  # hyphenated terms
        r'\b[a-z]+\s+[a-z]+\s+[a-z]+\b',  # 3-word technical phrases
        r'\b[a-z]+\s+[a-z]+\b',  # 2-word technical phrases
    ]
    
    question_lower = question.lower()
    for pattern in compound_patterns:
        matches = re.findall(pattern, question_lower)
        # Filter out common non-technical phrases
        filtered_matches = [m for m in matches if not any(common in m for common in 
                          ['of the', 'in the', 'for the', 'to the', 'at the', 'on the', 'with the'])]
        key_terms.update(filtered_matches)
    
    # Extract single important technical words
    important_word_pattern = r'\b(?:electrochemical|electrocatalyst|electrocatalysis|electroreduction|electrolysis|electrode|electrolyte|faradaic|overpotential|catalyst|selectivity|efficiency|current|density|potential|reduction|oxidation|co2|carbon|dioxide|copper|silver|gold|platinum|palladium|nickel|cobalt|iron|zinc|nanostructure|nanomaterial|surface|interface|adsorption|desorption|intermediate|mechanism|pathway|product|ethylene|ethanol|methane|formate|acetate|propanol|industrial|commercial|scaling|reactor|membrane|characterization|xrd|sem|tem|xps|raman|ftir|cyclic|voltammetry|impedance|chronoamperometry|performance|stability|durability|optimization)\w*\b'
    
    important_words = re.findall(important_word_pattern, question_lower)
    key_terms.update(important_words)
    
    # Convert to sorted list and limit
    return sorted(list(key_terms))[:15]  # Limit to top 15 terms

def identify_content_domain(text: str) -> str:
    """Identify the primary domain/field of the content with high precision."""
    text_lower = text.lower()
    
    # Define domain indicators with specific weight scores for electrochemical CO2 reduction
    domain_indicators = {
        'co2_electroreduction': {
            'keywords': ['co2 electroreduction', 'co2 reduction', 'carbon dioxide reduction', 'electrochemical co2', 
                        'co2 conversion', 'carbon dioxide electroreduction', 'co electroreduction', 'electrochemical reduction'],
            'weight': 15
        },
        'electrocatalysts': {
            'keywords': ['electrocatalyst', 'electrocatalysis', 'nano-electrocatalyst', 'catalyst design', 'cu catalyst',
                        'copper catalyst', 'transition metal catalyst', 'active sites', 'catalyst selectivity', 'faradaic efficiency'],
            'weight': 12
        },
        'electrochemical_processes': {
            'keywords': ['electrochemical', 'electrode', 'current density', 'overpotential', 'electrolyte', 'electroreduction',
                        'electrochemical cell', 'potential', 'galvanic', 'electrolysis', 'electrochemical conversion'],
            'weight': 10
        },
        'multi_carbon_products': {
            'keywords': ['c2+ products', 'c3+ products', 'multi-carbon', 'ethylene', 'ethanol', 'propanol', 'acetate',
                        'ethane', 'propane', 'c2 products', 'c3 products', 'multicarbon'],
            'weight': 10
        },
        'reaction_mechanisms': {
            'keywords': ['reaction mechanism', 'intermediates', 'reaction pathway', 'co intermediate', 'cho intermediate',
                        'surface binding', 'adsorption', 'desorption', 'surface chemistry', 'reaction kinetics'],
            'weight': 8
        },
        'industrial_applications': {
            'keywords': ['industrial implementation', 'scale-up', 'ampere-level', 'commercial', 'economic analysis',
                        'reactor design', 'flow cell', 'membrane electrode assembly', 'industrial scale'],
            'weight': 8
        },
        'materials_characterization': {
            'keywords': ['xrd', 'sem', 'tem', 'xps', 'raman', 'ftir', 'cv', 'lsv', 'eis', 'chronoamperometry',
                        'surface analysis', 'structural characterization', 'morphology', 'crystallinity'],
            'weight': 6
        }
    }
    
    domain_scores = {}
    
    for domain, info in domain_indicators.items():
        score = 0
        for keyword in info['keywords']:
            if keyword in text_lower:
                score += info['weight']
        domain_scores[domain] = score
    
    # Find the highest scoring domain
    if domain_scores:
        primary_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[primary_domain]
        
        # Only return a domain if it has a significant score
        if max_score >= 10:
            return primary_domain
    
    return 'general_electrochemical'

def extract_specific_domain_terms(text: str, domain: str) -> List[str]:
    """Extract domain-specific key terms from text based on identified domain."""
    text_lower = text.lower()
    key_terms = set()
    
    # Domain-specific term extraction patterns
    domain_patterns = {
        'co2_electroreduction': [
            r'\b(?:co2\s+(?:electroreduction|reduction|conversion))\b',
            r'\b(?:carbon\s+dioxide\s+(?:electroreduction|reduction))\b',
            r'\b(?:electrochemical\s+co2)\b',
            r'\b(?:co\s+electroreduction)\b',
        ],
        'electrocatalysts': [
            r'\b(?:electrocatalyst|electrocatalysis)\b',
            r'\b(?:nano-electrocatalyst|catalyst\s+design)\b',
            r'\b(?:cu\s+catalyst|copper\s+catalyst)\b',
            r'\b(?:transition\s+metal\s+catalyst)\b',
            r'\b(?:active\s+sites|catalyst\s+selectivity)\b',
            r'\b(?:faradaic\s+efficiency)\b',
        ],
        'electrochemical_processes': [
            r'\b(?:current\s+density|overpotential)\b',
            r'\b(?:electrode|electrolyte|electroreduction)\b',
            r'\b(?:electrochemical\s+cell|potential)\b',
            r'\b(?:galvanic|electrolysis)\b',
            r'\b(?:electrochemical\s+conversion)\b',
        ],
        'multi_carbon_products': [
            r'\b(?:c2\+\s+products|c3\+\s+products)\b',
            r'\b(?:multi-carbon|multicarbon)\b',
            r'\b(?:ethylene|ethanol|propanol)\b',
            r'\b(?:acetate|ethane|propane)\b',
            r'\b(?:c2\s+products|c3\s+products)\b',
        ],
        'reaction_mechanisms': [
            r'\b(?:reaction\s+mechanism|intermediates)\b',
            r'\b(?:reaction\s+pathway|co\s+intermediate)\b',
            r'\b(?:cho\s+intermediate|surface\s+binding)\b',
            r'\b(?:adsorption|desorption|surface\s+chemistry)\b',
            r'\b(?:reaction\s+kinetics)\b',
        ],
        'industrial_applications': [
            r'\b(?:industrial\s+implementation|scale-up)\b',
            r'\b(?:ampere-level|commercial)\b',
            r'\b(?:economic\s+analysis|reactor\s+design)\b',
            r'\b(?:flow\s+cell|membrane\s+electrode\s+assembly)\b',
            r'\b(?:industrial\s+scale)\b',
        ],
        'materials_characterization': [
            r'\b(?:XRD|SEM|TEM|XPS|Raman|FTIR)\b',
            r'\b(?:CV|LSV|EIS|chronoamperometry)\b',
            r'\b(?:surface\s+analysis|structural\s+characterization)\b',
            r'\b(?:morphology|crystallinity)\b',
        ]
    }
    
    # Extract terms specific to the identified domain
    if domain in domain_patterns:
        for pattern in domain_patterns[domain]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_terms.update(matches)
    
    # Also extract general electrochemical terms
    general_patterns = [
        r'\b(?:CO2|CO₂|CO|CH4|CH₄|C2H4|C₂H₄)\b',  # Chemical formulas
        r'\b(?:mA/cm²|V\s+vs\s+RHE|vs\s+RHE)\b',  # Electrochemical units
        r'\b(?:Cu|Ag|Au|Pd|Pt|Ni|Co|Fe|Zn)\b',  # Catalyst materials
        r'\b[A-Z]{2,}\b',  # Acronyms
        r'\b\w+(?:-\w+)+\b',  # Hyphenated technical terms
    ]
    
    for pattern in general_patterns:
        matches = re.findall(pattern, text)
        key_terms.update(matches)
    
    return list(key_terms)

def extract_dynamic_domain_terms(questions: List[str]) -> List[str]:
    """Extract domain-specific terms from research questions dynamically."""
    if not questions:
        return []
    
    # Join all questions and extract key terms
    all_text = " ".join(questions).lower()
    
    # Extract technical terms from questions
    domain_terms = set()
    
    # Extract compound technical terms
    compound_patterns = [
        r'\b[a-z]+(?:-[a-z]+)+\b',  # hyphenated terms like "two-dimensional"
        r'\b[A-Z][a-z]*[A-Z][a-z]*\b',  # CamelCase terms
        r'\b[A-Z]{2,}\b',  # Acronyms
        r'\b\w*[₀-₉]\w*\b',  # Chemical formulas with subscripts
    ]
    
    for pattern in compound_patterns:
        matches = re.findall(pattern, all_text)
        domain_terms.update(matches)
    
    # Extract single important keywords (nouns and adjectives)
    important_words = re.findall(r'\b(?:quantum|electronic|optical|magnetic|thermal|mechanical|chemical|biological|physical|computational|experimental|theoretical|semiconductor|nanoscale|atomic|molecular|crystalline|defect|interface|hetero|homo|strain|bandgap|mobility|carrier|transport|injection|recombination|efficiency|stability|performance|switching|leakage|threshold|voltage|current|resistance|conductivity|dielectric|phonon|plasmon|coupling|confinement|engineering|passivation|doping|epitaxial|monolayer|bilayer|multilayer|nanostructure|nanowire|nanotube|graphene|silicene|phosphorene)\w*\b', all_text)
    domain_terms.update(important_words)
    
    # Convert to sorted list for consistency
    return sorted(list(domain_terms))[:30]  # Limit to top 30 terms

def get_section_type_constraints(section_name: str) -> str:
    """Generate section-type-specific writing constraints for the three chapter types."""
    section_lower = section_name.lower()
    
    if "background" in section_lower or "background knowledge" in section_lower:
        return """
BACKGROUND KNOWLEDGE Section Constraints:
- Provide SPECIFIC technical details and mechanisms (not abstract overviews)
- Include precise material properties, device structures, and operational principles
- Use technical terminology and parameters with units
- Explain foundational concepts that directly support the research domain
- Keep content technically specific rather than broad conceptual descriptions
- Focus on established knowledge that provides concrete foundation for current research
- Avoid overly general statements - be specific to the technical domain
"""
    
    elif "recommendation" in section_lower or "future" in section_lower:
        return """
RESEARCH RECOMMENDATIONS Section Constraints:
- Use future-oriented language: "should investigate", "future studies should focus on", "further research is needed"
- Identify specific research gaps and opportunities
- Suggest concrete experimental approaches and methodologies
- Recommend particular directions for advancing the field
- Be prescriptive about what needs to be done next
- Focus on actionable research directions
- Bridge current limitations with proposed solutions
"""
    
    elif "current" in section_lower or "research" in section_lower:
        return """
CURRENT RESEARCH Section Constraints:
- Use temporal language: "recent advances", "emerging studies", "latest findings", "current investigations"
- Reference recent developments and ongoing research trends
- Highlight what is actively being studied NOW in the field
- Use progressive language: "researchers are exploring", "studies are revealing", "ongoing work focuses on"
- Emphasize cutting-edge aspects and novel approaches
- Show the dynamic, evolving nature of the research area
- Maintain forward-looking perspective on the field's direction
"""
    
    else:
        return """
GENERAL Section Constraints:
- Maintain academic tone and technical precision
- Use appropriate scientific language for the content type
- Ensure content matches the section's intended purpose
"""

class AG2ReviewParseError(Exception):
    """Raised when AG2 group review output cannot be parsed into required structure"""
    pass

def save_consolidated_output(data: Dict, section_name: str) -> str:
    """Save consolidated output for a single section."""
    try:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{section_name}_consolidated_{timestamp}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Save failed {section_name}: {e}")
        return ""

def extract_key_domain_terms(text: str, research_questions: List[str] = None) -> List[str]:
    """Extract key domain-specific terms from the original text and research questions for preservation."""
    
    key_terms = set()
    
    # If research questions provided, extract terms from them first
    if research_questions:
        domain_terms_from_questions = extract_dynamic_domain_terms(research_questions)
        key_terms.update(domain_terms_from_questions)
    
    # Extract terms from the text itself
    patterns = [
        r'\b[A-Z][a-z]*[A-Z][a-z]*\b',  # CamelCase terms
        r'\b[A-Z]{2,}\b',  # Acronyms
        r'\b\w*[₀-₉]\w*\b',  # Chemical formulas with subscripts
        r'\b\w+[₂₃₄₅₆₇₈₉]\w*\b',  # More chemical formulas
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        key_terms.update(matches)
    
    return list(key_terms)[:25]  # Limit to top 25 terms

def rewrite_with_ollama(original_text: str, improvement_points: List[str], section_topic: str = "", research_questions: List[str] = None) -> str:
    """Rewrite text using Ollama based on improvement points with enhanced topic preservation."""
    joined = "\n".join(f"{i+1}. {p}" for i, p in enumerate(improvement_points[:5]))  # Limit to 5 points
    
    # Extract key domain terms for preservation
    key_terms = extract_key_domain_terms(original_text, research_questions)
    key_terms_str = ", ".join(key_terms[:8]) if key_terms else "N/A"  # Limit to 8 terms
    
    # Extract main subject from original text
    first_sentence = original_text.split('.')[0].strip() if original_text else ""
    
    # Extract specific materials/topics mentioned in original text
    materials_mentioned = []
    electrochemical_terms = ["CO2 electroreduction", "CO2 reduction", "electrocatalyst", "Cu catalyst", 
                           "copper catalyst", "transition metal catalyst", "Faradaic efficiency", 
                           "current density", "overpotential", "C2+ products", "multi-carbon products",
                           "electrochemical conversion", "active sites", "reaction mechanism"]
    
    for term in electrochemical_terms:
        if term.lower() in original_text.lower():
            materials_mentioned.append(term)
    
    materials_str = ", ".join(materials_mentioned[:4]) if materials_mentioned else ""  # Limit materials
    
    # Build research question constraint (simplified)
    question_context = ""
    if research_questions:
        # Find most relevant questions based on key terms overlap
        relevant_questions = []
        for q in research_questions[:3]:  # Check only first 3 questions
            q_lower = q.lower()
            if any(term.lower() in q_lower for term in key_terms[:5]):
                relevant_questions.append(q[:80] + "...")  # Truncate questions
        
        if relevant_questions:
            question_context = f"Research context: {'; '.join(relevant_questions[:1])}"  # Only 1 question
    
    # Simplified, more concise prompt for electrochemical CO2 reduction
    prompt = f"""Academic rewriting task: Apply ONLY these specific improvements to the original electrochemical CO2 reduction text. Keep the same topic and terminology.

SUBJECT: {first_sentence[:100]}...
KEY TERMS TO PRESERVE: {key_terms_str}
{f"CORE ELECTROCHEMICAL CONCEPTS: {materials_str}" if materials_str else ""}
{question_context}

CONSTRAINTS:
1. Minimal changes only - do NOT rewrite completely
2. Stay within electrochemical CO2 reduction field - no topic drift
3. Preserve all citation markers [1], [2], etc.
4. Keep core electrochemical concepts and performance metrics intact
5. Maintain units (mA/cm², V vs RHE, % Faradaic efficiency)

IMPROVEMENTS TO APPLY:
{joined}

ORIGINAL TEXT:
{original_text}

OUTPUT: Improved text with minimal changes, same electrochemical topic and terminology."""
    
    # Check prompt length (rough estimate: 1 token ≈ 4 characters)
    estimated_tokens = len(prompt) // 4
    if estimated_tokens > 5800:  # Leave some buffer
        logger.warning(f"Prompt may be too long: ~{estimated_tokens} tokens (limit: 6000)")
        # Truncate original text if needed
        if len(original_text) > 1000:
            original_text_short = original_text[:1000] + "... [content truncated for processing]"
            prompt = prompt.replace(original_text, original_text_short)
            logger.info("Truncated original text to fit within token limit")
    
    try:
        improved = generate_text(
            prompt=prompt,
            model=MODEL_CONFIG.main_model,
            system="Apply minimal improvements only. Preserve topic and terminology. Output English only.",
            temperature=0.3,
            max_tokens=1800,
            task_type="section_rewrite"
        )
        
        # Debug logging
        logger.info(f"Rewrite completed. Original length: {len(original_text)}, Improved length: {len(improved)}")
        if len(improved) < 100:
            logger.warning(f"Rewritten text seems too short: '{improved[:200]}...'")
        
        return improved.strip()
        return improved.strip()
    except Exception as e:
        logger.error(f"Rewrite failed, returning original text: {e}")
        return original_text

def validate_topic_consistency_simple(original_text: str, rewritten_text: str, section_name: str, research_questions: List[str] = None) -> bool:
    """Validate that rewritten text maintains the same topic as original and stays relevant to research questions."""
    
    def extract_domain_indicators(text: str, questions: List[str] = None) -> set:
        """Extract domain-specific indicators from text."""
        import re
        
        indicators = set()
        
        # Scientific/technical terms
        tech_terms = set(re.findall(r'\b(?:electrochemical|electrocatalyst|electrocatalysis|electroreduction|electrolysis|electrode|electrolyte|faradaic|overpotential|catalyst|selectivity|efficiency|current|density|potential|reduction|oxidation|co2|carbon|dioxide|copper|silver|gold|platinum|palladium|nickel|cobalt|iron|zinc|nanostructure|nanomaterial|surface|interface|adsorption|desorption|intermediate|mechanism|pathway|product|ethylene|ethanol|methane|formate|acetate|propanol|industrial|commercial|scaling|reactor|membrane|characterization|xrd|sem|tem|xps|raman|ftir|cyclic|voltammetry|impedance|chronoamperometry|performance|stability|durability|optimization)\w*\b', text.lower()))
        indicators.update(tech_terms)
        
        # Chemical formulas and materials
        materials = set(re.findall(r'\b[A-Z][a-z]*[₀-₉]*[A-Z]*[a-z]*[₀-₉]*\b', text))
        indicators.update(materials)
        
        # Extract domain keywords dynamically from research questions if available
        if questions:
            question_terms = extract_dynamic_domain_terms(questions)
            # Check which question terms appear in the text
            text_lower = text.lower()
            for term in question_terms:
                if term.lower() in text_lower:
                    indicators.add(term.lower())
        else:
            # Fallback to general domain keywords
            domains = set()
            domain_keywords = {
                'materials': ['Cu', 'Ag', 'Au', 'Fe', 'Co', 'Ni', 'Zn', 'Pt', 'Pd', 'electrocatalyst', 'electrode', 'membrane', 'nanomaterial'],
                'physics': ['electron', 'photon', 'phonon', 'bandgap', 'mobility', 'conductivity'],
                'chemistry': ['synthesis', 'reaction', 'catalyst', 'molecule', 'ion', 'bond'],
                'engineering': ['device', 'fabrication', 'design', 'optimization', 'performance']
            }
            
            text_lower = text.lower()
            for domain, keywords in domain_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    domains.add(domain)
            indicators.update(domains)
        
        return indicators
    
    original_indicators = extract_domain_indicators(original_text, research_questions)
    rewritten_indicators = extract_domain_indicators(rewritten_text, research_questions)
    
    # Calculate overlap
    if not original_indicators:
        return True  # If no clear indicators, accept the rewrite
    
    overlap = len(original_indicators & rewritten_indicators)
    overlap_ratio = overlap / len(original_indicators)
    
    # Require at least 60% overlap of domain indicators (more strict than before)
    is_consistent = overlap_ratio >= 0.6
    
    if not is_consistent:
        logger.warning(f"Topic drift detected in {section_name}:")
        logger.warning(f"Original indicators: {sorted(original_indicators)}")
        logger.warning(f"Rewritten indicators: {sorted(rewritten_indicators)}")
        logger.warning(f"Overlap ratio: {overlap_ratio:.2f} (required: ≥0.6)")
        
        # Additional check: verify that core materials/concepts are preserved (more flexible)
        original_text_lower = original_text.lower()
        rewritten_text_lower = rewritten_text.lower()
        
        core_materials_check = True
        critical_terms_missing = []
        
        # More flexible core term checking with variants
        core_term_groups = [
            ["CO2 electroreduction", "CO2 reduction", "carbon dioxide reduction", "electrochemical reduction"],
            ["Faradaic efficiency", "current density", "overpotential"],
            ["electrocatalyst", "catalyst optimization", "active sites"],
            ["multi-carbon products", "ethylene", "ethanol", "C2+ products"],
            ["electrochemical conversion", "reaction mechanism", "intermediate"]
        ]
        
        for term_group in core_term_groups:
            original_has_any = any(term in original_text_lower for term in term_group)
            rewritten_has_any = any(term in rewritten_text_lower for term in term_group)
            
            if original_has_any and not rewritten_has_any:
                logger.warning(f"Core concept group missing: {term_group[0]} (and variants)")
                critical_terms_missing.append(term_group[0])
        
        # Only fail if MORE THAN HALF of critical term groups are missing
        if len(critical_terms_missing) > len(core_term_groups) // 2:
            logger.error(f"Topic drift: Too many core terms lost: {critical_terms_missing}")
            core_materials_check = False
        elif critical_terms_missing:
            logger.warning(f"Some core terms missing but within tolerance: {critical_terms_missing}")
        
        if not core_materials_check:
            logger.error("Topic drift includes significant loss of core material terms!")
            return False
    
    return is_consistent

def assess_quality_from_panel(panel_scores: Dict[str, Optional[float]], original_text: str, improved_text: str) -> Dict:
    """Construct quality assessment structure based on panel scores."""
    original_citations = set(re.findall(r'\[\d+\]', original_text))
    improved_citations = set(re.findall(r'\[\d+\]', improved_text))
    citations_preserved = original_citations.issubset(improved_citations)
    return {
        "metrics": panel_scores,
        "citations": {
            "preserved": citations_preserved,
            "missing": list(original_citations - improved_citations) if not citations_preserved else []
        }
    }

def validate_topic_consistency(original_text: str, improved_text: str, topic: str, research_questions: List[str] = None) -> Dict[str, Any]:
    """Validate whether the rewritten text maintains the same topic consistency as the original"""
    try:
        # Extract key technical terms
        technical_terms_original = set(re.findall(r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)*\b', original_text))
        technical_terms_improved = set(re.findall(r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)*\b', improved_text))
        
        # Check if core technical terms are preserved
        preserved_terms = technical_terms_original.intersection(technical_terms_improved)
        lost_terms = technical_terms_original - technical_terms_improved
        
        # Use research questions for more precise topic relevance check
        consistency_score = 1.0
        if research_questions:
            # Extract keywords from research questions
            question_keywords = extract_dynamic_domain_terms(research_questions)
            
            # Check which question keywords are included in original and improved text
            original_lower = original_text.lower()
            improved_lower = improved_text.lower()
            
            original_question_matches = sum(1 for keyword in question_keywords if keyword.lower() in original_lower)
            improved_question_matches = sum(1 for keyword in question_keywords if keyword.lower() in improved_lower)
            
            if original_question_matches > 0:
                consistency_score = improved_question_matches / original_question_matches
            else:
                # If original text has no matching keywords, use general method
                topic_lower = topic.lower()
                original_lower = original_text.lower()
                improved_lower = improved_text.lower()
                
                # General topic keywords
                general_keywords = [
                    'co2', 'carbon dioxide', 'electroreduction', 'electrochemical', 'electrocatalyst',
                    'catalyst', 'current density', 'faradaic efficiency', 'overpotential', 'electrode',
                    'electrolyte', 'multi-carbon', 'ethylene', 'ethanol', 'propanol', 'acetate',
                    'membrane', 'reactor', 'industrial', 'scaling', 'surface', 'interface', 'adsorption',
                    'desorption', 'intermediate', 'mechanism', 'pathway', 'product', 'characterization',
                    'xrd', 'sem', 'tem', 'xps', 'raman', 'ftir', 'cv', 'lsv', 'eis', 'chronoamperometry',
                    'stability', 'durability', 'optimization', 'nano', 'transition metal', 'cu', 'ag', 'au',
                    'pt', 'pd', 'ni', 'co', 'fe', 'zn'
                ]
                
                original_general_words = sum(1 for word in general_keywords if word in original_lower)
                improved_general_words = sum(1 for word in general_keywords if word in improved_lower)
                
                consistency_score = improved_general_words / max(original_general_words, 1) if original_general_words > 0 else 1.0
        else:
            # Original general check method
            topic_lower = topic.lower()
            original_lower = original_text.lower()
            improved_lower = improved_text.lower()
            
            topic_keywords = [ 'co2', 'carbon dioxide', 'electroreduction', 'electrochemical', 'electrocatalyst',
                'catalyst', 'current density', 'faradaic efficiency', 'overpotential', 'electrode',
                'electrolyte', 'multi-carbon', 'ethylene', 'ethanol', 'propanol', 'acetate',
                'membrane', 'reactor', 'industrial', 'scaling', 'surface', 'interface', 'adsorption',
                'desorption', 'intermediate', 'mechanism', 'pathway', 'product', 'characterization',
                'xrd', 'sem', 'tem', 'xps', 'raman', 'ftir', 'cv', 'lsv', 'eis', 'chronoamperometry',
                'stability', 'durability', 'optimization', 'nano', 'transition metal', 'cu', 'ag', 'au',
                'pt', 'pd', 'ni', 'co', 'fe', 'zn'
                ]
            
            original_topic_words = sum(1 for word in topic_keywords if word in original_lower)
            improved_topic_words = sum(1 for word in topic_keywords if word in improved_lower)
            
            consistency_score = improved_topic_words / max(original_topic_words, 1) if original_topic_words > 0 else 1.0
        
        return {
            "consistency_score": min(consistency_score, 1.0),
            "preserved_technical_terms": len(preserved_terms),
            "lost_technical_terms": len(lost_terms),
            "topic_preservation": consistency_score >= 0.7,
            "warning": "Major topic drift detected" if consistency_score < 0.5 else None
        }
    except Exception as e:
        logger.warning(f"Topic consistency validation failed: {e}")
        return {"consistency_score": 1.0, "warning": None}

# ============ Chemical tagging removed (placeholders retained for compatibility) ============
_CHEM_TAG_PATTERN = re.compile(r"$^")  # matches nothing

def strip_chem_tags(text: str) -> str:
    return text  # No-op; chemical tag system removed

def extract_last_asterisk_section(text: str) -> str:
    """Extract the last section enclosed in *** markers from a text."""
    if "***" not in text:
        return ""
    
    parts = text.split("***")
    if len(parts) < 3:
        return ""
    
    return parts[-2].strip()

def parse_improvement_points(text: str) -> List[str]:
    """Parse numbered improvement points from text."""
    points = []
    current_point = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a new numbered point
        if re.match(r'^\d+[\.\)\-:] ', line):
            if current_point:
                points.append(' '.join(current_point))
                current_point = []
            current_point.append(line)
        elif current_point:
            current_point.append(line)
            
    # Add the last point if exists
    if current_point:
        points.append(' '.join(current_point))
        
    return points


def ag2_group_review(section_name: str,
                     section_text: str,
                     referenced_papers: Optional[Dict] = None,
                     needed_agents: Optional[List[str]] = None) -> Dict:
    """Use simplified AG2 (multi-expert sequential dialogue simulation) to review a single section and return unified structure.

    Review roles (as needed):
      - technical_accuracy_agent
      - clarity_agent
      - structure_agent
      - fact_checking_agent (only enabled when referenced_papers is provided)
      - moderator_agent (always enabled, summarizes and outputs PART A + PART B)

    Moderator output mandatory format:
      PART A: *** 5-7 numbered suggestions ***
      PART B: ```json { ... scores/reasons/facts ... } ```
    If parsing fails, throws AG2ReviewParseError for upper layer fallback.
    """
    logger.info(f"[AG2] Start group review: section={section_name}")

    protected_text = section_text

    if needed_agents is None:
        needed_agents = ["technical_accuracy_agent", "clarity_agent", "structure_agent"]
        if referenced_papers:
            needed_agents.append("fact_checking_agent")

    needed_agents = list(dict.fromkeys(needed_agents))  

    # Construct the context of the references (for fact-checking)
    ref_context = ""
    if referenced_papers:
        lines = ["# Referenced Papers"]
        for title, meta in referenced_papers.items():
            cid = meta.get('citation_id', 'NA')
            lines.append(f"[{cid}] {title}\nAbstract: {meta.get('abstract','')[:400]}")
            if meta.get('chunks'):
                sample_chunk = meta['chunks'][0][:400]
                lines.append(f"Sample Chunk: {sample_chunk}")
        ref_context = "\n".join(lines)

    # General Prompt Segment
    protection_clause = (
        "Keep and prohibit modification of the following protected content: the text within the <CHEM_LATEX>, <CHEM_ENTITY>, and <CHEM_SYMBOL> tags, as well as all \\ce{...} (mhchem) formulas."
    )

    role_prompts = {
        "technical_accuracy_agent": (
            "You are a technical accuracy expert, focusing on factual correctness, precise terminology, and technical depth. Provide 5-7 executable improvement suggestions."
        ),
        "clarity_agent": (
            "You are a clarity and readability expert, focusing on logic, smooth flow, and concise expression. Provide 5-7 executable improvement suggestions."
        ),
        "structure_agent": (
            "You are a structural organization expert, focusing on paragraph organization, transitions, and overall structure. Provide 5-7 executable improvement suggestions."
        ),
        "fact_checking_agent": (
            "You are a fact and citation verification expert, identify statements not adequately supported by references or potentially incorrect, explain reasons or needed citations for each. Provide 5-7 improvement suggestions."
        )
    }

    raw_reviews: Dict[str, str] = {}
    cleaned_reviews: Dict[str, str] = {}

    def _single_call(agent_key: str) -> str:
        base_role = role_prompts[agent_key]
        prompt = f"""
Role: {agent_key}
Task: {base_role}
Section Name: {section_name}
{protection_clause}
Requirements:
1. Only output *** wrapped numbered list (5-7 items) with no other text
2. Each item focuses on an independent issue, specific and executable
3. Do not rewrite the main text, only provide improvement suggestions

Text to review:
{protected_text}
"""
        if agent_key == "fact_checking_agent" and referenced_papers:
            prompt += f"\nReference abstracts:\n{ref_context}\n"
        result = generate_text(
            prompt=prompt,
            model=MODEL_CONFIG.small_model,
            system="You are an academic review assistant, strictly follow the output format.",
            temperature=0.3,
            max_tokens=1200,
            task_type="agent_review"
        )
        return result.strip()

    # Get suggestions from each agent
    for agent in needed_agents:
        if agent not in role_prompts:
            continue
        try:
            txt = _single_call(agent)
            raw_reviews[agent] = txt
            cleaned_reviews[agent] = extract_last_asterisk_section(txt) or txt
        except Exception as e:
            raw_reviews[agent] = f"[error:{e}]"
            cleaned_reviews[agent] = ""

    # Construct moderator summary input
    review_blocks = []
    for k, v in cleaned_reviews.items():
        review_blocks.append(f"### {k}\n{v}")
    reviews_joined = "\n\n".join(review_blocks)

    citation_flag = "Yes" if referenced_papers else "No"

    moderator_prompt = f"""
You are the moderator, integrating multiple experts (their *** lists are provided).
Please output strictly two parts with no other content:
PART A: *** 5-7 comprehensive improvement suggestions (absorb and deduplicate all expert opinions; if fact_checking_agent exists, must cover all its concerns; each independent and executable) ***
PART B: A ```json code block with structure:
{{
  "scores": {{
    "technical_depth": 0~1,
    "clarity": 0~1,
    "structure": 0~1,
    "citation_accuracy": {"null" if not referenced_papers else "0~1"}
  }},
  "reasons": {{"technical_depth": "...", "clarity": "...", "structure": "...", "citation_accuracy": "... or null"}},
  "facts": [{{"claim":"...","citation_id":"[12]","status":"supported|contested|missing","note":"..."}}]
}}
Specifications:
- citation_accuracy should be {"null if no references else 0~1 score" if not referenced_papers else "scored 0~1 if references exist"}
- All scores ∈ [0,1] floating point (keep 2-3 digits)
- facts can be empty list
- No text other than PART A and PART B allowed
{protection_clause}

Section: {section_name}
Has citations: {citation_flag}
Main text:\n{protected_text[:3000]}

Expert original suggestion integration source (read-only):
{reviews_joined}
"""

    moderator_raw = generate_text(
        prompt=moderator_prompt,
        model=MODEL_CONFIG.main_model,
        system="You are a strict moderator, only output the specified two parts.",
        temperature=0.25,
        max_tokens=1800,
        task_type="agent_review"
    ).strip()

    raw_reviews["moderator_agent"] = moderator_raw

    # Parse PART A improvement suggestions
    improvements_block = extract_last_asterisk_section(moderator_raw)
    # Parse JSON
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", moderator_raw, re.DOTALL)
    if not json_match or not improvements_block:
        raise AG2ReviewParseError("Moderator output missing required sections")
    try:
        scores_payload = json.loads(json_match.group(1))
    except Exception as e:
        raise AG2ReviewParseError(f"JSON parse error: {e}")

    # Validate structure
    scores_obj = scores_payload.get("scores", {})
    reasons_obj = scores_payload.get("reasons", {})
    facts_list = scores_payload.get("facts", [])

    # Normalize scores
    def _norm(v):
        if v is None:
            return None
        try:
            f = float(v)
            return max(0.0, min(1.0, f))
        except (TypeError, ValueError):
            return 0.0

    panel_scores = {
        "technical_depth": _norm(scores_obj.get("technical_depth")),
        "clarity": _norm(scores_obj.get("clarity")),
        "structure": _norm(scores_obj.get("structure")),
        "citation_accuracy": None if (scores_obj.get("citation_accuracy") is None or not referenced_papers) else _norm(scores_obj.get("citation_accuracy"))
    }

    cleaned_reviews["moderator_agent"] = improvements_block

    result = {
        "section_name": section_name,
        "original_content": section_text,
        "raw_reviews": raw_reviews,
        "cleaned_reviews": cleaned_reviews,
        "panel_scores": panel_scores,
        "panel_reasons": reasons_obj,
        "panel_facts": facts_list,
    }

    logger.info(f"[AG2] Review completed: scores={panel_scores}")
    return result
# ================== Adapter End ==================

############################################################
# Real group chat review wrapper based on ag2_agents.GroupChat
############################################################
def determine_needed_agents(current_scores: Dict[str, float], previous_scores: Dict[str, float] = None, is_first_iteration: bool = True) -> List[str]:
    """Determine the expert agents that need to review.
    
    Args:
        current_scores: Current dimension scores
        previous_scores: Previous round dimension scores 
        is_first_iteration: Whether it's the first round of review
        
    Returns:
        List of agents that need to review
    """
    if is_first_iteration:
        # First round review: all agents participate
        return ["technical_accuracy_agent", "clarity_agent", "structure_agent", "fact_checking_agent"]
    
    needed_agents = []
    
    # Mapping relationship
    score_to_agent = {
        "technical_depth": "technical_accuracy_agent", 
        "clarity": "clarity_agent",
        "structure": "structure_agent",
        "citation_accuracy": "fact_checking_agent"
    }
    
    for score_key, agent_key in score_to_agent.items():
        current_val = current_scores.get(score_key, 0.0)
        previous_val = previous_scores.get(score_key, 0.0) if previous_scores else 0.0
        
        # Conditions for re-review:
        # 1. Current score below threshold (< 0.7)
        # 2. Score dropped by more than 0.05 compared to previous round
        # 3. If previous round met threshold but current round doesn't
        needs_review = (
            current_val < THRESHOLD or  # Below threshold
            (previous_scores and current_val < previous_val - 0.05) or  # Significant drop
            (previous_scores and previous_val >= THRESHOLD and current_val < THRESHOLD)  # From pass to fail
        )
        
        if needs_review:
            needed_agents.append(agent_key)
    
    # If no agents need review, skip detailed review
    if not needed_agents:
        logger.info("All modules have met standards and are stable, skipping detailed review")
        return []
    
    return needed_agents

def run_groupchat_review(section_name: str, section_text: str, chapter: int, needed_agents: List[str] = None) -> Dict[str, Any]:
    """Use AG2LiteratureReviewSystem's GroupChat method to perform review and improvement on a single section.

    Args:
        section_name: Section name
        section_text: Section content
        chapter: Chapter number
        needed_agents: List of agents to participate in review, None means all agents

    Returns structure aligned with previous iterations' review/rewrite needs:
        {
            'raw': Original system return,
            'improvement_points': (Empty list if cannot precisely extract points),
            'improved_text': Rewritten text (if exists),
            'reviews': {...expert original text...},
            'final_recommendations': Captain comprehensive text,
            'panel_scores': {...},
            'panel_reasons': {...},
            'panel_facts': [...]
        }

    Note: AG2LiteratureReviewSystem.review_literature has internally completed multi-round round_robin interactions.
    If Captain has output structured scoring (JSON), it will be parsed and returned; otherwise panel_scores may all be None.
    """
    ag2_system = AG2LiteratureReviewSystem()
    topic = f"Chapter {chapter} - {section_name}"
    
    # Get chapter-specific topic constraints from research questions
    chapter_topic_domain = get_chapter_topic_from_questions(chapter)
    key_terms = extract_key_terms_from_question(load_research_questions()[chapter-1] if chapter <= len(load_research_questions()) else "")
    
    # Enhanced topic string with domain-specific constraints
    enhanced_topic = f"{topic} | Domain: {chapter_topic_domain} | Key Terms: {', '.join(key_terms) if key_terms else 'N/A'}"
    
    # If specified agents for review are provided, log information
    if needed_agents is not None:
        if not needed_agents:
            logger.info(f"[STEP3][{enhanced_topic}] Skip review: all modules have met standards")
            # Return result maintaining original state
            return {
                'raw': {"skipped": True, "reason": "all_modules_passed"},
                'improvement_points': [],
                'improved_text': section_text,
                'reviews': {},
                'final_recommendations': "All modules have met standards, no improvement needed.",
                'panel_scores': {"technical_depth": 0.75, "clarity": 0.75, "structure": 0.75, "citation_accuracy": 0.75},
                'panel_reasons': {},
                'panel_facts': []
            }
        else:
            logger.info(f"[STEP3][{enhanced_topic}] Selective review：{', '.join(needed_agents)}")
    
    results = ag2_system.review_literature(section_text, enhanced_topic, needed_agents=needed_agents)
    # Proactively output the group chat message again at the step3 level to ensure it is written to step3.out (even if it has already been recorded internally)
    try:
        chat_hist = results.get('chat_history', []) if isinstance(results, dict) else []
        if chat_hist:
            logger.info(f"[STEP3][{topic}] GroupChat Again, recording begins (a total of {len(chat_hist)} messages)")
            for idx, msg in enumerate(chat_hist, 1):
                name = msg.get('name','?') if isinstance(msg, dict) else 'unknown'
                content = msg.get('content','') if isinstance(msg, dict) else str(msg)
                logger.info(f"[STEP3][{topic}] ---- Message {idx:02d} | {name} | len={len(content)} ----")
                for line in (content.splitlines() or ['']):
                    logger.info(line)
            logger.info(f"[STEP3][{topic}] GroupChat record end")
        else:
            logger.warning(f"[STEP3][{topic}] Failed to get chat_history, possibly parsing failed or empty")
    except Exception as _e:
        logger.warning(f"[STEP3][{topic}] Exception while recording chat_history: {_e}")
    improved = results.get('improved_content') or section_text

    # Extract improvement points
    rec_text = results.get('final_recommendations', '') or ''
    block = extract_last_asterisk_section(rec_text) or rec_text
    points = parse_improvement_points(block)

    panel_scores = results.get('panel_scores', {}) or {}
    # Unify four keys
    for k in ["technical_depth", "clarity", "structure", "citation_accuracy"]:
        panel_scores.setdefault(k, None)

    return {
        'raw': results,
        'improvement_points': points,
        'improved_text': improved,
        'reviews': results.get('reviews', {}),
        'final_recommendations': rec_text,
        'panel_scores': panel_scores,
        'panel_reasons': results.get('panel_reasons', {}),
        'panel_facts': results.get('panel_facts', [])
    }

def get_moderator_improvements_from_block(improvements_block: str) -> List[str]:
    section = extract_last_asterisk_section(improvements_block) or improvements_block
    return parse_improvement_points(section)

def derive_needed_agents(panel_scores: Dict[str, Optional[float]], referenced_papers: bool) -> List[str]:
    needed = []
    for key, agent_key in [
        ("technical_depth", "technical_accuracy_agent"),
        ("clarity", "clarity_agent"),
        ("structure", "structure_agent"),
    ]:
        v = panel_scores.get(key)
        if v is not None and v < THRESHOLD:
            needed.append(agent_key)
    if referenced_papers:
        v = panel_scores.get("citation_accuracy")
        if v is not None and v < THRESHOLD:
            needed.append("fact_checking_agent")
    return needed

def panel_scores_reach_threshold(panel_scores: Dict[str, Optional[float]], referenced_papers: bool) -> bool:
    base_ok = all((panel_scores.get(k) is not None and panel_scores.get(k) >= THRESHOLD) for k in ["technical_depth", "clarity", "structure"])
    if not base_ok:
        return False
    if referenced_papers:
        v = panel_scores.get("citation_accuracy")
        return v is not None and v >= THRESHOLD
    return True

def get_all_chapter_files() -> List[Path]:
    """Get all JSON files from the initial_chapters directory."""
    chapters_dir = Path("initial_chapters")
    if not chapters_dir.exists():
        raise FileNotFoundError(f"Directory {chapters_dir} not found")
    return sorted(chapters_dir.glob("chapter_*.json"))

def create_chapter_markdown(chapter_number: int, sections: dict, consolidated_outputs: dict) -> None:
    """Create a markdown file for a chapter combining all sections and references."""
    output_dir = Path("chapter_markdowns")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"chapter_{chapter_number}.md"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write chapter header
            f.write(f"# Chapter {chapter_number}\n\n")
            
            # Process each section in order
            for section_name, original_content in sections.items():
                f.write(f"## {section_name}\n\n")
                
                # Check if this is a references section
                if section_name.upper() in ['REFERENCES', 'REFERENCE', 'BIBLIOGRAPHY']:
                    # Write preserved references without any modifications
                    f.write(f"{original_content}\n\n")
                else:
                    # Get final improved content from consolidated output
                    section_output = consolidated_outputs.get(section_name, {})
                    final_result = section_output.get("final_result", {})
                    
                    # Add citation accuracy metrics if available and not a references section
                    if "quality_assessment" in final_result and "metrics" in final_result["quality_assessment"]:
                        citation_metrics = final_result["quality_assessment"]["metrics"].get("citation_accuracy", {})
                        if citation_metrics:
                            f.write(f"### Citation Accuracy Analysis\n\n")
                            f.write(f"**Overall Score: {citation_metrics.get('score', 0):.2f}**\n\n")
                            
                            # Add detailed citation analysis
                            f.write("#### Individual Citation Analysis\n\n")
                            for citation in citation_metrics.get('citation_analysis', []):
                                score_color = "green" if citation['score'] >= 0.7 else "orange" if citation['score'] >= 0.4 else "red"
                                f.write(f"- **Citation [{citation['citation_id']}]**: Score: <span style='color:{score_color}'>{citation['score']:.2f}</span>\n")
                                f.write(f"  - **Justification**: {citation['justification']}\n")
                                
                                # Add improvement suggestions for citations below threshold
                                if citation['score'] < 0.7:
                                    suggestions = citation_metrics.get('improvement_suggestions', {}).get(citation['citation_id'], {})
                                    if suggestions:
                                        f.write(f"  - **Suggestion**: {suggestions.get('suggestion', 'No specific suggestion')}\n")
                                f.write("\n")
                            
                            f.write("---\n\n")
                    
                    final_content = final_result.get("final_text", original_content)
                    f.write(f"{final_content}\n\n")
        
        logger.info(f"Created markdown file for Chapter {chapter_number}: {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating markdown file for Chapter {chapter_number}: {e}")
        logger.error(traceback.format_exc())

############################################################
# main process
############################################################

def main():
    try:
        # Load research questions
        research_questions = load_research_questions()
        logger.info(f"Loaded {len(research_questions)} research questions for dynamic topic constraint")
        
        chapter_files = get_all_chapter_files()
        if not chapter_files:
            logger.error("No chapter files found under initial_chapters")
            return
        logger.info(f"Processing {len(chapter_files)} chapter JSON files")

        for chapter_file in chapter_files:
            try:
                chapter_number = int(chapter_file.stem.split('_')[1])
                logger.info("\n" + '='*90)
                logger.info(f"Processing Chapter {chapter_number}")
                logger.info('='*90)
                with open(chapter_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                sections = data.get('sections', {})
                referenced_papers = data.get('referenced_papers', {})
                chapter_outputs = {}

                for section_name, section_text in sections.items():
                    if section_name.upper() in ['REFERENCES', 'REFERENCE', 'BIBLIOGRAPHY']:
                        logger.info(f"Skip references section {section_name}")
                        chapter_outputs[section_name] = {
                            "metadata": {"chapter": chapter_number, "section": section_name, "skipped": True},
                            "original_content": section_text,
                            "final_result": {"final_text": section_text, "status": "preserved"}
                        }
                        continue
                    if not section_text.strip():
                        logger.warning(f"Empty section skipped: {section_name}")
                        continue

                    logger.info(f"\n--- Chapter {chapter_number} Section: {section_name} ---")
                    consolidated = {
                        "metadata": {
                            "chapter": chapter_number,
                            "section": section_name,
                            "timestamp": time.strftime('%Y%m%d_%H%M%S'),
                            "model": MODEL_CONFIG.main_model,
                            "skipped": False
                        },
                        "iterations": [],
                        "final_result": None
                    }

                    current_text = section_text
                    needed_agents = None  # The first round: None -> ag2_group_review -> internal decision
                    last_panel_scores: Dict[str, Optional[float]] = {}
                    previous_panel_scores: Dict[str, Optional[float]] = {}  # Track the previous round's score

                    # === Iterative review + rewriting (baseline + rewrites) ===
                    for iteration in range(1, MAX_ITERATIONS + 1):
                        iteration_meta: Dict[str, Any] = {
                            "iteration_number": iteration,
                            "type": "baseline" if iteration == 1 else "rewrite",
                            "reviews": None,
                            "improvement_points": [],
                            "text": {"before": current_text, "after": None},
                            "panel_scores": {},
                            "panel_reasons": {},
                            "panel_facts": [],
                            "quality_assessment": {},
                            "rewrite_triggered": False,
                            "triggers": []
                        }
                        
                        # Identify the agents that require review
                        if iteration == 1:
                            # First round of review: All agents participated
                            selected_agents = None  # "None" indicates all agents.
                            logger.info(f"Iteration {iteration}: First complete review")
                        else:
                            # Subsequent iteration: Selective review
                            selected_agents = determine_needed_agents(
                                current_scores=last_panel_scores,
                                previous_scores=previous_panel_scores,
                                is_first_iteration=False
                            )
                            if not selected_agents:
                                logger.info(f"Iteration {iteration}: All modules meet standards, skip review")
                                # Mark as passed directly
                                panel_scores = last_panel_scores.copy()
                                # Build simplified review results
                                gc_review = {
                                    'panel_scores': panel_scores,
                                    'improvement_points': [],
                                    'improved_text': current_text,
                                    'raw': {"skipped": True, "reason": "all_modules_stable"}
                                }
                            else:
                                logger.info(f"Iteration {iteration}: Selective review agents -> {selected_agents}")

                        if iteration == 1 or selected_agents:
                            try:
                                gc_review = run_groupchat_review(section_name, current_text, chapter_number, selected_agents)
                            except Exception as e:
                                logger.error(f"GroupChat Call failed iteration {iteration}: {e}")
                                logger.error(traceback.format_exc())
                                # Treated as completely absent -> Forced rewrite
                                gc_review = {"panel_scores": {k: 0.0 for k in ["technical_depth","clarity","structure","citation_accuracy"]},
                                              "improvement_points": [],
                                              "improved_text": current_text,
                                              "raw": {"error": str(e)}}
                                iteration_meta["triggers"].append("groupchat_error")

                        improvement_points = gc_review.get('improvement_points', []) or []
                        improved_text_from_captain = gc_review.get('improved_text') or current_text
                        panel_scores = gc_review.get('panel_scores', {}) or {}

                        # For selective review, we need to merge the scores from the previous round
                        if iteration > 1 and selected_agents is not None and selected_agents:
                            # Only update the scores of the reviewed modules, keep the scores from the previous round for others
                            agent_to_score = {
                                "technical_accuracy_agent": "technical_depth",
                                "clarity_agent": "clarity", 
                                "structure_agent": "structure",
                                "fact_checking_agent": "citation_accuracy"
                            }
                            
                            merged_scores = last_panel_scores.copy()
                            for agent in selected_agents:
                                score_key = agent_to_score.get(agent)
                                if score_key and score_key in panel_scores:
                                    merged_scores[score_key] = panel_scores[score_key]
                                    logger.info(f"Updated {score_key}: {last_panel_scores.get(score_key, 0.0):.3f} -> {panel_scores[score_key]:.3f}")
                            
                            panel_scores = merged_scores
                        
                        for k in ["technical_depth","clarity","structure","citation_accuracy"]:
                            panel_scores.setdefault(k, None)
                        # Missing -> 0.0
                        for k,v in list(panel_scores.items()):
                            if v is None:
                                panel_scores[k] = 0.0
                                iteration_meta["triggers"].append(f"missing_{k}")

                        # Unsupported claim heuristic on ORIGINAL (or current) text
                        claim_pattern = re.compile(r"\b(suggests?|indicates?|implies?)\b", re.IGNORECASE)
                        unsupported_claim = False
                        for para in [p.strip() for p in current_text.split('\n') if p.strip()]:
                            if claim_pattern.search(para) and re.search(r'\[[0-9]+\]', para) is None:
                                unsupported_claim = True
                                iteration_meta["triggers"].append("unsupported_claim_no_citation")
                                break

                        # Pass check
                        all_pass = all(panel_scores.get(k,0) >= THRESHOLD for k in ["technical_depth","clarity","structure","citation_accuracy"])
                        need_rewrite = (not all_pass) or unsupported_claim
                        if not all_pass:
                            iteration_meta["triggers"].append("scores_below_threshold")

                        # Assemble iteration meta
                        iteration_meta.update({
                            "reviews": gc_review.get('raw'),
                            "improvement_points": improvement_points,
                            "panel_scores": panel_scores,
                            "panel_reasons": gc_review.get('panel_reasons', {}),
                            "panel_facts": gc_review.get('panel_facts', []),
                        })

                        # Baseline: Don't adopt Captain rewrite, only for recording (unless pass -> then accept improved)
                        if iteration == 1:
                            after_text = current_text if need_rewrite else improved_text_from_captain
                        else:
                            after_text = improved_text_from_captain  # Input from previous rewrite round is already current_text

                        qa = assess_quality_from_panel(panel_scores, current_text, after_text)
                        iteration_meta["quality_assessment"] = qa
                        iteration_meta["text"]["after"] = after_text
                        consolidated['iterations'].append(iteration_meta)
                        
                        # Save current round as previous for next round
                        previous_panel_scores = last_panel_scores.copy()
                        last_panel_scores = panel_scores.copy()

                        if not need_rewrite:
                            # Passed: Use after_text (possibly Captain's improved version)
                            current_text = after_text
                            logger.info(f"Section {section_name}: iteration {iteration} PASSED (all >= {THRESHOLD}).")
                            break

                        # Need rewrite
                        if iteration == MAX_ITERATIONS:
                            logger.info(f"Section {section_name}: reached MAX_ITERATIONS with unmet threshold; stopping.")
                            current_text = after_text  # Keep latest text (still original if baseline fail)
                            break

                        # Prepare rewrite (ignore Captain improved if baseline fail; we rewrite from original or last text)
                        base_for_rewrite = current_text if iteration == 1 else after_text
                        rewrite_points = improvement_points.copy()
                        if not rewrite_points:
                            # fallback synthesized points per low score
                            for k in ["technical_depth","clarity","structure","citation_accuracy"]:
                                if panel_scores.get(k,0) < THRESHOLD:
                                    rewrite_points.append(f"Increase {k} (current score {panel_scores.get(k,0):.2f}) with concrete, cited, and precise revisions.")
                            if unsupported_claim:
                                rewrite_points.append("Add supporting citations for inferential claims (suggests/indicates/implies) or qualify them.")
                        
                        # First rewrite attempt
                        rewritten_text = rewrite_with_ollama(base_for_rewrite, rewrite_points, section_name, research_questions)
                        
                        # Validate topic consistency
                        max_rewrite_attempts = 2
                        for attempt in range(max_rewrite_attempts):
                            if validate_topic_consistency_simple(section_text, rewritten_text, section_name, research_questions):
                                current_text = rewritten_text
                                break
                            else:
                                logger.warning(f"Rewrite attempt {attempt + 1} failed topic consistency check, retrying...")
                                # Enhanced prompt for retry
                                enhanced_points = [
                                    f"CRITICAL: Maintain the EXACT same subject matter as the original text about {section_text.split('.')[0]}",
                                    "Apply improvements to the EXISTING content without changing the fundamental topic",
                                ] + rewrite_points[:5]  # Limit to avoid overwhelming the model
                                
                                rewritten_text = rewrite_with_ollama(base_for_rewrite, enhanced_points, section_name, research_questions)
                        else:
                            # If all attempts fail, keep the original text
                            logger.error(f"All rewrite attempts failed topic consistency. Keeping original text for {section_name}")
                            current_text = base_for_rewrite
                            
                        logger.info(f"Section {section_name}: iteration {iteration} triggered rewrite -> proceeding to iteration {iteration+1}")
                        continue

                    consolidated['final_result'] = {
                        "original_text": section_text,
                        "final_text": strip_chem_tags(current_text) if REMOVE_CHEM_TAGS else current_text,
                        "total_iterations": len(consolidated['iterations']),
                        "final_status": "improved" if current_text != section_text else "unchanged",
                        "final_quality_assessment": {"metrics": last_panel_scores},
                        "referenced_papers_used": bool(referenced_papers)
                    }

                    out_path = save_consolidated_output(consolidated, f"chapter_{chapter_number}_{section_name}")
                    logger.info(f"Done: {section_name} -> {out_path}")
                    chapter_outputs[section_name] = consolidated

                create_chapter_markdown(chapter_number, sections, chapter_outputs)
                logger.info(f"Chapter {chapter_number} completed")
            except Exception as ce:
                logger.error(f"Chapter processing exception {chapter_file}: {ce}")
                logger.error(traceback.format_exc())
                continue

        logger.info("All chapters have been processed.")
    except Exception as e:
        logger.error(f"Main process error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
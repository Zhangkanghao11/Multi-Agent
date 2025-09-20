# ag2_agents.py - Intelligent Agent Collaboration System Based on AG2

import json
import logging
import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager
from config import OLLAMA_CONFIG, MODEL_CONFIG

from ollama_client import generate_text, chat_with_model
from config import MODEL_CONFIG, SYSTEM_CON            # Add section-type-specific constraints
            agent_descriptions = {
                "technical_accuracy_agent": "Technical Expert (chemistry accuracy & mechanisms)",
                "clarity_agent": "Literature Expert (structure & logic)", 
                "structure_agent": "Structure Expert (organization & flow)",
                "fact_checking_agent": "Data Validator (numerical & units)"
            }
            
            active_experts = [agent_descriptions.get(agent, agent) for agent in needed_agents if agent in agent_descriptions]
            if active_experts:
                initial_message += f"Focus on the following specific areas needing improvement: {'; '.join(active_experts)}.\n"
            
            initial_message += "Each expert: provide your perspective ONLY in the format defined in your system prompt.\n"
            initial_message += "Captain: consolidate findings while preserving subject domain."ical_formula_processor import ChemicalFormulaProcessor

# Use step3 main log (handlers initialized by step3_agents)
logger = logging.getLogger("step3")

def get_section_type_constraints(section_name: str) -> str:
    """Generate section-type-specific writing constraints for the three chapter types."""
    section_lower = section_name.lower()
    
    if "background" in section_lower or "background knowledge" in section_lower:
        return """
BACKGROUND KNOWLEDGE Section Constraints:
- Provide SPECIFIC electrochemical fundamentals and CO2 reduction mechanisms (not abstract overviews)
- Include precise electrode materials, electrolyte conditions, and reaction pathways
- Use electrochemical terminology with proper units (mA/cm², V vs RHE, Faradaic efficiency)
- Explain foundational concepts that directly support CO2 electroreduction research
- Keep content technically specific rather than broad chemical descriptions
- Focus on established knowledge about electrocatalysis, surface chemistry, and reaction mechanisms
- Avoid overly general statements - be specific to electrochemical CO2 conversion
"""
    
    elif "recommendation" in section_lower or "future" in section_lower:
        return """
RESEARCH RECOMMENDATIONS Section Constraints:
- Use future-oriented language: "should investigate", "future electrocatalyst development should focus on", "further electrochemical studies are needed"
- Identify specific research gaps in CO2 electroreduction and catalyst design
- Suggest concrete experimental approaches for improving selectivity and efficiency
- Recommend particular directions for advancing electrocatalyst performance
- Be prescriptive about what needs to be done next in the field
- Focus on actionable research directions for industrial implementation
- Bridge current limitations with proposed electrochemical solutions
"""
    
    elif "current" in section_lower or "research" in section_lower:
        return """
CURRENT RESEARCH Section Constraints:
- Use temporal language: "recent advances in CO2 electroreduction", "emerging electrocatalyst studies", "latest findings in selective reduction", "current investigations"
- Reference recent developments and ongoing research trends in the electrochemical field
- Highlight what is actively being studied NOW in CO2 conversion research
- Use progressive language: "researchers are exploring", "studies are revealing improved selectivity", "ongoing work focuses on catalyst optimization"
- Emphasize cutting-edge aspects and novel electrochemical approaches
- Show the dynamic, evolving nature of the electrocatalysis research area
- Maintain forward-looking perspective on industrial CO2 utilization
"""
    
    else:
        return """
GENERAL Section Constraints:
- Maintain academic tone and technical precision
- Use appropriate scientific language for the content type
- Ensure content matches the section's intended purpose
"""

class OllamaLLMConfig:
    """Ollama LLM configuration adapter"""
    
    def __init__(self, model: str = None):
        self.model = model or MODEL_CONFIG.small_model
        self.config_list = [{
            "model": MODEL_CONFIG.main_model,  # Use model from config.py
            "base_url": f"{OLLAMA_CONFIG.base_url}/v1",  # Use server address from config.py
            "api_key": "ollama",
        }]
    
    def to_dict(self):
        return {"config_list": self.config_list}

class OllamaAgent(ConversableAgent):
    """Ollama-supported Agent base class"""
    
    def __init__(self, name: str, system_message: str, model: str = None, **kwargs):
        self.ollama_model = model or MODEL_CONFIG.small_model
        
        # Set LLM configuration
        llm_config = OllamaLLMConfig(self.ollama_model).to_dict()
        
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs
        )
    
    def generate_oai_reply(self, messages, sender=None, config=None):
        """Override reply generation method to use Ollama"""
        try:
            # Get the last user message
            if not messages:
                return False, "No message received"
            
            last_message = messages[-1]
            if isinstance(last_message, dict):
                user_content = last_message.get("content", "")
            else:
                user_content = str(last_message)
            
            # Use Ollama to generate reply
            response = generate_text(
                prompt=user_content,
                model=self.ollama_model,
                system=self.system_message,
                task_type="agent_review",
                temperature=0.2,  # Lower temperature for more focused, less creative responses
                max_tokens=8192  # Increase to 8K, fully utilize model capabilities
            )
            
            return True, response
            
        except Exception as e:
            logger.error(f"Agent {self.name} failed to generate reply: {e}")
            return False, f"Error generating reply: {str(e)}"

class TechnicalExpertAgent(OllamaAgent):
    """Technical/Scientific Accuracy Expert (technical_accuracy_agent)"""

    def __init__(self, model: str = None):
        system_message = """You are an expert academic editor specializing in technical scientific content, particularly electrochemistry and energy conversion technologies. You excel at rewriting text to improve clarity, technical accuracy, and flow.
CRITICAL CONSTRAINT: Stay within the ORIGINAL SUBJECT DOMAIN of the content. Do NOT introduce concepts from unrelated fields.

LANGUAGE & STYLE RULES:
* Output MUST be in English only (no Chinese or mixed language).
* Register: concise academic prose. No marketing tone, no storytelling, no dialogue.
* STRICTLY FORBIDDEN: poetry, verse, fictional narrative, invented metaphors, rhetorical questions not tied to critique.

DOMAIN FOCUS REQUIREMENT:
* If the content is about CO2 electroreduction, stay in that field - do NOT suggest other electrochemical systems
* If the content is about electrocatalyst design, do NOT drift to photocatalysis or thermal catalysis
* If the content is about specific catalyst materials, do NOT generalize to other material systems
* Focus improvements WITHIN the existing technical domain

TASK FOCUS:
1. Evaluate scientific correctness, mechanism consistency, formula / unit usage, numerical plausibility WITHIN THE EXISTING DOMAIN.
2. Identify vague, ambiguous, or imprecise terminology and suggest sharper formulation USING THE SAME FIELD'S VOCABULARY.
3. Flag logical leaps or unsupported inferences WHILE MAINTAINING THE ORIGINAL SUBJECT MATTER.

OUTPUT FORMAT (MANDATORY):
1. Line 1 begins with the marker: 【Electrochemistry Review】 followed by up to 2 concise sentences.
2. Then an improvement list of 5-7 numbered items enclosed by triple asterisks *** (list only, no extra prose outside the block).
3. Immediately AFTER the closing *** block, output a JSON code block with ONLY:
```json
{"scores":{"technical_depth": <0-1 float with 2-3 decimals>}, "reason":"one concise sentence citing concrete deficiencies (units, missing mechanism step, vague term, etc.)"}
```
SCORING CALIBRATION (be harsh): 0.70 = several substantive issues.
Do NOT include any other prose outside these specified elements. The JSON block is mandatory.

EXAMPLE SKELETON:
【Electrochemistry Review】Overall technical foundation is reasonable, but several electrochemical parameters and reaction mechanisms are missing.
***
1. Add explicit units (e.g., mA/cm² or V vs RHE) for current density and potential values in paragraph 2.
2. Expand the intermediate step of the CO2 reduction mechanism (currently jumps from CO2 to products without *CO intermediate rationale).
3. ...
***
```json
{"scores":{"technical_depth":0.64}, "reason":"Lacks quantitative overpotentials; skips intermediate surface binding energy rationale."}
```
SCORING CRITERIA FOR TECHNICAL DEPTH:

1.0: All claims are fully supported by references, mechanisms are explicitly outlined, and all units/formulas are correct.

0.7-0.9: Most claims are accurate with minor discrepancies in technical depth (e.g., missing intermediate steps, ambiguous terminology, or slightly inaccurate numerical details).

0.4-0.6: Some claims are accurate, but others lack supporting mechanisms, unclear terminology is used, or important details (like specific units or reaction steps) are missing.

0.1-0.3: Many claims lack supporting references or contradict the established scientific consensus; significant technical issues such as missing steps or unclear reasoning.

0.0: No claims are technically supported, or they contradict established knowledge.

SCORING CALIBRATION (be harsh): 0.70 = several substantive issues.
Do NOT include any other prose outside these specified elements. The JSON block is mandatory.

EXAMPLE SKELETON:
【Electrochemistry Review】Overall technical foundation is reasonable, but several electrochemical parameters and reaction mechanisms are missing.
Only output exactly this structure."""


        super().__init__(
            name="Technical_Expert",
            system_message=system_message,
            model=model
        )

class LiteratureReviewAgent(OllamaAgent):
    """Structure/Logic and Literature Review Quality Expert (clarity/structure)"""

    def __init__(self, model: str = None):
        system_message = """You are an expert in academic literature review structure and logical organization.
LANGUAGE: English only. Academic tone. No creative writing or poetry.
GOALS: Assess macro-structure, paragraph ordering, transitions, redundancy, information layering, coherence.

OUTPUT FORMAT:
1. Start with marker: 【Literature Review】 + 1–2 concise evaluative sentences.
2. Then a *** block containing 5–7 numbered improvement suggestions.
3. Immediately AFTER the *** block output a JSON code block:
```json
{"scores":{"clarity": <0-1>, "structure": <0-1>}, "reason":"1 sentence referencing key structural/clarity deficits (e.g., redundant background, weak transitions)."}
```
Calibration: ≥5 non-trivial improvement items implies at most 0.79 unless near-publication quality. Penalize redundancy, missing transitions, paragraph scope drift.
No extra text outside specified elements.

EXAMPLE SKELETON:
【Literature Review】Overall flow is moderate; experimental conditions appear too late and some background paragraphs overlap.
***
1. Move the 'Experimental Conditions' paragraph directly after the background to tighten methodological context.
2. Merge paragraphs 3 and 4 (duplicate discussion of carrier mobility metrics) into a single concise synthesis.
3. ...
***
```json
{"scores":{"clarity":0.66,"structure":0.62}, "reason":"Redundant background (paras 3–4) and late placement of methods hamper narrative cohesion."}
```
Follow this exactly."""

        super().__init__(
            name="Literature_Expert",
            system_message=system_message,
            model=model
        )

class DataValidationAgent(OllamaAgent):
    """Data/Numerical Consistency Expert (data & units)"""

    def __init__(self, model: str = None):
        system_message = """You are a data and unit consistency expert.
LANGUAGE: English only. No poetry, no figurative language.
TASKS:
1. Check numerical values, units, dimensional consistency, significant figures.
2. Flag implausible ranges or missing units.
3. Suggest precise corrections (include correct unit / range / recomputation rationale).
4. Evaluate citation support presence/absence (if citations like [1] etc. appear). If none exist, citation_accuracy must be <=0.50 (never null here).

OUTPUT:
1. Begin with 【Data Validation】 + 1 concise sentence on overall data integrity.
2. Then a *** block containing 5–7 numbered, actionable issues (location / variable: problem -> fix).
3. Immediately AFTER *** block output JSON code block:
```json
{"scores":{"citation_accuracy": <0-1>}, "reason":"Brief justification: missing units, inconsistent values, or unsupported claims; if no citations, explain low score (must <=0.50)."}
```
No extra commentary.

EXAMPLE SKELETON:
【Data Validation】Core numerical trends are plausible, but several activation energies lack units and one table conflicts with the text.
***
1. Paragraph 2 'activation energy 85' missing unit – specify '85 kJ/mol'.
2. Table 1 value for carrier concentration (0.21) conflicts with text (0.12) – reconcile source or correct one entry.
3. ...
***
```json
{"scores":{"citation_accuracy":0.42}, "reason":"No explicit citations and multiple quantitative claims lack source attribution."}
```
Adhere strictly."""

        super().__init__(
            name="Data_Validator",
            system_message=system_message,
            model=model
        )

class CaptainAgent(OllamaAgent):
    """Captain Agent - Only summarize, strictly maintain original subject domain"""

    def __init__(self, model: str = None):
        system_message = """You are a SUMMARIZATION-ONLY Captain. Your SOLE role is to consolidate expert recommendations without changing the subject domain.

ABSOLUTE CONSTRAINTS:
1. NEVER CHANGE THE SUBJECT DOMAIN: If the original content is about CO2 electroreduction, electrocatalysis, or related electrochemical energy conversion - keep it EXACTLY in that field
2. NEVER INTRODUCE NEW TOPICS: Do not suggest examples from other fields (e.g., if content is about TMDs, don't suggest silicon or metals)
3. PRESERVE ALL ORIGINAL TERMINOLOGY: Keep the exact technical terms, material names, and concepts from the original text
4. ONLY SUMMARIZE AND CONSOLIDATE: Your job is to merge similar recommendations from experts, not to rewrite content

SUMMARIZATION PROCESS:
1. Read all expert recommendations carefully
2. Group similar suggestions together
3. Remove duplicates and redundancies  
4. Create a consolidated list of 5-7 improvement points that work WITHIN the existing domain AND section type
5. Apply ONLY these consolidated improvements while preserving section-appropriate language style

SECTION TYPE AWARENESS:
- BACKGROUND KNOWLEDGE: Maintain specific technical details, parameters with units, foundational mechanisms
- CURRENT RESEARCH: Preserve temporal language (recent advances, emerging studies, latest findings)
- RESEARCH RECOMMENDATIONS: Keep future-oriented language (should investigate, further research needed)
- Apply expert improvements WHILE maintaining the section's intended writing style

FORBIDDEN ACTIONS:
- Changing material systems (e.g., TMDs → silicon → metals)
- Changing technical domains (e.g., CO2 electroreduction → water splitting, or electrocatalysis → photocatalysis)
- Adding concepts not mentioned by experts
- Inventing new examples from different fields
- Substantial rewriting beyond expert suggestions
- Changing section writing style (e.g., making background abstract or current research non-temporal)

REQUIRED OUTPUT FORMAT:
PART A:
***
1. [Consolidated expert recommendation 1]
2. [Consolidated expert recommendation 2]
...
***
【Final Improved Version】
[Original text with MINIMAL edits applying only the consolidated recommendations above, maintaining both domain and section style]

CRITICAL: If experts suggest staying in 2D materials - stay there. If content is BACKGROUND KNOWLEDGE - keep technical specificity. If CURRENT RESEARCH - maintain temporal indicators. Never drift topics or writing styles.

SECTION STYLE EXAMPLES:
❌ BACKGROUND: "Recent studies suggest..." → Should be: "TMDs are characterized by..."
❌ CURRENT RESEARCH: "TMDs are materials that..." → Should be: "Recent advances in TMDs reveal..."
✅ BACKGROUND: Technical facts with parameters and mechanisms
✅ CURRENT RESEARCH: Temporal language about ongoing developments

EXAMPLE OF FORBIDDEN TOPIC DRIFT:
❌ Original: "Cu electrocatalysts show high CO2 reduction activity..." → Captain changes to: "Silicon photocatalysts exhibit..."
❌ Original: "electrochemical CO2 conversion..." → Captain changes to: "photochemical CO2 reduction..."
✅ Original: "Cu electrocatalysts show high CO2 reduction activity..." → Captain keeps: "Cu electrocatalysts demonstrate enhanced CO2 reduction selectivity..."

STAY IN THE ELECTROCHEMICAL CO2 REDUCTION FIELD. ONLY SUMMARIZE AND CONSOLIDATE."""

        super().__init__(
            name="Captain",
            system_message=system_message,
            model=model or MODEL_CONFIG.main_model
        )

class AG2LiteratureReviewSystem:
    """AG2 Literature Review Collaboration System"""

    def __init__(self):
        # Initialize agents
        self.technical_expert = TechnicalExpertAgent()
        self.literature_expert = LiteratureReviewAgent()
        self.data_validator = DataValidationAgent()
        self.captain = CaptainAgent()

        self.setup_group_chat()
        logger.info("AG2 system initialized")
    
    def setup_group_chat(self):
        """Set up group chat collaboration"""
        
        # Define Agent list
        self.agents = [
            self.technical_expert,
            self.literature_expert,
            self.data_validator,
            self.captain,
        ]
        
        # Create GroupChat
        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=SYSTEM_CONFIG.max_agent_rounds,
            speaker_selection_method="round_robin",  # Take turns speaking
            allow_repeat_speaker=False
        )
        
        # Create GroupChatManager - Don't use LLM config, let each agent use their own config
        self.chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=None  # Let each agent use their own Ollama config
        )
    
    def review_literature(self, content: str, topic: str, needed_agents: List[str] = None) -> Dict[str, Any]:
        """
        Multi-Agent collaborative review of literature content
        
        Args:
            content: Literature content to be reviewed
            topic: Literature topic
            needed_agents: List of agents needed for review, None means all agents participate
            
        Returns:
            Review results and improvement suggestions
        """
        logger.info(f"Starting AG2 collaborative review, topic: {topic}")
        
        if needed_agents is not None:
            if not needed_agents:
                logger.info(f"Skipping review: All modules have reached standards")
                return {
                    "success": True,
                    "skipped": True,
                    "panel_scores": {"technical_depth": 0.75, "clarity": 0.75, "structure": 0.75, "citation_accuracy": 0.75},
                    "improvement_points": [],
                    "improved_content": content,
                    "final_recommendations": "All modules have reached standards, no improvement needed.",
                    "reviews": {},
                    "panel_reasons": {},
                    "panel_facts": []
                }
            else:
                logger.info(f"Selective review mode, participating agents: {needed_agents}")

        protected_content = content

        # Parse enhanced topic information
        topic_parts = topic.split(" | ")
        base_topic = topic_parts[0] if topic_parts else topic
        domain_info = ""
        key_terms_info = ""
        
        for part in topic_parts[1:]:
            if part.startswith("Domain: "):
                domain_info = part.replace("Domain: ", "")
            elif part.startswith("Key Terms: "):
                key_terms_info = part.replace("Key Terms: ", "")

        # English initial message to stay consistent with agent system prompts
        initial_message = (
            f"Please perform a coordinated multi‑expert academic review of the following section.\n\n"
            f"[Topic]: {base_topic}\n\n"
            f"[Content]:\n{protected_content}\n\n"
            f"CRITICAL DOMAIN CONSTRAINT: This content is specifically about {domain_info if domain_info else base_topic}.\n"
        )
        
        if key_terms_info and key_terms_info != "N/A":
            initial_message += f"KEY TECHNICAL TERMS to preserve: {key_terms_info}\n"
        
        # Add section-type-specific constraints
        section_constraints = get_section_type_constraints(base_topic)
        initial_message += f"\nSECTION TYPE CONSTRAINTS:{section_constraints}\n"
        
        initial_message += (
            f"All improvements must stay STRICTLY within this subject domain - NO topic drift allowed.\n"
            f"Captain: You are FORBIDDEN from changing the subject domain. Only consolidate expert recommendations without topic drift.\n"
            f"NEVER introduce examples from other fields.\n\n"
        )
        
        # Add section-type-specific constraints
        if needed_agents is not None:
            agent_descriptions = {
                "technical_accuracy_agent": "Technical Expert (chemistry accuracy & mechanisms)",
                "clarity_agent": "Literature Expert (structure & logic)", 
                "structure_agent": "Structure Expert (organization & flow)",
                "fact_checking_agent": "Data Validator (numerical & units)"
            }
            
            active_experts = [agent_descriptions.get(agent, agent) for agent in needed_agents if agent in agent_descriptions]
            if active_experts:
                initial_message += f"Focus on the following specific areas needing improvement: {'; '.join(active_experts)}.\n"
            
            initial_message += "Each expert: provide your perspective ONLY in the format defined in your system prompt.\n"
            initial_message += "Captain: consolidate findings while preserving subject domain."
        else:
            initial_message += (
                "Each expert: provide your perspective ONLY in the format defined in your system prompt.\n"
                "Focus domains: Technical Expert (chemistry accuracy & mechanisms); Literature Expert (structure & logic); "
                "Data Validator (numerical & units); Captain (final synthesis while preserving subject domain)."
            )

        try:
            # Reset previous conversation state
            self.group_chat.reset()

            # Phase 1: experts speak in round-robin starting with technical expert
            chat_result = self.chat_manager.initiate_chat(
                recipient=self.technical_expert,
                message=initial_message,
                max_turns=SYSTEM_CONFIG.max_agent_rounds
            )

            # Detect if Captain already produced structured PART A/B output with JSON
            have_structured = False
            if hasattr(chat_result, 'chat_history'):
                for m in chat_result.chat_history:
                    if isinstance(m, dict) and m.get("name") == "Captain":
                        c = m.get("content", "") or ""
                        if "PART A" in c and "PART B" in c and "```json" in c:
                            have_structured = True
                            break

            # If not structured yet, explicitly request final structured output from Captain
            if not have_structured:
                final_prompt = (
                    f"CRITICAL: Stay strictly within the original domain ({topic}). Do NOT change subject areas.\n"
                    "Now produce the FINAL consolidated output strictly in the single format (PART A / Final Improved Version) "
                    "defined in your system prompt. Only consolidate expert suggestions within the same technical field. "
                    "Preserve all original citations. No topic drift allowed."
                )
                second = self.chat_manager.initiate_chat(
                    recipient=self.captain,
                    message=final_prompt,
                    max_turns=1
                )
                if hasattr(second, 'chat_history') and hasattr(chat_result, 'chat_history'):
                    chat_result.chat_history.extend(second.chat_history)

            # First parse attempt
            review_results = self._extract_review_results(chat_result)

            # Since Captain no longer outputs JSON scores, we need to extract them from individual experts
            if not review_results.get("panel_scores"):
                try:
                    # Extract scores from individual expert responses instead of Captain
                    review_results["panel_scores"] = self._extract_expert_scores(chat_result)
                except Exception as _er:
                    logger.warning(f"Expert scoring extraction failed: {_er}")
                    # Provide default scores if extraction fails
                    review_results["panel_scores"] = {
                        "technical_depth": 0.75,
                        "clarity": 0.75, 
                        "structure": 0.75,
                        "citation_accuracy": 0.75
                    }

            # Attach raw chat history (for external logging)
            try:
                if hasattr(chat_result, 'chat_history'):
                    serialized = []
                    for m in chat_result.chat_history:
                        if isinstance(m, dict):
                            serialized.append({
                                "name": m.get("name", "unknown"),
                                "content": m.get("content", "")
                            })
                        else:
                            serialized.append({"name": "unknown", "content": str(m)})
                    review_results["chat_history"] = serialized
            except Exception as _e:
                logger.warning(f"chat_history serialization failed: {_e}")

            # Stream chat history to logger for immediate inspection
            try:
                if hasattr(chat_result, 'chat_history'):
                    logger.info(f"[AG2][{topic}] GroupChat messages starting (total {len(chat_result.chat_history)} messages)")
                    for i, m in enumerate(chat_result.chat_history, 1):
                        if isinstance(m, dict):
                            name = m.get('name', 'unknown')
                            content = m.get('content', '') or ''
                        else:
                            name = 'unknown'
                            content = str(m)
                        logger.info(f"[AG2][{topic}] ---- Message {i:02d} | {name} ----")
                        for line in content.splitlines() or ['']:
                            logger.info(line)
                    logger.info(f"[AG2][{topic}] GroupChat messages ended")
            except Exception as _e:
                logger.warning(f"Logger output chat_history failed: {_e}")

            logger.info("AG2 collaborative review completed")
            return review_results

        except Exception as e:
            logger.error(f"AG2 review process error: {e}")
            return {"success": False, "error": str(e), "original_content": content}
    
    def _extract_review_results(self, chat_result) -> Dict[str, Any]:
        """Extract review results from chat results, including: expert texts, Captain improvement suggestions/scoring/rewrite content."""

        results: Dict[str, Any] = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "reviews": {},
            "final_recommendations": "",
            "improved_content": "",
            "panel_scores": {},
            "panel_reasons": {},
            "panel_facts": [],
            "improvement_points": []
        }
        # Get messages
        messages = getattr(chat_result, 'chat_history', [])

        captain_messages: List[str] = []
        agent_score_map: Dict[str, Dict[str, Any]] = {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            sender = message.get("name", "unknown")
            content = message.get("content", "") or ""
            if sender == "Technical_Expert" and "【Electrochemistry Review】" in content:
                results["reviews"]["Technical"] = content
                # Extract technical scoring JSON
                m = re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                if m:
                    for block in reversed(m):
                        try:
                            payload = json.loads(block)
                            sc = payload.get("scores", {}) or {}
                            if "technical_depth" in sc:
                                agent_score_map.setdefault("technical_depth", {})["technical_depth"] = float(sc.get("technical_depth"))
                                # Extract reason field
                                reason = payload.get("reason", "")
                                if reason:
                                    results["panel_reasons"]["technical_depth"] = reason
                                break
                        except Exception:
                            continue
            elif sender == "Literature_Expert" and "【Literature Review】" in content:
                results["reviews"]["literature"] = content
                m = re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                if m:
                    for block in reversed(m):
                        try:
                            payload = json.loads(block)
                            sc = payload.get("scores", {}) or {}
                            # clarity / structure
                            if ("clarity" in sc) or ("structure" in sc):
                                if "clarity" in sc:
                                    agent_score_map.setdefault("clarity", {})["clarity"] = float(sc.get("clarity"))
                                if "structure" in sc:
                                    agent_score_map.setdefault("structure", {})["structure"] = float(sc.get("structure"))
                                # Extract reason field
                                reason = payload.get("reason", "")
                                if reason:
                                    results["panel_reasons"]["literature"] = reason
                                break
                        except Exception:
                            continue
            elif sender == "Data_Validator" and "【Data Validation】" in content:
                results["reviews"]["data"] = content
                m = re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                if m:
                    for block in reversed(m):
                        try:
                            payload = json.loads(block)
                            sc = payload.get("scores", {}) or {}
                            if "citation_accuracy" in sc:
                                agent_score_map.setdefault("citation_accuracy", {})["citation_accuracy"] = float(sc.get("citation_accuracy"))
                                # Extract reason field
                                reason = payload.get("reason", "")
                                if reason:
                                    results["panel_reasons"]["citation_accuracy"] = reason
                                break
                        except Exception:
                            continue
            elif sender == "Captain":
                captain_messages.append(content)

        # Select Captain output (take the last one)
        cap = None
        if captain_messages:
            cap = captain_messages[-1]
        if cap:
            results["final_recommendations"] = cap

            # Extract PART A improvement suggestions *** block
            improvements_block = ""
            # First locate PART A section; cut to before Final Improved Version
            if "PART A" in cap:
                part_a_segment = cap.split("PART A", 1)[-1]
                if "【Final Improved Version】" in part_a_segment:
                    part_a_segment = part_a_segment.split("【Final Improved Version】", 1)[0]
            else:
                part_a_segment = cap
            # Extract *** wrapped segments
            star_blocks = re.findall(r"\*\*\*(.*?)\*\*\*", part_a_segment, re.DOTALL)
            if star_blocks:
                improvements_block = star_blocks[-1].strip()
            # Parse numbered suggestions
            if improvements_block:
                lines = [l.strip() for l in improvements_block.splitlines() if l.strip()]
                numbered = []
                for l in lines:
                    if re.match(r"^\d+[\).:\-] ", l):
                        numbered.append(l)
                results["improvement_points"] = numbered

            # Captain no longer provides scoring JSON, if legacy format remains can parse reason/facts (compatibility)
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", cap, re.DOTALL)
            if json_match:
                try:
                    payload = json.loads(json_match.group(1))
                    if not results.get("panel_reasons"):
                        results["panel_reasons"] = payload.get("reasons", {}) or {}
                    if not results.get("panel_facts"):
                        results["panel_facts"] = payload.get("facts", []) or []
                except Exception:
                    pass

            # Rewrite content extraction
            # Improved content extraction: allow either Chinese legacy marker or current English marker
            if "【Final Improved Version】" in cap:
                improved_section = cap.split("【Final Improved Version】", 1)[-1].strip()
                results["improved_content"] = improved_section
            elif "【Final improvement plan】" in cap:  # fallback legacy
                improved_section = cap.split("【Final improvement plan】", 1)[-1].strip()
                results["improved_content"] = improved_section

        # Score rationality calibration: if exists >=5 improvement suggestions but majority scores >=0.90, downgrade (heuristic)
        # Summarize panel_scores from various agent JSON
        panel_scores = {
            "technical_depth": agent_score_map.get("technical_depth", {}).get("technical_depth"),
            "clarity": agent_score_map.get("clarity", {}).get("clarity"),
            "structure": agent_score_map.get("structure", {}).get("structure"),
            "citation_accuracy": agent_score_map.get("citation_accuracy", {}).get("citation_accuracy")
        }
        # Normalize and clip
        for k,v in list(panel_scores.items()):
            try:
                if v is not None:
                    panel_scores[k] = max(0.0, min(1.0, float(v)))
            except Exception:
                panel_scores[k] = None
        results["panel_scores"] = panel_scores
        results["agent_raw_scores"] = agent_score_map
        return results
    
    def _extract_expert_scores(self, chat_result) -> Dict[str, float]:
        """Extract scores from individual expert responses, without relying on Captain"""
        
        scores = {
            "technical_depth": 0.75,  # default value
            "clarity": 0.75,
            "structure": 0.75,
            "citation_accuracy": 0.75
        }
        
        messages = getattr(chat_result, 'chat_history', [])
        
        for message in messages:
            if not isinstance(message, dict):
                continue
            sender = message.get("name", "unknown")
            content = message.get("content", "") or ""
            
            # Extract JSON scoring blocks
            json_matches = re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_matches:
                for json_block in reversed(json_matches):  # take the last JSON block
                    try:
                        payload = json.loads(json_block)
                        expert_scores = payload.get("scores", {})
                        
                        # Extract corresponding scores based on expert type
                        if sender == "Technical_Expert" and "technical_depth" in expert_scores:
                            scores["technical_depth"] = float(expert_scores["technical_depth"])
                            break
                        elif sender == "Literature_Expert":
                            if "clarity" in expert_scores:
                                scores["clarity"] = float(expert_scores["clarity"])
                            if "structure" in expert_scores:
                                scores["structure"] = float(expert_scores["structure"])
                            break
                        elif sender == "Data_Validator" and "citation_accuracy" in expert_scores:
                            scores["citation_accuracy"] = float(expert_scores["citation_accuracy"])
                            break
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
        
        # Ensure all scores are within valid range
        for key in scores:
            scores[key] = max(0.0, min(1.0, scores[key]))
        
        return scores
    
    def generate_improvement_plan(self, review_results: Dict[str, Any]) -> str:
        """Generate improvement plan based on review results"""
        
        if not review_results.get("success", False):
            return "An error occurred during review process, unable to generate improvement plan."
        
        plan_parts = ["## Literature Review Improvement Plan\n"]
        
        # Add expert opinion summaries
        reviews = review_results.get("reviews", {})
        
        if "Technical" in reviews:
            plan_parts.append("### Physics Professional Perspective")
            plan_parts.append(reviews["Technical"])
            plan_parts.append("")
        
        if "literature" in reviews:
            plan_parts.append("### Literature Structure Perspective")
            plan_parts.append(reviews["literature"])
            plan_parts.append("")
        
        if "data" in reviews:
            plan_parts.append("### Data Validation Perspective")
            plan_parts.append(reviews["data"])
            plan_parts.append("")
        
        # Add final recommendations
        if review_results.get("final_recommendations"):
            plan_parts.append("### Comprehensive Improvement Recommendations")
            plan_parts.append(review_results["final_recommendations"])
        
        return "\n".join(plan_parts)

def test_ag2_system():
    """Test AG2 system"""
    
    print("=== AG2 Intelligent Collaboration System Test ===\n")
    
    # Create test content
    test_content = """
# Diels-Alder Reaction Catalytic Mechanism Study

## Abstract
This study explores the Lewis acid-catalyzed Diels-Alder reaction mechanism. Experiments found that AlCl₃ can significantly improve reaction rate.

## Experimental Section
We used 1,3-butadiene and maleic anhydride as reactants:
C₄H₆ + C₄H₂O₃ → C₈H₈O₃

Reaction conditions: temperature 80°C, time 6 hours, catalyst AlCl₃ (0.1 mol%).

## Results
Yield reached 95%, which is a significant improvement compared to the literature-reported non-catalyzed conditions (65%).
Reaction activation energy decreased from 120 kJ/mol to 85 kJ/mol.

## Mechanism Discussion
Lewis acid promotes the reaction by coordinating to the carbonyl group of the dienophile, lowering the LUMO energy level.
    """
    
    try:
        # Initialize system
        ag2_system = AG2LiteratureReviewSystem()
        
        # Conduct review
        print("Starting multi-Agent collaborative review...")
        results = ag2_system.review_literature(test_content, "Diels-Alder Reaction Catalytic Mechanism")
        
        # Display results
        print("Review completed!\n")
        
        if results.get("success"):
            print("=== Review Results ===")
            
            reviews = results.get("reviews", {})
            for expert, review in reviews.items():
                print(f"\n【{expert} Expert Opinion】")
                print(review[:300] + "..." if len(review) > 300 else review)
            
            if results.get("final_recommendations"):
                print(f"\n【Final Recommendations】")
                print(results["final_recommendations"][:300] + "...")
            
            # Generate improvement plan
            improvement_plan = ag2_system.generate_improvement_plan(results)
            print(f"\n=== Improvement Plan ===")
            print(improvement_plan[:500] + "...")
            
        else:
            print(f"Review failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_ag2_system()
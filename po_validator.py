#!/usr/bin/env python3
"""
PO File Validator with Local LLM (Ollama)
Production-ready CLI script for validating large .po files (65k+ records, EN→RU)
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import polib
import requests
from tqdm import tqdm

# Configuration constants
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"  # Default model, can be overridden via CLI
DEFAULT_RESUME_FILE = "resume_state.json"
DEFAULT_OUTPUT_PO = "validated_output.po"
DEFAULT_ISSUES_FILE = "issues.jsonl"

# LLM Prompt Template
LLM_SYSTEM_PROMPT = "You are a strict translation QA validator. You analyze English-to-Russian pairs. Output ONLY valid JSON. No explanations, no markdown, no code blocks."

LLM_USER_TEMPLATE = """Check semantic correctness of this translation pair.
SOURCE: {msgid}
TARGET: {msgstr}

Rules:
1. Does TARGET convey the EXACT same meaning as SOURCE? (Flag obvious mismatches like 'hello' → 'как дела?')
2. Ignore minor stylistic differences. Focus on meaning, context, and key terms.
3. Return ONLY this JSON format:
{{"is_correct": boolean, "reason": "string or null"}}
If correct: {{"is_correct": true, "reason": null}}
If incorrect: {{"is_correct": false, "reason": "brief 1-sentence explanation of mismatch"}}"""

# Regex patterns for structural validation
PATTERNS = {
    'python_format': r'\{[^}]*\}',  # {0}, {name}, etc.
    'printf_format': r'%[sdfcoxXeEgG]',  # %s, %d, %f, etc.
    'newline': r'\\n',
    'html_tag': r'<[^>]+>',
    'escape_char': r'\\[tnr"\']',
}


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def load_resume_state(resume_file: str) -> Dict[str, Any]:
    """Load resume state from JSON file if it exists."""
    path = Path(resume_file)
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
                logging.info(f"Loaded resume state from {resume_file}")
                return state
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to load resume state: {e}. Starting fresh.")
    
    return {
        'last_index': -1,
        'issues': [],
        'processed_count': 0,
        'error_count': 0
    }


def save_resume_state(state: Dict[str, Any], resume_file: str):
    """Save current progress to resume state file."""
    try:
        with open(resume_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logging.error(f"Failed to save resume state: {e}")


def extract_placeholders(text: str) -> set:
    """Extract all placeholders and special patterns from text."""
    placeholders = set()
    for pattern_name, pattern in PATTERNS.items():
        matches = re.findall(pattern, text)
        placeholders.update(matches)
    return placeholders


def count_sentences(text: str) -> int:
    """Count sentences in text (approximate)."""
    # Simple sentence counting based on common terminators
    sentences = re.split(r'[.!?]+', text.strip())
    return len([s for s in sentences if s.strip()])


def count_lines(text: str) -> int:
    """Count non-empty lines in text."""
    return len([line for line in text.split('\n') if line.strip()])


def validate_structure(msgid: str, msgstr: str) -> Tuple[bool, Optional[str]]:
    """
    Stage 1: Fast structural validation using regex and counts.
    Returns (is_valid, reason_if_invalid)
    """
    # Check placeholder consistency
    id_placeholders = extract_placeholders(msgid)
    str_placeholders = extract_placeholders(msgstr)
    
    if id_placeholders != str_placeholders:
        missing_in_str = id_placeholders - str_placeholders
        extra_in_str = str_placeholders - id_placeholders
        reason_parts = []
        if missing_in_str:
            reason_parts.append(f"Missing in translation: {missing_in_str}")
        if extra_in_str:
            reason_parts.append(f"Extra in translation: {extra_in_str}")
        return False, "; ".join(reason_parts)
    
    # Check sentence count (allow some flexibility)
    id_sentences = count_sentences(msgid)
    str_sentences = count_sentences(msgstr)
    
    # Allow ±1 sentence difference for natural language variations
    if abs(id_sentences - str_sentences) > 1:
        return False, f"Sentence count mismatch: source={id_sentences}, target={str_sentences}"
    
    # Check line count for multi-line strings
    id_lines = count_lines(msgid)
    str_lines = count_lines(msgstr)
    
    if abs(id_lines - str_lines) > 1:
        return False, f"Line count mismatch: source={id_lines}, target={str_lines}"
    
    return True, None


def call_ollama_llm(msgid: str, msgstr: str, model: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Call local Ollama LLM for semantic validation.
    Returns parsed JSON response or error dict.
    """
    user_prompt = LLM_USER_TEMPLATE.format(msgid=msgid, msgstr=msgstr)
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for consistent output
            "max_tokens": 200
        }
    }
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        result = response.json()
        content = result.get("message", {}).get("content", "")
        
        # Parse JSON from response
        # Handle potential markdown code blocks
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            parsed = json.loads(content)
            return {
                "success": True,
                "data": parsed
            }
        except json.JSONDecodeError as e:
            logging.debug(f"JSON parse error: {e}, content: {content[:100]}")
            return {
                "success": False,
                "error": "llm_parse_error",
                "raw_content": content
            }
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": "timeout"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "connection_error"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"http_error: {str(e)}"}


def validate_with_retry(msgid: str, msgstr: str, model: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Call LLM with exponential backoff retry logic.
    """
    delays = [1, 2, 4]
    
    for attempt in range(max_retries):
        result = call_ollama_llm(msgid, msgstr, model)
        
        if result["success"]:
            return result
        
        # Don't retry on parse errors - treat as validation failure
        if result["error"] == "llm_parse_error":
            return result
        
        if attempt < max_retries - 1:
            delay = delays[attempt]
            logging.warning(f"LLM request failed ({result['error']}), retrying in {delay}s...")
            time.sleep(delay)
        else:
            logging.error(f"LLM request failed after {max_retries} attempts: {result['error']}")
            return result
    
    return result


def add_freez_flag(entry: polib.POEntry):
    """Add #freez flag to a PO entry."""
    if 'freez' not in entry.flags:
        entry.flags.append('freez')


def process_po_file(
    input_path: str,
    output_path: str,
    issues_path: str,
    resume_path: str,
    model: str,
    resume: bool = False,
    verbose: bool = False
):
    """Main processing function for PO file validation."""
    
    # Load or initialize resume state
    state = load_resume_state(resume_path) if resume else {
        'last_index': -1,
        'issues': [],
        'processed_count': 0,
        'error_count': 0
    }
    
    start_index = state['last_index'] + 1 if resume else 0
    
    # Load PO file
    logging.info(f"Loading PO file: {input_path}")
    try:
        po_file = polib.pofile(input_path)
    except Exception as e:
        logging.error(f"Failed to load PO file: {e}")
        return
    
    total_entries = len(po_file)
    logging.info(f"Total entries: {total_entries}, starting from index: {start_index}")
    
    # Initialize issues list from resume state
    issues = state['issues']
    processed_count = state['processed_count']
    error_count = state['error_count']
    
    # Create progress bar
    pbar = tqdm(
        total=total_entries - start_index,
        initial=0,
        desc="Validating",
        unit="entry",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}, {elapsed}<{remaining}]'
    )
    
    try:
        for idx in range(start_index, total_entries):
            entry = po_file[idx]
            
            # Skip entries without msgid or msgstr
            if not entry.msgid or not entry.msgstr:
                pbar.update(1)
                continue
            
            # Handle plural forms
            msgid = entry.msgid
            msgstr = entry.msgstr
            
            # For plural forms, check all variants
            if entry.msgid_plural:
                # Combine all plural forms for validation
                msgstr_combined = ' | '.join(entry.msgstr_plural.values())
                msgid_combined = f"{entry.msgid} | {entry.msgid_plural}"
            else:
                msgstr_combined = msgstr
                msgid_combined = msgid
            
            is_valid = True
            reason = None
            issue_type = None
            
            # Stage 1: Structural validation
            struct_valid, struct_reason = validate_structure(msgid_combined, msgstr_combined)
            
            if not struct_valid:
                is_valid = False
                reason = struct_reason
                issue_type = "structural"
                add_freez_flag(entry)
                logging.debug(f"Index {idx}: Structural validation failed - {reason}")
            else:
                # Stage 2: Semantic validation with LLM
                llm_result = validate_with_retry(msgid_combined, msgstr_combined, model)
                
                if not llm_result["success"]:
                    is_valid = False
                    reason = f"LLM error: {llm_result.get('error', 'unknown')}"
                    issue_type = "semantic"
                    add_freez_flag(entry)
                    error_count += 1
                    logging.debug(f"Index {idx}: LLM validation failed - {reason}")
                else:
                    llm_data = llm_result["data"]
                    if isinstance(llm_data, dict) and llm_data.get("is_correct") is False:
                        is_valid = False
                        reason = llm_data.get("reason", "Semantic mismatch detected")
                        issue_type = "semantic"
                        add_freez_flag(entry)
                        logging.debug(f"Index {idx}: Semantic validation failed - {reason}")
            
            # Record issue if validation failed
            if not is_valid:
                issue_record = {
                    "index": idx,
                    "msgid": msgid,
                    "msgstr": msgstr,
                    "reason": reason,
                    "type": issue_type
                }
                issues.append(issue_record)
                
                # Write issue immediately to JSONL
                with open(issues_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(issue_record, ensure_ascii=False) + '\n')
            
            # Update state
            processed_count += 1
            state['last_index'] = idx
            state['issues'] = issues
            state['processed_count'] = processed_count
            state['error_count'] = error_count
            
            # Save checkpoint every 100 entries
            if processed_count % 100 == 0:
                save_resume_state(state, resume_path)
            
            # Update progress bar
            pbar.set_postfix({
                'errors': error_count,
                'issues': len(issues)
            })
            pbar.update(1)
    
    finally:
        pbar.close()
    
    # Save final state
    save_resume_state(state, resume_path)
    
    # Write validated PO file
    logging.info(f"Saving validated PO file: {output_path}")
    po_file.save(output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print(f"Total entries processed: {processed_count}")
    print(f"Issues found: {len(issues)}")
    print(f"LLM errors: {error_count}")
    print(f"Output PO file: {output_path}")
    print(f"Issues report: {issues_path}")
    print(f"Resume state: {resume_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate .po files using local LLM (Ollama) for translation quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.po
  %(prog)s input.po --model llama3.2 --resume
  %(prog)s input.po --output validated.po --issues issues.jsonl
        """
    )
    
    parser.add_argument(
        'input',
        help='Input .po file to validate'
    )
    parser.add_argument(
        '--output', '-o',
        default=DEFAULT_OUTPUT_PO,
        help=f'Output validated .po file (default: {DEFAULT_OUTPUT_PO})'
    )
    parser.add_argument(
        '--issues', '-i',
        default=DEFAULT_ISSUES_FILE,
        help=f'Output issues report in JSONL format (default: {DEFAULT_ISSUES_FILE})'
    )
    parser.add_argument(
        '--resume-state', '-r',
        default=DEFAULT_RESUME_FILE,
        help=f'Resume state file (default: {DEFAULT_RESUME_FILE})'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--model', '-m',
        default=OLLAMA_MODEL,
        help=f'Ollama model to use (default: {OLLAMA_MODEL})'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate input file exists
    if not Path(args.input).exists():
        logging.error(f"Input file not found: {args.input}")
        return 1
    
    # Clear issues file if not resuming
    if not args.resume and Path(args.issues).exists():
        Path(args.issues).unlink()
        logging.info("Cleared existing issues file")
    
    try:
        process_po_file(
            input_path=args.input,
            output_path=args.output,
            issues_path=args.issues,
            resume_path=args.resume_state,
            model=args.model,
            resume=args.resume,
            verbose=args.verbose
        )
        return 0
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user. Progress saved to resume state.")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

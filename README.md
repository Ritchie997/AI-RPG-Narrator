# PO File Validator with Local LLM (Ollama)

Production-ready CLI script for validating large .po files (65k+ records, EN→RU) using local LLM via Ollama.

## Features

- **Two-stage validation**:
  - **Stage 1 (Fast)**: Regex-based structural validation (placeholders, special symbols, sentence/line counts)
  - **Stage 2 (Semantic)**: LLM-based semantic validation via Ollama
- **Resume capability**: Automatically saves progress and can resume from interruptions
- **Retry logic**: Exponential backoff (1s, 2s, 4s) for network errors/timeouts
- **Progress tracking**: Real-time tqdm progress bar with error counters
- **Output formats**: 
  - Validated `.po` file with `#freez` flags on problematic entries
  - JSONL report with detailed issue information

## Requirements

- Python 3.10+
- Ollama running locally (default: `http://localhost:11434`)
- Required packages: `polib`, `requests`, `tqdm`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python po_validator.py input.po
```

### With Custom Options

```bash
python po_validator.py input.po \
  --model llama3.2 \
  --output validated_output.po \
  --issues issues.jsonl \
  --verbose
```

### Resume from Interruption

```bash
python po_validator.py input.po --resume
```

## Command-Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `validated_output.po` | Output validated .po file |
| `--issues` | `-i` | `issues.jsonl` | Output issues report (JSONL) |
| `--resume-state` | `-r` | `resume_state.json` | Resume state file |
| `--resume` | | | Resume from last checkpoint |
| `--model` | `-m` | `llama3.2` | Ollama model to use |
| `--verbose` | `-v` | | Enable verbose logging |

## Output Files

### `validated_output.po`
Copy of the original .po file with `#freez` flag added before problematic entries:

```po
#, freez
msgid "Hello"
msgstr "как дела?"
```

### `issues.jsonl`
JSON Lines format with one issue per line:

```json
{"index": 42, "msgid": "Hello", "msgstr": "как дела?", "reason": "Semantic mismatch detected", "type": "semantic"}
{"index": 157, "msgid": "Welcome {name}", "msgstr": "Добро пожаловать", "reason": "Missing in translation: {'{name}'}", "type": "structural"}
```

### `resume_state.json`
Checkpoint file for resuming interrupted processing:

```json
{
  "last_index": 1234,
  "issues": [...],
  "processed_count": 1235,
  "error_count": 5
}
```

## Validation Rules

### Stage 1: Structural Validation
- **Placeholders**: `{0}`, `{name}`, `%s`, `%d`, `\n`, `<tag>`, etc.
- **Sentence count**: Allows ±1 difference for natural language variations
- **Line count**: Allows ±1 difference for multi-line strings

### Stage 2: Semantic Validation (LLM)
- Checks if target conveys the exact same meaning as source
- Flags obvious mismatches (e.g., "hello" → "как дела?")
- Ignores minor stylistic differences
- Focuses on meaning, context, and key terms

## Hardware Considerations

Optimized for RTX 3050 with limited VRAM:
- Sequential processing (no batching)
- Isolated requests (no context accumulation)
- Low temperature (0.1) for consistent LLM output
- Max tokens: 200 per response

## Example Workflow

```bash
# Start validation
python po_validator.py translations.po --model llama3.2 --verbose

# If interrupted, resume from checkpoint
python po_validator.py translations.po --resume

# Review issues
cat issues.jsonl | jq .

# Check flagged entries in output
grep -B1 "#freez" validated_output.po
```

## Error Handling

- **Network errors**: 3 retries with exponential backoff
- **JSON parse errors**: Treated as validation failures
- **Timeouts**: 30-second timeout per request
- **Keyboard interrupt**: Graceful shutdown with progress saved

## License

MIT

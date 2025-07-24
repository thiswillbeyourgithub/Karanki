# Karanki

**Bidirectional synchronization between Karakeep highlights and Anki flashcards**

Karanki creates Anki flashcards from your Karakeep highlights and organizes them by color-coded retention levels. It provides bidirectional sync, keeping your highlights and flashcards in sync across both platforms.

## Features

- **Color-based organization**: Highlights are sorted into decks by color with optimized retention rates:
  - Red highlights → Red deck (95% retention)
  - Yellow highlights → Yellow deck (90% retention) 
  - Blue highlights → Blue deck (85% retention)
  - Green highlights → Green deck (80% retention)

- **Bidirectional synchronization**: Changes in Anki are reflected back to Karakeep
- **Tag synchronization**: Optional sync of tags between platforms
- **State management**: Tracks sync status to avoid duplicates
- **Incremental sync**: Only processes new/changed content
- **Robust error handling**: Retry mechanisms and detailed logging

## Requirements

- Python 3.8+
- Anki with [AnkiConnect](https://ankiweb.net/shared/info/2055492159) add-on installed
- Access to a Karakeep instance
- Karakeep API credentials

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure Anki is running with AnkiConnect add-on active

## Usage

### Basic Usage

```bash
python karanki.py --karakeep-base-url https://your-karakeep-instance.com
```

### Advanced Options

```bash
python karanki.py \
  --karakeep-base-url https://karakeep.example.com \
  --deck-path "My Karakeep Cards" \
  --sync-tags \
  --anki-tag-prefix "karakeep" \
  --verbose \
  --limit 100
```

### Command Line Options

- `--karakeep-base-url`: **Required.** Base URL for your Karakeep instance
- `--deck-path`: Base deck name for Karakeep cards (default: "Karakeep")
- `--sync-state-path`: Path to sync state file (default: "sync_state.json")
- `--sync-tags`: Enable bidirectional tag synchronization
- `--anki-tag-prefix`: Prefix for Anki tags from Karakeep (default: "karakeep")
- `--verbose`: Enable verbose logging output
- `--limit`: Limit creation to N oldest highlights
- `--only-sync`: Only sync existing cards without creating new ones

## Configuration

### Environment Variables

Set these environment variables or configure them in your Karakeep instance:

- Karakeep API credentials (refer to karakeep-python-api documentation)
- `KARAKEEP_PYTHON_API_VERBOSE`: Enable verbose API logging

### Anki Setup

1. Install AnkiConnect add-on in Anki
2. Ensure Anki is running before starting sync
3. The tool will create deck structure automatically

## Project Structure

```
Karanki/
├── karanki.py                 # CLI entry point
├── requirements.txt        # Python dependencies 
├── utils/
│   ├── __init__.py
│   ├── sync.py            # Main sync orchestration
│   ├── anki_manager.py    # Anki integration
│   ├── karakeep_manager.py # Karakeep integration  
│   ├── state_manager.py   # Sync state tracking
│   └── exceptions.py      # Custom exceptions
├── karanki_debug.log      # Debug log file
└── sync_state.json        # Sync state (created automatically)
```

## Architecture

The tool consists of several specialized managers:

- **KarankiBidirSync** (`sync.py`): Main orchestration class
- **AnkiManager** (`anki_manager.py`): Handles Anki operations via py-ankiconnect
- **KarakeepManager** (`karakeep_manager.py`): Manages Karakeep API interactions  
- **SyncStateManager** (`state_manager.py`): Tracks synchronization state

## Dependencies

### Core
- `click`: Command line interface
- `loguru`: Structured logging

### Integration
- `py-ankiconnect`: Anki integration
- `karakeep-python-api`: Karakeep API client

### Text Processing  
- `beautifulsoup4`: HTML parsing
- `chonkie[semantic]`: Text chunking
- `corpus-matcher`: Content matching

### Utilities
- `pydantic`: Data validation
- `rtoml`: Configuration parsing
- `numpy`: Numerical operations
- `requests`: HTTP operations

## Logging

The tool provides comprehensive logging:

- Debug logs are saved to `karanki_debug.log` 
- Use `--verbose` flag for detailed console output
- Logs include timestamps, levels, and source locations

## Error Handling

- Automatic retry for transient failures
- Graceful handling of network issues  
- State preservation across runs
- Detailed error reporting

## Version

Current version: 0.1.0

---

*Created with assistance from [aider.chat](https://github.com/Aider-AI/aider/)*

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
- **State management**: Tracks sync status to avoid duplicates with automatic backups
- **Incremental sync**: Only processes new/changed content
- **Robust error handling**: Retry mechanisms and detailed logging
- **Parallel processing**: Efficient parallel bookmark fetching for faster syncs
- **HTML-aware cloze generation**: Preserves formatting and structure in flashcards
- **Semantic chunking**: Intelligent context extraction using semantic boundaries
- **Cross-platform paths**: Uses platformdirs for consistent file locations across operating systems

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
  --create-only-n 50
```

### Sync Existing Cards Only

```bash
# Only sync existing cards without creating new ones
python karanki.py --only-sync
```

### Debug Mode

```bash
# Enable debug mode for post-mortem debugging
python karanki.py --debug
```

### Command Line Options

- `--karakeep-base-url`: Base URL for your Karakeep instance (defaults to `KARAKEEP_PYTHON_API_ENDPOINT` environment variable)
- `--deck-path`: Base deck name for Karakeep cards (default: "Karakeep")
- `--sync-state-path`: Path to sync state file (default: uses platformdirs user data directory)
- `--sync-tags`: Enable bidirectional tag synchronization (default: enabled)
- `--anki-tag-prefix`: Prefix for Anki tags from Karakeep (default: "karakeep")
- `--verbose`: Enable verbose logging output
- `-n, --create-only-n`: Limit creation to N new notes per run (-1 for unlimited)
- `--only-sync`: Only sync existing cards without creating new ones
- `--debug`: Enable debug mode for post-mortem debugging
- `--version`: Show version information

## Configuration

### Environment Variables

Set these environment variables or configure them in your Karakeep instance:

- `KARAKEEP_PYTHON_API_ENDPOINT`: Base URL for your Karakeep instance (can be overridden with `--karakeep-base-url`)
- Karakeep API credentials (refer to karakeep-python-api documentation)
- `KARAKEEP_PYTHON_API_VERBOSE`: Enable verbose API logging

### Anki Setup

1. Install [AnkiConnect](https://ankiweb.net/shared/info/2055492159) add-on in Anki
2. Ensure Anki is running before starting sync
3. Create the required deck structure manually:
   - Create main deck: `Karakeep`
   - Create color sub-decks: `Karakeep::Red`, `Karakeep::Yellow`, `Karakeep::Blue`, `Karakeep::Green`
4. Configure FSRS parameters for each color deck in Anki (cannot be automated):
   - Red: 95% retention
   - Yellow: 90% retention
   - Blue: 85% retention
   - Green: 80% retention

### File Locations

Karanki uses platformdirs for cross-platform file storage:

- **Sync state**: Stored in user data directory (e.g., `~/.local/share/karanki/sync_state.json` on Linux)
- **Debug logs**: Stored in user cache directory (e.g., `~/.cache/karanki/karanki_debug.log` on Linux)
- **Lock file**: Stored in user cache directory to prevent concurrent execution

The exact paths are shown when you run the tool with `--verbose` flag.

## Project Structure

```
Karanki/
├── karanki/
│   ├── karanki.py             # CLI entry point
│   └── utils/
│       ├── __init__.py
│       ├── sync.py            # Main sync orchestration
│       ├── anki_manager.py    # Anki integration
│       ├── karakeep_manager.py # Karakeep integration  
│       ├── state_manager.py   # Sync state tracking
│       └── exceptions.py      # Custom exceptions
├── requirements.txt           # Python dependencies
├── bumpver.toml              # Version management
├── .pre-commit-config.yaml   # Pre-commit hooks
└── README.md                 # This file

User data directories (created automatically):
├── ~/.local/share/karanki/sync_state.json     # Sync state (Linux)
└── ~/.cache/karanki/
    ├── karanki_debug.log                      # Debug logs
    └── karanki.lock                           # Execution lock
```

## Architecture

The tool consists of several specialized managers:

- **KarankiBidirSync** (`sync.py`): Main orchestration class that coordinates the sync process
- **AnkiManager** (`anki_manager.py`): Handles all Anki operations via py-ankiconnect
  - Creates and updates notes with HTML-aware cloze generation
  - Uses semantic chunking for intelligent context extraction
  - Validates deck structure and manages note types
- **KarakeepManager** (`karakeep_manager.py`): Manages Karakeep API interactions
  - Fetches highlights and bookmarks with parallel processing
  - Handles color updates and tag synchronization
  - Implements retry logic for network resilience
- **SyncStateManager** (`state_manager.py`): Tracks synchronization state
  - Maintains mappings between highlights and notes
  - Provides automatic state backups
  - Handles recovery from failures

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

- Debug logs are automatically saved to cache directory (location shown on startup)
- Use `--verbose` flag for detailed console output
- Logs include timestamps, levels, and source locations
- Separate log rotation keeps file sizes manageable (10 MB limit)

## Error Handling

- Automatic retry for transient failures with exponential backoff
- Graceful handling of network issues  
- State preservation across runs with automatic backups
- Detailed error reporting with context
- Failed creation tracking to avoid repeated errors
- Lock file prevents concurrent execution
- Debug mode available for post-mortem debugging

## Contributing

This project is part of the [Karakeep Community Scripts](https://github.com/thiswillbeyourgithub/karakeep_python_api).

Contributions are welcome! Please ensure:
- Code follows the project conventions (see `../../../attila_scripts/ubiquitous_scripts/aider_conventions.txt`)
- Type hints and docstrings are included
- Pre-commit hooks pass (black formatting)

---

*Created with assistance from [aider.chat](https://github.com/Aider-AI/aider/)*

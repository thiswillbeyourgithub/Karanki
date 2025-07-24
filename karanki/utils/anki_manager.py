"""
Anki integration manager

Handles all interactions with Anki through py_ankiconnect, including
deck management, note creation, and notetype handling.

Created with assistance from aider.chat (https://github.com/Aider-AI/aider/)
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import asdict, is_dataclass
from loguru import logger
from py_ankiconnect import PyAnkiconnect
from bs4 import BeautifulSoup
from chonkie import SemanticChunker
from corpus_matcher import find_best_substring_match

from .exceptions import AnkiConnectionError, DeckNotFoundError, ContentProcessingError

# Beartype decorator pattern for optional runtime type checking
try:
    from beartype import beartype as optional_typecheck
except ImportError:
    from typing import Callable

    def optional_typecheck(callable_obj: Callable) -> Callable:
        """Dummy decorator if beartype is not installed."""
        return callable_obj


class AnkiManager:
    """
    Manager class for all Anki-related operations.

    This class handles communication with Anki through AnkiConnect,
    manages decks, and handles note creation/updating.
    """

    KARAKEEP_NOTETYPE = "Karakeep Highlight"

    @optional_typecheck
    def __init__(self, config):
        """
        Initialize the Anki manager.

        Parameters
        ----------
        config : SyncConfig
            Configuration object containing sync settings
        """
        self.config = config
        self.akc = PyAnkiconnect()

        logger.debug("AnkiManager initialized")

    @optional_typecheck
    def validate_connection(self) -> None:
        """
        Validate that Anki is running and AnkiConnect is available.

        Raises
        ------
        AnkiConnectionError
            If unable to connect to Anki
        """
        try:
            # Test connection by getting deck names
            deck_names = self.akc("deckNames")
            logger.debug(
                f"Successfully connected to Anki. Found {len(deck_names)} decks"
            )
        except Exception as e:
            error_msg = f"Failed to connect to Anki. Is Anki running with AnkiConnect addon? Error: {e}"
            logger.error(error_msg)
            raise AnkiConnectionError(error_msg) from e

    @optional_typecheck
    def check_decks_exist(self) -> None:
        """
        Check that required deck structure exists.
        Crashes with helpful error message if any deck is missing, as per whitepaper spec.

        Raises
        ------
        DeckNotFoundError
            If any required deck is missing
        """
        logger.info("Checking required deck structure exists...")

        try:
            existing_decks = self.akc("deckNames")

            # Check main deck
            missing_decks = []
            if self.config.deck_path not in existing_decks:
                missing_decks.append(self.config.deck_path)

            # Check color sub-decks
            required_subdecks = ["Red", "Yellow", "Blue", "Green"]
            for subdeck in required_subdecks:
                full_deck_name = f"{self.config.deck_path}::{subdeck}"
                if full_deck_name not in existing_decks:
                    missing_decks.append(full_deck_name)

            if missing_decks:
                error_msg = (
                    f"Required Anki decks are missing: {', '.join(missing_decks)}\n\n"
                    f"Please create these decks in Anki before running Karanki:\n"
                    + "\n".join(f"  - {deck}" for deck in missing_decks)
                    + f"\n\nIMPORTANT: After creating the color-coded decks, manually configure "
                    f"their FSRS parameters in Anki for optimal retention:\n"
                    f"  - {self.config.deck_path}::Red: 95% retention (Options → FSRS)\n"
                    f"  - {self.config.deck_path}::Yellow: 90% retention (Options → FSRS)\n"
                    f"  - {self.config.deck_path}::Blue: 85% retention (Options → FSRS)\n"
                    f"  - {self.config.deck_path}::Green: 80% retention (Options → FSRS)\n\n"
                    f"Note: FSRS parameters cannot be set programmatically and must be "
                    f"configured manually in Anki's deck options."
                )
                logger.error(error_msg)
                raise DeckNotFoundError(error_msg)

            logger.info("All required decks exist")

        except DeckNotFoundError:
            raise  # Re-raise our custom error
        except Exception as e:
            error_msg = f"Failed to check deck structure: {e}"
            logger.error(error_msg)
            raise AnkiConnectionError(error_msg) from e

    @optional_typecheck
    def ensure_notetype_exists(self) -> None:
        """
        Ensure the Karakeep notetype exists, creating it if necessary.

        The notetype is a cloze type with fields for:
        - Text (the context with cloze)
        - Source (link back to Karakeep)
        - Tags (for tag syncing)
        """
        logger.info("Checking/creating Karakeep notetype...")

        try:
            # Check if notetype already exists
            model_names = self.akc("modelNames")

            if self.KARAKEEP_NOTETYPE in model_names:
                logger.debug(f"Notetype '{self.KARAKEEP_NOTETYPE}' already exists")
                return

            # Create the notetype
            self._create_karakeep_notetype()
            logger.info(f"Created notetype '{self.KARAKEEP_NOTETYPE}'")

        except Exception as e:
            error_msg = f"Failed to ensure notetype exists: {e}"
            logger.error(error_msg)
            raise AnkiConnectionError(error_msg) from e

    @optional_typecheck
    def _create_karakeep_notetype(self) -> None:
        """Create the Karakeep highlight notetype."""

        # Define the notetype structure
        notetype_config = {
            "modelName": self.KARAKEEP_NOTETYPE,
            "inOrderFields": ["Text", "Header", "Source", "Tags", "Metadata"],
            "css": """
.card {
    font-family: arial;
    font-size: 16px;
    text-align: left;
    color: black;
    background-color: white;
    margin: 20px;
}

.cloze {
    font-weight: bold;
    color: blue;
}

.source {
    font-size: 12px;
    color: #666;
    margin-top: 20px;
    border-top: 1px solid #ccc;
    padding-top: 10px;
}

.tags {
    font-size: 11px;
    color: #888;
    margin-top: 10px;
}

.header {
    font-size: 18px;
    font-weight: bold;
    color: #333;
    margin-bottom: 15px;
    border-bottom: 2px solid #ddd;
    padding-bottom: 5px;
}
            """,
            "cardTemplates": [
                {
                    "Name": "Cloze",
                    "Front": '{{#Header}}<div class="header"><h3>{{Header}}</h3></div>{{/Header}}{{cloze:Text}}',
                    "Back": """{{#Header}}<div class=\"header\"><h3>{{Header}}</h3></div>{{/Header}}{{cloze:Text}}
<div class="source">
    <strong>Source:</strong> {{Source}}
</div>
{{#Tags}}
<div class="tags">
    <strong>Tags:</strong> {{Tags}}
</div>
{{/Tags}}
{{#Metadata}}
<div class="metadata" style="display: none;">
    {{Metadata}}
</div>
{{/Metadata}}""",
                }
            ],
            "isCloze": True,
        }

        # Create the notetype
        self.akc("createModel", **notetype_config)

    @optional_typecheck
    def _extract_text_from_html(self, html_content: str) -> str:
        """
        Extract clean text from HTML content using BeautifulSoup.

        Parameters
        ----------
        html_content : str
            HTML content from bookmark

        Returns
        -------
        str
            Clean text content
        """
        # Set size limits to prevent memory issues
        MAX_HTML_SIZE = 10 * 1024 * 1024  # 10MB
        MAX_TEXT_SIZE = 5 * 1024 * 1024  # 5MB

        if len(html_content) > MAX_HTML_SIZE:
            logger.warning(
                f"HTML content too large ({len(html_content)} bytes), truncating to {MAX_HTML_SIZE} bytes"
            )
            html_content = html_content[:MAX_HTML_SIZE]

        try:
            soup = BeautifulSoup(html_content, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            # Get text and preserve formatting
            text = soup.get_text()

            # Split into lines and clean up while preserving structure
            lines = text.splitlines()
            cleaned_lines = []

            for line in lines:
                # Only strip leading/trailing whitespace, keep empty lines for formatting
                cleaned_line = line.strip()
                cleaned_lines.append(cleaned_line)

            # Join with newlines to preserve original structure
            text = "\n".join(cleaned_lines)

            # Only remove truly excessive newlines (4 or more consecutive)
            text = re.sub(r"\n{4,}", "\n\n\n", text)

            # Apply text size limit
            if len(text) > MAX_TEXT_SIZE:
                logger.warning(
                    f"Extracted text too large ({len(text)} bytes), truncating to {MAX_TEXT_SIZE} bytes"
                )
                text = text[:MAX_TEXT_SIZE]
                # Ensure we don't cut in the middle of a word
                last_space = text.rfind(" ")
                if last_space > MAX_TEXT_SIZE * 0.95:  # If we're close to the limit
                    text = text[:last_space]

            return text
        except Exception as e:
            logger.error(f"Failed to extract text from HTML: {e}")
            raise ContentProcessingError(
                f"Failed to extract text from HTML: {e}"
            ) from e

    @optional_typecheck
    def _find_highlight_in_text(
        self, highlight_text: str, full_text: str
    ) -> Optional[tuple]:
        """
        Find the position of highlight text within the full text.
        Uses corpus_matcher as fallback when exact matching fails.

        Parameters
        ----------
        highlight_text : str
            The highlighted text to find
        full_text : str
            The full text content to search in

        Returns
        -------
        Optional[tuple]
            (start_pos, end_pos) if found, None otherwise
        """
        # Simple exact match first
        start_pos = full_text.find(highlight_text)
        if start_pos != -1:
            return (start_pos, start_pos + len(highlight_text))

        # Try with normalized whitespace
        normalized_highlight = re.sub(r"\s+", " ", highlight_text.strip())
        normalized_text = re.sub(r"\s+", " ", full_text)

        start_pos = normalized_text.find(normalized_highlight)
        if start_pos != -1:
            return (start_pos, start_pos + len(normalized_highlight))

        # Fallback to corpus_matcher for fuzzy matching
        logger.debug(
            f"Exact match failed, trying corpus_matcher for: {highlight_text[:50]}..."
        )
        try:
            result = find_best_substring_match(highlight_text, full_text)
            if result.matches:
                best_match = result.matches[0]
                start_pos = full_text.find(best_match)
                if start_pos != -1:
                    logger.debug(f"Found fuzzy match: {best_match[:50]}...")
                    return (start_pos, start_pos + len(best_match))
        except Exception as e:
            logger.warning(f"corpus_matcher failed: {e}")

        logger.warning(
            f"Could not find highlight text in content: {highlight_text[:50]}..."
        )
        return None

    @optional_typecheck
    def _create_context_with_cloze(
        self, full_text: str, highlight_text: str, highlight_pos: tuple
    ) -> str:
        """
        Create context around highlight with cloze deletion using semantic chunking.

        Parameters
        ----------
        full_text : str
            Full text content
        highlight_text : str
            Text to be cloze-deleted
        highlight_pos : tuple
            (start_pos, end_pos) of highlight in full text

        Returns
        -------
        str
            Text with cloze deletion around highlight
        """
        try:
            # Initialize semantic chunker with multilingual model as per spec
            chunker = SemanticChunker(
                embedding_model="minishlab/potion-multilingual-128M",
                threshold=0.5,
                chunk_size=512,
                min_sentences=1,
            )

            # Create chunks
            chunks = chunker.chunk(full_text)

            start_pos, end_pos = highlight_pos

            # Find which chunk contains the highlight
            target_chunk = None
            for chunk in chunks:
                if chunk.start_index <= start_pos < chunk.end_index:
                    target_chunk = chunk
                    break

            if not target_chunk:
                # Fallback: create context manually
                logger.debug("Chunk not found, using manual context creation")
                context_size = 200
                context_start = max(0, start_pos - context_size)
                context_end = min(len(full_text), end_pos + context_size)
                context = full_text[context_start:context_end]

                # Adjust positions relative to context
                highlight_start_in_context = start_pos - context_start
                highlight_end_in_context = end_pos - context_start
            else:
                context = target_chunk.text
                # Adjust positions relative to chunk
                highlight_start_in_context = start_pos - target_chunk.start_index
                highlight_end_in_context = end_pos - target_chunk.start_index

            # Validate positions
            if (
                highlight_start_in_context < 0
                or highlight_end_in_context > len(context)
                or highlight_start_in_context >= highlight_end_in_context
            ):
                logger.warning(
                    "Invalid highlight positions in context, using simple cloze"
                )
                return f"{{{{c1::{highlight_text}}}}}"

            # Create cloze deletion
            before_highlight = context[:highlight_start_in_context]
            after_highlight = context[highlight_end_in_context:]

            # Create cloze with c1 (first cloze)
            cloze_text = (
                f"{before_highlight}{{{{c1::{highlight_text}}}}}{after_highlight}"
            )

            return cloze_text.strip()

        except Exception as e:
            logger.error(f"Failed to create context with cloze: {e}")
            # Fallback to simple cloze
            return f"{{{{c1::{highlight_text}}}}}"

    @optional_typecheck
    def create_note(
        self, highlight_data: Dict[str, Any], bookmark_data: Dict[str, Any]
    ) -> str:
        """
        Create a new Anki note from highlight data.

        Parameters
        ----------
        highlight_data : Dict[str, Any]
            Highlight information from Karakeep (can be dataclass or dict)
        bookmark_data : Dict[str, Any]
            Bookmark information from Karakeep (can be dataclass or dict)

        Returns
        -------
        str
            The ID of the created note
        """
        try:
            # Convert dataclasses to dictionaries if needed
            if is_dataclass(highlight_data):
                highlight_data = asdict(highlight_data)
            if is_dataclass(bookmark_data):
                bookmark_data = asdict(bookmark_data)

            # Extract data
            highlight_text = highlight_data.get("text", "")
            highlight_color = highlight_data.get("color", "yellow")
            bookmark_id = highlight_data.get("bookmarkId", "")

            # Get HTML content and convert to text
            html_content = bookmark_data.get("content", {}).get("htmlContent", "")
            if not html_content:
                raise ContentProcessingError("No HTML content found in bookmark")

            full_text = self._extract_text_from_html(html_content)

            # Find highlight position in text
            highlight_pos = self._find_highlight_in_text(highlight_text, full_text)
            if not highlight_pos:
                # Fallback: just create simple cloze
                cloze_text = f"{{{{c1::{highlight_text}}}}}"
            else:
                # Create context with cloze deletion
                cloze_text = self._create_context_with_cloze(
                    full_text, highlight_text, highlight_pos
                )

            # Create source link
            base_url = self.config.karakeep_base_url.replace("/api/v1", "")
            source_url = f"{base_url}/dashboard/preview/{bookmark_id}"
            bookmark_title = bookmark_data.get("title", "") or "Untitled"
            if bookmark_title == "Untitled":
                bookmark_title = (
                    bookmark_data.get("content", {}).get("title", "") or "Untitled"
                )
            source_link = f'<a href="{source_url}">{bookmark_title}</a>'

            # Prepare tags field (for future tag sync)
            tags_field = ""
            if self.config.sync_tags:
                bookmark_tags = bookmark_data.get("tags", [])
                if bookmark_tags:
                    tag_names = [tag.get("name", "") for tag in bookmark_tags]
                    prefixed_tags = [
                        f"{self.config.anki_tag_prefix}::{tag}"
                        for tag in tag_names
                        if tag
                    ]
                    tags_field = ", ".join(prefixed_tags)

            # Create metadata field with TOML format
            import rtoml

            metadata = {
                "version": "0.1.0",
                "highlight_id": highlight_data.get("id", ""),
                "bookmark_id": bookmark_id,
                "color": highlight_color,
                "created_at": highlight_data.get("createdAt", ""),
                "karakeep_base_url": self.config.karakeep_base_url,
                "sync_tags": self.config.sync_tags,
                "anki_tag_prefix": (
                    self.config.anki_tag_prefix if self.config.sync_tags else None
                ),
            }
            metadata_toml = rtoml.dumps(metadata)

            # Determine target deck using hierarchical structure
            deck_name = self._get_deck_name_for_color(highlight_color)

            # Create the note
            note_config = {
                "deckName": deck_name,
                "modelName": self.KARAKEEP_NOTETYPE,
                "fields": {
                    "Text": cloze_text,
                    "Header": source_link,
                    "Source": source_url,
                    "Tags": tags_field,
                    "Metadata": metadata_toml,
                },
                "tags": ["karakeep", f"karakeep::{highlight_color}"],
            }

            if self.config.sync_tags and tags_field:
                # Add individual tags to Anki note tags
                individual_tags = [tag.strip() for tag in tags_field.split(",")]
                note_config["tags"].extend(individual_tags)

            note_id = self.akc("addNote", note=note_config)
            logger.debug(
                f"Created note {note_id} for highlight {highlight_data.get('id', 'unknown')}"
            )

            return str(note_id)

        except Exception as e:
            error_msg = f"Failed to create note for highlight: {e}"
            logger.error(error_msg)
            raise ContentProcessingError(error_msg) from e

    @optional_typecheck
    def update_note(self, note_id: str, fields: Dict[str, str]) -> None:
        """
        Update an existing Anki note.

        Parameters
        ----------
        note_id : str
            The ID of the note to update
        fields : Dict[str, str]
            Fields to update
        """
        try:
            self.akc("updateNote", id=note_id, fields=fields)
            logger.debug(f"Updated note {note_id}")
        except Exception as e:
            logger.error(f"Failed to update note {note_id}: {e}")
            raise

    @optional_typecheck
    def find_notes_in_deck(self, deck_name: str) -> List[str]:
        """
        Find all notes in a specific deck.

        Parameters
        ----------
        deck_name : str
            Name of the deck to search

        Returns
        -------
        List[str]
            List of note IDs
        """
        try:
            query = f"deck:{deck_name}"
            note_ids = self.akc("findNotes", query=query)
            logger.debug(f"Found {len(note_ids)} notes in deck '{deck_name}'")
            return note_ids
        except Exception as e:
            logger.error(f"Failed to find notes in deck '{deck_name}': {e}")
            raise

    @optional_typecheck
    def suspend_note(self, note_id: str) -> None:
        """
        Suspend a note (and its cards).

        Parameters
        ----------
        note_id : str
            The ID of the note to suspend
        """
        try:
            # Get cards for this note and suspend them
            card_ids = self.akc("findCards", query=f"nid:{note_id}")
            if card_ids:
                self.akc("suspend", cards=card_ids)
                logger.debug(f"Suspended {len(card_ids)} cards for note {note_id}")
        except Exception as e:
            logger.error(f"Failed to suspend note {note_id}: {e}")
            raise

    @optional_typecheck
    def check_note_has_unlinked_tag(self, note_id: str) -> bool:
        """
        Check if a note has the karakeep::unlinked tag.

        Parameters
        ----------
        note_id : str
            The ID of the note to check

        Returns
        -------
        bool
            True if note has karakeep::unlinked tag, False otherwise
        """
        try:
            # Get note info including tags
            note_info = self.akc("notesInfo", notes=[note_id])
            if not note_info:
                return False

            note_tags = note_info[0].get("tags", [])
            return "karakeep::unlinked" in note_tags

        except Exception as e:
            logger.error(f"Failed to check tags for note {note_id}: {e}")
            return False

    @optional_typecheck
    def _get_deck_name_for_color(self, color: str) -> str:
        """
        Get the full hierarchical deck name for a given color.

        Parameters
        ----------
        color : str
            Highlight color (red, yellow, blue, green)

        Returns
        -------
        str
            Full deck name including hierarchy
        """
        # Color to deck suffix mapping
        color_deck_mapping = {
            "red": "Red",
            "yellow": "Yellow",
            "blue": "Blue",
            "green": "Green",
        }

        if color in color_deck_mapping:
            suffix = color_deck_mapping[color]
            return f"{self.config.deck_path}::{suffix}"
        else:
            logger.warning(f"Unknown color '{color}', using default deck")
            return self.config.deck_path

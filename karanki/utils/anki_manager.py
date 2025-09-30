"""
Anki integration manager

Handles all interactions with Anki through py_ankiconnect, including
deck management, note creation, and notetype handling.

Created with assistance from aider.chat (https://github.com/Aider-AI/aider/)
"""

import re
from typing import Dict, List, Optional, Any, Literal
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


@optional_typecheck
def adjust_index_to_sentence(
    index: int,
    sentences_indexes: List[List[int]],
    only: Optional[Literal["start", "end"]] = None,
) -> Optional[int]:
    """
    Adjust an index to align with sentence boundaries.

    Parameters
    ----------
    index : int
        The index to adjust
    sentences_indexes : List[List[int]]
        List of [start, end] sentence boundaries
    only : Optional[Literal["start", "end"]]
        If specified, only adjust to sentence start or end

    Returns
    -------
    Optional[int]
        Adjusted index, or None if no suitable sentence found
    """
    best_match = None

    for isind, sind in enumerate(sentences_indexes):
        start, end = sind
        if not (start <= index <= end):
            continue
        if only == "start":
            best_match = start
            break
        elif only == "end":
            best_match = end
            break
        else:
            candidates = []
            if isind != 0:
                candidates.extend(sentences_indexes[isind - 1])
            candidates.extend([start, end])
            if isind < len(sentences_indexes) - 1:
                candidates.extend(sentences_indexes[isind + 1])
            if not (min(candidates) <= index <= max(candidates)):
                continue
            diffs = [abs(index - elem) for elem in candidates]
            best_match = candidates[diffs.index(min(diffs))]
            break

    return best_match


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
        Also updates the templates if the notetype already exists to keep them in sync with code changes.

        The notetype is a cloze type with fields for:
        - Text (the context with cloze)
        - Source (link back to Karakeep)
        """
        logger.info("Checking/creating Karakeep notetype...")

        try:
            # Check if notetype already exists
            model_names = self.akc("modelNames")

            if self.KARAKEEP_NOTETYPE in model_names:
                logger.debug(
                    f"Notetype '{self.KARAKEEP_NOTETYPE}' already exists, updating templates..."
                )
                self._update_karakeep_notetype_templates()
                logger.info(f"Updated notetype '{self.KARAKEEP_NOTETYPE}' templates")
                return

            # Create the notetype
            self._create_karakeep_notetype()
            logger.info(f"Created notetype '{self.KARAKEEP_NOTETYPE}'")

        except Exception as e:
            error_msg = f"Failed to ensure notetype exists: {e}"
            logger.error(error_msg)
            raise AnkiConnectionError(error_msg) from e

    @optional_typecheck
    def _get_karakeep_templates(self) -> Dict[str, Dict[str, str]]:
        """Get the standard Karakeep notetype templates."""
        return {
            "Cloze": {
                "Front": '{{#Header}}<div class="header"><h3>{{Header}}</h3></div>{{/Header}}{{cloze:Text}}',
                "Back": """{{#Header}}<div class=\"header\"><h3>{{Header}}</h3></div>{{/Header}}{{cloze:Text}}
{{#OriginalHighlight}}
<div class="original-highlight">
    <strong>Original Highlight:</strong><br>
    {{OriginalHighlight}}
</div>
{{/OriginalHighlight}}
<div class="source">
    <strong>Source:</strong> {{Source}}
</div>
{{#Metadata}}
<div class="metadata" style="display: none;">
    {{Metadata}}
</div>
{{/Metadata}}""",
            }
        }

    @optional_typecheck
    def _create_karakeep_notetype(self) -> None:
        """Create the Karakeep highlight notetype."""

        # Define the notetype structure
        notetype_config = {
            "modelName": self.KARAKEEP_NOTETYPE,
            "inOrderFields": [
                "Text",
                "Header",
                "Source",
                "OriginalHighlight",
                "Metadata",
            ],
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

.header {
    font-size: 18px;
    font-weight: bold;
    color: #333;
    margin-bottom: 15px;
    border-bottom: 2px solid #ddd;
    padding-bottom: 5px;
}

.original-highlight {
    font-size: 14px;
    color: #444;
    background-color: #f9f9f9;
    border-left: 3px solid #4CAF50;
    padding: 10px;
    margin-top: 15px;
    margin-bottom: 15px;
    font-style: italic;
}
            """,
            "cardTemplates": [
                {
                    "Name": template_name,
                    "Front": template_data["Front"],
                    "Back": template_data["Back"],
                }
                for template_name, template_data in self._get_karakeep_templates().items()
            ],
            "isCloze": True,
        }

        # Create the notetype
        self.akc("createModel", **notetype_config)

    @optional_typecheck
    def _update_karakeep_notetype_templates(self) -> None:
        """Update the templates of the existing Karakeep notetype."""

        # Get the current template definitions
        templates = self._get_karakeep_templates()

        # Use updateModelTemplates to update the existing notetype
        self.akc(
            "updateModelTemplates",
            model={"name": self.KARAKEEP_NOTETYPE, "templates": templates},
        )

        logger.debug(f"Updated templates for notetype '{self.KARAKEEP_NOTETYPE}'")

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

            # Convert block-level elements to newlines before extracting text
            # This ensures paragraph breaks and line breaks are preserved
            block_elements = soup.find_all(
                [
                    "p",
                    "div",
                    "br",
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "h5",
                    "h6",
                    "li",
                    "blockquote",
                    "pre",
                ]
            )
            for element in block_elements:
                if element.name == "br":
                    element.replace_with("\n")
                else:
                    # Add newlines before and after block elements
                    if element.string:
                        element.string.replace_with("\n" + element.get_text() + "\n")
                    else:
                        # For elements with nested content, insert newlines around them
                        if element.previous_sibling:
                            element.insert_before("\n")
                        if element.next_sibling:
                            element.insert_after("\n")

            # Get text with newline separator to preserve structure
            text = soup.get_text(separator="\n")

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
            # But preserve paragraph breaks (double newlines)
            text = re.sub(r"\n{4,}", "\n\n\n", text)

            # Ensure we don't accidentally remove all paragraph breaks
            # Convert any remaining multiple spaces to single spaces while preserving newlines
            text = re.sub(r"[ \t]+", " ", text)

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
        # Initialize semantic chunker with multilingual model as per spec
        chunker = SemanticChunker(
            embedding_model="minishlab/potion-multilingual-128M",
            threshold=0.5,
            chunk_size=max(512, len(highlight_text) + 256),
            min_sentences=1,
        )

        # strip the text, adjusting the highlight position if needed
        # end portion, adjusting the last index if the highlight is until the end
        highlight_pos = list(highlight_pos)
        diff_len = len(full_text) - len(full_text.rstrip())
        if diff_len > 0 and highlight_pos[-1] == len(full_text):
            full_text = full_text.rstrip()
            highlight_pos[-1] -= diff_len
        # start portion: adjusting both positions
        full_text_stripped = full_text.strip()
        diff_len = len(full_text) - len(full_text_stripped)
        if diff_len > 0:
            full_text = full_text.rstrip()
            highlight_pos[0] -= diff_len
            highlight_pos[1] -= diff_len
        start_pos, end_pos = highlight_pos

        # Create chunks
        chunks = chunker.chunk(full_text)

        # Find which chunk contains the highlight start
        target_chunk = None
        target_chunk_idx = None
        for ichunk, chunk in enumerate(chunks):
            if chunk.start_index <= start_pos < chunk.end_index:
                target_chunk = chunk
                target_chunk_idx = ichunk
                break

        if target_chunk is None:
            logger.warning(f"Couldn't find chunk for highlight, using simple cloze")
            return f"{{{{c1::{highlight_text}}}}}"

        # Determine if we need to include the next chunk
        # If highlight extends beyond current chunk and there's a next chunk available
        if end_pos > target_chunk.end_index and target_chunk_idx < len(chunks) - 1:
            next_chunk = chunks[target_chunk_idx + 1]
            context = target_chunk.text + " " + next_chunk.text
            context_start_index = target_chunk.start_index
        else:
            context = target_chunk.text
            context_start_index = target_chunk.start_index

        # Calculate positions relative to context
        highlight_start_in_context = start_pos - context_start_index
        highlight_end_in_context = end_pos - context_start_index

        # Build sentence indices relative to context (not full_text)
        sentences_indexes: List[List[int]] = []
        for chunk in chunks:
            for sentence in chunk.sentences:
                # Convert sentence indices to be relative to context
                sentence_start = sentence.start_index - context_start_index
                sentence_end = sentence.end_index - context_start_index
                # Only include sentences that are within our context bounds
                if 0 <= sentence_start < len(context) and 0 < sentence_end <= len(
                    context
                ):
                    sentences_indexes.append([sentence_start, sentence_end])

        # Validate positions before adjustment
        if highlight_end_in_context > len(context):
            logger.warning(
                f"Highlight end ({highlight_end_in_context}) exceeds context length ({len(context)}), using simple cloze"
            )
            return f"{{{{c1::{highlight_text}}}}}"

        if highlight_start_in_context >= len(context) or highlight_start_in_context < 0:
            logger.warning(
                f"Highlight start ({highlight_start_in_context}) out of context bounds ({len(context)}), using simple cloze"
            )
            return f"{{{{c1::{highlight_text}}}}}"

        if highlight_start_in_context == highlight_end_in_context:
            logger.warning(
                f"Highlight start and end are the same ({highlight_start_in_context}), using simple cloze"
            )
            return f"{{{{c1::{highlight_text}}}}}"

        # Ensure we have valid sentence indices
        if not sentences_indexes:
            logger.warning("No sentence indices found, using simple cloze")
            return f"{{{{c1::{highlight_text}}}}}"

        # Adjust indexes to sentence borders
        adjusted_start = adjust_index_to_sentence(
            index=highlight_start_in_context,
            sentences_indexes=sentences_indexes,
            only="start",
        )
        adjusted_end = adjust_index_to_sentence(
            index=highlight_end_in_context,
            sentences_indexes=sentences_indexes,
            only="end",
        )

        # Validate adjusted positions
        if adjusted_start is None or adjusted_end is None:
            logger.warning(
                "Failed to adjust to sentence boundaries, using simple cloze"
            )
            return f"{{{{c1::{highlight_text}}}}}"

        if adjusted_start == adjusted_end:
            logger.warning(
                f"Adjusted borders are the same ({adjusted_start}), using simple cloze"
            )
            return f"{{{{c1::{highlight_text}}}}}"

        if adjusted_end <= 0 or adjusted_start < 0:
            logger.warning(
                f"Invalid adjusted positions (start: {adjusted_start}, end: {adjusted_end}), using simple cloze"
            )
            return f"{{{{c1::{highlight_text}}}}}"

        if adjusted_end > len(context) or adjusted_start >= len(context):
            logger.warning(
                f"Adjusted positions exceed context (start: {adjusted_start}, end: {adjusted_end}, len: {len(context)}), using simple cloze"
            )
            return f"{{{{c1::{highlight_text}}}}}"

        if adjusted_end <= adjusted_start:
            logger.warning(
                f"Wrong order of adjusted borders (start: {adjusted_start}, end: {adjusted_end}), using simple cloze"
            )
            return f"{{{{c1::{highlight_text}}}}}"

        # Create cloze deletion
        before_highlight = context[:adjusted_start]
        # we use actual_highlight instead of highlight_text because otherwise the newlines are removed
        actual_highlight = context[adjusted_start:adjusted_end]
        after_highlight = context[adjusted_end:]

        # Strip leading and trailing whitespace from the cloze content
        # and move it to before/after sections to avoid awkward formatting
        leading_stripped = actual_highlight.lstrip()
        leading_whitespace = actual_highlight[
            : len(actual_highlight) - len(leading_stripped)
        ]
        before_highlight += leading_whitespace

        actual_highlight_clean = leading_stripped.rstrip()
        trailing_whitespace = leading_stripped[len(actual_highlight_clean) :]
        after_highlight = trailing_whitespace + after_highlight

        actual_highlight = actual_highlight_clean

        # remove cloze markers that could be there by chance
        before_highlight = (
            before_highlight.replace("{{", "{ { ")
            .replace("}}", "} } ")
            .replace("::", ": : ")
        )
        actual_highlight = (
            actual_highlight.replace("{{", "{ { ")
            .replace("}}", "} } ")
            .replace("::", ": : ")
        )
        after_highlight = (
            after_highlight.replace("{{", "{ { ")
            .replace("}}", "} } ")
            .replace("::", ": : ")
        )

        # Create cloze with c1 (first cloze)
        cloze_text = (
            f"{before_highlight}{{{{c1::{actual_highlight}}}}} {after_highlight}"
        )
        return cloze_text.strip().replace("\n", "<br>")

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
            bookmark_url = f"{base_url}/dashboard/preview/{bookmark_id}"
            bookmark_title = bookmark_data.get("title", "") or "Untitled"
            if bookmark_title == "Untitled":
                bookmark_title = (
                    bookmark_data.get("content", {}).get("title", "") or "Untitled"
                )
            bookmark_link = f'<a href="{bookmark_url}">{bookmark_title}</a>'
            original_url = bookmark_data.get("content", {}).get("url", None) or "No URL"
            original_link = f'<a href="{original_url}">{original_url}</a>'

            # Prepare tags (for future tag sync)
            tags_to_add = []
            if self.config.sync_tags:
                bookmark_tags = bookmark_data.get("tags", [])
                if bookmark_tags:
                    tag_names = [tag.get("name", "") for tag in bookmark_tags]
                    prefixed_tags = [
                        f"{self.config.anki_tag_prefix}::{tag}"
                        for tag in tag_names
                        if tag
                    ]
                    tags_to_add.extend(prefixed_tags)

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
            metadata_toml = rtoml.dumps(metadata).strip().replace("\n", "<br>")

            # Determine target deck using hierarchical structure
            deck_name = self._get_deck_name_for_color(highlight_color)

            # Create the note
            note_config = {
                "deckName": deck_name,
                "modelName": self.KARAKEEP_NOTETYPE,
                "fields": {
                    "Text": cloze_text,
                    "Header": bookmark_link,
                    "Source": original_url,
                    "OriginalHighlight": highlight_text.replace("\n", "<br>"),
                    "Metadata": metadata_toml,
                },
                "tags": ["karakeep", f"karakeep::color::{highlight_color}"],
            }

            if self.config.sync_tags:
                # Add individual tags to Anki note tags
                note_config["tags"].extend(tags_to_add)

            note_id = self.akc("addNote", note=note_config)
            logger.debug(
                f"Created note {note_id} for highlight {highlight_data.get('id', 'unknown')}"
            )

            # Breakpoint for debugging Text field parsing issues with newlines
            breakpoint()

            return str(note_id)

        except Exception as e:
            # Enhance error message with identifying information for debugging
            highlight_id = highlight_data.get("id", "unknown")
            bookmark_id = highlight_data.get("bookmarkId", "unknown")
            highlight_text = highlight_data.get("text", "")
            # Truncate highlight text for readability
            highlight_text_preview = (
                highlight_text[:100] + "..."
                if len(highlight_text) > 100
                else highlight_text
            )
            bookmark_title = bookmark_data.get("title", "") or bookmark_data.get(
                "content", {}
            ).get("title", "Untitled")

            error_msg = (
                f"Failed to create note for highlight: {e}\n"
                f"  Highlight ID: {highlight_id}\n"
                f"  Bookmark ID: {bookmark_id}\n"
                f"  Bookmark title: {bookmark_title}\n"
                f"  Highlight text preview: {highlight_text_preview}"
            )
            logger.error(error_msg)
            raise ContentProcessingError(error_msg) from e

    @optional_typecheck
    def update_note(
        self, note_id: str, fields: Dict[str, str], tags: Optional[List[str]] = None
    ) -> None:
        """
        Update an existing Anki note.

        Parameters
        ----------
        note_id : str
            The ID of the note to update
        fields : Dict[str, str]
            Fields to update
        tags : Optional[List[str]]
            Tags to set on the note (replaces existing tags)
        """
        try:
            # Convert note_id to integer for AnkiConnect API
            note_id_int = int(note_id)

            # Update fields if provided
            if fields:
                self.akc("updateNote", id=note_id_int, fields=fields)
                logger.debug(f"Updated fields for note {note_id}")

            # Update tags if provided
            if tags is not None:
                # Get current tags to remove them first
                note_info = self.akc("notesInfo", notes=[note_id_int])
                if note_info:
                    current_tags = note_info[0].get("tags", [])

                    # Remove all current tags
                    if current_tags:
                        self.akc(
                            "removeTags",
                            notes=[note_id_int],
                            tags=" ".join(current_tags),
                        )

                    # Add new tags
                    if tags:
                        self.akc("addTags", notes=[note_id_int], tags=" ".join(tags))

                logger.debug(f"Updated tags for note {note_id}: {tags}")

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
            note_ids = [str(nid) for nid in self.akc("findNotes", query=query)]
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
            # Convert note_id to integer for AnkiConnect API
            note_id_int = int(note_id)

            # Get cards for this note and suspend them
            card_ids = self.akc("findCards", query=f"nid:{note_id_int}")
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
            # Convert note_id to integer for AnkiConnect API
            note_id_int = int(note_id)

            # Get note info including tags
            note_info = self.akc("notesInfo", notes=[note_id_int])
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

"""
Core synchronization logic between Karakeep and Anki

This module contains the main KarankiBidirSync class that handles the
bidirectional synchronization of highlights and flashcards.

Created with assistance from aider.chat (https://github.com/Aider-AI/aider/)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from py_ankiconnect import PyAnkiconnect
from karakeep_python_api import KarakeepAPI
from platformdirs import user_data_dir, user_cache_dir

from .anki_manager import AnkiManager
from .karakeep_manager import KarakeepManager
from .state_manager import SyncStateManager

# Beartype decorator pattern for optional runtime type checking
try:
    from beartype import beartype as optional_typecheck
except ImportError:
    from typing import Callable

    def optional_typecheck(callable_obj: Callable) -> Callable:
        """Dummy decorator if beartype is not installed."""
        return callable_obj


def get_default_sync_state_path() -> str:
    """Get the default sync state file path using platformdirs."""
    data_dir = Path(user_data_dir("karanki", "karanki"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / "sync_state.json")


def get_default_lock_file_path() -> str:
    """Get the default lock file path using platformdirs."""
    cache_dir = Path(user_cache_dir("karanki", "karanki"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / "karanki.lock")


def get_default_debug_log_path() -> str:
    """Get the default debug log file path using platformdirs."""
    cache_dir = Path(user_cache_dir("karanki", "karanki"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / "karanki_debug.log")


@dataclass
class SyncConfig:
    """Configuration for the synchronization process"""

    deck_path: str
    sync_state_path: str
    karakeep_base_url: str
    sync_tags: bool
    anki_tag_prefix: str
    create_only_n: int = -1
    only_sync: bool = False


class KarankiBidirSync:
    """
    Main class for bidirectional synchronization between Karakeep and Anki.

    This class orchestrates the sync process, handling highlights from Karakeep
    and converting them into Anki flashcards with appropriate deck placement
    based on highlight colors.
    """

    # Color to deck mapping with retention levels
    COLOR_DECK_MAPPING = {
        "red": {"deck_suffix": "Red", "retention": 95},
        "yellow": {"deck_suffix": "Yellow", "retention": 90},
        "blue": {"deck_suffix": "Blue", "retention": 85},
        "green": {"deck_suffix": "Green", "retention": 80},
    }

    @optional_typecheck
    def __init__(
        self,
        deck_path: str = "Karakeep",
        sync_state_path: Optional[str] = None,
        karakeep_base_url: str = "",
        sync_tags: bool = False,
        anki_tag_prefix: str = "karakeep",
        create_only_n: int = -1,
        only_sync: bool = False,
        debug: bool = False,
    ):
        """
        Initialize the synchronization manager.

        Parameters
        ----------
        deck_path : str
            Base deck name for organizing Karakeep cards
        sync_state_path : Optional[str]
            Path to the JSON file storing sync state (uses platformdirs default if None)
        karakeep_base_url : str
            Base URL for the Karakeep instance
        sync_tags : bool
            Whether to enable tag synchronization
        anki_tag_prefix : str
            Prefix for Anki tags created from Karakeep tags
        create_only_n : int
            Maximum number of new notes to create per run (-1 for unlimited)
        only_sync : bool
            If True, only sync existing cards without creating new ones
        """
        # Use platformdirs default paths if not provided
        using_default_sync_state = sync_state_path is None
        if sync_state_path is None:
            sync_state_path = get_default_sync_state_path()

        self.config = SyncConfig(
            deck_path=deck_path,
            sync_state_path=sync_state_path,
            karakeep_base_url=karakeep_base_url,
            sync_tags=sync_tags,
            anki_tag_prefix=anki_tag_prefix,
            create_only_n=create_only_n,
            only_sync=only_sync,
        )
        self.debug = debug

        logger.debug(f"Initializing KarankiBidirSync with config: {self.config}")

        # Log file locations for user visibility
        lock_file_path = get_default_lock_file_path()
        debug_log_path = get_default_debug_log_path()

        if using_default_sync_state:
            logger.info(
                f"Using default sync state file location (platformdirs): {self.config.sync_state_path}"
            )
        else:
            logger.info(f"Using sync state file: {self.config.sync_state_path}")
        logger.info(f"Using lock file: {lock_file_path}")
        logger.info(f"Using debug log: {debug_log_path}")

        # Validate configuration
        self._validate_config()

        # Initialize managers
        self.anki_manager = AnkiManager(self.config)
        self.karakeep_manager = KarakeepManager(self.config)
        self.state_manager = SyncStateManager(self.config)

        # Lock file for preventing concurrent execution
        self.lock_file = Path(lock_file_path)

    @optional_typecheck
    def run(self) -> None:
        """
        Execute the complete synchronization process.

        This method follows the workflow outlined in the whitepaper:
        1. Check deck and notetype existence
        2. Load highlights and sync state
        3. Process various sync scenarios
        4. Update sync state
        """
        logger.info("Starting Karanki synchronization process")

        # Check for concurrent execution
        self._acquire_lock()

        try:
            # Phase 1: Validate prerequisites
            self._validate_prerequisites()

            # Phase 2: Load data
            highlights = self._load_highlights()
            sync_state = self._load_sync_state()

            # Phase 3: Process synchronization
            self._process_sync(highlights, sync_state)

            # Phase 4: Final state save (redundant but ensures consistency)
            # State is already saved after each iteration, but we do a final save
            # to update the last_updated timestamp
            self._save_sync_state(sync_state)

            logger.info("Synchronization process completed successfully")

        except Exception as e:
            logger.error(f"Synchronization failed: {e}")
            raise
        finally:
            # Always release the lock
            self._release_lock()

    @optional_typecheck
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        if not self.config.karakeep_base_url:
            from .exceptions import SyncLogicError

            raise SyncLogicError("karakeep_base_url is required")

        if not self.config.karakeep_base_url.startswith(("http://", "https://")):
            from .exceptions import SyncLogicError

            raise SyncLogicError(
                "karakeep_base_url must start with http:// or https://"
            )

        # Remove trailing slash for consistency
        if self.config.karakeep_base_url.endswith("/"):
            self.config.karakeep_base_url = self.config.karakeep_base_url.rstrip("/")

        logger.debug("Configuration validation passed")

    @optional_typecheck
    def _get_deck_name_for_color(self, color: str) -> str:
        """Get the full hierarchical deck name for a given color."""
        if color in self.COLOR_DECK_MAPPING:
            suffix = self.COLOR_DECK_MAPPING[color]["deck_suffix"]
            return f"{self.config.deck_path}::{suffix}"
        return self.config.deck_path

    @optional_typecheck
    def _validate_prerequisites(self) -> None:
        """Validate that required Anki decks and notetypes exist."""
        logger.info("Validating prerequisites...")

        # Check Anki connection and decks
        self.anki_manager.validate_connection()
        self.anki_manager.check_decks_exist()
        self.anki_manager.ensure_notetype_exists()

        # Check Karakeep connection
        self.karakeep_manager.validate_connection()

        logger.info("Prerequisites validation completed")

    @optional_typecheck
    def _load_highlights(self) -> List[Dict[str, Any]]:
        """Load highlights from Karakeep with optional sorting."""
        logger.info("Loading highlights from Karakeep...")
        highlights = self.karakeep_manager.get_all_highlights()

        # Sort highlights by creation date (oldest first) for consistent ordering
        try:
            highlights.sort(key=lambda h: h.get("createdAt", ""))
            logger.debug(f"Sorted {len(highlights)} highlights by creation date")
        except Exception as e:
            logger.warning(f"Failed to sort highlights by creation date: {e}")

        logger.info(f"Loaded {len(highlights)} highlights from Karakeep")
        return highlights

    @optional_typecheck
    def _load_sync_state(self) -> Dict[str, Any]:
        """Load the synchronization state from file."""
        logger.info("Loading sync state...")
        sync_state = self.state_manager.load_state()
        logger.info(
            f"Loaded sync state with {len(sync_state.get('mappings', []))} existing mappings"
        )
        return sync_state

    @optional_typecheck
    def _process_sync(
        self, highlights: List[Dict[str, Any]], sync_state: Dict[str, Any]
    ) -> None:
        """Process the main synchronization logic following whitepaper workflow."""
        logger.info("Processing synchronization...")

        sync_stats = {
            "new_notes_created": 0,
            "notes_suspended": 0,
            "color_mismatches_fixed": 0,
            "missing_from_both": 0,
            "tags_synced": 0,
            "errors": 0,
        }

        try:
            # Step 1: Get all existing Anki notes from Karakeep-related decks
            logger.info("Step 1/7: Gathering existing Anki notes...")
            anki_notes = self._get_all_karakeep_notes()
            logger.info(
                f"Found {len(anki_notes)} existing Anki notes in Karakeep decks"
            )

            # Step 2: Build lookup dictionaries for efficient processing
            logger.info("Step 2/7: Building lookup tables...")
            highlights_by_id = {h["id"]: h for h in highlights}
            state_mappings = {
                m["highlight_id"]: m for m in sync_state.get("mappings", [])
            }
            anki_notes_by_id = {note_id: note for note_id, note in anki_notes.items()}

            # Step 3: Check for entries missing from both Karakeep and Anki
            logger.info("Step 3/7: Checking for entries missing from both sources...")
            missing_count = self._handle_missing_from_both(
                state_mappings, highlights_by_id, anki_notes_by_id, sync_state
            )
            sync_stats["missing_from_both"] = missing_count

            # Step 4: Handle new highlights (in Karakeep but not in state/Anki)
            if self.config.only_sync:
                logger.info(
                    "Step 4/7: Skipping creation of new highlights (only-sync mode)"
                )
                sync_stats["new_notes_created"] = 0
            else:
                logger.info("Step 4/7: Creating notes for new highlights...")
                new_count = self._handle_new_highlights(
                    highlights_by_id, state_mappings, sync_state
                )
                sync_stats["new_notes_created"] = new_count

            # Step 5: Check for unlinked Anki cards and color mismatches
            logger.info(
                "Step 5/7: Validating Anki cards and fixing color mismatches..."
            )
            mismatch_count = self._handle_anki_card_validation(
                anki_notes_by_id, state_mappings, highlights_by_id
            )
            sync_stats["color_mismatches_fixed"] = mismatch_count

            # Step 6: Handle highlights missing from Karakeep (suspend cards)
            logger.info("Step 6/7: Handling highlights missing from Karakeep...")
            suspended_count = self._handle_missing_highlights(
                state_mappings, highlights_by_id, sync_state
            )
            sync_stats["notes_suspended"] = suspended_count

            # Step 7: Tag synchronization (if enabled)
            if self.config.sync_tags:
                logger.info("Step 7/8: Synchronizing tags...")
                tag_count = self._sync_tags(
                    highlights_by_id, state_mappings, sync_state
                )
                sync_stats["tags_synced"] = tag_count
            else:
                logger.info("Step 7/8: Skipping tag synchronization (disabled)")

            # Step 8: Update Text fields with current parser
            logger.info("Step 8/8: Updating Text fields with current parser...")
            updated_count = self._handle_text_field_updates(
                highlights_by_id, state_mappings, sync_state
            )
            sync_stats["text_fields_updated"] = updated_count

            # Report final statistics
            logger.info("Sync processing completed successfully")
            logger.info(
                f"Sync Summary: {sync_stats['new_notes_created']} new notes, "
                f"{sync_stats['notes_suspended']} suspended, "
                f"{sync_stats['color_mismatches_fixed']} color fixes, "
                f"{sync_stats['missing_from_both']} missing from both, "
                f"{sync_stats['tags_synced']} tags synced, "
                f"{sync_stats['text_fields_updated']} text fields updated"
            )

        except Exception as e:
            sync_stats["errors"] += 1
            logger.error(
                f"Sync processing failed at step with partial stats: {sync_stats}"
            )
            raise

    @optional_typecheck
    def _save_sync_state(self, sync_state: Dict[str, Any]) -> None:
        """Save the updated synchronization state."""
        logger.info("Saving sync state...")
        self.state_manager.save_state(sync_state)
        logger.info("Sync state saved successfully")

    @optional_typecheck
    def _get_all_karakeep_notes(self) -> Dict[str, Dict[str, Any]]:
        """Get all notes from Karakeep-related decks."""
        all_notes = {}

        # Get notes from color decks (hierarchical structure)
        for color in self.COLOR_DECK_MAPPING.keys():
            deck_name = self._get_deck_name_for_color(color)
            note_ids = self.anki_manager.find_notes_in_deck(deck_name)
            for note_id in note_ids:
                all_notes[note_id] = {"deck": deck_name, "color": color}

        # Get notes from main Karakeep deck
        main_deck_notes = self.anki_manager.find_notes_in_deck(self.config.deck_path)
        for note_id in main_deck_notes:
            all_notes[note_id] = {"deck": self.config.deck_path, "color": "unknown"}

        return all_notes

    @optional_typecheck
    def _handle_missing_from_both(
        self,
        state_mappings: Dict[str, Dict],
        highlights_by_id: Dict[str, Dict],
        anki_notes_by_id: Dict[str, Dict],
        sync_state: Dict[str, Any],
    ) -> int:
        """Handle entries missing from both Karakeep and Anki."""
        count = 0
        for highlight_id, mapping in state_mappings.items():
            note_id = mapping["note_id"]

            # Check if missing from both
            if (
                highlight_id not in highlights_by_id
                and note_id not in anki_notes_by_id
                and mapping["status"] != "missing_from_both"
            ):

                logger.info(
                    f"Highlight {highlight_id} and note {note_id} missing from both - updating status"
                )
                self.state_manager.update_mapping_status(
                    sync_state, highlight_id, "missing_from_both"
                )
                # Save state after each modification to preserve progress
                self._save_sync_state(sync_state)
                count += 1
        return count

    @optional_typecheck
    def _fetch_bookmarks_parallel(
        self, bookmark_ids: List[str], max_workers: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch multiple bookmarks in parallel using thread pool.

        Parameters
        ----------
        bookmark_ids : List[str]
            List of bookmark IDs to fetch
        max_workers : int
            Maximum number of parallel threads (default: 10)

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mapping of bookmark_id -> bookmark_data for successful fetches
        """
        bookmark_data_map = {}
        failed_bookmarks = []

        logger.debug(
            f"Fetching {len(bookmark_ids)} bookmarks in parallel with {max_workers} workers"
        )

        # Use ThreadPoolExecutor to fetch bookmarks in parallel
        # Threading (not multiprocessing) is appropriate here because the work is I/O-bound
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fetch tasks
            future_to_bookmark_id = {
                executor.submit(
                    self.karakeep_manager.get_bookmark_content, bookmark_id
                ): bookmark_id
                for bookmark_id in bookmark_ids
            }

            # Collect results as they complete
            for future in as_completed(future_to_bookmark_id):
                bookmark_id = future_to_bookmark_id[future]
                try:
                    bookmark_data = future.result()
                    bookmark_data_map[bookmark_id] = bookmark_data
                    logger.debug(f"Successfully fetched bookmark {bookmark_id}")
                except Exception as e:
                    logger.error(f"Failed to fetch bookmark {bookmark_id}: {e}")
                    failed_bookmarks.append(bookmark_id)
                    if self.debug:
                        raise

        if failed_bookmarks:
            logger.warning(
                f"Failed to fetch {len(failed_bookmarks)} bookmarks: {failed_bookmarks}"
            )

        logger.info(
            f"Successfully fetched {len(bookmark_data_map)}/{len(bookmark_ids)} bookmarks"
        )
        return bookmark_data_map

    @optional_typecheck
    def _handle_new_highlights(
        self,
        highlights_by_id: Dict[str, Dict],
        state_mappings: Dict[str, Dict],
        sync_state: Dict[str, Any],
    ) -> int:
        """Handle highlights that exist in Karakeep but not in sync state or Anki."""
        # Collect all new highlights that need bookmark data
        new_highlights = [
            (highlight_id, highlight)
            for highlight_id, highlight in highlights_by_id.items()
            if highlight_id not in state_mappings
        ]

        if not new_highlights:
            logger.info("No new highlights to process")
            return 0

        # Sort by creation date (oldest first) to ensure consistent ordering across runs
        try:
            new_highlights.sort(key=lambda x: x[1].get("createdAt", ""))
            logger.debug(
                f"Sorted {len(new_highlights)} new highlights by creation date"
            )
        except Exception as e:
            logger.warning(f"Failed to sort new highlights by creation date: {e}")

        # Apply create_only_n limit if specified (only when > 0)
        if (
            self.config.create_only_n > 0
            and len(new_highlights) > self.config.create_only_n
        ):
            original_count = len(new_highlights)
            new_highlights = new_highlights[: self.config.create_only_n]
            logger.info(
                f"Limiting creation to {self.config.create_only_n} new notes (out of {original_count} total new highlights)"
            )

        logger.info(f"Processing {len(new_highlights)} new highlights")

        # Extract unique bookmark IDs and fetch them in parallel
        bookmark_ids = list(
            set(highlight["bookmarkId"] for _, highlight in new_highlights)
        )
        logger.info(
            f"Fetching {len(bookmark_ids)} unique bookmarks in parallel for {len(new_highlights)} highlights"
        )
        bookmark_data_map = self._fetch_bookmarks_parallel(bookmark_ids)

        # Now process each highlight sequentially with pre-fetched bookmark data
        count = 0
        for highlight_id, highlight in new_highlights:
            try:
                bookmark_id = highlight["bookmarkId"]

                # Check if we successfully fetched this bookmark
                if bookmark_id not in bookmark_data_map:
                    logger.error(
                        f"Skipping highlight {highlight_id}: bookmark {bookmark_id} fetch failed"
                    )
                    continue

                logger.info(f"New highlight found: {highlight_id} - creating Anki note")

                bookmark_data = bookmark_data_map[bookmark_id]

                # Create Anki note
                note_id = self.anki_manager.create_note(highlight, bookmark_data)

                # Update sync state
                color = highlight.get("color", "yellow")
                deck_name = self._get_deck_name_for_color(color)

                self.state_manager.add_mapping(
                    sync_state, highlight_id, note_id, bookmark_id, color, deck_name
                )

                # Clear any previous failed creation error for this highlight
                self.state_manager.clear_failed_creation(sync_state, highlight_id)

                # Save state after each modification to preserve progress
                self._save_sync_state(sync_state)
                count += 1

            except Exception as e:
                logger.error(f"Failed to create note for highlight {highlight_id}: {e}")

                # Record the failed creation in sync state
                bookmark_id = highlight.get("bookmarkId", "unknown")
                error_message = str(e)
                self.state_manager.add_failed_creation(
                    sync_state, highlight_id, bookmark_id, error_message
                )
                # Save state to persist the error record
                self._save_sync_state(sync_state)

                if self.debug:
                    raise
                # Continue processing other highlights
                continue

        logger.info(f"Successfully created {count} new notes")
        return count

    @optional_typecheck
    def _handle_anki_card_validation(
        self,
        anki_notes_by_id: Dict[str, Dict],
        state_mappings: Dict[str, Dict],
        highlights_by_id: Dict[str, Dict],
    ) -> int:
        """Validate Anki cards and handle color mismatches."""
        mismatch_count = 0
        for note_id, note_info in anki_notes_by_id.items():
            # Find corresponding mapping
            mapping = None
            for m in state_mappings.values():
                if m["note_id"] == note_id:
                    mapping = m
                    break

            if not mapping:
                # Check if card has karakeep::unlinked tag
                if not self.anki_manager.check_note_has_unlinked_tag(note_id):
                    error_msg = (
                        f"Anki note {note_id} in Karakeep deck but no mapping found "
                        f"and missing 'karakeep::unlinked' tag. This suggests an "
                        f"inconsistent state."
                    )
                    logger.error(error_msg)
                    from .exceptions import SyncLogicError

                    raise SyncLogicError(error_msg)
                else:
                    logger.debug(f"Note {note_id} has unlinked tag, skipping")
                continue

            highlight_id = mapping["highlight_id"]
            if highlight_id in highlights_by_id:
                highlight = highlights_by_id[highlight_id]
                current_color = highlight.get("color", "yellow")
                expected_deck = self._get_deck_name_for_color(current_color)
                actual_deck = note_info["deck"]

                # Check for color/deck mismatch
                if expected_deck != actual_deck:
                    logger.info(
                        f"Deck mismatch for highlight {highlight_id}: expected {expected_deck}, found {actual_deck}"
                    )
                    # Update Karakeep highlight color to match Anki deck
                    # Create deck-to-color mapping using hierarchical deck names
                    deck_to_color = {}
                    for color, mapping in self.COLOR_DECK_MAPPING.items():
                        deck_name = self._get_deck_name_for_color(color)
                        deck_to_color[deck_name] = color

                    new_color = deck_to_color.get(actual_deck)

                    if new_color:
                        self.karakeep_manager.update_highlight_color(
                            highlight_id, new_color
                        )
                        logger.info(
                            f"Updated highlight {highlight_id} color to {new_color}"
                        )
                        mismatch_count += 1

        return mismatch_count

    @optional_typecheck
    def _handle_missing_highlights(
        self,
        state_mappings: Dict[str, Dict],
        highlights_by_id: Dict[str, Dict],
        sync_state: Dict[str, Any],
    ) -> int:
        """Handle highlights missing from Karakeep but present in state and Anki."""
        count = 0
        for highlight_id, mapping in state_mappings.items():
            if highlight_id not in highlights_by_id and mapping["status"] == "active":

                logger.info(
                    f"Highlight {highlight_id} missing from Karakeep - suspending Anki note"
                )

                # Suspend the Anki note
                note_id = mapping["note_id"]
                self.anki_manager.suspend_note(note_id)

                # Update status
                self.state_manager.update_mapping_status(
                    sync_state, highlight_id, "missing_from_karakeep"
                )
                # Save state after each modification to preserve progress
                self._save_sync_state(sync_state)
                count += 1
        return count

    @optional_typecheck
    def _sync_tags(
        self,
        highlights_by_id: Dict[str, Dict],
        state_mappings: Dict[str, Dict],
        sync_state: Dict[str, Any],
    ) -> int:
        """Handle tag synchronization between Karakeep and Anki."""
        logger.info("Syncing tags between Karakeep and Anki...")

        # Collect highlights that need tag syncing
        highlights_to_sync = []
        for highlight_id, mapping in state_mappings.items():
            if mapping["status"] != "active":
                continue

            if highlight_id not in highlights_by_id:
                continue

            highlight = highlights_by_id[highlight_id]
            bookmark_id = highlight.get("bookmarkId")
            note_id = mapping["note_id"]

            if bookmark_id:
                highlights_to_sync.append((highlight_id, bookmark_id, note_id))

        if not highlights_to_sync:
            logger.info("No highlights to sync tags for")
            return 0

        # Fetch all bookmark data in parallel (which includes tags)
        bookmark_ids = [bookmark_id for _, bookmark_id, _ in highlights_to_sync]
        logger.info(f"Fetching tags for {len(bookmark_ids)} bookmarks in parallel")
        bookmark_data_map = self._fetch_bookmarks_parallel(bookmark_ids)

        # Process tag updates
        count = 0
        for highlight_id, bookmark_id, note_id in highlights_to_sync:
            if bookmark_id not in bookmark_data_map:
                logger.warning(
                    f"Skipping tag sync for highlight {highlight_id}: bookmark {bookmark_id} fetch failed"
                )
                continue

            try:
                bookmark_data = bookmark_data_map[bookmark_id]
                bookmark_tags = bookmark_data.get("tags", [])

                if bookmark_tags:
                    # Convert to Anki tag format with prefix
                    anki_tags = []
                    for tag in bookmark_tags:
                        tag_name = tag.get("name", "").strip()
                        if tag_name:
                            anki_tag = f"{self.config.anki_tag_prefix}::{tag_name}"
                            anki_tags.append(anki_tag)

                    if anki_tags:
                        # Update the note's tags
                        self.anki_manager.update_note(note_id, {}, tags=anki_tags)
                        logger.debug(f"Updated tags for note {note_id}: {anki_tags}")
                        count += 1

            except Exception as e:
                logger.warning(f"Failed to sync tags for highlight {highlight_id}: {e}")
                if self.debug:
                    raise

        logger.info(f"Tag synchronization completed: updated {count} notes")
        return count

    @optional_typecheck
    def _handle_text_field_updates(
        self,
        highlights_by_id: Dict[str, Dict],
        state_mappings: Dict[str, Dict],
        sync_state: Dict[str, Any],
    ) -> int:
        """
        Update Text fields of existing notes with current parser logic.

        This allows parser improvements to be applied to previously created notes.

        Parameters
        ----------
        highlights_by_id : Dict[str, Dict]
            All highlights indexed by ID
        state_mappings : Dict[str, Dict]
            All state mappings indexed by highlight ID
        sync_state : Dict[str, Any]
            The sync state data

        Returns
        -------
        int
            Number of notes updated
        """
        # Collect highlights that need Text field updates
        highlights_to_update = []
        for highlight_id, mapping in state_mappings.items():
            if mapping["status"] != "active":
                continue

            if highlight_id not in highlights_by_id:
                continue

            highlight = highlights_by_id[highlight_id]
            bookmark_id = highlight.get("bookmarkId")
            note_id = mapping["note_id"]

            if bookmark_id:
                highlights_to_update.append(
                    (highlight_id, bookmark_id, note_id, highlight)
                )

        if not highlights_to_update:
            logger.info("No notes to update Text fields for")
            return 0

        # Fetch all bookmark data in parallel
        bookmark_ids = [bookmark_id for _, bookmark_id, _, _ in highlights_to_update]
        logger.info(
            f"Fetching {len(bookmark_ids)} bookmarks in parallel for Text field updates"
        )
        bookmark_data_map = self._fetch_bookmarks_parallel(bookmark_ids)

        # Update Text fields
        count = 0
        for highlight_id, bookmark_id, note_id, highlight in highlights_to_update:
            if bookmark_id not in bookmark_data_map:
                logger.warning(
                    f"Skipping Text field update for highlight {highlight_id}: "
                    f"bookmark {bookmark_id} fetch failed"
                )
                continue

            try:
                bookmark_data = bookmark_data_map[bookmark_id]

                # Update the Text field if it has changed
                if self.anki_manager.update_note_text_field(
                    note_id, highlight, bookmark_data
                ):
                    count += 1

            except Exception as e:
                logger.warning(
                    f"Failed to update Text field for highlight {highlight_id}: {e}"
                )
                if self.debug:
                    raise

        logger.info(f"Text field updates completed: updated {count} notes")
        return count

    @optional_typecheck
    def _acquire_lock(self) -> None:
        """
        Acquire lock to prevent concurrent execution.

        Raises
        ------
        SyncLogicError
            If another instance is already running
        """
        from .exceptions import SyncLogicError

        if self.lock_file.exists():
            try:
                with open(self.lock_file, "r") as f:
                    existing_pid = f.read().strip()

                # Check if the process is still running
                try:
                    os.kill(
                        int(existing_pid), 0
                    )  # Send signal 0 to check if process exists
                    error_msg = f"Another Karanki sync process is already running (PID: {existing_pid})"
                    logger.error(error_msg)
                    raise SyncLogicError(error_msg)
                except (OSError, ValueError):
                    # Process doesn't exist, remove stale lock file
                    logger.warning(
                        f"Removing stale lock file (PID {existing_pid} not running)"
                    )
                    self.lock_file.unlink()

            except Exception as e:
                if isinstance(e, SyncLogicError):
                    raise
                logger.warning(f"Error reading lock file, removing it: {e}")
                try:
                    self.lock_file.unlink()
                except:
                    pass

        # Create lock file with current PID
        try:
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.lock_file, "w") as f:
                f.write(str(os.getpid()))
            logger.debug(f"Acquired lock: {self.lock_file}")
        except Exception as e:
            raise SyncLogicError(f"Failed to create lock file: {e}")

    @optional_typecheck
    def _release_lock(self) -> None:
        """Release the execution lock by removing the lock file."""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
                logger.debug(f"Released lock: {self.lock_file}")
        except Exception as e:
            logger.warning(f"Failed to remove lock file: {e}")

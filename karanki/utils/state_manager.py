"""
Sync state management

Handles loading, saving, and managing the synchronization state between
Karakeep highlights and Anki notes.

Created with assistance from aider.chat (https://github.com/Aider-AI/aider/)
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger

from .exceptions import StateFileError

# Beartype decorator pattern for optional runtime type checking
try:
    from beartype import beartype as optional_typecheck
except ImportError:
    from typing import Callable

    def optional_typecheck(callable_obj: Callable) -> Callable:
        """Dummy decorator if beartype is not installed."""
        return callable_obj


@dataclass
class SyncMapping:
    """Represents a mapping between a Karakeep highlight and an Anki note."""

    highlight_id: str
    note_id: str
    bookmark_id: str
    color: str
    status: str  # active, missing_from_karakeep, missing_from_anki, missing_from_both
    created_at: str
    updated_at: str
    anki_deck: str
    anki_tag_prefix: Optional[str] = None


class SyncStateManager:
    """
    Manager class for synchronization state persistence.

    This class handles loading and saving the sync state file that tracks
    the mapping between Karakeep highlights and Anki notes.
    """

    @optional_typecheck
    def __init__(self, config):
        """
        Initialize the sync state manager.

        Parameters
        ----------
        config : SyncConfig
            Configuration object containing sync settings
        """
        self.config = config
        self.state_file = Path(config.sync_state_path)

        logger.debug(f"SyncStateManager initialized with state file: {self.state_file}")

    @optional_typecheck
    def load_state(self) -> Dict[str, Any]:
        """
        Load the synchronization state from file.

        Returns
        -------
        Dict[str, Any]
            The sync state data, or empty structure if file doesn't exist

        Raises
        ------
        StateFileError
            If the state file exists but cannot be loaded
        """
        if not self.state_file.exists():
            logger.info(
                f"State file {self.state_file} does not exist, starting with empty state"
            )
            return self._create_empty_state()

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)

            # Validate state structure
            self._validate_state_structure(state_data)

            logger.info(f"Loaded sync state from {self.state_file}")
            logger.debug(
                f"State contains {len(state_data.get('mappings', []))} mappings"
            )

            return state_data

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in state file {self.state_file}: {e}"
            logger.error(error_msg)
            raise StateFileError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load state file {self.state_file}: {e}"
            logger.error(error_msg)
            raise StateFileError(error_msg)

    @optional_typecheck
    def save_state(self, state_data: Dict[str, Any]) -> None:
        """
        Save the synchronization state to file.

        Parameters
        ----------
        state_data : Dict[str, Any]
            The sync state data to save

        Raises
        ------
        StateFileError
            If the state cannot be saved
        """
        try:
            # Create backup before saving new state
            self._create_backup()

            # Update metadata
            state_data["last_updated"] = datetime.now().isoformat()
            state_data["version"] = "1.0"

            # Ensure parent directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            # Write state file atomically by writing to temp file first
            # Append .tmp to preserve the original extension
            temp_file = Path(str(self.state_file) + ".tmp")

            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(state_data, f, indent=2, ensure_ascii=False)
                    # Ensure all data is written to disk before rename
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename - this is guaranteed to be atomic on POSIX systems
                temp_file.replace(self.state_file)
            except Exception:
                # Clean up temp file if something went wrong
                if temp_file.exists():
                    temp_file.unlink()
                raise

            logger.info(f"Saved sync state to {self.state_file}")
            logger.debug(
                f"State contains {len(state_data.get('mappings', []))} mappings"
            )

        except Exception as e:
            error_msg = f"Failed to save state file {self.state_file}: {e}"
            logger.error(error_msg)
            raise StateFileError(error_msg)

    @optional_typecheck
    def add_mapping(
        self,
        state_data: Dict[str, Any],
        highlight_id: str,
        note_id: str,
        bookmark_id: str,
        color: str,
        anki_deck: str,
    ) -> None:
        """
        Add a new mapping to the state data.

        Parameters
        ----------
        state_data : Dict[str, Any]
            The sync state data to modify
        highlight_id : str
            Karakeep highlight ID
        note_id : str
            Anki note ID
        bookmark_id : str
            Karakeep bookmark ID
        color : str
            Highlight color
        anki_deck : str
            Target Anki deck name
        """
        now = datetime.now().isoformat()

        mapping = SyncMapping(
            highlight_id=highlight_id,
            note_id=note_id,
            bookmark_id=bookmark_id,
            color=color,
            status="active",
            created_at=now,
            updated_at=now,
            anki_deck=anki_deck,
            anki_tag_prefix=(
                self.config.anki_tag_prefix if self.config.sync_tags else None
            ),
        )

        state_data["mappings"].append(asdict(mapping))
        logger.debug(f"Added mapping: highlight {highlight_id} -> note {note_id}")

    @optional_typecheck
    def update_mapping_status(
        self, state_data: Dict[str, Any], highlight_id: str, new_status: str
    ) -> bool:
        """
        Update the status of an existing mapping.

        Parameters
        ----------
        state_data : Dict[str, Any]
            The sync state data to modify
        highlight_id : str
            Karakeep highlight ID
        new_status : str
            New status for the mapping

        Returns
        -------
        bool
            True if mapping was found and updated, False otherwise
        """
        for mapping in state_data["mappings"]:
            if mapping["highlight_id"] == highlight_id:
                mapping["status"] = new_status
                mapping["updated_at"] = datetime.now().isoformat()
                logger.debug(
                    f"Updated mapping status for highlight {highlight_id} to {new_status}"
                )
                return True

        logger.warning(f"No mapping found for highlight {highlight_id}")
        return False

    @optional_typecheck
    def find_mapping_by_highlight(
        self, state_data: Dict[str, Any], highlight_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find a mapping by highlight ID.

        Parameters
        ----------
        state_data : Dict[str, Any]
            The sync state data to search
        highlight_id : str
            Karakeep highlight ID to find

        Returns
        -------
        Optional[Dict[str, Any]]
            The mapping data if found, None otherwise
        """
        for mapping in state_data["mappings"]:
            if mapping["highlight_id"] == highlight_id:
                return mapping
        return None

    @optional_typecheck
    def find_mapping_by_note(
        self, state_data: Dict[str, Any], note_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find a mapping by Anki note ID.

        Parameters
        ----------
        state_data : Dict[str, Any]
            The sync state data to search
        note_id : str
            Anki note ID to find

        Returns
        -------
        Optional[Dict[str, Any]]
            The mapping data if found, None otherwise
        """
        for mapping in state_data["mappings"]:
            if mapping["note_id"] == note_id:
                return mapping
        return None

    @optional_typecheck
    def add_failed_creation(
        self,
        state_data: Dict[str, Any],
        highlight_id: str,
        bookmark_id: str,
        error_message: str,
    ) -> None:
        """
        Record a failed note creation attempt.

        Parameters
        ----------
        state_data : Dict[str, Any]
            The sync state data to modify
        highlight_id : str
            Karakeep highlight ID that failed
        bookmark_id : str
            Karakeep bookmark ID for the highlight
        error_message : str
            The error message from the failure
        """
        # Ensure failed_creations list exists (for backwards compatibility)
        if "failed_creations" not in state_data:
            state_data["failed_creations"] = []

        # Check if error already exists for this highlight
        for error_entry in state_data["failed_creations"]:
            if error_entry.get("highlight_id") == highlight_id:
                # Update existing error entry
                error_entry["error_message"] = error_message
                error_entry["last_failed_at"] = datetime.now().isoformat()
                error_entry["bookmark_id"] = bookmark_id
                logger.debug(
                    f"Updated existing error entry for highlight {highlight_id}"
                )
                return

        # Create new error entry
        error_entry = {
            "highlight_id": highlight_id,
            "bookmark_id": bookmark_id,
            "error_message": error_message,
            "first_failed_at": datetime.now().isoformat(),
            "last_failed_at": datetime.now().isoformat(),
        }

        state_data["failed_creations"].append(error_entry)
        logger.debug(
            f"Recorded failed creation for highlight {highlight_id}: {error_message}"
        )

    @optional_typecheck
    def clear_failed_creation(
        self, state_data: Dict[str, Any], highlight_id: str
    ) -> bool:
        """
        Clear a failed creation record when we successfully create the note.

        Parameters
        ----------
        state_data : Dict[str, Any]
            The sync state data to modify
        highlight_id : str
            Karakeep highlight ID to clear

        Returns
        -------
        bool
            True if an error was cleared, False if no error existed
        """
        # Ensure failed_creations list exists
        if "failed_creations" not in state_data:
            return False

        # Find and remove error entry
        for i, error_entry in enumerate(state_data["failed_creations"]):
            if error_entry.get("highlight_id") == highlight_id:
                state_data["failed_creations"].pop(i)
                logger.debug(
                    f"Cleared failed creation error for highlight {highlight_id}"
                )
                return True

        return False

    @optional_typecheck
    def _create_empty_state(self) -> Dict[str, Any]:
        """Create an empty state structure."""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "mappings": [],
            "failed_creations": [],
            "config": {
                "anki_tag_prefix": self.config.anki_tag_prefix,
                "karakeep_base_url": self.config.karakeep_base_url,
                "sync_tags": self.config.sync_tags,
            },
        }

    @optional_typecheck
    def _validate_state_structure(self, state_data: Dict[str, Any]) -> None:
        """
        Validate that the state data has the expected structure.

        Parameters
        ----------
        state_data : Dict[str, Any]
            State data to validate

        Raises
        ------
        StateFileError
            If the state structure is invalid
        """
        required_keys = ["version", "mappings"]
        for key in required_keys:
            if key not in state_data:
                raise StateFileError(f"State file missing required key: {key}")

        if not isinstance(state_data["mappings"], list):
            raise StateFileError("State file 'mappings' must be a list")

        # Validate mapping structure
        for i, mapping in enumerate(state_data["mappings"]):
            required_mapping_keys = ["highlight_id", "note_id", "status"]
            for key in required_mapping_keys:
                if key not in mapping:
                    raise StateFileError(f"Mapping {i} missing required key: {key}")

    @optional_typecheck
    def _create_backup(self) -> None:
        """
        Create a backup of the current state file.
        Keeps up to 5 most recent backups.
        """
        if not self.state_file.exists():
            return

        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.state_file.with_suffix(f".backup_{timestamp}")

            # Copy current state to backup
            shutil.copy2(self.state_file, backup_file)
            logger.debug(f"Created backup: {backup_file}")

            # Clean up old backups (keep only 5 most recent)
            self._cleanup_old_backups()

        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
            # Don't fail the main operation if backup fails

    @optional_typecheck
    def _cleanup_old_backups(self) -> None:
        """Remove old backup files, keeping only the 5 most recent."""
        try:
            backup_pattern = f"{self.state_file.stem}.backup_*"
            backup_files = list(self.state_file.parent.glob(backup_pattern))

            # Sort by modification time, newest first
            backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            # Remove files beyond the 5 most recent
            for old_backup in backup_files[5:]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    @optional_typecheck
    def restore_from_backup(self, backup_timestamp: Optional[str] = None) -> bool:
        """
        Restore state from a backup file.

        Parameters
        ----------
        backup_timestamp : Optional[str]
            Specific backup timestamp to restore. If None, restores most recent backup.

        Returns
        -------
        bool
            True if restore was successful, False otherwise
        """
        try:
            if backup_timestamp:
                backup_file = self.state_file.with_suffix(f".backup_{backup_timestamp}")
            else:
                # Find most recent backup
                backup_pattern = f"{self.state_file.stem}.backup_*"
                backup_files = list(self.state_file.parent.glob(backup_pattern))
                if not backup_files:
                    logger.error("No backup files found")
                    return False
                backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                backup_file = backup_files[0]

            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False

            # Restore from backup
            shutil.copy2(backup_file, self.state_file)
            logger.info(f"Restored state from backup: {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False

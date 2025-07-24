"""
Custom exceptions for Karanki

Defines custom exception classes used throughout the Karanki synchronization
process for better error handling and user feedback.

Created with assistance from aider.chat (https://github.com/Aider-AI/aider/)
"""


class KarankiError(Exception):
    """Base exception class for all Karanki-related errors."""

    pass


class AnkiConnectionError(KarankiError):
    """Raised when unable to connect to or communicate with Anki."""

    pass


class KarakeepConnectionError(KarankiError):
    """Raised when unable to connect to or communicate with Karakeep."""

    pass


class DeckNotFoundError(KarankiError):
    """Raised when required Anki decks are not found."""

    pass


class StateFileError(KarankiError):
    """Raised when there are issues with the sync state file."""

    pass


class ContentProcessingError(KarankiError):
    """Raised when there are issues processing bookmark content."""

    pass


class SyncLogicError(KarankiError):
    """Raised when there are issues with the synchronization logic."""

    pass

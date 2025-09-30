#!/usr/bin/env python3
"""
Karanki - Sync Karakeep highlights with Anki cards

This tool provides bidirectional synchronization between Karakeep highlights
and Anki flashcards, organizing them by color-coded retention levels.

Created with assistance from aider.chat (https://github.com/Aider-AI/aider/)
"""

import os
import click
from loguru import logger
from pathlib import Path
from typing import Optional

# Import using relative path to avoid sys.path manipulation
try:
    from utils.sync import KarankiBidirSync, get_default_debug_log_path
except ImportError:
    # Fallback for when running from different directories
    import sys

    sys.path.append(str(Path(__file__).parent))
    from utils.sync import KarankiBidirSync, get_default_debug_log_path

# Beartype decorator pattern for optional runtime type checking
try:
    from beartype import beartype as optional_typecheck
except ImportError:
    from typing import Callable

    def optional_typecheck(callable_obj: Callable) -> Callable:
        """Dummy decorator if beartype is not installed."""
        return callable_obj


VERSION = "1.2.0"


@click.command()
@click.option(
    "--deck-path",
    default="Karakeep",
    help="Base deck name for Karakeep cards (default: 'Karakeep')",
)
@click.option(
    "--sync-state-path",
    default=None,
    help="Path to sync state file (default: uses platformdirs user data directory)",
)
@click.option(
    "--karakeep-base-url",
    required=False,
    default=None,
    help="Base URL for Karakeep instance (e.g., https://karakeep.example.com). Defaults to KARAKEEP_PYTHON_API_ENDPOINT environment variable.",
)
@click.option(
    "--sync-tags",
    is_flag=True,
    default=True,
    help="Enable bidirectional tag synchronization between Karakeep and Anki",
)
@click.option(
    "--anki-tag-prefix",
    default="karakeep",
    help="Prefix for Anki tags from Karakeep (default: 'karakeep')",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging output",
)
@click.option(
    "-n",
    "--create-only-n",
    type=int,
    default=-1,
    help="Limit creation to N new notes per run (-1 for unlimited, 0 is invalid)",
)
@click.option(
    "--only-sync",
    is_flag=True,
    help="Only sync existing cards without creating new ones",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode for post-mortem debugging (disables exception handling)",
)
@click.version_option(version=VERSION)
@optional_typecheck
def main(
    deck_path: str,
    sync_state_path: Optional[str],
    karakeep_base_url: str,
    sync_tags: bool,
    anki_tag_prefix: str,
    verbose: bool,
    create_only_n: int,
    only_sync: bool,
    debug: bool,
) -> None:
    """
    Sync Karakeep highlights with Anki cards.

    This tool creates Anki flashcards from your Karakeep highlights, organizing
    them into different decks based on highlight color:
    - Red highlights → Red deck (95% retention)
    - Yellow highlights → Yellow deck (90% retention)
    - Blue highlights → Blue deck (85% retention)
    - Green highlights → Green deck (80% retention)
    """
    # Get Karakeep base URL from environment variable if not provided via CLI
    if karakeep_base_url is None:
        karakeep_base_url = os.environ.get("KARAKEEP_PYTHON_API_ENDPOINT")
        if karakeep_base_url is None:
            error_msg = (
                "Karakeep base URL not specified. Please either:\n"
                "  1. Set the KARAKEEP_PYTHON_API_ENDPOINT environment variable, or\n"
                "  2. Pass --karakeep-base-url option on the command line"
            )
            logger.error(error_msg)
            raise click.ClickException(error_msg)
        logger.debug(
            f"Using Karakeep base URL from KARAKEEP_PYTHON_API_ENDPOINT: {karakeep_base_url}"
        )

    # Set up logging
    logger.remove()  # Remove default handler

    # Use platformdirs for debug log location
    debug_log_path = get_default_debug_log_path()

    if verbose:
        logger.add(
            debug_log_path,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
        )
        logger.add(
            lambda msg: click.echo(msg, err=True),
            level="DEBUG",
            format="{time:HH:mm:ss} | {level} | {message}",
        )
    else:
        logger.add(
            debug_log_path,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
        )
        logger.add(
            lambda msg: click.echo(msg, err=True),
            level="INFO",
            format="{time:HH:mm:ss} | {level} | {message}",
        )

    # Validate create_only_n parameter
    if create_only_n == 0:
        error_msg = "--create-only-n cannot be 0. Use -1 for unlimited or a positive number to limit."
        logger.error(error_msg)
        raise click.ClickException(error_msg)

    logger.info(f"Starting Karanki v{VERSION}")
    logger.debug(f"Config: deck_path={deck_path}, sync_state_path={sync_state_path}")
    logger.debug(
        f"Config: karakeep_base_url={karakeep_base_url}, sync_tags={sync_tags}"
    )
    logger.debug(f"Config: anki_tag_prefix={anki_tag_prefix}, verbose={verbose}")
    logger.debug(
        f"Config: create_only_n={create_only_n}, only_sync={only_sync}, debug={debug}"
    )

    try:
        # Initialize the sync manager
        sync_manager = KarankiBidirSync(
            deck_path=deck_path,
            sync_state_path=sync_state_path,
            karakeep_base_url=karakeep_base_url,
            sync_tags=sync_tags,
            anki_tag_prefix=anki_tag_prefix,
            create_only_n=create_only_n,
            only_sync=only_sync,
            debug=debug,
        )

        # Run the synchronization
        sync_manager.run()

        logger.info("Synchronization completed successfully")

    except Exception as e:
        if debug:
            # In debug mode, log the error but don't catch it
            # This allows pdb post-mortem debugging
            logger.error(f"Synchronization failed: {e}")
            logger.debug(
                "Debug mode enabled - re-raising exception for post-mortem debugging"
            )
            raise
        else:
            # Normal mode - log and re-raise
            logger.error(f"Synchronization failed: {e}")
            raise


if __name__ == "__main__":
    main()

"""
Karakeep integration manager

Handles all interactions with Karakeep through the karakeep_python_api,
including highlight retrieval and bookmark content processing.

Created with assistance from aider.chat (https://github.com/Aider-AI/aider/)
"""

from typing import Dict, List, Optional, Any, Callable
import json
import time
from functools import wraps
from loguru import logger
from karakeep_python_api import KarakeepAPI

from .exceptions import KarakeepConnectionError

# Beartype decorator pattern for optional runtime type checking
try:
    from beartype import beartype as optional_typecheck
except ImportError:
    from typing import Callable

    def optional_typecheck(callable_obj: Callable) -> Callable:
        """Dummy decorator if beartype is not installed."""
        return callable_obj


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator for API calls that may fail due to network issues.

    Parameters
    ----------
    max_attempts : int
        Maximum number of retry attempts
    delay : float
        Initial delay between retries in seconds
    backoff : float
        Multiplier for delay between retries
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        # Last attempt, re-raise the exception
                        raise e

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

            # This should never be reached, but just in case
            raise last_exception

        return wrapper

    return decorator


@optional_typecheck
def karakeepdata_to_dict(obj: Any) -> Any:
    """
    Recursively convert objects to dictionaries, handling JSON serialization issues.

    This function attempts to convert objects to dictionaries and recursively
    processes nested structures. For objects that can't be JSON serialized,
    it tries to call .dict() method if available.

    Parameters
    ----------
    obj : Any
        Object to convert to dictionary

    Returns
    -------
    Any
        Dictionary representation of the object, or the original value if conversion fails
    """
    # First try to convert the main object to dict if it's not already a basic type
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            obj = obj.dict()
        except Exception:
            pass
    elif not isinstance(obj, (dict, list, str, int, float, bool, type(None))):
        try:
            obj = dict(obj)
        except Exception:
            pass

    # Now recursively process the structure
    try:
        # Test if it's JSON serializable as-is
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass

    # If not JSON serializable, recursively process
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            result[key] = karakeepdata_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [karakeepdata_to_dict(item) for item in obj]
    elif hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return karakeepdata_to_dict(obj.dict())
        except Exception:
            pass

    # If all else fails, try to convert to string for JSON compatibility
    try:
        json.dumps(str(obj))
        return str(obj)
    except Exception:
        return str(obj)


class KarakeepManager:
    """
    Manager class for all Karakeep-related operations.

    This class handles communication with Karakeep, manages highlights,
    and processes bookmark content.
    """

    @optional_typecheck
    def __init__(self, config):
        """
        Initialize the Karakeep manager.

        Parameters
        ----------
        config : SyncConfig
            Configuration object containing sync settings
        """
        self.config = config
        self.karakeep_api = KarakeepAPI()

        logger.debug("KarakeepManager initialized")

    @optional_typecheck
    @retry_on_failure(max_attempts=3, delay=2.0)
    def validate_connection(self) -> None:
        """
        Validate that Karakeep API is accessible.

        Raises
        ------
        KarakeepConnectionError
            If unable to connect to Karakeep
        """
        try:
            # Test connection by getting user info
            user_info = dict(self.karakeep_api.get_current_user_info())
            logger.debug(
                f"Successfully connected to Karakeep for user: {user_info.get('id', 'unknown')}"
            )
        except Exception as e:
            error_msg = (
                f"Failed to connect to Karakeep. Check API credentials. Error: {e}"
            )
            logger.error(error_msg)
            raise KarakeepConnectionError(error_msg) from e

    @optional_typecheck
    @retry_on_failure(max_attempts=3, delay=1.0)
    def get_all_highlights(self) -> List[Dict[str, Any]]:
        """
        Retrieve all highlights from Karakeep with pagination.

        Returns
        -------
        List[Dict[str, Any]]
            List of all highlights
        """
        logger.info("Fetching all highlights from Karakeep...")

        all_highlights = []
        cursor = None

        try:
            while True:
                # Get a batch of highlights
                response = karakeepdata_to_dict(
                    self.karakeep_api.get_all_highlights(cursor=cursor, limit=100)
                )

                # Validate response structure
                if not isinstance(response, dict):
                    raise ValueError(f"Expected dict response, got {type(response)}")

                if "highlights" not in response:
                    raise ValueError("Response missing 'highlights' field")

                highlights = response.get("highlights", [])
                if not isinstance(highlights, list):
                    raise ValueError(
                        f"Expected highlights to be list, got {type(highlights)}"
                    )

                # Validate each highlight has required fields
                valid_highlights = []
                for h in highlights:
                    highlight_dict = karakeepdata_to_dict(h)
                    if not isinstance(highlight_dict, dict):
                        logger.warning(f"Skipping malformed highlight: {h}")
                        continue
                    if "id" not in highlight_dict:
                        logger.warning(
                            f"Skipping highlight without ID: {highlight_dict}"
                        )
                        continue
                    valid_highlights.append(highlight_dict)

                all_highlights.extend(valid_highlights)

                logger.debug(
                    f"Fetched {len(valid_highlights)} valid highlights (total so far: {len(all_highlights)})"
                )

                # Check if there are more pages
                cursor = response.get("nextCursor")
                if not cursor:
                    break

            logger.info(
                f"Retrieved {len(all_highlights)} total highlights from Karakeep"
            )
            return all_highlights

        except Exception as e:
            error_msg = f"Failed to retrieve highlights from Karakeep: {e}"
            logger.error(error_msg)
            raise KarakeepConnectionError(error_msg) from e

    @optional_typecheck
    @retry_on_failure(max_attempts=3, delay=1.0)
    def get_bookmark_content(self, bookmark_id: str) -> Dict[str, Any]:
        """
        Get bookmark content including HTML for text processing.

        Parameters
        ----------
        bookmark_id : str
            ID of the bookmark to retrieve

        Returns
        -------
        Dict[str, Any]
            Bookmark data including content
        """
        # Validate input
        if not bookmark_id or not isinstance(bookmark_id, str):
            raise ValueError("bookmark_id must be a non-empty string")

        try:
            response = karakeepdata_to_dict(
                self.karakeep_api.get_a_single_bookmark(
                    bookmark_id=bookmark_id, include_content=True
                )
            )
            # API returns a list with a single bookmark according to spec
            if isinstance(response, list) and len(response) > 0:
                bookmark_data = response[0]
            else:
                bookmark_data = response

            # Validate bookmark data structure
            if not isinstance(bookmark_data, dict):
                raise ValueError(
                    f"Expected bookmark to be dict, got {type(bookmark_data)}"
                )

            if "id" not in bookmark_data:
                raise ValueError("Bookmark data missing required 'id' field")

            logger.debug(f"Retrieved bookmark content for {bookmark_id}")
            assert json.dumps(
                bookmark_data
            ), "Bookmark does appear to contain things that are not yet turned into a dict"
            return bookmark_data

        except Exception as e:
            error_msg = f"Failed to get bookmark content for {bookmark_id}: {e}"
            logger.error(error_msg)
            raise KarakeepConnectionError(error_msg) from e

    @optional_typecheck
    @retry_on_failure(max_attempts=3, delay=1.0)
    def update_highlight_color(self, highlight_id: str, color: str) -> None:
        """
        Update the color of a highlight in Karakeep.

        Parameters
        ----------
        highlight_id : str
            ID of the highlight to update
        color : str
            New color for the highlight (yellow, red, green, blue)
        """
        # Validate color input
        valid_colors = {"red", "yellow", "blue", "green"}
        if color not in valid_colors:
            raise ValueError(f"Invalid color '{color}'. Must be one of: {valid_colors}")

        if not highlight_id or not isinstance(highlight_id, str):
            raise ValueError("highlight_id must be a non-empty string")

        try:
            self.karakeep_api.update_a_highlight(highlight_id=highlight_id, color=color)
            logger.debug(f"Updated highlight {highlight_id} color to {color}")

        except Exception as e:
            error_msg = f"Failed to update highlight {highlight_id} color: {e}"
            logger.error(error_msg)
            raise KarakeepConnectionError(error_msg) from e

    @optional_typecheck
    def get_bookmark_tags(self, bookmark_id: str) -> List[Dict[str, Any]]:
        """
        Get tags for a specific bookmark.

        Parameters
        ----------
        bookmark_id : str
            ID of the bookmark

        Returns
        -------
        List[Dict[str, Any]]
            List of tag information
        """
        try:
            bookmark_data = karakeepdata_to_dict(self.get_bookmark_content(bookmark_id))
            tags = bookmark_data.get("tags", [])
            logger.debug(f"Retrieved {len(tags)} tags for bookmark {bookmark_id}")
            return tags

        except Exception as e:
            error_msg = f"Failed to get tags for bookmark {bookmark_id}: {e}"
            logger.error(error_msg)
            raise KarakeepConnectionError(error_msg) from e

"""Data loading utilities for constructicon patterns."""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path


def get_data_path() -> Path:
    """Get path to the data directory."""
    return Path(__file__).parent


def load_constructicon_patterns() -> List[Dict[str, str]]:
    """Load constructicon patterns from the bundled JSON file.

    Returns:
        List of dictionaries with 'id' and 'pattern' keys

    Raises:
        FileNotFoundError: If constructicon.json is not found
        json.JSONDecodeError: If JSON file is malformed
    """
    data_path = get_data_path() / 'constructicon.json'

    if not data_path.exists():
        raise FileNotFoundError(f"Constructicon data not found at {data_path}")

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            patterns = json.load(f)

        # Validate format
        if not isinstance(patterns, list):
            raise ValueError("Constructicon data must be a list")

        for i, pattern in enumerate(patterns):
            if not isinstance(pattern, dict) or 'id' not in pattern or 'pattern' not in pattern:
                raise ValueError(f"Invalid pattern format at index {i}. Expected dict with 'id' and 'pattern' keys")

        return patterns

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in constructicon.json: {e}")


def validate_custom_patterns(patterns: List[Dict[str, str]]) -> None:
    """Validate format of custom patterns.

    Args:
        patterns: List of pattern dictionaries

    Raises:
        ValueError: If patterns format is invalid
    """
    if not isinstance(patterns, list):
        raise ValueError("Patterns must be a list")

    for i, pattern in enumerate(patterns):
        if not isinstance(pattern, dict):
            raise ValueError(f"Pattern at index {i} must be a dictionary")
        if 'id' not in pattern or 'pattern' not in pattern:
            raise ValueError(f"Pattern at index {i} must have 'id' and 'pattern' keys")
        if not isinstance(pattern['id'], str) or not isinstance(pattern['pattern'], str):
            raise ValueError(f"Pattern at index {i}: 'id' and 'pattern' must be strings")

"""Postprocessing utilities for span predictions."""

import string
from typing import Tuple


def apply_punctuation_heuristics(span_text: str,
                                 pattern: str,
                                 original_text: str,
                                 span_start: int) -> Tuple[str,
                                                           int,
                                                           int]:
    """Apply punctuation heuristics to span predictions.

    According to the heuristic: if prediction ends with punctuation,
    check if pattern ends with same punctuation. If not, remove it.
    Also remove any punctuation from the beginning.

    Args:
        span_text: Predicted span text
        pattern: Original pattern text
        original_text: Full original text
        span_start: Start position of span in original text

    Returns:
        Tuple of (cleaned_span_text, new_start, new_end)
    """
    if not span_text:
        return span_text, span_start, span_start

    cleaned_span = span_text
    start_offset = 0
    end_offset = 0

    # Remove punctuation from the beginning
    while cleaned_span and cleaned_span[0] in string.punctuation:
        cleaned_span = cleaned_span[1:]
        start_offset += 1

    # Handle punctuation at the end
    if cleaned_span and cleaned_span[-1] in string.punctuation:
        span_punct = cleaned_span[-1]

        # Check if pattern ends with the same punctuation
        if not (pattern.rstrip().endswith(span_punct)):
            # Remove punctuation from span
            cleaned_span = cleaned_span[:-1].rstrip()
            end_offset = -1

    # Calculate new positions
    new_start = span_start + start_offset
    new_end = span_start + len(span_text) + end_offset

    # Make sure we don't go beyond the original span
    new_end = max(new_start, new_end)

    return cleaned_span.strip(), new_start, new_end


def validate_span(
        span_text: str,
        original_text: str,
        span_start: int,
        span_end: int) -> bool:
    """Validate that span coordinates are correct.

    Args:
        span_text: Span text
        original_text: Original text
        span_start: Start position
        span_end: End position

    Returns:
        True if span is valid, False otherwise
    """
    if span_start < 0 or span_end > len(
            original_text) or span_start >= span_end:
        return False

    extracted_text = original_text[span_start:span_end]
    return extracted_text.strip() == span_text.strip()

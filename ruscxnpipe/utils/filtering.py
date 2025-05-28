"""Filtering utilities for pattern matching."""

import re
from typing import List, Set
from pymystem3 import Mystem
import logging


logger = logging.getLogger(__name__)


class PatternFilter:
    """Filter patterns based on anchor words and morphological patterns."""

    def __init__(self):
        """Initialize the pattern filter with Mystem lemmatizer."""
        self._mystem = Mystem(disambiguation=False)

    def _is_morphological_pattern(self, pattern: str) -> bool:
        """Check if a pattern is morphological.

        Args:
            pattern: Pattern text to check

        Returns:
            True if pattern is morphological, False otherwise
        """
        # Remove passage: prefix if present
        if pattern.startswith("passage: "):
            pattern = pattern[9:]

        # Check for tilde sign
        if "~" in pattern:
            return True

        # Check for Cyrillic character followed by hyphen in the same word
        # or hyphen followed by Cyrillic character in the same word
        words = pattern.split()
        for word in words:
            if re.search(r'[а-яёА-ЯЁ]-', word) or re.search(r'-[а-яёА-ЯЁ]', word):
                return True

        return False


    def _extract_russian_lemmas(self, text: str) -> List[str]:
        """Extract Russian words from text and lemmatize them.

        Args:
            text: Input text

        Returns:
            List of Russian lemmas
        """
        # Strip any prefixes like "query: " or "passage: " if they exist
        if text.startswith("query: "):
            text = text[7:]
        elif text.startswith("passage: "):
            text = text[9:]

        # Lemmatize the text
        lemmas = self._mystem.lemmatize(text)
        # Keep only words containing Russian letters
        russian_lemmas = [
            word for word in lemmas
            if re.search('[а-яёА-ЯЁ]', word) and word.strip()
        ]
        return russian_lemmas


    def filter_patterns_for_queries(self, queries: List[str], patterns: List[str]) -> List[List[int]]:
        """Filter patterns for each query based on anchor words.

        Args:
            queries: List of query texts
            patterns: List of pattern texts

        Returns:
            List of lists, where each inner list contains valid pattern indices for the corresponding query
        """
        # Lemmatize queries
        query_lemmas = [self._extract_russian_lemmas(query) for query in queries]

        # Lemmatize patterns and check if they are morphological
        pattern_lemmas = [self._extract_russian_lemmas(pattern) for pattern in patterns]
        is_morphological = [self._is_morphological_pattern(pattern) for pattern in patterns]

        valid_patterns_for_queries = []

        for query_lem in query_lemmas:
            # Skip if the query has no lemmas
            if not query_lem:
                valid_patterns_for_queries.append(list(range(len(patterns))))
                continue

            # Find valid patterns for this query
            valid_patterns = []
            for j, pattern_lem in enumerate(pattern_lemmas):
                # Check if pattern is morphological - if yes, it's valid for all queries
                if is_morphological[j]:
                    valid_patterns.append(j)
                # Check if at least one pattern lemma is in the query or pattern is without anchors
                elif not pattern_lem or any(lemma in query_lem for lemma in pattern_lem):
                    valid_patterns.append(j)

            # If no valid patterns found, use all patterns to avoid empty results
            if not valid_patterns:
                valid_patterns = list(range(len(patterns)))

            valid_patterns_for_queries.append(valid_patterns)

        return valid_patterns_for_queries

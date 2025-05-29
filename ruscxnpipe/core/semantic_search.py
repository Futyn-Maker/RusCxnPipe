"""Semantic search component for finding construction candidates."""

import pickle
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer
import time

from ..data.loader import load_constructicon_patterns, validate_custom_patterns
from ..models.config import get_model_config
from ..utils.filtering import PatternFilter


logger = logging.getLogger(__name__)


class SemanticSearch:
    """Semantic search for Russian constructicon patterns."""

    def __init__(self,
                 model_name: Optional[str] = None,
                 query_prefix: Optional[str] = None,
                 pattern_prefix: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 use_filtering: bool = True):
        """Initialize semantic search component.

        Args:
            model_name: Name of the sentence transformer model
            query_prefix: Prefix to add to queries
            pattern_prefix: Prefix to add to patterns
            cache_dir: Directory for caching embeddings
            use_filtering: Whether to use rule-based filtering
        """
        # Load configuration
        config = get_model_config('semantic_search')

        self.model_name = model_name or config['model_name']
        self.query_prefix = query_prefix or config['prefixes']['query_prefix']
        self.pattern_prefix = pattern_prefix or config['prefixes']['pattern_prefix']
        self.use_filtering = use_filtering

        # Initialize model
        self._model = None

        # Set up caching
        self.cache_dir = Path(
            cache_dir) if cache_dir else self._get_default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / \
            f"embeddings_cache_{self._get_model_hash()}.pkl"

        # Load cache
        self._embedding_cache = self._load_cache()

        # Initialize filter
        if self.use_filtering:
            self._filter = PatternFilter()

        # Load default patterns
        self._default_patterns = load_constructicon_patterns()

        logger.info(
            f"Initialized SemanticSearch with model: {self.model_name}")

    def _get_default_cache_dir(self) -> Path:
        """Get default cache directory in user's home."""
        return Path.home() / '.cache' / 'ruscxnpipe'

    def _get_model_hash(self) -> str:
        """Get a hash of the model name for cache naming."""
        import hashlib
        return hashlib.md5(self.model_name.encode()).hexdigest()[:8]

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load embedding cache from disk.

        Returns:
            Dictionary mapping pattern IDs to their data and embeddings
        """
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(
                    f"Loaded embedding cache with {len(cache)} entries")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        return {}

    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._embedding_cache, f)
            logger.debug(
                f"Saved embedding cache with {len(self._embedding_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    @property
    def model(self) -> SentenceTransformer:
        """Get the sentence transformer model, loading it if necessary."""
        if self._model is None:
            logger.info(f"Loading model: {self.model_name}")
            start_time = time.time()
            self._model = SentenceTransformer(self.model_name)
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
        return self._model

    def _get_pattern_embeddings(
            self, patterns: List[Dict[str, str]]) -> Tuple[np.ndarray, List[str]]:
        """Get embeddings for patterns, using cache when possible.

        Args:
            patterns: List of pattern dictionaries with 'id' and 'pattern' keys

        Returns:
            Tuple of (embeddings array, list of pattern texts)
        """
        embeddings_list = []
        pattern_texts = []
        patterns_to_encode = []
        patterns_to_encode_indices = []

        # Check cache for each pattern
        for i, pattern_dict in enumerate(patterns):
            pattern_id = pattern_dict['id']
            pattern_text = pattern_dict['pattern']
            pattern_texts.append(pattern_text)

            # Check if pattern is in cache and matches
            if (pattern_id in self._embedding_cache and
                    self._embedding_cache[pattern_id]['pattern'] == pattern_text):
                embeddings_list.append(
                    self._embedding_cache[pattern_id]['embedding'])
            else:
                # Need to encode this pattern
                patterns_to_encode.append(self.pattern_prefix + pattern_text)
                patterns_to_encode_indices.append(i)
                embeddings_list.append(None)  # Placeholder

        # Encode missing patterns in batch
        if patterns_to_encode:
            logger.info(f"Encoding {len(patterns_to_encode)} new patterns")
            start_time = time.time()
            new_embeddings = self.model.encode(
                patterns_to_encode,
                normalize_embeddings=True,
                show_progress_bar=len(patterns_to_encode) > 10
            )
            encode_time = time.time() - start_time
            logger.info(f"Patterns encoded in {encode_time:.2f} seconds")

            # Update cache and embeddings list
            for idx, pattern_idx in enumerate(patterns_to_encode_indices):
                pattern_dict = patterns[pattern_idx]
                embedding = new_embeddings[idx]

                # Update cache
                self._embedding_cache[pattern_dict['id']] = {
                    'pattern': pattern_dict['pattern'],
                    'embedding': embedding
                }

                # Update embeddings list
                embeddings_list[pattern_idx] = embedding

            # Save updated cache
            self._save_cache()

        # Convert to numpy array
        embeddings = np.vstack(embeddings_list)
        return embeddings, pattern_texts

    def find_candidates(self,
                        queries: List[str],
                        patterns: Optional[List[Dict[str, str]]] = None,
                        n: int = 15,
                        batch_size: int = 32) -> List[Dict[str, Any]]:
        """Find construction candidates for given queries.

        Args:
            queries: List of query texts
            patterns: Custom patterns (if None, uses default constructicon patterns)
            n: Number of top candidates to return per query
            batch_size: Batch size for encoding

        Returns:
            List of results, one per query, each containing:
            - 'query': original query text
            - 'candidates': list of n candidates with 'id', 'pattern', 'similarity', 'rank'
        """
        start_time = time.time()

        # Use default patterns if none provided
        if patterns is None:
            patterns = self._default_patterns
        else:
            validate_custom_patterns(patterns)

        logger.info(
            f"Processing {len(queries)} queries against {len(patterns)} patterns")

        # Get pattern embeddings
        pattern_embeddings, pattern_texts = self._get_pattern_embeddings(
            patterns)

        # Encode queries
        logger.info(f"Encoding {len(queries)} queries")
        query_start = time.time()
        prefixed_queries = [self.query_prefix + query for query in queries]
        query_embeddings = self.model.encode(
            prefixed_queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(queries) > 10
        )
        query_time = time.time() - query_start
        logger.info(f"Queries encoded in {query_time:.2f} seconds")

        # Apply filtering if enabled
        valid_patterns_for_queries = None
        if self.use_filtering:
            logger.info("Applying rule-based filtering")
            filter_start = time.time()
            valid_patterns_for_queries = self._filter.filter_patterns_for_queries(
                queries, pattern_texts)
            filter_time = time.time() - filter_start
            logger.info(f"Filtering completed in {filter_time:.2f} seconds")

        # Calculate similarities and find top candidates
        results = []
        for i, query in enumerate(queries):
            # Get valid pattern indices for this query
            if valid_patterns_for_queries is not None:
                valid_indices = valid_patterns_for_queries[i]
            else:
                valid_indices = list(range(len(patterns)))

            # Calculate similarities only for valid patterns
            query_embedding = query_embeddings[i]
            valid_pattern_embeddings = pattern_embeddings[valid_indices]

            similarities = np.dot(valid_pattern_embeddings, query_embedding)

            # Get top n candidates
            top_indices = np.argsort(similarities)[::-1][:n]

            candidates = []
            for rank, idx in enumerate(top_indices, 1):
                original_idx = valid_indices[idx]
                candidates.append({
                    'id': patterns[original_idx]['id'],
                    'pattern': patterns[original_idx]['pattern'],
                    'similarity': float(similarities[idx]),
                    'rank': rank
                })

            results.append({
                'query': query,
                'candidates': candidates
            })

        total_time = time.time() - start_time
        logger.info(f"Semantic search completed in {total_time:.2f} seconds")

        return results

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Embedding cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache.

        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self._embedding_cache),
            'cache_file': str(self.cache_file),
            'cache_exists': self.cache_file.exists()
        }

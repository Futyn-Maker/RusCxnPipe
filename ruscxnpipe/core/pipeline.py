"""Complete pipeline for Russian constructicon pattern extraction."""

import logging
from typing import List, Dict, Any, Optional
import time

from .semantic_search import SemanticSearch
from .classification import ConstructionClassifier
from .span_prediction import SpanPredictor
from ..models.config import get_model_config


logger = logging.getLogger(__name__)


class RusCxnPipe:
    """Complete pipeline for Russian constructicon pattern extraction."""

    def __init__(self,
                 # Semantic search parameters
                 semantic_model: Optional[str] = None,
                 query_prefix: Optional[str] = None,
                 pattern_prefix: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 use_filtering: bool = True,

                 # Classification parameters
                 classification_model: Optional[str] = None,
                 classification_query_prefix: Optional[str] = None,
                 classification_pattern_prefix: Optional[str] = None,
                 classification_threshold: float = 0.5,
                 classification_device: Optional[str] = None,

                 # Span prediction parameters
                 span_model: Optional[str] = None,
                 span_model_args: Optional[Dict[str, Any]] = None,
                 span_use_cuda: Optional[bool] = None,

                 # Custom patterns
                 custom_patterns: Optional[List[Dict[str, str]]] = None):
        """Initialize the complete RusCxnPipe pipeline.

        Args:
            # Semantic search parameters
            semantic_model: Name of the semantic search model
            query_prefix: Prefix for queries in semantic search
            pattern_prefix: Prefix for patterns in semantic search
            cache_dir: Directory for caching embeddings
            use_filtering: Whether to use rule-based filtering

            # Classification parameters
            classification_model: Name of the classification model
            classification_query_prefix: Prefix for queries in classification
            classification_pattern_prefix: Prefix for patterns in classification
            classification_threshold: Classification threshold
            classification_device: Device for classification model

            # Span prediction parameters
            span_model: Name/path of the span prediction model
            span_model_args: Additional arguments for span prediction model
            span_use_cuda: Whether to use CUDA for span prediction

            # Data parameters
            custom_patterns: Custom patterns to use instead of default constructicon
        """
        self.custom_patterns = custom_patterns

        # Initialize semantic search component
        logger.info("Initializing semantic search component...")
        self.semantic_search = SemanticSearch(
            model_name=semantic_model,
            query_prefix=query_prefix,
            pattern_prefix=pattern_prefix,
            cache_dir=cache_dir,
            use_filtering=use_filtering
        )

        # Initialize classification component
        logger.info("Initializing classification component...")
        self.classifier = ConstructionClassifier(
            model_name=classification_model,
            query_prefix=classification_query_prefix,
            pattern_prefix=classification_pattern_prefix,
            threshold=classification_threshold,
            device=classification_device
        )

        # Initialize span prediction component
        logger.info("Initializing span prediction component...")
        self.span_predictor = SpanPredictor(
            model_name=span_model,
            model_args=span_model_args,
            use_cuda=span_use_cuda
        )

        logger.info("RusCxnPipe pipeline initialized successfully")

    def process_texts(self,
                      examples: List[str],
                      n_candidates: int = 15,
                      semantic_batch_size: int = 32,
                      classification_batch_size: int = 32,
                      span_batch_size: Optional[int] = None) -> List[Dict[str,
                                                                          Any]]:
        """Process text examples through the complete pipeline.

        Args:
            examples: List of text examples to process
            n_candidates: Number of candidates to retrieve from semantic search
            semantic_batch_size: Batch size for semantic search
            classification_batch_size: Batch size for classification
            span_batch_size: Batch size for span prediction

        Returns:
            List of results, one per example. Each result contains:
            - 'example': original text
            - 'constructions': list of found constructions with 'id', 'pattern', and 'span'
        """
        start_time = time.time()

        if not examples:
            return []

        logger.info(
            f"Processing {
                len(examples)} examples through complete pipeline")

        # Step 1: Semantic search to find candidates
        logger.info("Step 1: Finding candidates with semantic search...")
        search_start = time.time()
        search_results = self.semantic_search.find_candidates(
            queries=examples,
            patterns=self.custom_patterns,
            n=n_candidates,
            batch_size=semantic_batch_size
        )
        search_time = time.time() - search_start
        logger.info(f"Semantic search completed in {search_time:.2f} seconds")

        # Step 2: Classification to filter candidates
        logger.info("Step 2: Classifying candidates...")
        classification_start = time.time()

        # Prepare candidates for classification
        candidates_for_classification = []
        for result in search_results:
            candidates_for_classification.append(result['candidates'])

        classification_results = self.classifier.classify_candidates(
            queries=examples,
            candidates_list=candidates_for_classification,
            batch_size=classification_batch_size
        )
        classification_time = time.time() - classification_start
        logger.info(
            f"Classification completed in {
                classification_time:.2f} seconds")

        # Step 3: Filter only positive classifications and prepare for span
        # prediction
        logger.info(
            "Step 3: Preparing positive candidates for span prediction...")
        span_input = []
        result_mapping = []  # To track which results belong to which original example

        for i, (example, classified_candidates) in enumerate(
                zip(examples, classification_results)):
            # Filter only positive candidates
            positive_candidates = [
                candidate for candidate in classified_candidates
                if candidate.get('is_present', 0) == 1
            ]

            if positive_candidates:
                # Prepare patterns for span prediction (remove extra fields)
                patterns_for_span = []
                for candidate in positive_candidates:
                    patterns_for_span.append({
                        'id': candidate['id'],
                        'pattern': candidate['pattern']
                    })

                span_input.append({
                    'example': example,
                    'patterns': patterns_for_span
                })
                result_mapping.append((i, positive_candidates))

        # Step 4: Span prediction
        logger.info(
            f"Step 4: Predicting spans for {
                len(span_input)} examples with positive candidates...")
        span_start = time.time()

        if span_input:
            span_results = self.span_predictor.predict_spans(
                examples_with_patterns=span_input,
                batch_size=span_batch_size
            )
        else:
            span_results = []

        span_time = time.time() - span_start
        logger.info(f"Span prediction completed in {span_time:.2f} seconds")

        # Step 5: Combine results
        logger.info("Step 5: Combining final results...")
        final_results = []

        # Initialize all results with empty constructions
        for example in examples:
            final_results.append({
                'example': example,
                'constructions': []
            })

        # Fill in results for examples with positive candidates
        for span_result, (original_idx, original_candidates) in zip(
                span_results, result_mapping):
            constructions = []

            for pattern_result in span_result['patterns']:
                construction = {
                    'id': pattern_result['id'],
                    'pattern': pattern_result['pattern'],
                    'span': pattern_result['span']
                }
                constructions.append(construction)

            final_results[original_idx]['constructions'] = constructions

        total_time = time.time() - start_time

        # Log summary statistics
        total_constructions = sum(
            len(result['constructions']) for result in final_results)
        examples_with_constructions = sum(
            1 for result in final_results if result['constructions'])

        logger.info(f"Pipeline completed in {total_time:.2f} seconds")
        logger.info(
            f"Found {total_constructions} constructions in {examples_with_constructions}/{
                len(examples)} examples")

        return final_results

    def process_text(self,
                     text: str,
                     n_candidates: int = 15,
                     semantic_batch_size: int = 32,
                     classification_batch_size: int = 32,
                     span_batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Process a single text example (convenience method).

        Args:
            text: Text example to process
            n_candidates: Number of candidates to retrieve from semantic search
            semantic_batch_size: Batch size for semantic search
            classification_batch_size: Batch size for classification
            span_batch_size: Batch size for span prediction

        Returns:
            Dictionary with 'example' and 'constructions' keys
        """
        results = self.process_texts(
            examples=[text],
            n_candidates=n_candidates,
            semantic_batch_size=semantic_batch_size,
            classification_batch_size=classification_batch_size,
            span_batch_size=span_batch_size
        )

        return results[0] if results else {
            'example': text, 'constructions': []}

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline components.

        Returns:
            Dictionary with information about all pipeline components
        """
        return {
            'semantic_search': self.semantic_search.get_cache_info(),
            'classification': self.classifier.get_model_info(),
            'span_prediction': self.span_predictor.get_model_info(),
            'custom_patterns': len(
                self.custom_patterns) if self.custom_patterns else None}

    def clear_cache(self) -> None:
        """Clear all caches used by the pipeline."""
        self.semantic_search.clear_cache()
        logger.info("Pipeline caches cleared")

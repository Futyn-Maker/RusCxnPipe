"""Classification component for determining construction presence."""

import torch
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from tqdm.auto import tqdm

from ..models.config import get_model_config


logger = logging.getLogger(__name__)


class ConstructionClassifier:
    """Classifier for determining if constructions are present in text."""

    def __init__(self,
                 model_name: Optional[str] = None,
                 query_prefix: Optional[str] = None,
                 pattern_prefix: Optional[str] = None,
                 threshold: float = 0.5,
                 device: Optional[str] = None):
        """Initialize construction classifier.

        Args:
            model_name: Name of the classification model
            query_prefix: Prefix to add to queries
            pattern_prefix: Prefix to add to patterns
            threshold: Classification threshold (default 0.5)
            device: Device to run model on ('cpu', 'cuda', or None for auto)
        """
        # Load configuration
        config = get_model_config('classification')

        self.model_name = model_name or config['model_name']
        self.query_prefix = query_prefix or config['prefixes']['query_prefix']
        self.pattern_prefix = pattern_prefix or config['prefixes']['pattern_prefix']
        self.threshold = threshold

        # Set device
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model and tokenizer
        self._model = None
        self._tokenizer = None

        logger.info(
            f"Initialized ConstructionClassifier with model: {
                self.model_name}")
        logger.info(f"Using device: {self.device}")

    @property
    def model(self) -> AutoModelForSequenceClassification:
        """Get the classification model, loading it if necessary."""
        if self._model is None:
            logger.info(f"Loading classification model: {self.model_name}")
            start_time = time.time()
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2
            )
            self._model.to(self.device)
            self._model.eval()
            load_time = time.time() - start_time
            logger.info(
                f"Classification model loaded in {
                    load_time:.2f} seconds")
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer, loading it if necessary."""
        if self._tokenizer is None:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def _prepare_inputs(
            self, queries: List[str], candidates_list: List[List[Dict[str, Any]]]) -> tuple:
        """Prepare inputs for the classifier.

        Args:
            queries: List of query texts
            candidates_list: List of candidate lists for each query

        Returns:
            Tuple of (patterns, examples, query_indices, candidate_indices)
        """
        patterns = []
        examples = []
        query_indices = []
        candidate_indices = []

        for query_idx, (query, candidates) in enumerate(
                zip(queries, candidates_list)):
            for candidate_idx, candidate in enumerate(candidates):
                # Extract pattern from candidate (ignore extra fields like
                # similarity, rank)
                pattern = candidate['pattern']

                patterns.append(self.pattern_prefix + pattern)
                examples.append(self.query_prefix + query)
                query_indices.append(query_idx)
                candidate_indices.append(candidate_idx)

        return patterns, examples, query_indices, candidate_indices

    def _tokenize_batch(
            self,
            patterns: List[str],
            examples: List[str],
            batch_size: int = 32) -> List[Dict]:
        """Tokenize inputs in batches without padding, like in training.

        Args:
            patterns: List of pattern texts with prefixes
            examples: List of example texts with prefixes
            batch_size: Batch size for tokenization

        Returns:
            List of tokenized inputs (no tensors, just token IDs and attention masks as lists)
        """
        tokenized_inputs = []

        for i in tqdm(range(0, len(patterns), batch_size), desc="Tokenizing"):
            batch_patterns = patterns[i:i + batch_size]
            batch_examples = examples[i:i + batch_size]

            batch_tokens = self.tokenizer(
                batch_patterns,
                batch_examples,
                truncation="only_second"
            )

            # Store each sample individually (as lists, not tensors)
            for j in range(len(batch_patterns)):
                tokenized_inputs.append({
                    # This is a list, not tensor
                    'input_ids': batch_tokens['input_ids'][j],
                    # This is a list, not tensor
                    'attention_mask': batch_tokens['attention_mask'][j]
                })

        return tokenized_inputs

    def _predict_batch(
            self,
            tokenized_inputs: List[Dict],
            batch_size: int = 32) -> List[float]:
        """Make predictions with dynamic padding per batch.

        Args:
            tokenized_inputs: List of tokenized inputs (with lists)
            batch_size: Batch size for prediction

        Returns:
            List of prediction probabilities for positive class
        """
        predictions = []

        with torch.no_grad():
            for i in tqdm(
                    range(0, len(tokenized_inputs), batch_size),
                    desc="Predicting"):
                batch_inputs = tokenized_inputs[i:i + batch_size]

                # Find max length in this batch
                max_length = max(len(inp['input_ids']) for inp in batch_inputs)

                # Pad each sequence to max_length
                padded_input_ids = []
                padded_attention_masks = []

                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1

                for inp in batch_inputs:
                    input_ids = inp['input_ids']
                    attention_mask = inp['attention_mask']

                    # Pad to max_length
                    padding_length = max_length - len(input_ids)

                    if padding_length > 0:
                        input_ids = input_ids + [pad_token_id] * padding_length
                        attention_mask = attention_mask + [0] * padding_length

                    padded_input_ids.append(input_ids)
                    padded_attention_masks.append(attention_mask)

                # Convert to tensors
                input_ids = torch.tensor(
                    padded_input_ids,
                    dtype=torch.long).to(
                    self.device)
                attention_mask = torch.tensor(
                    padded_attention_masks,
                    dtype=torch.long).to(
                    self.device)

                # Make predictions
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask)
                logits = outputs.logits

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)

                # Get probabilities for positive class (class 1)
                positive_probs = probs[:, 1].cpu().numpy()
                predictions.extend(positive_probs.tolist())

        return predictions

    def classify_candidates(self,
                            queries: List[str],
                            candidates_list: List[List[Dict[str, Any]]],
                            batch_size: int = 32) -> List[List[Dict[str, Any]]]:
        """Classify construction candidates for presence in queries.

        Args:
            queries: List of query texts
            candidates_list: List of candidate lists for each query.
                           Each candidate should have at least 'id' and 'pattern' keys.
                           Extra fields (like 'similarity', 'rank') are preserved but ignored.
            batch_size: Batch size for processing

        Returns:
            List of results, one per query. Each result is a list of candidates
            with added 'is_present' field (1 if present, 0 if not present).
            All original fields are preserved.
        """
        start_time = time.time()

        if not queries:
            return []

        if len(queries) != len(candidates_list):
            raise ValueError(
                "Length of queries and candidates_list must match")

        logger.info(f"Classifying candidates for {len(queries)} queries")

        # Count total candidates
        total_candidates = sum(len(candidates)
                               for candidates in candidates_list)
        logger.info(f"Total candidates to classify: {total_candidates}")

        if total_candidates == 0:
            return [[] for _ in queries]

        # Prepare inputs
        patterns, examples, query_indices, candidate_indices = self._prepare_inputs(
            queries, candidates_list)

        # Tokenize inputs
        logger.info("Tokenizing inputs...")
        tokenize_start = time.time()
        tokenized_inputs = self._tokenize_batch(patterns, examples, batch_size)
        tokenize_time = time.time() - tokenize_start
        logger.info(f"Tokenization completed in {tokenize_time:.2f} seconds")

        # Make predictions
        logger.info("Making predictions...")
        predict_start = time.time()
        predictions = self._predict_batch(tokenized_inputs, batch_size)
        predict_time = time.time() - predict_start
        logger.info(f"Predictions completed in {predict_time:.2f} seconds")

        # Convert probabilities to binary predictions
        binary_predictions = [
            1 if prob >= self.threshold else 0 for prob in predictions]

        # Organize results back to original structure
        results = [[] for _ in queries]

        for i, (query_idx, candidate_idx) in enumerate(
                zip(query_indices, candidate_indices)):
            # Get original candidate and copy all fields
            original_candidate = candidates_list[query_idx][candidate_idx].copy(
            )

            # Add prediction
            original_candidate['is_present'] = binary_predictions[i]

            results[query_idx].append(original_candidate)

        total_time = time.time() - start_time

        # Log statistics
        positive_predictions = sum(binary_predictions)
        logger.info(f"Classification completed in {total_time:.2f} seconds")
        logger.info(f"Positive predictions: {positive_predictions}/{total_candidates} "
                    f"({positive_predictions / total_candidates * 100:.1f}%)")

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'threshold': self.threshold,
            'query_prefix': self.query_prefix,
            'pattern_prefix': self.pattern_prefix,
            'model_loaded': self._model is not None,
            'tokenizer_loaded': self._tokenizer is not None
        }

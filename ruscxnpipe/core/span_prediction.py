"""Span prediction component for finding construction boundaries."""

import logging
from typing import List, Dict, Any, Optional
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
import time

from ..models.config import get_model_config
from ..utils.postprocessing import apply_punctuation_heuristics, validate_span


logger = logging.getLogger(__name__)


class SpanPredictor:
    """Span predictor for finding construction boundaries in text."""

    def __init__(self,
                 model_name: Optional[str] = None,
                 model_args: Optional[Dict[str, Any]] = None,
                 use_cuda: Optional[bool] = None):
        """Initialize span predictor.

        Args:
            model_name: Name/path of the QA model
            model_args: Additional arguments for the QA model
            use_cuda: Whether to use CUDA (None for auto-detection)
        """
        # Load configuration
        config = get_model_config('span_prediction')

        self.model_name = model_name or config['model_name']
        if self.model_name is None:
            raise ValueError("model_name must be provided for span prediction")

        # Set up model arguments
        default_args = {
            'reprocess_input_data': True,
            'overwrite_output_dir': True,
            'use_cached_eval_features': False,
            'output_dir': 'temp_qa_outputs',
            'cache_dir': 'temp_qa_cache',
            'silent': True
        }

        if model_args:
            default_args.update(model_args)

        self.model_args = QuestionAnsweringArgs()
        for key, value in default_args.items():
            setattr(self.model_args, key, value)

        # Initialize model
        self._model = None
        self.use_cuda = use_cuda

        logger.info(f"Initialized SpanPredictor with model: {self.model_name}")

    @property
    def model(self) -> QuestionAnsweringModel:
        """Get the QA model, loading it if necessary."""
        if self._model is None:
            logger.info(f"Loading QA model: {self.model_name}")
            start_time = time.time()

            self._model = QuestionAnsweringModel(
                'xlmroberta',
                self.model_name,
                args=self.model_args,
                use_cuda=self.use_cuda
            )

            load_time = time.time() - start_time
            logger.info(f"QA model loaded in {load_time:.2f} seconds")

        return self._model

    def _prepare_qa_input(
            self, examples_with_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare input data for QA model.

        Args:
            examples_with_patterns: List of examples with patterns

        Returns:
            List formatted for QA model prediction
        """
        qa_input = []

        for i, item in enumerate(examples_with_patterns):
            example_text = item['example']
            patterns = item['patterns']

            # Create QAS for each pattern
            qas = []
            for j, pattern_dict in enumerate(patterns):
                qas.append({
                    "question": pattern_dict['pattern'],
                    "id": f"{i}_{j}"  # Unique ID for each question
                })

            qa_input.append({
                "context": example_text,
                "qas": qas
            })

        return qa_input

    def _parse_predictions(self,
                           predictions: List[Dict[str,
                                                  Any]],
                           examples_with_patterns: List[Dict[str,
                                                             Any]]) -> List[Dict[str,
                                                                                 Any]]:
        """Parse QA model predictions back to original format.

        Args:
            predictions: Raw predictions from QA model
            examples_with_patterns: Original input data

        Returns:
            Results with span information added to each pattern
        """
        results = []

        # Create mapping from prediction ID to original data
        id_to_data = {}
        for i, item in enumerate(examples_with_patterns):
            for j, pattern_dict in enumerate(item['patterns']):
                pred_id = f"{i}_{j}"
                id_to_data[pred_id] = {
                    'example_idx': i,
                    'pattern_idx': j,
                    'example_text': item['example'],
                    'pattern_dict': pattern_dict.copy()
                }

        # Process predictions
        pred_dict = {pred['id']: pred for pred in predictions}

        # Group results by example
        example_results = {}
        for pred_id, data in id_to_data.items():
            example_idx = data['example_idx']
            if example_idx not in example_results:
                example_results[example_idx] = {
                    'example': data['example_text'],
                    'patterns': []
                }

            pattern_result = data['pattern_dict'].copy()

            # Extract span information from prediction
            if pred_id in pred_dict:
                pred = pred_dict[pred_id]
                answer_text = ""
                span_start = 0

                # Extract answer from prediction
                if 'answer' in pred and pred['answer']:
                    answer_text = pred['answer'][0] if isinstance(
                        pred['answer'], list) else pred['answer']

                    # Find position in text
                    if answer_text:
                        # Remove trailing period if present but not in original
                        # text
                        if answer_text.endswith(
                                '.') and not data['example_text'].endswith('.'):
                            answer_text = answer_text.rstrip('.')

                        span_start = data['example_text'].find(answer_text)

                        if span_start >= 0:
                            span_end = span_start + len(answer_text)

                            # Apply punctuation heuristics
                            cleaned_span, new_start, new_end = apply_punctuation_heuristics(
                                answer_text, pattern_result['pattern'], data['example_text'], span_start)

                            # Validate the span
                            if validate_span(
                                    cleaned_span, data['example_text'], new_start, new_end):
                                pattern_result['span'] = {
                                    'span_string': cleaned_span,
                                    'span_start': new_start,
                                    'span_end': new_end
                                }
                            else:
                                # Fallback to original prediction if heuristics
                                # fail
                                pattern_result['span'] = {
                                    'span_string': answer_text,
                                    'span_start': span_start,
                                    'span_end': span_end
                                }
                        else:
                            # Could not find answer in text
                            pattern_result['span'] = {
                                'span_string': "",
                                'span_start': 0,
                                'span_end': 0
                            }
                    else:
                        # Empty answer
                        pattern_result['span'] = {
                            'span_string': "",
                            'span_start': 0,
                            'span_end': 0
                        }
                else:
                    # No answer in prediction
                    pattern_result['span'] = {
                        'span_string': "",
                        'span_start': 0,
                        'span_end': 0
                    }
            else:
                # No prediction found for this ID
                pattern_result['span'] = {
                    'span_string': "",
                    'span_start': 0,
                    'span_end': 0
                }

            example_results[example_idx]['patterns'].append(pattern_result)

        # Convert to list format
        for i in sorted(example_results.keys()):
            results.append(example_results[i])

        return results

    def predict_spans(self,
                      examples_with_patterns: List[Dict[str, Any]],
                      batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Predict spans for construction patterns in examples.

        Args:
            examples_with_patterns: List of dictionaries with 'example' (str) and
                                  'patterns' (list of dicts with 'id' and 'pattern' keys)
            batch_size: Batch size for processing (uses model default if None)

        Returns:
            List of results with span information added to each pattern dictionary.
            Each pattern dict will have an added 'span' key containing:
            - 'span_string': the predicted span text
            - 'span_start': start index in the example
            - 'span_end': end index in the example
        """
        start_time = time.time()

        if not examples_with_patterns:
            return []

        logger.info(
            f"Predicting spans for {
                len(examples_with_patterns)} examples")

        # Count total patterns
        total_patterns = sum(len(item['patterns'])
                             for item in examples_with_patterns)
        logger.info(f"Total patterns to process: {total_patterns}")

        if total_patterns == 0:
            return [{'example': item['example'], 'patterns': []}
                    for item in examples_with_patterns]

        # Prepare input for QA model
        logger.info("Preparing QA input...")
        qa_input = self._prepare_qa_input(examples_with_patterns)

        # Make predictions
        logger.info("Making span predictions...")
        predict_start = time.time()
        predictions, _ = self.model.predict(qa_input)
        predict_time = time.time() - predict_start
        logger.info(
            f"Span predictions completed in {
                predict_time:.2f} seconds")

        # Parse predictions back to original format
        logger.info("Processing prediction results...")
        results = self._parse_predictions(predictions, examples_with_patterns)

        total_time = time.time() - start_time
        logger.info(f"Span prediction completed in {total_time:.2f} seconds")

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'model_loaded': self._model is not None,
            'use_cuda': self.use_cuda,
            'model_args': self.model_args.__dict__ if self.model_args else None
        }

"""Model configuration and default settings."""

from typing import Dict, Any


# Default model configurations
DEFAULT_MODELS = {
    'semantic_search': 'Futyn-Maker/ruscxn-embedder',
    'classification': 'Futyn-Maker/ruscxn-classifier',
    'span_prediction': "Futyn-Maker/ruscxn-span_predictor"
}

# Default prefixes for embedding models
DEFAULT_PREFIXES = {
    'query_prefix': 'Instruct: Given a sentence, find the constructions of the Russian Constructicon that it contains\nQuery: ',
    'pattern_prefix': ''}

# Default prefixes for classification model
DEFAULT_CLASSIFICATION_PREFIXES = {
    'query_prefix': 'query: ',
    'pattern_prefix': 'passage: '
}

# Default parameters
DEFAULT_PARAMS = {
    'semantic_search': {
        'batch_size': 32,
        'n_candidates': 15,
        'normalize_embeddings': True
    },
    'classification': {
        'batch_size': 32,
        'threshold': 0.5
    },
    'span_prediction': {
        'batch_size': 32
    }
}


def get_model_config(component: str) -> Dict[str, Any]:
    """Get configuration for a specific component.

    Args:
        component: Component name ('semantic_search', 'classification', 'span_prediction')

    Returns:
        Configuration dictionary for the component
    """
    config = {
        'model_name': DEFAULT_MODELS.get(component),
        'params': DEFAULT_PARAMS.get(component, {})
    }

    if component == 'semantic_search':
        config['prefixes'] = DEFAULT_PREFIXES
    elif component == 'classification':
        config['prefixes'] = DEFAULT_CLASSIFICATION_PREFIXES
    else:
        config['prefixes'] = {}

    return config

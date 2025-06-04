# RusCxnPipe

A Python library for automatic extraction of Russian Constructicon constructions from text, developed as part of my computational linguistics thesis research.

## Overview

RusCxnPipe implements a complete pipeline for identifying and locating Russian Constructicon constructions in arbitrary text. The library combines modern NLP techniques with linguistic knowledge to provide accurate construction detection and precise span identification.

### Features

- **Semantic Search**: Find construction candidates using fine-tuned sentence embeddings ([ruscxn-embedder](https://huggingface.co/Futyn-Maker/ruscxn-embedder))
- **Binary Classification**: Determine which constructions are actually present in text ([ruscxn-classifier](https://huggingface.co/Futyn-Maker/ruscxn-classifier))
- **Span Prediction**: Locate exact construction boundaries within text ([ruscxn-span-predictor](https://huggingface.co/Futyn-Maker/ruscxn-span-predictor))
- **Rule-based Filtering**: pre-filtering using anchor word matching
- **Embedding Caching**: Fast repeated processing with automatic caching
- **Flexible Usage**: Use individual components or the complete pipeline

### Algorithm

The pipeline processes text through four main stages:

1. **Rule-based filtering** removes obviously unsuitable constructions based on anchor word presence
2. **Semantic search** using a fine-tuned Sentence-BERT model ([ruscxn-embedder](https://huggingface.co/Futyn-Maker/ruscxn-embedder)) computes semantic vectors and retrieves the top *n* most similar construction patterns
3. **Binary classification** with a fine-tuned encoder model ([ruscxn-classifier](https://huggingface.co/Futyn-Maker/ruscxn-classifier)) determines which constructions are actually present in the text
4. **Span prediction** using a fine-tuned QA model ([ruscxn-span-predictor](https://huggingface.co/Futyn-Maker/ruscxn-span-predictor)) finds exact text fragments corresponding to each construction, enhanced with punctuation heuristics

The library provides both simplicity (process text with default settings) and flexibility (customize models, patterns, and parameters for research needs).

## Installation

### Requirements

- **Python**: 3.10 or higher
- **Hardware**: GPU recommended (even 4GB VRAM is sufficient), CPU operation possible but significantly slower
- **Dependencies**: PyTorch, Transformers, Sentence Transformers, SimpleTransformers

### Install from GitHub

```bash
pip install git+https://github.com/Futyn-Maker/ruscxnpipe.git
```

### Development Installation

```bash
git clone https://github.com/Futyn-Maker/ruscxnpipe.git
cd ruscxnpipe
pip install -e .
```

## Usage

### Basic Usage

**Single text processing:**

```python
from ruscxnpipe import RusCxnPipe

# Initialize pipeline
pipe = RusCxnPipe()

# Process text
result = pipe.process_text("Что вам здесь нужно?")

print(f"Found {len(result['constructions'])} constructions:")
for construction in result['constructions']:
    print(f"- {construction['pattern']}")
    print(f"  Span: '{construction['span']['span_string']}'")
    print(f"  Position: {construction['span']['span_start']}-{construction['span']['span_end']}")
```

**Multiple texts processing:**

```python
texts = [
    "Что вам здесь нужно?",
    "Мои друзья разъехались и исчезли кто где.",
    "Таня танцевала без устали, танцевала со всеми подряд."
]

results = pipe.process_texts(texts)

for i, result in enumerate(results):
    print(f"Text {i+1}: {result['example']}")
    print(f"Constructions: {len(result['constructions'])}")
```

The `process_texts()` method returns a list of dictionaries, each containing:
- `example`: the original text
- `constructions`: list of found constructions with `id`, `pattern`, and `span` information

The `process_text()` method is implemented for convenience and returns a single dictionary with the keys `example` and `constructions`.

### Advanced Usage

**Custom parameters and patterns:**

```python
from ruscxnpipe import RusCxnPipe

# Define custom patterns
custom_patterns = [
    {"id": "pattern_1", "pattern": "что NP-Dat Cop нужно/надо?"},
    {"id": "pattern_2", "pattern": "VP кто где"},
    {"id": "pattern_3", "pattern": "VP без устали"}
]

# Initialize with custom settings
pipe = RusCxnPipe(
    classification_threshold=0.7,  # Higher precision
    custom_patterns=custom_patterns,
    use_filtering=True
)

# Process with custom parameters
results = pipe.process_texts(
    examples=["Что вам здесь нужно?"],
    n_candidates=10,  # Fewer candidates for speed
    semantic_batch_size=16
)
```

### Individual Components

**Semantic search only:**

```python
from ruscxnpipe import SemanticSearch

search = SemanticSearch()
candidates = search.find_candidates(
    queries=["Мои друзья разъехались и исчезли кто где."],
    n=5
)

for candidate in candidates[0]['candidates']:
    print(f"{candidate['pattern']} (similarity: {candidate['similarity']:.3f})")
```

The `find_candidates()` method returns a list with `query` and `candidates` fields, where each candidate includes `id`, `pattern`, `similarity`, and `rank`.

**Classification only:**

```python
from ruscxnpipe import ConstructionClassifier

classifier = ConstructionClassifier()
results = classifier.classify_candidates(
    queries=["Таня танцевала без устали."],
    candidates_list=[[
        {"id": "1", "pattern": "VP без устали"},
        {"id": "2", "pattern": "VP кто где"}
    ]]
)

for candidate in results[0]:
    status = "PRESENT" if candidate['is_present'] == 1 else "NOT PRESENT"
    print(f"{candidate['pattern']}: {status}")
```

The `classify_candidates()` method adds an `is_present` field (1 or 0) to each candidate while preserving all original fields.

**Span prediction only:**

```python
from ruscxnpipe import SpanPredictor

predictor = SpanPredictor()
results = predictor.predict_spans([{
    "example": "Что вам здесь нужно?",
    "patterns": [{"id": "1", "pattern": "что NP-Dat Cop нужно/надо?"}]
}])

span = results[0]['patterns'][0]['span']
print(f"Span: '{span['span_string']}' at positions {span['span_start']}-{span['span_end']}")
```

The `predict_spans()` method adds a `span` field to each pattern containing `span_string`, `span_start`, and `span_end`.

## Demo

Try the interactive web demo: **[RusCxnPipe Demo](https://huggingface.co/spaces/Futyn-Maker/RusCxnPipe)**

The demo provides two interfaces:
- **Full Pipeline**: Complete analysis of Russian text
- **Span Prediction**: Direct span detection for specified patterns

## Models

The library uses three specialized models trained on Russian Constructicon data:

- **[ruscxn-embedder](https://huggingface.co/Futyn-Maker/ruscxn-embedder)**: Sentence-BERT model for semantic similarity
- **[ruscxn-classifier](https://huggingface.co/Futyn-Maker/ruscxn-classifier)**: Binary classifier for construction presence
- **[ruscxn-span-predictor](https://huggingface.co/Futyn-Maker/ruscxn-span-predictor)**: QA model for span boundary detection

## Performance Notes

- For optimal accuracy, process individual sentences rather than long texts
- GPU usage significantly improves processing speed
- Embedding caching makes repeated processing very fast
- Rule-based filtering reduces computational overhead and increases accuracy

## License

MIT License. See [LICENSE](LICENSE) file for details.

## Links

- **GitHub Repository**: [https://github.com/Futyn-Maker/ruscxnpipe](https://github.com/Futyn-Maker/ruscxnpipe)
- **Interactive Demo**: [https://huggingface.co/spaces/Futyn-Maker/RusCxnPipe](https://huggingface.co/spaces/Futyn-Maker/RusCxnPipe)
- **Russian Constructicon**: [https://constructicon.ruscorpora.ru/](https://constructicon.ruscorpora.ru/)

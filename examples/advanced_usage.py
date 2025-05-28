"""Advanced usage examples for RusCxnPipe."""

from ruscxnpipe import RusCxnPipe


def main():
    # Custom patterns
    custom_patterns = [
        {"id": "custom_1", "pattern": "что NP-Dat Cop нужно?"},
        {"id": "custom_2", "pattern": "при чём здесь NP-Nom?"}
    ]

    # Initialize pipeline with custom settings
    pipe = RusCxnPipe(
        # Semantic search settings
        semantic_model="Futyn-Maker/ruscxn-embedder",
        use_filtering=True,

        # Classification settings
        classification_model="Futyn-Maker/ruscxn-classifier",
        classification_threshold=0.7,  # Higher threshold

        # Span prediction settings
        span_model="Futyn-Maker/ruscxn-span_predictor",

        # Use custom patterns
        custom_patterns=custom_patterns
    )

    # Process with custom parameters
    texts = ["Что вам здесь нужно?", "При чём здесь эта история?"]

    results = pipe.process_texts(
        examples=texts,
        n_candidates=10,  # Fewer candidates
        semantic_batch_size=16,
        classification_batch_size=16
    )

    for result in results:
        print(f"Text: {result['example']}")
        for construction in result['constructions']:
            print(f"  Construction: {construction['id']}")
            print(f"  Pattern: {construction['pattern']}")
            print(f"  Span: '{construction['span']['span_string']}'")

    # Get pipeline information
    info = pipe.get_pipeline_info()
    print(f"\nPipeline info: {info}")


if __name__ == "__main__":
    main()

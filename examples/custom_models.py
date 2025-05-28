"""Example of using RusCxnPipe with custom models."""

from ruscxnpipe import SemanticSearch, ConstructionClassifier, SpanPredictor, RusCxnPipe


def main():
    # Example 1: Using individual components
    print("=== Using individual components ===")

    # Semantic search only
    search = SemanticSearch()
    candidates = search.find_candidates(
        queries=["Что вам здесь нужно?"],
        n=5
    )
    print(f"Found {len(candidates[0]['candidates'])} candidates")

    # Classification only
    classifier = ConstructionClassifier()
    classified = classifier.classify_candidates(
        queries=["Что вам здесь нужно?"],
        candidates_list=[candidates[0]['candidates']]
    )
    positive_candidates = [c for c in classified[0] if c['is_present'] == 1]
    print(f"Positive candidates: {len(positive_candidates)}")

    # Example 2: Custom model configurations
    print("\n=== Using custom model configurations ===")

    pipe = RusCxnPipe(
        # Custom semantic model
        semantic_model="your/custom/semantic/model",
        query_prefix="Custom query prefix: ",

        # Custom classification model
        classification_model="your/custom/classification/model",
        classification_threshold=0.8,

        # Custom span model
        span_model="your/custom/span/model",
        span_model_args={
            'max_seq_length': 512,
            'doc_stride': 128
        }
    )

    # Process text
    result = pipe.process_text("Что вам здесь нужно?")
    print(f"Processed: {result['example']}")
    print(f"Found: {len(result['constructions'])} constructions")


if __name__ == "__main__":
    main()

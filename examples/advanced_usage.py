"""Advanced usage with parameter settings."""

from ruscxnpipe import RusCxnPipe

def main():
    # Custom patterns
    custom_patterns = [
        {"id": "pattern_1", "pattern": "что NP-Dat Cop нужно/надо?"},
        {"id": "pattern_2", "pattern": "VP кто где"},
        {"id": "pattern_3", "pattern": "VP без устали"},
        {"id": "pattern_4", "pattern": "VP всё/все подряд"}
    ]
    
    # Initialize pipeline with custom settings
    pipe = RusCxnPipe(
        # Semantic search settings
        semantic_model="Futyn-Maker/ruscxn-embedder",
        use_filtering=True,
        
        # Classification settings
        classification_model="Futyn-Maker/ruscxn-classifier",
        classification_threshold=0.7,  # Higher threshold for more precision
        
        # Span prediction settings
        span_model="Futyn-Maker/ruscxn-span-predictor",
        
        # Use custom patterns instead of full constructicon
        custom_patterns=custom_patterns
    )
    
    # Test texts that should match our custom patterns
    texts = [
        "Что вам здесь нужно?",
        "Мои друзья разъехались и исчезли кто где.",
        "Таня танцевала без устали, танцевала со всеми подряд."
    ]
    
    # Process with custom parameters
    results = pipe.process_texts(
        examples=texts,
        n_candidates=10,  # Fewer candidates for faster processing
        semantic_batch_size=16,
        classification_batch_size=16
    )
    
    print("=== Advanced Processing Results ===")
    for i, result in enumerate(results, 1):
        print(f"\nText {i}: {result['example']}")
        print(f"Constructions found: {len(result['constructions'])}")
        
        for construction in result['constructions']:
            print(f"  - Construction ID: {construction['id']}")
            print(f"    Pattern: {construction['pattern']}")
            print(f"    Span: '{construction['span']['span_string']}'")
    
    # Get pipeline information
    info = pipe.get_pipeline_info()
    print(f"\n=== Pipeline Information ===")
    print(f"Custom patterns used: {info['custom_patterns']}")
    print(f"Cache info: {info['semantic_search']}")

if __name__ == "__main__":
    main()

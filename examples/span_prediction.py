"""Using SpanPredictor component separately."""

from ruscxnpipe import SpanPredictor

def main():
    # Initialize span predictor
    predictor = SpanPredictor(
        model_name="Futyn-Maker/ruscxn-span-predictor"
    )
    
    # Prepare examples with their known patterns
    examples_with_patterns = [
        {
            "example": "Что вам здесь нужно?",
            "patterns": [
                {"id": "pattern_1", "pattern": "что NP-Dat Cop нужно/надо?"}
            ]
        },
        {
            "example": "Мои друзья разъехались и исчезли кто где.",
            "patterns": [
                {"id": "pattern_2", "pattern": "VP кто где"},
                {"id": "pattern_3", "pattern": "VP кто PronInt"}
            ]
        },
        {
            "example": "Таня танцевала без устали, танцевала со всеми подряд.",
            "patterns": [
                {"id": "pattern_4", "pattern": "VP без устали"},
                {"id": "pattern_5", "pattern": "VP всё/все подряд"}
            ]
        }
    ]
    
    # Predict spans
    results = predictor.predict_spans(examples_with_patterns)
    
    print("=== Span Prediction Results ===")
    for i, result in enumerate(results, 1):
        print(f"\nExample {i}: {result['example']}")
        print("Predicted spans:")
        
        for pattern_result in result['patterns']:
            span_info = pattern_result['span']
            print(f"  - Pattern: {pattern_result['pattern']}")
            
            if span_info['span_string']:
                print(f"    Span: '{span_info['span_string']}'")
                print(f"    Position: {span_info['span_start']}-{span_info['span_end']}")
                
                # Show text with highlighted span
                text = result['example']
                before = text[:span_info['span_start']]
                span = text[span_info['span_start']:span_info['span_end']]
                after = text[span_info['span_end']:]
                print(f"    Highlighted: {before}[{span}]{after}")
            else:
                print(f"    Span: No span found")
    
    # Get model information
    model_info = predictor.get_model_info()
    print(f"\n=== Model Information ===")
    print(f"Model: {model_info['model_name']}")
    print(f"Model loaded: {model_info['model_loaded']}")

if __name__ == "__main__":
    main()

"""Basic usage: predicting multiple texts."""

from ruscxnpipe import RusCxnPipe

def main():
    # Initialize pipeline with default settings
    pipe = RusCxnPipe()
    
    # Process multiple texts
    texts = [
        "Что вам здесь нужно?",
        "Мои друзья разъехались и исчезли кто где.",
        "Таня танцевала без устали, танцевала со всеми подряд."
    ]
    
    results = pipe.process_texts(texts)
    
    for i, result in enumerate(results, 1):
        print(f"\n=== Text {i} ===")
        print(f"Text: {result['example']}")
        print(f"Found {len(result['constructions'])} constructions:")
        
        for j, construction in enumerate(result['constructions'], 1):
            print(f"  {j}. Pattern: {construction['pattern']}")
            print(f"     Span: '{construction['span']['span_string']}'")
            print(f"     Position: {construction['span']['span_start']}-{construction['span']['span_end']}")

if __name__ == "__main__":
    main()

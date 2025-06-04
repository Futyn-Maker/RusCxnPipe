"""Basic usage: single text prediction."""

from ruscxnpipe import RusCxnPipe

def main():
    # Initialize pipeline with default settings
    pipe = RusCxnPipe()
    
    # Process single text
    text = "Что вам здесь нужно?"
    result = pipe.process_text(text)
    
    print(f"Text: {result['example']}")
    print(f"Found {len(result['constructions'])} constructions:")
    
    for i, construction in enumerate(result['constructions'], 1):
        print(f"\n{i}. Construction ID: {construction['id']}")
        print(f"   Pattern: {construction['pattern']}")
        print(f"   Span: '{construction['span']['span_string']}'")
        print(f"   Position: {construction['span']['span_start']}-{construction['span']['span_end']}")

if __name__ == "__main__":
    main()

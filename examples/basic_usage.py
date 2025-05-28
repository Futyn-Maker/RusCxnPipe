"""Basic usage example for RusCxnPipe."""

from ruscxnpipe import RusCxnPipe


def main():
    # Initialize pipeline with default settings
    pipe = RusCxnPipe()

    # Process single text
    text = "Что вам здесь нужно?"
    result = pipe.process_text(text)

    print(f"Text: {result['example']}")
    print(f"Found {len(result['constructions'])} constructions:")
    for construction in result['constructions']:
        print(f"  - Pattern: {construction['pattern']}")
        print(f"    Span: '{construction['span']['span_string']}'")
        print(
            f"    Position: {
                construction['span']['span_start']}-{
                construction['span']['span_end']}")

    # Process multiple texts
    texts = [
        "Что вам здесь нужно?",
        "Как дела?",
        "При чём здесь эта история?"
    ]

    results = pipe.process_texts(texts)

    for i, result in enumerate(results):
        print(f"\nText {i + 1}: {result['example']}")
        print(f"Constructions found: {len(result['constructions'])}")


if __name__ == "__main__":
    main()

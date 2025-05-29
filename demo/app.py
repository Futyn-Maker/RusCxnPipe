"""Gradio demo application for RusCxnPipe."""

import gradio as gr
import logging
from typing import List, Dict, Any


# Set up logging to avoid cluttering the interface
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

try:
    from ruscxnpipe import RusCxnPipe, SpanPredictor
except ImportError:
    # For development/testing when library isn't installed
    import sys
    import os
    sys.path.append(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))
    from ruscxnpipe import RusCxnPipe, SpanPredictor


# Initialize models at startup
print("🚀 Initializing RusCxnPipe models...")
try:
    PIPELINE = RusCxnPipe(
        semantic_model="Futyn-Maker/ruscxn-embedder",
        classification_model="Futyn-Maker/ruscxn-classifier",
        span_model="Futyn-Maker/ruscxn-span-predictor"
    )
    SPAN_PREDICTOR = SpanPredictor(
        model_name="Futyn-Maker/ruscxn-span-predictor")
    print("✅ Models initialized successfully!")
    MODELS_LOADED = True
    MODEL_ERROR = None
except Exception as e:
    print(f"❌ Error initializing models: {str(e)}")
    PIPELINE = None
    SPAN_PREDICTOR = None
    MODELS_LOADED = False
    MODEL_ERROR = str(e)


def highlight_span(
        text: str,
        span_start: int,
        span_end: int,
        span_string: str) -> str:
    """Highlight a span in text using HTML."""
    if span_start < 0 or span_end > len(text) or span_start >= span_end:
        return text

    # Ensure the span matches
    actual_span = text[span_start:span_end]
    if actual_span.strip() != span_string.strip():
        # Fallback: try to find the span in the text
        span_start = text.find(span_string)
        if span_start >= 0:
            span_end = span_start + len(span_string)
        else:
            return text

    # Create highlighted version
    before = text[:span_start]
    highlighted = text[span_start:span_end]
    after = text[span_end:]

    return f'{before}<mark style="background-color: #64b5f6; color: #1565c0; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{highlighted}</mark>{after}'


def create_construction_link(construction_id: str, pattern: str) -> str:
    """Create a clickable link to the construction page."""
    url = f"https://constructicon.ruscorpora.ru/construction/{construction_id}"
    return f'<a href="{url}" target="_blank" style="color: #1976d2; text-decoration: none; font-weight: bold; border-bottom: 1px dotted #1976d2;">{pattern}</a>'


def format_pipeline_results(results: Dict[str, Any]) -> str:
    """Format the pipeline results as HTML."""
    if not results or not results['constructions']:
        return "<div style='padding: 20px; text-align: center; color: #666;'>No constructions found in the text.</div>"

    constructions = results['constructions']
    original_text = results['example']

    html_parts = []
    html_parts.append("<div style='font-family: Arial, sans-serif;'>")

    # Header
    html_parts.append(
        "<h3 style='color: #333; margin-bottom: 20px;'>Found {} construction(s):</h3>".format(
            len(constructions)))

    # Process each construction
    for i, construction in enumerate(constructions, 1):
        construction_id = construction['id']
        pattern = construction['pattern']
        span_info = construction['span']

        # Construction header with link
        html_parts.append(
            "<div style='margin-bottom: 25px; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #fafafa;'>")
        html_parts.append(
            f"<h4 style='margin: 0 0 10px 0; color: #333;'>{i}. {create_construction_link(construction_id, pattern)}</h4>")

        # Highlighted text
        if span_info['span_string']:
            highlighted_text = highlight_span(
                original_text,
                span_info['span_start'],
                span_info['span_end'],
                span_info['span_string']
            )
            html_parts.append(
                f"<div style='font-size: 16px; line-height: 1.5; margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 4px; border: 1px solid #ddd; color: #333;'>{highlighted_text}</div>")

            # Span details
            html_parts.append(
                "<div style='margin-top: 8px; font-size: 12px; color: #666;'>")
            html_parts.append(
                f"Span: \"{span_info['span_string']}\" (positions {span_info['span_start']}-{span_info['span_end']})")
            html_parts.append("</div>")
        else:
            html_parts.append(
                f"<div style='font-size: 16px; line-height: 1.5; margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 4px; border: 1px solid #ddd; color: #333;'>{original_text}</div>")
            html_parts.append(
                "<div style='margin-top: 8px; font-size: 12px; color: #999;'>No specific span identified</div>")

        html_parts.append("</div>")

    html_parts.append("</div>")
    return "".join(html_parts)


def format_span_results(text: str, results: List[Dict[str, Any]]) -> str:
    """Format span prediction results as HTML."""
    if not results or not results[0]['patterns']:
        return "<div style='padding: 20px; text-align: center; color: #666;'>No patterns processed.</div>"

    patterns = results[0]['patterns']

    html_parts = []
    html_parts.append("<div style='font-family: Arial, sans-serif;'>")

    # Header
    html_parts.append(
        f"<h3 style='color: #333; margin-bottom: 20px;'>Span predictions for {len(patterns)} pattern(s):</h3>")

    # Process each pattern
    for i, pattern_info in enumerate(patterns, 1):
        pattern = pattern_info['pattern']
        span_info = pattern_info['span']

        html_parts.append(
            "<div style='margin-bottom: 25px; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #fafafa;'>")
        html_parts.append(
            f"<h4 style='margin: 0 0 10px 0; color: #333;'>{i}. {pattern}</h4>")

        # Highlighted text
        if span_info['span_string']:
            highlighted_text = highlight_span(
                text,
                span_info['span_start'],
                span_info['span_end'],
                span_info['span_string']
            )
            html_parts.append(
                f"<div style='font-size: 16px; line-height: 1.5; margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 4px; border: 1px solid #ddd; color: #333;'>{highlighted_text}</div>")

            # Span details
            html_parts.append(
                "<div style='margin-top: 8px; font-size: 12px; color: #666;'>")
            html_parts.append(
                f"Span: \"{span_info['span_string']}\" (positions {span_info['span_start']}-{span_info['span_end']})")
            html_parts.append("</div>")
        else:
            html_parts.append(
                f"<div style='font-size: 16px; line-height: 1.5; margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 4px; border: 1px solid #ddd; color: #333;'>{text}</div>")
            html_parts.append(
                "<div style='margin-top: 8px; font-size: 12px; color: #999;'>No span found for this pattern</div>")

        html_parts.append("</div>")

    html_parts.append("</div>")
    return "".join(html_parts)


def process_full_pipeline(text: str, n_candidates: int) -> str:
    """Process text through the full pipeline."""
    if not text.strip():
        return "<div style='padding: 20px; text-align: center; color: #666;'>Please enter some text to analyze.</div>"

    if not MODELS_LOADED:
        return f"<div style='color: red; padding: 20px;'>Error: {MODEL_ERROR}</div>"

    try:
        results = PIPELINE.process_text(
            text.strip(), n_candidates=n_candidates)
        return format_pipeline_results(results)
    except Exception as e:
        return f"<div style='color: red; padding: 20px;'>Error processing text: {str(e)}</div>"


def process_span_prediction(text: str, patterns_text: str) -> str:
    """Process text for span prediction only."""
    if not text.strip():
        return "<div style='padding: 20px; text-align: center; color: #666;'>Please enter some text to analyze.</div>"

    if not patterns_text.strip():
        return "<div style='padding: 20px; text-align: center; color: #666;'>Please enter some patterns to search for.</div>"

    if not MODELS_LOADED:
        return f"<div style='color: red; padding: 20px;'>Error: {MODEL_ERROR}</div>"

    # Parse patterns
    patterns = [p.strip()
                for p in patterns_text.strip().split('\n') if p.strip()]
    if not patterns:
        return "<div style='padding: 20px; text-align: center; color: #666;'>No valid patterns found.</div>"

    # Prepare input for span predictor
    examples_with_patterns = [{'example': text.strip(),
                               'patterns': [{'id': f'pattern_{i}',
                                             'pattern': pattern} for i,
                                            pattern in enumerate(patterns)]}]

    try:
        results = SPAN_PREDICTOR.predict_spans(examples_with_patterns)
        return format_span_results(text.strip(), results)
    except Exception as e:
        return f"<div style='color: red; padding: 20px;'>Error processing spans: {str(e)}</div>"

# Create the Gradio interface


def create_demo():
    """Create the Gradio demo interface."""

    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gr-button {
        background: linear-gradient(90deg, #1976d2, #42a5f5);
        border: none;
        color: white;
        font-weight: bold;
    }
    .gr-button:hover {
        background: linear-gradient(90deg, #1565c0, #2196f3);
    }
    """

    with gr.Blocks(css=css, title="RusCxnPipe Demo", theme=gr.themes.Soft()) as demo:

        # Header
        gr.Markdown("""
        # 🔍 RusCxnPipe: Russian Constructicon Pattern Extractor

        **Automatically identify and locate Russian constructicon patterns in text**

        This tool uses advanced NLP models to find linguistic constructions from the Russian Constructicon database in your text.
        It performs semantic search, classification, and span prediction to provide accurate results with precise text locations.

        """)

        with gr.Tabs():
            # Tab 1: Full Pipeline
            with gr.Tab("🚀 Full Pipeline", id="pipeline"):
                gr.Markdown("""
                ### Complete Analysis
                Enter Russian text to automatically find all constructicon patterns present in it.
                The system will search through the database, classify candidates, and highlight exact locations.
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Text",
                            placeholder="Мои друзья разъехались и исчезли кто где.",
                            lines=3,
                            value="Мои друзья разъехались и исчезли кто где.")

                        n_candidates = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=15,
                            step=5,
                            label="Number of semantic search candidates",
                            info="More candidates = more thorough search but slower processing and higher probability of false-positives"
                        )

                        analyze_btn = gr.Button(
                            "🔍 Analyze Text", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        results_html = gr.HTML(
                            label="Results",
                            value="<div style='padding: 40px; text-align: center; color: #666; border: 2px dashed #ccc; border-radius: 8px;'>Enter text and click 'Analyze Text' to see results</div>"
                        )

                # Examples
                gr.Markdown("### 📝 Try these examples:")
                example_texts = [
                    "Мои друзья разъехались и исчезли кто где.",
                    "Петр так и замер на месте.",
                    "Таня танцевала без устали, танцевала со всеми подряд."
                ]

                with gr.Row():
                    for example in example_texts:
                        gr.Button(f'"{example}"', size="sm").click(
                            lambda x=example: x, outputs=text_input
                        )

                analyze_btn.click(
                    fn=process_full_pipeline,
                    inputs=[text_input, n_candidates],
                    outputs=results_html
                )

            # Tab 2: Span Prediction Only
            with gr.Tab("🎯 Span Prediction", id="spans"):
                gr.Markdown("""
                ### Pattern Span Detection
                Enter text and specific patterns to find where exactly these patterns occur in the text.
                This skips the search and classification steps, directly predicting span boundaries.
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        span_text_input = gr.Textbox(
                            label="Text",
                            placeholder="Мои друзья разъехались и исчезли кто где.",
                            lines=3,
                            value="Мои друзья разъехались и исчезли кто где.")

                        patterns_input = gr.Textbox(
                            label="Patterns (one per line)",
                            placeholder="VP кто PronInt\nVP кто где",
                            lines=5,
                            value="VP кто PronInt\nVP кто где"
                        )

                        predict_btn = gr.Button(
                            "🎯 Predict Spans", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        span_results_html = gr.HTML(
                            label="Span Results",
                            value="<div style='padding: 40px; text-align: center; color: #666; border: 2px dashed #ccc; border-radius: 8px;'>Enter text and patterns, then click 'Predict Spans' to see results</div>"
                        )

                predict_btn.click(
                    fn=process_span_prediction,
                    inputs=[span_text_input, patterns_input],
                    outputs=span_results_html
                )

        # Footer
        gr.Markdown("""
        ---
        **About RusCxnPipe**: This tool is based on fine-tuned transformer models trained on Russian Constructicon data.
        The pipeline combines semantic search, classification, and span prediction to achieve high accuracy in construction detection.

        **Models used**:
        - Semantic: [ruscxn-embedder](https://huggingface.co/Futyn-Maker/ruscxn-embedder)
        - Classification: [ruscxn-classifier](https://huggingface.co/Futyn-Maker/ruscxn-classifier)
        - Span prediction: [ruscxn-span-predictor](https://huggingface.co/Futyn-Maker/ruscxn-span-predictor)

        📚 [Russian Constructicon Database](https://constructicon.ruscorpora.ru/) | 💻 [Source Code](https://github.com/Futyn-Maker/ruscxnpipe)
        """)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",  # For Hugging Face Spaces
        server_port=7860,       # Default port for Spaces
        show_error=True
    )

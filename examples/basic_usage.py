"""
Basic usage examples for the transformer visualization library.

This script demonstrates how to create architecture diagrams
styled after the Transformer Circuits publications.
"""

import sys
sys.path.insert(0, "..")

from transformer_viz import (
    TransformerVisualizer,
    VisualizationConfig,
    TransformerLensAdapter,
)
from transformer_viz.visualizer import visualize_transformer


def example_gpt2_small():
    """Visualize GPT-2 Small architecture."""
    print("Creating GPT-2 Small visualization...")

    viz = TransformerVisualizer()
    viz.from_pretrained("gpt2-small")

    # Show only first 4 layers for clarity
    fig, ax = viz.draw(max_layers=4, detailed=False)
    viz.save("gpt2_small_architecture.png", dpi=150)
    print("Saved: gpt2_small_architecture.png")

    return fig, ax


def example_custom_model():
    """Visualize a custom model configuration."""
    print("Creating custom model visualization...")

    # Define a custom architecture
    custom_config = {
        "n_layers": 6,
        "d_model": 512,
        "n_heads": 8,
        "d_head": 64,
        "d_mlp": 2048,
        "d_vocab": 32000,
        "n_ctx": 2048,
        "model_name": "My Custom Transformer",
    }

    viz = TransformerVisualizer()
    viz.from_dict(custom_config)

    fig, ax = viz.draw(detailed=True)
    viz.save("custom_model_architecture.png", dpi=150)
    print("Saved: custom_model_architecture.png")

    return fig, ax


def example_minimal_style():
    """Create a minimal/clean visualization."""
    print("Creating minimal style visualization...")

    # Use minimal config for cleaner look
    config = VisualizationConfig.minimal()

    viz = TransformerVisualizer(config)
    viz.from_pretrained("pythia-70m")

    fig, ax = viz.draw()
    viz.save("pythia_minimal.png", dpi=150)
    print("Saved: pythia_minimal.png")

    return fig, ax


def example_detailed_style():
    """Create a detailed visualization with all components."""
    print("Creating detailed visualization...")

    config = VisualizationConfig.detailed()

    viz = TransformerVisualizer(config)
    viz.from_pretrained("gpt2")

    fig, ax = viz.draw(max_layers=3, detailed=True)
    viz.save("gpt2_detailed.png", dpi=150)
    print("Saved: gpt2_detailed.png")

    return fig, ax


def example_attention_heads():
    """Visualize attention head layout."""
    print("Creating attention head visualization...")

    viz = TransformerVisualizer()
    viz.from_pretrained("gpt2-medium")

    fig, ax = viz.draw_attention_pattern()
    viz.save("attention_heads.png", dpi=150)
    print("Saved: attention_heads.png")

    return fig, ax


def example_single_layer():
    """Visualize a single layer in detail."""
    print("Creating single layer visualization...")

    viz = TransformerVisualizer()
    viz.from_pretrained("gpt2")

    fig, ax = viz.draw_single_layer(layer_idx=0, detailed=True)
    viz.save("single_layer_detail.png", dpi=150)
    print("Saved: single_layer_detail.png")

    return fig, ax


def example_with_transformerlens():
    """Example using an actual TransformerLens model."""
    print("Creating visualization from TransformerLens model...")

    try:
        from transformer_lens import HookedTransformer

        # Load a small model
        model = HookedTransformer.from_pretrained("gpt2-small")

        viz = TransformerVisualizer()
        viz.from_model(model)

        fig, ax = viz.draw(max_layers=4)
        viz.save("transformerlens_model.png", dpi=150)
        print("Saved: transformerlens_model.png")

        return fig, ax

    except ImportError:
        print("TransformerLens not installed. Skipping this example.")
        print("Install with: pip install transformer-lens")
        return None, None


def example_dark_theme():
    """Create a visualization with dark theme."""
    print("Creating dark theme visualization...")

    config = VisualizationConfig.dark_theme()

    viz = TransformerVisualizer(config)
    viz.from_pretrained("gpt2")

    fig, ax = viz.draw(max_layers=3)
    viz.save("gpt2_dark_theme.png", dpi=150)
    print("Saved: gpt2_dark_theme.png")

    return fig, ax


def example_svg_output():
    """Save as SVG for crisp vector graphics."""
    print("Creating SVG visualization...")

    viz = TransformerVisualizer()
    viz.from_pretrained("gpt2-small")

    fig, ax = viz.draw(max_layers=3)
    viz.save("gpt2_architecture.svg")
    print("Saved: gpt2_architecture.svg")

    return fig, ax


def example_quick_function():
    """Use the convenience function for quick visualization."""
    print("Using quick visualization function...")

    # One-liner to visualize and save
    fig, ax = visualize_transformer(
        "gpt2",
        max_layers=4,
        save_path="quick_viz.png"
    )
    print("Saved: quick_viz.png")

    return fig, ax


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend

    print("=" * 50)
    print("Transformer Architecture Visualization Examples")
    print("=" * 50)
    print()

    # Run all examples
    example_gpt2_small()
    print()

    example_custom_model()
    print()

    example_minimal_style()
    print()

    example_detailed_style()
    print()

    example_attention_heads()
    print()

    example_single_layer()
    print()

    example_dark_theme()
    print()

    example_svg_output()
    print()

    example_quick_function()
    print()

    # Try TransformerLens example if available
    example_with_transformerlens()
    print()

    print("=" * 50)
    print("All examples completed!")
    print("=" * 50)

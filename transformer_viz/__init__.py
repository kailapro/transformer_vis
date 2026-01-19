"""
Transformer Architecture Visualization Library

A visualization library for transformer models, compatible with TransformerLens,
styled after the diagrams in Anthropic's Transformer Circuits publications.

Interactive usage (recommended):
    >>> from transformer_viz import visualize
    >>> visualize("gpt2", max_layers=4)

Static image usage:
    >>> from transformer_viz import TransformerVisualizer
    >>> viz = TransformerVisualizer()
    >>> viz.from_pretrained("gpt2").draw()
"""

# Interactive visualization (primary API)
from .interactive import InteractiveTransformerViz, visualize

# Static visualization (matplotlib-based)
from .visualizer import TransformerVisualizer
from .components import (
    ResidualStream,
    AttentionBlock,
    MLPBlock,
    EmbeddingLayer,
    UnembeddingLayer,
)
from .config import VisualizationConfig
from .model_adapter import TransformerLensAdapter, ModelArchitecture

__version__ = "0.1.0"
__all__ = [
    # Primary interactive API
    "visualize",
    "InteractiveTransformerViz",
    # Static visualization
    "TransformerVisualizer",
    # Components
    "ResidualStream",
    "AttentionBlock",
    "MLPBlock",
    "EmbeddingLayer",
    "UnembeddingLayer",
    # Config and adapters
    "VisualizationConfig",
    "TransformerLensAdapter",
    "ModelArchitecture",
]

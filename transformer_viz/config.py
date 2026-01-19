"""
Configuration for transformer visualizations.

Defines colors, styles, and layout parameters inspired by Transformer Circuits diagrams.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class VisualizationConfig:
    """Configuration for transformer architecture visualization."""

    # Canvas settings
    figure_width: float = 12.0
    layer_height: float = 2.0
    component_spacing: float = 0.3

    # Residual stream
    residual_stream_width: float = 0.15
    residual_stream_color: str = "#E8E8E8"
    residual_stream_edge_color: str = "#CCCCCC"

    # Attention block colors (Transformer Circuits style - blues/teals)
    attention_block_color: str = "#4ECDC4"
    attention_block_edge_color: str = "#2C9E97"
    attention_head_colors: list = field(default_factory=lambda: [
        "#45B7AA", "#3DA89C", "#359A8E", "#2D8C80",
        "#257E72", "#1D7064", "#156256", "#0D5448"
    ])

    # MLP block colors (warm oranges/reds)
    mlp_block_color: str = "#FF6B6B"
    mlp_block_edge_color: str = "#E54545"
    mlp_inner_color: str = "#FFA07A"

    # Embedding colors
    embedding_color: str = "#9B59B6"
    embedding_edge_color: str = "#7D3C98"
    unembedding_color: str = "#3498DB"
    unembedding_edge_color: str = "#2980B9"

    # Layer norm colors
    layer_norm_color: str = "#95A5A6"
    layer_norm_edge_color: str = "#7F8C8D"

    # Text and labels
    font_family: str = "sans-serif"
    font_size: float = 10.0
    label_font_size: float = 8.0
    dimension_font_size: float = 7.0
    title_font_size: float = 14.0
    text_color: str = "#2C3E50"
    dimension_color: str = "#7F8C8D"

    # Arrow/connection styling
    arrow_color: str = "#34495E"
    arrow_width: float = 1.5
    arrow_head_size: float = 8.0

    # Component dimensions
    block_width: float = 2.0
    block_height: float = 0.6
    attention_head_width: float = 0.4
    attention_head_height: float = 0.4

    # Background
    background_color: str = "#FFFFFF"
    grid_color: str = "#F5F5F5"
    show_grid: bool = False

    # Show options
    show_dimensions: bool = True
    show_layer_numbers: bool = True
    show_residual_connections: bool = True
    show_layer_norms: bool = True
    compact_mode: bool = False

    def get_attention_head_color(self, head_idx: int) -> str:
        """Get color for a specific attention head."""
        return self.attention_head_colors[head_idx % len(self.attention_head_colors)]

    @classmethod
    def minimal(cls) -> "VisualizationConfig":
        """Create a minimal configuration with fewer details."""
        return cls(
            show_dimensions=False,
            show_layer_norms=False,
            compact_mode=True,
            layer_height=1.5,
        )

    @classmethod
    def detailed(cls) -> "VisualizationConfig":
        """Create a detailed configuration showing all components."""
        return cls(
            show_dimensions=True,
            show_layer_norms=True,
            show_residual_connections=True,
            layer_height=2.5,
        )

    @classmethod
    def dark_theme(cls) -> "VisualizationConfig":
        """Create a dark theme configuration."""
        return cls(
            background_color="#1E1E1E",
            text_color="#E0E0E0",
            dimension_color="#A0A0A0",
            residual_stream_color="#3A3A3A",
            residual_stream_edge_color="#4A4A4A",
            arrow_color="#B0B0B0",
            grid_color="#2A2A2A",
        )

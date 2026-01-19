"""
Visual components for transformer architecture diagrams.

Each component represents a part of the transformer architecture that can be
drawn on the visualization canvas.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.collections import PatchCollection
import numpy as np

from .config import VisualizationConfig


@dataclass
class BoundingBox:
    """Represents the bounding box of a component."""
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def top(self) -> float:
        return self.y + self.height

    @property
    def bottom(self) -> float:
        return self.y

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width


class Component:
    """Base class for all visual components."""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.bbox: Optional[BoundingBox] = None

    def draw(self, ax: plt.Axes, x: float, y: float) -> BoundingBox:
        """Draw the component on the given axes at position (x, y)."""
        raise NotImplementedError


class ResidualStream(Component):
    """Represents the residual stream flowing through the transformer."""

    def __init__(self, config: VisualizationConfig, d_model: int, height: float):
        super().__init__(config)
        self.d_model = d_model
        self.height = height

    def draw(self, ax: plt.Axes, x: float, y: float) -> BoundingBox:
        """Draw a vertical residual stream segment."""
        width = self.config.residual_stream_width

        # Draw the residual stream as a filled rectangle
        rect = FancyBboxPatch(
            (x - width / 2, y),
            width,
            self.height,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=self.config.residual_stream_color,
            edgecolor=self.config.residual_stream_edge_color,
            linewidth=1.5,
            zorder=1,
        )
        ax.add_patch(rect)

        self.bbox = BoundingBox(x - width / 2, y, width, self.height)
        return self.bbox

    def draw_with_dimension(self, ax: plt.Axes, x: float, y: float) -> BoundingBox:
        """Draw residual stream with dimension label."""
        bbox = self.draw(ax, x, y)

        if self.config.show_dimensions:
            ax.text(
                x + self.config.residual_stream_width / 2 + 0.1,
                y + self.height / 2,
                f"d={self.d_model}",
                fontsize=self.config.dimension_font_size,
                color=self.config.dimension_color,
                va="center",
                ha="left",
                fontfamily=self.config.font_family,
            )

        return bbox


class AttentionBlock(Component):
    """Represents a multi-head attention block."""

    def __init__(
        self,
        config: VisualizationConfig,
        n_heads: int,
        d_head: int,
        d_model: int,
        layer_idx: int,
    ):
        super().__init__(config)
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = d_model
        self.layer_idx = layer_idx

    def draw(self, ax: plt.Axes, x: float, y: float) -> BoundingBox:
        """Draw the attention block with individual heads."""
        cfg = self.config

        # Calculate dimensions
        total_width = cfg.block_width
        head_spacing = 0.05
        available_width = total_width - (self.n_heads + 1) * head_spacing

        # For many heads, we show a summary view
        show_individual_heads = self.n_heads <= 8

        if show_individual_heads:
            head_width = available_width / self.n_heads
            head_height = cfg.attention_head_height

            # Draw individual attention heads
            for i in range(self.n_heads):
                head_x = x + head_spacing + i * (head_width + head_spacing)
                head_color = cfg.get_attention_head_color(i)

                head_rect = FancyBboxPatch(
                    (head_x, y),
                    head_width,
                    head_height,
                    boxstyle="round,pad=0.01,rounding_size=0.03",
                    facecolor=head_color,
                    edgecolor=cfg.attention_block_edge_color,
                    linewidth=1,
                    zorder=3,
                )
                ax.add_patch(head_rect)

            # Draw head label
            if cfg.show_dimensions:
                ax.text(
                    x + total_width / 2,
                    y + head_height + 0.08,
                    f"{self.n_heads}×({self.d_head})",
                    fontsize=cfg.dimension_font_size,
                    color=cfg.dimension_color,
                    ha="center",
                    va="bottom",
                    fontfamily=cfg.font_family,
                )
        else:
            # Summary view for many heads
            block_rect = FancyBboxPatch(
                (x, y),
                total_width,
                cfg.block_height,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=cfg.attention_block_color,
                edgecolor=cfg.attention_block_edge_color,
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(block_rect)

            # Label inside
            ax.text(
                x + total_width / 2,
                y + cfg.block_height / 2,
                f"Attn ({self.n_heads}h)",
                fontsize=cfg.label_font_size,
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
                fontfamily=cfg.font_family,
            )

            if cfg.show_dimensions:
                ax.text(
                    x + total_width + 0.1,
                    y + cfg.block_height / 2,
                    f"{self.n_heads}×{self.d_head}",
                    fontsize=cfg.dimension_font_size,
                    color=cfg.dimension_color,
                    ha="left",
                    va="center",
                    fontfamily=cfg.font_family,
                )

        height = cfg.attention_head_height if show_individual_heads else cfg.block_height
        self.bbox = BoundingBox(x, y, total_width, height)
        return self.bbox

    def draw_detailed(self, ax: plt.Axes, x: float, y: float) -> BoundingBox:
        """Draw detailed attention block showing Q, K, V, O projections."""
        cfg = self.config
        total_width = cfg.block_width
        inner_height = cfg.block_height * 1.5

        # Outer container
        outer_rect = FancyBboxPatch(
            (x, y),
            total_width,
            inner_height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#E8F4F4",
            edgecolor=cfg.attention_block_edge_color,
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(outer_rect)

        # Q, K, V, O boxes
        qkvo_width = (total_width - 0.3) / 4
        qkvo_height = inner_height * 0.5
        qkvo_y = y + inner_height * 0.25

        labels = ["Q", "K", "V", "O"]
        colors = ["#5DADE2", "#48C9B0", "#F5B041", "#EC7063"]

        for i, (label, color) in enumerate(zip(labels, colors)):
            qkvo_x = x + 0.05 + i * (qkvo_width + 0.05)
            qkvo_rect = FancyBboxPatch(
                (qkvo_x, qkvo_y),
                qkvo_width,
                qkvo_height,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=color,
                edgecolor="white",
                linewidth=1,
                zorder=3,
            )
            ax.add_patch(qkvo_rect)

            ax.text(
                qkvo_x + qkvo_width / 2,
                qkvo_y + qkvo_height / 2,
                label,
                fontsize=cfg.label_font_size,
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
                fontfamily=cfg.font_family,
            )

        # Title
        ax.text(
            x + total_width / 2,
            y + inner_height - 0.08,
            "Multi-Head Attention",
            fontsize=cfg.label_font_size - 1,
            color=cfg.text_color,
            ha="center",
            va="top",
            fontfamily=cfg.font_family,
        )

        self.bbox = BoundingBox(x, y, total_width, inner_height)
        return self.bbox


class MLPBlock(Component):
    """Represents an MLP (feed-forward) block."""

    def __init__(
        self,
        config: VisualizationConfig,
        d_model: int,
        d_mlp: int,
        layer_idx: int,
    ):
        super().__init__(config)
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.layer_idx = layer_idx

    def draw(self, ax: plt.Axes, x: float, y: float) -> BoundingBox:
        """Draw the MLP block."""
        cfg = self.config
        width = cfg.block_width
        height = cfg.block_height

        # Main MLP block
        rect = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=cfg.mlp_block_color,
            edgecolor=cfg.mlp_block_edge_color,
            linewidth=1.5,
            zorder=3,
        )
        ax.add_patch(rect)

        # Label
        ax.text(
            x + width / 2,
            y + height / 2,
            "MLP",
            fontsize=cfg.label_font_size,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            fontfamily=cfg.font_family,
        )

        # Dimension label
        if cfg.show_dimensions:
            ax.text(
                x + width + 0.1,
                y + height / 2,
                f"{self.d_model}→{self.d_mlp}→{self.d_model}",
                fontsize=cfg.dimension_font_size,
                color=cfg.dimension_color,
                ha="left",
                va="center",
                fontfamily=cfg.font_family,
            )

        self.bbox = BoundingBox(x, y, width, height)
        return self.bbox

    def draw_detailed(self, ax: plt.Axes, x: float, y: float) -> BoundingBox:
        """Draw detailed MLP block showing expansion and contraction."""
        cfg = self.config
        width = cfg.block_width
        height = cfg.block_height * 1.2

        # Outer container
        outer_rect = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#FFE8E8",
            edgecolor=cfg.mlp_block_edge_color,
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(outer_rect)

        # Up projection (narrow to wide)
        up_width = width * 0.35
        up_height = height * 0.6
        up_x = x + width * 0.1
        up_y = y + (height - up_height) / 2

        up_rect = FancyBboxPatch(
            (up_x, up_y),
            up_width,
            up_height,
            boxstyle="round,pad=0.01",
            facecolor=cfg.mlp_inner_color,
            edgecolor=cfg.mlp_block_edge_color,
            linewidth=1,
            zorder=3,
        )
        ax.add_patch(up_rect)

        ax.text(
            up_x + up_width / 2,
            up_y + up_height / 2,
            "↑",
            fontsize=cfg.label_font_size + 2,
            color="white",
            ha="center",
            va="center",
            fontfamily=cfg.font_family,
        )

        # Down projection
        down_width = width * 0.35
        down_x = x + width * 0.55
        down_rect = FancyBboxPatch(
            (down_x, up_y),
            down_width,
            up_height,
            boxstyle="round,pad=0.01",
            facecolor=cfg.mlp_block_color,
            edgecolor=cfg.mlp_block_edge_color,
            linewidth=1,
            zorder=3,
        )
        ax.add_patch(down_rect)

        ax.text(
            down_x + down_width / 2,
            up_y + up_height / 2,
            "↓",
            fontsize=cfg.label_font_size + 2,
            color="white",
            ha="center",
            va="center",
            fontfamily=cfg.font_family,
        )

        # Dimensions
        if cfg.show_dimensions:
            ax.text(
                x + width / 2,
                y + height + 0.05,
                f"d_mlp={self.d_mlp}",
                fontsize=cfg.dimension_font_size,
                color=cfg.dimension_color,
                ha="center",
                va="bottom",
                fontfamily=cfg.font_family,
            )

        self.bbox = BoundingBox(x, y, width, height)
        return self.bbox


class EmbeddingLayer(Component):
    """Represents the token embedding layer."""

    def __init__(self, config: VisualizationConfig, d_vocab: int, d_model: int):
        super().__init__(config)
        self.d_vocab = d_vocab
        self.d_model = d_model

    def draw(self, ax: plt.Axes, x: float, y: float) -> BoundingBox:
        """Draw the embedding layer."""
        cfg = self.config
        width = cfg.block_width
        height = cfg.block_height

        rect = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=cfg.embedding_color,
            edgecolor=cfg.embedding_edge_color,
            linewidth=1.5,
            zorder=3,
        )
        ax.add_patch(rect)

        ax.text(
            x + width / 2,
            y + height / 2,
            "Embed",
            fontsize=cfg.label_font_size,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            fontfamily=cfg.font_family,
        )

        if cfg.show_dimensions:
            ax.text(
                x + width + 0.1,
                y + height / 2,
                f"{self.d_vocab}→{self.d_model}",
                fontsize=cfg.dimension_font_size,
                color=cfg.dimension_color,
                ha="left",
                va="center",
                fontfamily=cfg.font_family,
            )

        self.bbox = BoundingBox(x, y, width, height)
        return self.bbox


class UnembeddingLayer(Component):
    """Represents the unembedding (output) layer."""

    def __init__(self, config: VisualizationConfig, d_model: int, d_vocab: int):
        super().__init__(config)
        self.d_model = d_model
        self.d_vocab = d_vocab

    def draw(self, ax: plt.Axes, x: float, y: float) -> BoundingBox:
        """Draw the unembedding layer."""
        cfg = self.config
        width = cfg.block_width
        height = cfg.block_height

        rect = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=cfg.unembedding_color,
            edgecolor=cfg.unembedding_edge_color,
            linewidth=1.5,
            zorder=3,
        )
        ax.add_patch(rect)

        ax.text(
            x + width / 2,
            y + height / 2,
            "Unembed",
            fontsize=cfg.label_font_size,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            fontfamily=cfg.font_family,
        )

        if cfg.show_dimensions:
            ax.text(
                x + width + 0.1,
                y + height / 2,
                f"{self.d_model}→{self.d_vocab}",
                fontsize=cfg.dimension_font_size,
                color=cfg.dimension_color,
                ha="left",
                va="center",
                fontfamily=cfg.font_family,
            )

        self.bbox = BoundingBox(x, y, width, height)
        return self.bbox


class LayerNorm(Component):
    """Represents a layer normalization component."""

    def __init__(self, config: VisualizationConfig, d_model: int, label: str = "LN"):
        super().__init__(config)
        self.d_model = d_model
        self.label = label

    def draw(self, ax: plt.Axes, x: float, y: float) -> BoundingBox:
        """Draw the layer norm component."""
        cfg = self.config
        width = cfg.block_width * 0.3
        height = cfg.block_height * 0.5

        rect = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=cfg.layer_norm_color,
            edgecolor=cfg.layer_norm_edge_color,
            linewidth=1,
            zorder=3,
        )
        ax.add_patch(rect)

        ax.text(
            x + width / 2,
            y + height / 2,
            self.label,
            fontsize=cfg.dimension_font_size,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            fontfamily=cfg.font_family,
        )

        self.bbox = BoundingBox(x, y, width, height)
        return self.bbox


class Connection:
    """Draws connections/arrows between components."""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def draw_arrow(
        self,
        ax: plt.Axes,
        start: Tuple[float, float],
        end: Tuple[float, float],
        curved: bool = False,
        color: Optional[str] = None,
    ):
        """Draw an arrow from start to end."""
        cfg = self.config
        arrow_color = color or cfg.arrow_color

        if curved:
            # Curved arrow for residual connections
            style = "arc3,rad=0.3"
            arrow = FancyArrowPatch(
                start,
                end,
                connectionstyle=style,
                arrowstyle=f"->,head_width={cfg.arrow_head_size/30},head_length={cfg.arrow_head_size/30}",
                color=arrow_color,
                linewidth=cfg.arrow_width,
                zorder=1,
            )
        else:
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle=f"->,head_width={cfg.arrow_head_size/30},head_length={cfg.arrow_head_size/30}",
                color=arrow_color,
                linewidth=cfg.arrow_width,
                zorder=1,
            )

        ax.add_patch(arrow)

    def draw_residual_bypass(
        self,
        ax: plt.Axes,
        start_y: float,
        end_y: float,
        x: float,
        offset: float = 0.3,
    ):
        """Draw a residual bypass connection around a component."""
        cfg = self.config

        # Draw curved bypass
        mid_x = x - offset
        path_style = f"arc3,rad=-0.2"

        arrow = FancyArrowPatch(
            (x, start_y),
            (x, end_y),
            connectionstyle=f"bar,fraction=-0.3",
            arrowstyle=f"->,head_width={cfg.arrow_head_size/40},head_length={cfg.arrow_head_size/40}",
            color=cfg.residual_stream_edge_color,
            linewidth=cfg.arrow_width * 0.8,
            linestyle="--",
            zorder=0,
        )
        ax.add_patch(arrow)

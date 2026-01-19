"""
Main visualizer for transformer architectures.

Creates complete architecture diagrams combining all components.
"""

from typing import Optional, Union, Any, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from .config import VisualizationConfig
from .model_adapter import ModelArchitecture, TransformerLensAdapter
from .components import (
    ResidualStream,
    AttentionBlock,
    MLPBlock,
    EmbeddingLayer,
    UnembeddingLayer,
    LayerNorm,
    Connection,
    BoundingBox,
)


class TransformerVisualizer:
    """
    Main class for visualizing transformer architectures.

    Supports TransformerLens models, config dicts, and pretrained model names.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualizer.

        Args:
            config: Visualization configuration. Uses defaults if not provided.
        """
        self.config = config or VisualizationConfig()
        self.arch: Optional[ModelArchitecture] = None
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

    def from_model(self, model: Any) -> "TransformerVisualizer":
        """
        Load architecture from a TransformerLens HookedTransformer model.

        Args:
            model: A HookedTransformer instance

        Returns:
            self for method chaining
        """
        self.arch = TransformerLensAdapter.from_hooked_transformer(model)
        return self

    def from_config(self, cfg: Any) -> "TransformerVisualizer":
        """
        Load architecture from a TransformerLens config.

        Args:
            cfg: A HookedTransformerConfig instance

        Returns:
            self for method chaining
        """
        self.arch = TransformerLensAdapter.from_config(cfg)
        return self

    def from_dict(self, config: dict) -> "TransformerVisualizer":
        """
        Load architecture from a dictionary.

        Args:
            config: Dictionary with model configuration

        Returns:
            self for method chaining
        """
        self.arch = TransformerLensAdapter.from_dict(config)
        return self

    def from_pretrained(self, model_name: str) -> "TransformerVisualizer":
        """
        Load architecture for a known pretrained model.

        Args:
            model_name: Name like "gpt2", "gpt2-medium", "pythia-70m"

        Returns:
            self for method chaining
        """
        self.arch = TransformerLensAdapter.from_pretrained_name(model_name)
        return self

    def draw(
        self,
        max_layers: Optional[int] = None,
        detailed: bool = False,
        show_title: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Draw the transformer architecture diagram.

        Args:
            max_layers: Maximum number of layers to show (None = all)
            detailed: Show detailed internal structure of blocks
            show_title: Include model name as title

        Returns:
            Tuple of (Figure, Axes)
        """
        if self.arch is None:
            raise ValueError("No architecture loaded. Use from_model(), from_dict(), etc.")

        cfg = self.config
        arch = self.arch

        # Determine how many layers to show
        n_layers_to_show = min(arch.n_layers, max_layers) if max_layers else arch.n_layers
        show_ellipsis = max_layers and arch.n_layers > max_layers

        # Calculate figure dimensions
        # Each layer needs space for: residual + attn + residual + mlp + residual
        components_per_layer = 2 if arch.d_mlp > 0 else 1
        layer_height = cfg.layer_height

        # Total height: embed + layers + unembed + spacing
        total_layers = n_layers_to_show + (1 if show_ellipsis else 0)
        fig_height = (total_layers * layer_height) + 3.0  # Extra for embed/unembed

        self.fig, self.ax = plt.subplots(figsize=(cfg.figure_width, fig_height))
        ax = self.ax

        # Set up the canvas
        ax.set_xlim(0, cfg.figure_width)
        ax.set_ylim(0, fig_height)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(cfg.background_color)

        # Center position for main components
        center_x = cfg.figure_width / 2
        block_start_x = center_x - cfg.block_width / 2

        current_y = 0.5  # Start from bottom

        # Draw embedding layer
        embed = EmbeddingLayer(cfg, arch.d_vocab, arch.d_model)
        embed_bbox = embed.draw(ax, block_start_x, current_y)
        current_y = embed_bbox.top + cfg.component_spacing

        # Connection helper
        conn = Connection(cfg)

        # Draw each transformer layer
        for layer_idx in range(n_layers_to_show):
            layer_start_y = current_y

            # Layer number label
            if cfg.show_layer_numbers:
                ax.text(
                    block_start_x - 0.5,
                    current_y + layer_height / 2,
                    f"L{layer_idx}",
                    fontsize=cfg.label_font_size,
                    color=cfg.text_color,
                    ha="right",
                    va="center",
                    fontweight="bold",
                    fontfamily=cfg.font_family,
                )

            # Draw residual stream segment (before attention)
            if cfg.show_residual_connections:
                res_stream = ResidualStream(cfg, arch.d_model, cfg.component_spacing)
                res_stream.draw(ax, center_x, current_y)
            current_y += cfg.component_spacing

            # Layer Norm before attention
            if cfg.show_layer_norms:
                ln1 = LayerNorm(cfg, arch.d_model, "LN1")
                ln_width = cfg.block_width * 0.3
                ln1.draw(ax, center_x - ln_width / 2, current_y)
                current_y += cfg.block_height * 0.5 + cfg.component_spacing * 0.5

            # Attention block
            attn = AttentionBlock(cfg, arch.n_heads, arch.d_head, arch.d_model, layer_idx)
            if detailed:
                attn_bbox = attn.draw_detailed(ax, block_start_x, current_y)
            else:
                attn_bbox = attn.draw(ax, block_start_x, current_y)
            current_y = attn_bbox.top + cfg.component_spacing

            # MLP block (if present)
            if arch.d_mlp > 0:
                # Residual stream segment between attention and MLP
                if cfg.show_residual_connections:
                    res_stream = ResidualStream(cfg, arch.d_model, cfg.component_spacing)
                    res_stream.draw(ax, center_x, current_y)
                current_y += cfg.component_spacing

                # Layer Norm before MLP
                if cfg.show_layer_norms:
                    ln2 = LayerNorm(cfg, arch.d_model, "LN2")
                    ln2.draw(ax, center_x - ln_width / 2, current_y)
                    current_y += cfg.block_height * 0.5 + cfg.component_spacing * 0.5

                # MLP block
                mlp = MLPBlock(cfg, arch.d_model, arch.d_mlp, layer_idx)
                if detailed:
                    mlp_bbox = mlp.draw_detailed(ax, block_start_x, current_y)
                else:
                    mlp_bbox = mlp.draw(ax, block_start_x, current_y)
                current_y = mlp_bbox.top + cfg.component_spacing

            # Draw residual bypass arrows for this layer
            if cfg.show_residual_connections:
                # Bypass around attention
                bypass_x = block_start_x - 0.15
                conn.draw_arrow(
                    ax,
                    (bypass_x, layer_start_y + cfg.component_spacing),
                    (bypass_x, attn_bbox.top),
                    curved=False,
                    color=cfg.residual_stream_edge_color,
                )

        # Ellipsis for truncated layers
        if show_ellipsis:
            ax.text(
                center_x,
                current_y + 0.3,
                f"⋮\n({arch.n_layers - n_layers_to_show} more layers)",
                fontsize=cfg.font_size,
                color=cfg.text_color,
                ha="center",
                va="center",
                fontfamily=cfg.font_family,
            )
            current_y += layer_height * 0.7

        # Final residual stream
        if cfg.show_residual_connections:
            res_stream = ResidualStream(cfg, arch.d_model, cfg.component_spacing)
            res_stream.draw(ax, center_x, current_y)
        current_y += cfg.component_spacing

        # Final layer norm
        if cfg.show_layer_norms and arch.has_final_ln:
            ln_final = LayerNorm(cfg, arch.d_model, "LN")
            ln_final.draw(ax, center_x - ln_width / 2, current_y)
            current_y += cfg.block_height * 0.5 + cfg.component_spacing

        # Unembedding layer
        unembed = UnembeddingLayer(cfg, arch.d_model, arch.d_vocab_out or arch.d_vocab)
        unembed.draw(ax, block_start_x, current_y)

        # Title
        if show_title:
            title_text = arch.model_name
            if arch.n_ctx:
                title_text += f" (ctx={arch.n_ctx})"

            ax.text(
                center_x,
                fig_height - 0.3,
                title_text,
                fontsize=cfg.title_font_size,
                color=cfg.text_color,
                ha="center",
                va="top",
                fontweight="bold",
                fontfamily=cfg.font_family,
            )

            # Subtitle with key dimensions
            subtitle = f"d_model={arch.d_model}, n_layers={arch.n_layers}, n_heads={arch.n_heads}"
            ax.text(
                center_x,
                fig_height - 0.6,
                subtitle,
                fontsize=cfg.label_font_size,
                color=cfg.dimension_color,
                ha="center",
                va="top",
                fontfamily=cfg.font_family,
            )

        plt.tight_layout()
        return self.fig, ax

    def draw_single_layer(
        self,
        layer_idx: int = 0,
        detailed: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Draw a detailed view of a single transformer layer.

        Args:
            layer_idx: Which layer to visualize
            detailed: Show internal structure

        Returns:
            Tuple of (Figure, Axes)
        """
        if self.arch is None:
            raise ValueError("No architecture loaded.")

        cfg = self.config
        arch = self.arch

        fig_width = cfg.figure_width
        fig_height = 4.0

        self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height))
        ax = self.ax

        ax.set_xlim(0, fig_width)
        ax.set_ylim(0, fig_height)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(cfg.background_color)

        center_x = fig_width / 2
        current_y = 0.5

        # Input residual stream label
        ax.text(
            center_x,
            current_y,
            f"Input: residual stream (d={arch.d_model})",
            fontsize=cfg.label_font_size,
            color=cfg.text_color,
            ha="center",
            va="bottom",
            fontfamily=cfg.font_family,
        )
        current_y += 0.4

        # Residual stream
        res = ResidualStream(cfg, arch.d_model, 0.3)
        res.draw(ax, center_x, current_y)
        current_y += 0.5

        # Attention block (detailed)
        attn_x = center_x - cfg.block_width - 0.5
        attn = AttentionBlock(cfg, arch.n_heads, arch.d_head, arch.d_model, layer_idx)
        attn_bbox = attn.draw_detailed(ax, attn_x, current_y)

        # MLP block (detailed)
        if arch.d_mlp > 0:
            mlp_x = center_x + 0.5
            mlp = MLPBlock(cfg, arch.d_model, arch.d_mlp, layer_idx)
            mlp_bbox = mlp.draw_detailed(ax, mlp_x, current_y)

        current_y = max(attn_bbox.top, mlp_bbox.top if arch.d_mlp > 0 else attn_bbox.top) + 0.5

        # Output residual stream
        res_out = ResidualStream(cfg, arch.d_model, 0.3)
        res_out.draw(ax, center_x, current_y)

        # Title
        ax.text(
            center_x,
            fig_height - 0.2,
            f"Layer {layer_idx} Detail",
            fontsize=cfg.title_font_size,
            color=cfg.text_color,
            ha="center",
            va="top",
            fontweight="bold",
            fontfamily=cfg.font_family,
        )

        plt.tight_layout()
        return self.fig, ax

    def draw_attention_pattern(
        self,
        n_heads: Optional[int] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Draw a schematic of the attention head pattern.

        Args:
            n_heads: Override number of heads (uses model's n_heads if None)

        Returns:
            Tuple of (Figure, Axes)
        """
        if self.arch is None:
            raise ValueError("No architecture loaded.")

        cfg = self.config
        arch = self.arch
        n_heads = n_heads or arch.n_heads

        # Grid layout for heads
        cols = min(8, n_heads)
        rows = (n_heads + cols - 1) // cols

        fig_width = cols * 1.2 + 1
        fig_height = rows * 1.2 + 1.5

        self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height))
        ax = self.ax

        ax.set_xlim(0, fig_width)
        ax.set_ylim(0, fig_height)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(cfg.background_color)

        head_size = 0.8
        spacing = 1.2

        for i in range(n_heads):
            row = i // cols
            col = i % cols

            x = col * spacing + 0.7
            y = fig_height - (row + 1) * spacing - 0.5

            color = cfg.get_attention_head_color(i)

            rect = FancyBboxPatch(
                (x, y),
                head_size,
                head_size,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=color,
                edgecolor=cfg.attention_block_edge_color,
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(rect)

            ax.text(
                x + head_size / 2,
                y + head_size / 2,
                f"H{i}",
                fontsize=cfg.label_font_size,
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
                fontfamily=cfg.font_family,
            )

        # Title
        ax.text(
            fig_width / 2,
            fig_height - 0.2,
            f"Attention Heads ({n_heads}×d_head={arch.d_head})",
            fontsize=cfg.title_font_size,
            color=cfg.text_color,
            ha="center",
            va="top",
            fontweight="bold",
            fontfamily=cfg.font_family,
        )

        plt.tight_layout()
        return self.fig, ax

    def save(
        self,
        filepath: str,
        dpi: int = 150,
        transparent: bool = False,
    ):
        """
        Save the current figure to a file.

        Args:
            filepath: Output path (supports .png, .svg, .pdf)
            dpi: Resolution for raster formats
            transparent: Use transparent background
        """
        if self.fig is None:
            raise ValueError("No figure to save. Call draw() first.")

        self.fig.savefig(
            filepath,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
            facecolor=self.config.background_color if not transparent else "none",
        )

    def show(self):
        """Display the current figure."""
        if self.fig is None:
            raise ValueError("No figure to show. Call draw() first.")
        plt.show()


def visualize_transformer(
    model_or_config: Union[Any, dict, str],
    config: Optional[VisualizationConfig] = None,
    max_layers: Optional[int] = None,
    detailed: bool = False,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Convenience function to visualize a transformer in one call.

    Args:
        model_or_config: HookedTransformer, config dict, or model name string
        config: Visualization configuration
        max_layers: Maximum layers to show
        detailed: Show detailed internal structure
        save_path: Optional path to save the figure

    Returns:
        Tuple of (Figure, Axes)
    """
    viz = TransformerVisualizer(config)

    if isinstance(model_or_config, str):
        viz.from_pretrained(model_or_config)
    elif isinstance(model_or_config, dict):
        viz.from_dict(model_or_config)
    else:
        viz.from_model(model_or_config)

    fig, ax = viz.draw(max_layers=max_layers, detailed=detailed)

    if save_path:
        viz.save(save_path)

    return fig, ax

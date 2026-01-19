"""
Model adapter for extracting architecture information from TransformerLens models.
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict


@dataclass
class ModelArchitecture:
    """Container for transformer model architecture specifications."""

    # Core dimensions
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    d_mlp: int
    d_vocab: int

    # Optional dimensions
    n_ctx: Optional[int] = None  # Context length
    d_vocab_out: Optional[int] = None  # Output vocab size (if different)

    # Architecture variants
    model_name: str = "Transformer"
    is_gpt2_style: bool = True  # Attention before MLP in each layer
    has_layer_norm: bool = True
    has_final_ln: bool = True
    has_positional_embedding: bool = True

    # Attention variants
    has_qkv_bias: bool = False
    has_rotary: bool = False
    has_parallel_attention: bool = False  # Attention and MLP in parallel

    @property
    def total_params_estimate(self) -> int:
        """Estimate total parameter count."""
        # Embedding
        embed_params = self.d_vocab * self.d_model

        # Positional embedding (if applicable)
        pos_params = self.n_ctx * self.d_model if self.n_ctx else 0

        # Per-layer attention params: Q, K, V, O projections
        attn_params = 4 * self.d_model * self.d_model

        # Per-layer MLP params
        mlp_params = 2 * self.d_model * self.d_mlp

        # Per-layer layer norm params
        ln_params = 4 * self.d_model if self.has_layer_norm else 0

        # Total per layer
        layer_params = attn_params + mlp_params + ln_params

        # Unembedding
        unembed_params = self.d_model * (self.d_vocab_out or self.d_vocab)

        # Final layer norm
        final_ln_params = self.d_model if self.has_final_ln else 0

        return embed_params + pos_params + (layer_params * self.n_layers) + unembed_params + final_ln_params


class TransformerLensAdapter:
    """Adapter to extract architecture info from TransformerLens HookedTransformer models."""

    @staticmethod
    def from_hooked_transformer(model: Any) -> ModelArchitecture:
        """
        Extract architecture from a TransformerLens HookedTransformer model.

        Args:
            model: A HookedTransformer instance from TransformerLens

        Returns:
            ModelArchitecture with extracted specifications
        """
        cfg = model.cfg

        return ModelArchitecture(
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_head=cfg.d_head,
            d_mlp=cfg.d_mlp,
            d_vocab=cfg.d_vocab,
            n_ctx=cfg.n_ctx,
            d_vocab_out=getattr(cfg, "d_vocab_out", None),
            model_name=getattr(cfg, "model_name", "Transformer"),
            is_gpt2_style=True,  # TransformerLens uses GPT-2 style by default
            has_layer_norm=True,
            has_final_ln=True,
            has_positional_embedding=not getattr(cfg, "positional_embedding_type", "") == "rotary",
            has_qkv_bias=getattr(cfg, "attn_only", False) is False,
            has_rotary=getattr(cfg, "positional_embedding_type", "") == "rotary",
            has_parallel_attention=getattr(cfg, "parallel_attn_mlp", False),
        )

    @staticmethod
    def from_config(cfg: Any) -> ModelArchitecture:
        """
        Extract architecture from a TransformerLens HookedTransformerConfig.

        Args:
            cfg: A HookedTransformerConfig instance

        Returns:
            ModelArchitecture with extracted specifications
        """
        return ModelArchitecture(
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_head=cfg.d_head,
            d_mlp=cfg.d_mlp if cfg.d_mlp else cfg.d_model * 4,
            d_vocab=cfg.d_vocab,
            n_ctx=cfg.n_ctx,
            d_vocab_out=getattr(cfg, "d_vocab_out", None),
            model_name=getattr(cfg, "model_name", "Transformer"),
        )

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> ModelArchitecture:
        """
        Create architecture from a dictionary specification.

        Args:
            config: Dictionary with model configuration

        Returns:
            ModelArchitecture with specifications from dict
        """
        return ModelArchitecture(
            n_layers=config["n_layers"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            d_head=config.get("d_head", config["d_model"] // config["n_heads"]),
            d_mlp=config.get("d_mlp", config["d_model"] * 4),
            d_vocab=config["d_vocab"],
            n_ctx=config.get("n_ctx"),
            d_vocab_out=config.get("d_vocab_out"),
            model_name=config.get("model_name", "Transformer"),
        )

    @staticmethod
    def from_pretrained_name(model_name: str) -> ModelArchitecture:
        """
        Get architecture for well-known pretrained models.

        Args:
            model_name: Name of the pretrained model (e.g., "gpt2", "gpt2-medium")

        Returns:
            ModelArchitecture for the named model
        """
        # Common model configurations
        configs = {
            "gpt2": {
                "n_layers": 12, "d_model": 768, "n_heads": 12,
                "d_head": 64, "d_mlp": 3072, "d_vocab": 50257, "n_ctx": 1024,
            },
            "gpt2-small": {
                "n_layers": 12, "d_model": 768, "n_heads": 12,
                "d_head": 64, "d_mlp": 3072, "d_vocab": 50257, "n_ctx": 1024,
            },
            "gpt2-medium": {
                "n_layers": 24, "d_model": 1024, "n_heads": 16,
                "d_head": 64, "d_mlp": 4096, "d_vocab": 50257, "n_ctx": 1024,
            },
            "gpt2-large": {
                "n_layers": 36, "d_model": 1280, "n_heads": 20,
                "d_head": 64, "d_mlp": 5120, "d_vocab": 50257, "n_ctx": 1024,
            },
            "gpt2-xl": {
                "n_layers": 48, "d_model": 1600, "n_heads": 25,
                "d_head": 64, "d_mlp": 6400, "d_vocab": 50257, "n_ctx": 1024,
            },
            "gpt-neo-125m": {
                "n_layers": 12, "d_model": 768, "n_heads": 12,
                "d_head": 64, "d_mlp": 3072, "d_vocab": 50257, "n_ctx": 2048,
            },
            "pythia-70m": {
                "n_layers": 6, "d_model": 512, "n_heads": 8,
                "d_head": 64, "d_mlp": 2048, "d_vocab": 50304, "n_ctx": 2048,
            },
            "pythia-160m": {
                "n_layers": 12, "d_model": 768, "n_heads": 12,
                "d_head": 64, "d_mlp": 3072, "d_vocab": 50304, "n_ctx": 2048,
            },
            "pythia-410m": {
                "n_layers": 24, "d_model": 1024, "n_heads": 16,
                "d_head": 64, "d_mlp": 4096, "d_vocab": 50304, "n_ctx": 2048,
            },
            # Toy models for testing
            "attn-only-1l": {
                "n_layers": 1, "d_model": 512, "n_heads": 8,
                "d_head": 64, "d_mlp": 0, "d_vocab": 50257, "n_ctx": 1024,
            },
            "attn-only-2l": {
                "n_layers": 2, "d_model": 512, "n_heads": 8,
                "d_head": 64, "d_mlp": 0, "d_vocab": 50257, "n_ctx": 1024,
            },
        }

        if model_name.lower() not in configs:
            raise ValueError(
                f"Unknown model: {model_name}. Known models: {list(configs.keys())}"
            )

        config = configs[model_name.lower()]
        config["model_name"] = model_name
        return TransformerLensAdapter.from_dict(config)

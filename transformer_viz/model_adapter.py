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
    n_key_value_heads: Optional[int] = None  # For grouped-query attention

    # Architecture variants
    model_name: str = "Transformer"
    is_gpt2_style: bool = True  # Attention before MLP in each layer
    has_layer_norm: bool = True
    has_final_ln: bool = True
    has_positional_embedding: bool = True

    # LayerNorm variants
    ln_type: str = "pre"  # "pre" (GPT-2 style) or "post" (original transformer)

    # Positional embedding variants
    pos_embed_type: str = "learned"  # "learned", "rotary", "alibi", "sinusoidal", "none"

    # Attention variants
    has_qkv_bias: bool = False
    has_rotary: bool = False
    has_parallel_attention: bool = False  # Attention and MLP in parallel
    attn_type: str = "multi-head"  # "multi-head", "multi-query", "grouped-query"

    # MLP variants
    activation: str = "gelu"  # "gelu", "relu", "silu", "swiglu", "geglu"
    mlp_type: str = "standard"  # "standard", "gated" (for SwiGLU/GeGLU)
    d_mlp_gate: Optional[int] = None  # For gated MLPs

    # Embedding variants
    tied_embeddings: bool = False  # Whether embed and unembed share weights

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

        # Determine positional embedding type
        pos_embed_type_raw = getattr(cfg, "positional_embedding_type", "standard")
        if pos_embed_type_raw == "rotary":
            pos_embed_type = "rotary"
        elif pos_embed_type_raw == "alibi":
            pos_embed_type = "alibi"
        elif pos_embed_type_raw == "shortformer":
            pos_embed_type = "learned"
        else:
            pos_embed_type = "learned"

        # Determine activation function
        act_fn = getattr(cfg, "act_fn", "gelu")
        if "gelu" in act_fn.lower():
            activation = "gelu"
        elif "relu" in act_fn.lower():
            activation = "relu"
        elif "silu" in act_fn.lower() or "swish" in act_fn.lower():
            activation = "silu"
        else:
            activation = act_fn

        # Determine attention type
        n_kv_heads = getattr(cfg, "n_key_value_heads", None)
        if n_kv_heads is not None and n_kv_heads < cfg.n_heads:
            if n_kv_heads == 1:
                attn_type = "multi-query"
            else:
                attn_type = "grouped-query"
        else:
            attn_type = "multi-head"

        # Check for gated MLP (SwiGLU, GeGLU)
        gated_mlp = getattr(cfg, "gated_mlp", False)
        mlp_type = "gated" if gated_mlp else "standard"

        return ModelArchitecture(
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_head=cfg.d_head,
            d_mlp=cfg.d_mlp,
            d_vocab=cfg.d_vocab,
            n_ctx=cfg.n_ctx,
            d_vocab_out=getattr(cfg, "d_vocab_out", None),
            n_key_value_heads=n_kv_heads,
            model_name=getattr(cfg, "model_name", "Transformer"),
            is_gpt2_style=True,
            has_layer_norm=True,
            has_final_ln=True,
            has_positional_embedding=pos_embed_type != "rotary" and pos_embed_type != "alibi",
            ln_type="pre",  # TransformerLens uses pre-LN
            pos_embed_type=pos_embed_type,
            has_qkv_bias=getattr(cfg, "attn_only", False) is False,
            has_rotary=pos_embed_type == "rotary",
            has_parallel_attention=getattr(cfg, "parallel_attn_mlp", False),
            attn_type=attn_type,
            activation=activation,
            mlp_type=mlp_type,
            tied_embeddings=getattr(cfg, "tie_word_embeddings", False),
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
            n_key_value_heads=config.get("n_key_value_heads"),
            model_name=config.get("model_name", "Transformer"),
            ln_type=config.get("ln_type", "pre"),
            pos_embed_type=config.get("pos_embed_type", "learned"),
            has_positional_embedding=config.get("has_positional_embedding", True),
            attn_type=config.get("attn_type", "multi-head"),
            activation=config.get("activation", "gelu"),
            mlp_type=config.get("mlp_type", "standard"),
            tied_embeddings=config.get("tied_embeddings", False),
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
            # GPT-2 family (pre-LN, GELU, learned pos embed)
            "gpt2": {
                "n_layers": 12, "d_model": 768, "n_heads": 12,
                "d_head": 64, "d_mlp": 3072, "d_vocab": 50257, "n_ctx": 1024,
                "activation": "gelu", "ln_type": "pre", "pos_embed_type": "learned",
            },
            "gpt2-small": {
                "n_layers": 12, "d_model": 768, "n_heads": 12,
                "d_head": 64, "d_mlp": 3072, "d_vocab": 50257, "n_ctx": 1024,
                "activation": "gelu", "ln_type": "pre", "pos_embed_type": "learned",
            },
            "gpt2-medium": {
                "n_layers": 24, "d_model": 1024, "n_heads": 16,
                "d_head": 64, "d_mlp": 4096, "d_vocab": 50257, "n_ctx": 1024,
                "activation": "gelu", "ln_type": "pre", "pos_embed_type": "learned",
            },
            "gpt2-large": {
                "n_layers": 36, "d_model": 1280, "n_heads": 20,
                "d_head": 64, "d_mlp": 5120, "d_vocab": 50257, "n_ctx": 1024,
                "activation": "gelu", "ln_type": "pre", "pos_embed_type": "learned",
            },
            "gpt2-xl": {
                "n_layers": 48, "d_model": 1600, "n_heads": 25,
                "d_head": 64, "d_mlp": 6400, "d_vocab": 50257, "n_ctx": 1024,
                "activation": "gelu", "ln_type": "pre", "pos_embed_type": "learned",
            },
            # GPT-Neo
            "gpt-neo-125m": {
                "n_layers": 12, "d_model": 768, "n_heads": 12,
                "d_head": 64, "d_mlp": 3072, "d_vocab": 50257, "n_ctx": 2048,
                "activation": "gelu", "ln_type": "pre", "pos_embed_type": "learned",
            },
            # Pythia family (rotary pos embed)
            "pythia-70m": {
                "n_layers": 6, "d_model": 512, "n_heads": 8,
                "d_head": 64, "d_mlp": 2048, "d_vocab": 50304, "n_ctx": 2048,
                "activation": "gelu", "ln_type": "pre", "pos_embed_type": "rotary",
                "has_positional_embedding": False,
            },
            "pythia-160m": {
                "n_layers": 12, "d_model": 768, "n_heads": 12,
                "d_head": 64, "d_mlp": 3072, "d_vocab": 50304, "n_ctx": 2048,
                "activation": "gelu", "ln_type": "pre", "pos_embed_type": "rotary",
                "has_positional_embedding": False,
            },
            "pythia-410m": {
                "n_layers": 24, "d_model": 1024, "n_heads": 16,
                "d_head": 64, "d_mlp": 4096, "d_vocab": 50304, "n_ctx": 2048,
                "activation": "gelu", "ln_type": "pre", "pos_embed_type": "rotary",
                "has_positional_embedding": False,
            },
            # LLaMA family (RoPE, SwiGLU, RMSNorm-style)
            "llama-7b": {
                "n_layers": 32, "d_model": 4096, "n_heads": 32,
                "d_head": 128, "d_mlp": 11008, "d_vocab": 32000, "n_ctx": 2048,
                "activation": "silu", "mlp_type": "gated", "ln_type": "pre",
                "pos_embed_type": "rotary", "has_positional_embedding": False,
            },
            "llama-13b": {
                "n_layers": 40, "d_model": 5120, "n_heads": 40,
                "d_head": 128, "d_mlp": 13824, "d_vocab": 32000, "n_ctx": 2048,
                "activation": "silu", "mlp_type": "gated", "ln_type": "pre",
                "pos_embed_type": "rotary", "has_positional_embedding": False,
            },
            "llama-2-7b": {
                "n_layers": 32, "d_model": 4096, "n_heads": 32,
                "d_head": 128, "d_mlp": 11008, "d_vocab": 32000, "n_ctx": 4096,
                "activation": "silu", "mlp_type": "gated", "ln_type": "pre",
                "pos_embed_type": "rotary", "has_positional_embedding": False,
            },
            # Mistral (GQA)
            "mistral-7b": {
                "n_layers": 32, "d_model": 4096, "n_heads": 32, "n_key_value_heads": 8,
                "d_head": 128, "d_mlp": 14336, "d_vocab": 32000, "n_ctx": 8192,
                "activation": "silu", "mlp_type": "gated", "ln_type": "pre",
                "pos_embed_type": "rotary", "has_positional_embedding": False,
                "attn_type": "grouped-query",
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

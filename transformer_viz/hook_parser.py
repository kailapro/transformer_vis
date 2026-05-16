"""
Parse TransformerLens hook names to extract component information.

Every HookedTransformer hookpoint maps to a visual region in the architecture
diagram. The HOOKPOINT_MAP below defines this mapping exhaustively.

Visual regions in the diagram (bottom to top):
  ┌─────────────────────────────────────────────────────────────┐
  │  tokens → [embed] → + ← [pos_embed]                       │
  │                     │ (residual stream, dashed line)        │
  │  ┌─ Layer X ────────┼──────────────────────────────────┐   │
  │  │                   ├──→ [LN1] ──→ [h0][h1]...[h11] ─┤   │
  │  │                   ⊕ ← attention output               │   │
  │  │                   │                                   │   │
  │  │                   ├──→ [LN2] ──→ [MLP] ─────────────┤   │
  │  │                   ⊕ ← MLP output                     │   │
  │  └───────────────────┼──────────────────────────────────┘   │
  │                      │                                      │
  │  [ln_final] → [unembed] → logits                           │
  └─────────────────────────────────────────────────────────────┘

Hookpoint → Visual region mapping:

  HOOK NAME                         COMPONENT TYPE   VISUAL REGION
  ─────────────────────────────────────────────────────────────────
  hook_embed                        embed            Embed block
  hook_pos_embed                    pos_embed        Pos_embed block

  blocks.X.hook_resid_pre           resid            Residual: layer entry → attn ⊕
  blocks.X.hook_resid_mid           resid            Residual: attn ⊕ → MLP ⊕
  blocks.X.hook_resid_post          resid            Residual: MLP ⊕ → next LN branch

  blocks.X.ln1.hook_scale           ln1              Pre-attention LN block
  blocks.X.ln1.hook_normalized      ln1              Pre-attention LN block

  blocks.X.hook_attn_in             attn_in          Arrow: residual → LN1 → heads
  blocks.X.hook_q_input             attn_in          Arrow: residual → LN1 → heads
  blocks.X.hook_k_input             attn_in          Arrow: residual → LN1 → heads
  blocks.X.hook_v_input             attn_in          Arrow: residual → LN1 → heads

  blocks.X.attn.hook_q              attention        Attention head blocks
  blocks.X.attn.hook_k              attention        Attention head blocks
  blocks.X.attn.hook_v              attention        Attention head blocks
  blocks.X.attn.hook_z              attention        Attention head blocks
  blocks.X.attn.hook_attn_scores    attention        Attention head blocks
  blocks.X.attn.hook_pattern        attention        Attention head blocks
  blocks.X.attn.hook_result         attention        Attention head blocks

  blocks.X.hook_attn_out            attn_out         Arrow: heads → ⊕ on residual

  blocks.X.ln2.hook_scale           ln2              Pre-MLP LN block
  blocks.X.ln2.hook_normalized      ln2              Pre-MLP LN block

  blocks.X.hook_mlp_in              mlp_in           Arrow: residual → LN2 → MLP
  blocks.X.mlp.hook_pre             mlp              MLP block
  blocks.X.mlp.hook_post            mlp              MLP block
  blocks.X.hook_mlp_out             mlp_out          Arrow: MLP → ⊕ on residual

  ln_final.hook_scale               ln_final         Final LayerNorm block
  ln_final.hook_normalized          ln_final         Final LayerNorm block
"""

import re
import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable, Tuple, Any

# Hooks beyond this count can generate enough SVG to crash the Jupyter kernel
_HOOK_COUNT_WARNING_THRESHOLD = 200

# Distinct colors for different hook functions
HOOK_COLORS = [
    '#FFD93D',  # Yellow
    '#6BCB77',  # Green
    '#4D96FF',  # Blue
    '#FF6B6B',  # Red/coral
    '#C9B1FF',  # Purple
    '#FF9F45',  # Orange
    '#00D9FF',  # Cyan
    '#FF6BFF',  # Pink
]


# ── Hookpoint → component type mapping ──────────────────────────────────────
#
# Each entry is (regex_pattern, component_type).
# Patterns are tried in order; first match wins.
# Groups: group(1) = layer index (if present), remainder = subcomponent.
#
# "Layer-level" hooks use blocks.{layer}.hook_* format.
# "Sub-module" hooks use blocks.{layer}.{module}.hook_* format.

# Patterns for specific layer (blocks.{digit}.*)
_LAYER_PATTERNS: List[Tuple[str, str]] = [
    # Residual stream
    (r'blocks\.(\d+)\.hook_resid_(pre|mid|post)',  'resid'),

    # Pre-attention LayerNorm
    (r'blocks\.(\d+)\.ln1\.(.+)',                  'ln1'),

    # Attention inputs (block-level hooks for Q/K/V inputs)
    (r'blocks\.(\d+)\.hook_attn_in',               'attn_in'),
    (r'blocks\.(\d+)\.hook_q_input',               'attn_in'),
    (r'blocks\.(\d+)\.hook_k_input',               'attn_in'),
    (r'blocks\.(\d+)\.hook_v_input',               'attn_in'),

    # Attention internals (sub-module hooks)
    (r'blocks\.(\d+)\.attn\.(.+)',                  'attention'),

    # Attention output (block-level)
    (r'blocks\.(\d+)\.hook_attn_out',              'attn_out'),

    # Pre-MLP LayerNorm
    (r'blocks\.(\d+)\.ln2\.(.+)',                  'ln2'),

    # MLP input (block-level)
    (r'blocks\.(\d+)\.hook_mlp_in',               'mlp_in'),

    # MLP internals (sub-module hooks)
    (r'blocks\.(\d+)\.mlp\.(.+)',                  'mlp'),

    # MLP output (block-level)
    (r'blocks\.(\d+)\.hook_mlp_out',              'mlp_out'),
]

# Same patterns but with wildcard layer (blocks.*.*)
_WILDCARD_PATTERNS: List[Tuple[str, str]] = [
    (r'blocks\.\*\.hook_resid_(pre|mid|post)',     'resid'),
    (r'blocks\.\*\.ln1\.(.+)',                     'ln1'),
    (r'blocks\.\*\.hook_attn_in',                  'attn_in'),
    (r'blocks\.\*\.hook_q_input',                  'attn_in'),
    (r'blocks\.\*\.hook_k_input',                  'attn_in'),
    (r'blocks\.\*\.hook_v_input',                  'attn_in'),
    (r'blocks\.\*\.attn\.(.+)',                    'attention'),
    (r'blocks\.\*\.hook_attn_out',                 'attn_out'),
    (r'blocks\.\*\.ln2\.(.+)',                     'ln2'),
    (r'blocks\.\*\.hook_mlp_in',                   'mlp_in'),
    (r'blocks\.\*\.mlp\.(.+)',                     'mlp'),
    (r'blocks\.\*\.hook_mlp_out',                  'mlp_out'),
]

# Global hooks (no layer index)
_GLOBAL_PATTERNS: List[Tuple[str, str]] = [
    (r'hook_embed$',                               'embed'),
    (r'hook_pos_embed$',                           'pos_embed'),
    (r'ln_final\.(.+)',                            'ln_final'),
]


@dataclass
class ParsedHook:
    """Parsed information from a TransformerLens hook specification."""
    hook_name: str           # Original: 'blocks.1.attn.hook_pattern'
    component_type: str      # 'attention', 'mlp', 'ln1', 'ln2', 'embed', 'pos_embed', 'resid', ...
    layer_idx: Optional[int] # 1 (or None for global hooks)
    subcomponent: str        # 'hook_pattern', 'hook_q', etc.
    function_name: str       # 'hook_function' (extracted from function.__name__)
    head_indices: Optional[List[int]] = None  # Specific heads to highlight (None = all heads)
    wildcard_layer: bool = False  # True when blocks.* wildcard — matches all layers


# Type alias for hook specifications
HookSpec = Tuple[str, Callable]  # Basic: (hook_name, hook_fn)
# Extended format: (hook_name, hook_fn, {'heads': [0, 2]})


def parse_hook(hook_spec) -> ParsedHook:
    """
    Parse a TransformerLens hook specification into structured data.

    Args:
        hook_spec: Tuple of (hook_name, hook_function) or
                   (hook_name, hook_function, options_dict)
                   e.g., ('blocks.1.attn.hook_pattern', my_hook_fn)
                   or ('blocks.1.attn.hook_pattern', my_hook_fn, {'heads': [0, 2]})

    Returns:
        ParsedHook with extracted component information
    """
    # Handle both 2-tuple and 3-tuple formats
    if len(hook_spec) == 2:
        hook_name, hook_fn = hook_spec
        options = {}
    elif len(hook_spec) == 3:
        hook_name, hook_fn, options = hook_spec
        if options is None:
            options = {}
    else:
        raise ValueError(f"Hook spec must be 2 or 3 elements, got {len(hook_spec)}")

    # Extract function name
    function_name = getattr(hook_fn, '__name__', 'anonymous')

    # Extract head indices from options
    head_indices = options.get('heads', None)

    # Try wildcard patterns first (blocks.*.*)
    for pattern, component_type in _WILDCARD_PATTERNS:
        m = re.match(pattern, hook_name)
        if m:
            subcomponent = m.group(0).split('.')[-1]  # last segment as subcomponent
            # For attention, pass head_indices
            hi = head_indices if component_type in ('attention', 'attn_in', 'attn_out') else None
            return ParsedHook(hook_name, component_type, None, subcomponent, function_name, hi, wildcard_layer=True)

    # Try specific-layer patterns (blocks.{digit}.*)
    for pattern, component_type in _LAYER_PATTERNS:
        m = re.match(pattern, hook_name)
        if m:
            layer_idx = int(m.group(1))
            subcomponent = hook_name.split('.')[-1]  # last segment
            hi = head_indices if component_type in ('attention', 'attn_in', 'attn_out') else None
            return ParsedHook(hook_name, component_type, layer_idx, subcomponent, function_name, hi)

    # Try global patterns
    for pattern, component_type in _GLOBAL_PATTERNS:
        m = re.match(pattern, hook_name)
        if m:
            subcomponent = hook_name
            return ParsedHook(hook_name, component_type, None, subcomponent, function_name)

    # Fallback for unrecognized patterns
    return ParsedHook(hook_name, 'unknown', None, hook_name, function_name)


def assign_colors(parsed_hooks: List[ParsedHook]) -> Dict[str, str]:
    """
    Assign colors to unique function names.

    Args:
        parsed_hooks: List of ParsedHook objects

    Returns:
        Dictionary mapping function names to hex color strings
    """
    unique_functions = []
    seen = set()
    for h in parsed_hooks:
        if h.function_name not in seen:
            unique_functions.append(h.function_name)
            seen.add(h.function_name)

    return {fn: HOOK_COLORS[i % len(HOOK_COLORS)] for i, fn in enumerate(unique_functions)}


def process_hooks(hooks: List) -> Dict[str, Any]:
    """
    Process a list of hook specifications into data for visualization.

    Args:
        hooks: List of hook specs. Each can be:
               - (hook_name, hook_function) - highlights all heads for attention
               - (hook_name, hook_function, {'heads': [0, 2]}) - highlights specific heads

    Returns:
        Dictionary with 'hooks' list and 'legend' list for JavaScript
    """
    if not hooks:
        return {'hooks': [], 'legend': []}

    if len(hooks) > _HOOK_COUNT_WARNING_THRESHOLD:
        warnings.warn(
            f"You passed {len(hooks)} hooks to the visualizer, but rendering more than "
            f"{_HOOK_COUNT_WARNING_THRESHOLD} can generate very large SVG and may crash "
            f"the Jupyter kernel.\n\n"
            f"Tip: pass one representative hook per component type instead of all hooks. "
            f"For example, if you want to show all hookpoints that exist in the model, "
            f"use one hook per component (embed, pos_embed, attn, mlp, resid, ln1, ln2, "
            f"ln_final) rather than iterating model.hook_dict.keys().",
            UserWarning,
            stacklevel=2,
        )

    parsed = [parse_hook(h) for h in hooks]
    color_map = assign_colors(parsed)

    hook_data = {
        'hooks': [
            {
                'layer': h.layer_idx,
                'component': h.component_type,
                'function': h.function_name,
                'color': color_map[h.function_name],
                'hookName': h.hook_name,
                'subcomponent': h.subcomponent,
                'heads': h.head_indices,  # None means all heads, list means specific heads
                'wildcardLayer': h.wildcard_layer,
            }
            for h in parsed
        ],
        'legend': [
            {'function': fn, 'color': color}
            for fn, color in color_map.items()
        ]
    }

    return hook_data

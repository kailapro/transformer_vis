"""
Parse TransformerLens hook names to extract component information.

Hook name format examples:
- blocks.1.attn.hook_pattern -> Layer 1 attention
- blocks.2.mlp.hook_post -> Layer 2 MLP
- blocks.0.ln1.hook_normalized -> Layer 0 pre-attention LayerNorm
- blocks.1.ln2.hook_normalized -> Layer 1 pre-MLP LayerNorm
- hook_embed -> Embedding
- hook_pos_embed -> Positional embedding
- ln_final.hook_normalized -> Final LayerNorm
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable, Tuple, Any

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


@dataclass
class ParsedHook:
    """Parsed information from a TransformerLens hook specification."""
    hook_name: str           # Original: 'blocks.1.attn.hook_pattern'
    component_type: str      # 'attention', 'mlp', 'ln1', 'ln2', 'embed', 'pos_embed', 'resid', 'ln_final'
    layer_idx: Optional[int] # 1 (or None for global hooks)
    subcomponent: str        # 'hook_pattern', 'hook_q', etc.
    function_name: str       # 'hook_function' (extracted from function.__name__)
    head_indices: Optional[List[int]] = None  # Specific heads to highlight (None = all heads)


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

    # Parse the hook name to determine component type and layer
    component_type = 'unknown'
    layer_idx = None
    subcomponent = hook_name

    # Pattern: blocks.{L}.attn.* -> attention
    attn_match = re.match(r'blocks\.(\d+)\.attn\.(.+)', hook_name)
    if attn_match:
        layer_idx = int(attn_match.group(1))
        subcomponent = attn_match.group(2)
        component_type = 'attention'
        return ParsedHook(hook_name, component_type, layer_idx, subcomponent, function_name, head_indices)

    # Pattern: blocks.{L}.hook_attn_* -> attention (alternative format)
    attn_alt_match = re.match(r'blocks\.(\d+)\.hook_attn_(.+)', hook_name)
    if attn_alt_match:
        layer_idx = int(attn_alt_match.group(1))
        subcomponent = 'hook_attn_' + attn_alt_match.group(2)
        component_type = 'attention'
        return ParsedHook(hook_name, component_type, layer_idx, subcomponent, function_name, head_indices)

    # Pattern: blocks.{L}.mlp.* -> mlp
    mlp_match = re.match(r'blocks\.(\d+)\.mlp\.(.+)', hook_name)
    if mlp_match:
        layer_idx = int(mlp_match.group(1))
        subcomponent = mlp_match.group(2)
        component_type = 'mlp'
        return ParsedHook(hook_name, component_type, layer_idx, subcomponent, function_name)

    # Pattern: blocks.{L}.hook_mlp_* -> mlp (alternative format)
    mlp_alt_match = re.match(r'blocks\.(\d+)\.hook_mlp_(.+)', hook_name)
    if mlp_alt_match:
        layer_idx = int(mlp_alt_match.group(1))
        subcomponent = 'hook_mlp_' + mlp_alt_match.group(2)
        component_type = 'mlp'
        return ParsedHook(hook_name, component_type, layer_idx, subcomponent, function_name)

    # Pattern: blocks.{L}.ln1.* -> ln1 (pre-attention LayerNorm)
    ln1_match = re.match(r'blocks\.(\d+)\.ln1\.(.+)', hook_name)
    if ln1_match:
        layer_idx = int(ln1_match.group(1))
        subcomponent = ln1_match.group(2)
        component_type = 'ln1'
        return ParsedHook(hook_name, component_type, layer_idx, subcomponent, function_name)

    # Pattern: blocks.{L}.ln2.* -> ln2 (pre-MLP LayerNorm)
    ln2_match = re.match(r'blocks\.(\d+)\.ln2\.(.+)', hook_name)
    if ln2_match:
        layer_idx = int(ln2_match.group(1))
        subcomponent = ln2_match.group(2)
        component_type = 'ln2'
        return ParsedHook(hook_name, component_type, layer_idx, subcomponent, function_name)

    # Pattern: blocks.{L}.hook_resid_* -> resid
    resid_match = re.match(r'blocks\.(\d+)\.hook_resid_(.+)', hook_name)
    if resid_match:
        layer_idx = int(resid_match.group(1))
        subcomponent = 'hook_resid_' + resid_match.group(2)
        component_type = 'resid'
        return ParsedHook(hook_name, component_type, layer_idx, subcomponent, function_name)

    # Pattern: hook_embed -> embed
    if hook_name == 'hook_embed':
        component_type = 'embed'
        subcomponent = 'hook_embed'
        return ParsedHook(hook_name, component_type, layer_idx, subcomponent, function_name)

    # Pattern: hook_pos_embed -> pos_embed
    if hook_name == 'hook_pos_embed':
        component_type = 'pos_embed'
        subcomponent = 'hook_pos_embed'
        return ParsedHook(hook_name, component_type, layer_idx, subcomponent, function_name)

    # Pattern: ln_final.* -> ln_final
    ln_final_match = re.match(r'ln_final\.(.+)', hook_name)
    if ln_final_match:
        component_type = 'ln_final'
        subcomponent = ln_final_match.group(1)
        return ParsedHook(hook_name, component_type, layer_idx, subcomponent, function_name)

    # Fallback for unrecognized patterns
    return ParsedHook(hook_name, component_type, layer_idx, subcomponent, function_name)


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
            }
            for h in parsed
        ],
        'legend': [
            {'function': fn, 'color': color}
            for fn, color in color_map.items()
        ]
    }

    return hook_data

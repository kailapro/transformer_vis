"""
Test script for hook visualization feature.

This script tests:
1. Hook parsing functionality
2. Color assignment
3. Visualization with hooks
4. Head-specific highlighting

Run this script and open test_hooks.html in a browser to verify:
- Layer 1, head 0 highlighted in color A (yellow)
- Layer 1, head 2 highlighted in color B (green)
- Layer 2 MLP highlighted in color B (green)
- Layer 0 pre-attention LN highlighted in color A (same as my_hook)
- Legend shows: my_hook -> color A, ablation_hook -> color B
"""

from transformer_viz import visualize
from transformer_viz.hook_parser import parse_hook, assign_colors, process_hooks, ParsedHook


def my_hook(tensor, hook):
    """Example hook function."""
    return tensor


def ablation_hook(tensor, hook):
    """Example ablation hook."""
    return tensor * 0


def head_specific_hook(tensor, hook):
    """Example hook that modifies specific heads."""
    # In practice, this would modify only certain heads
    tensor[:, 2, :, :] = 0  # Zero out head 2
    return tensor


def test_hook_parsing():
    """Test that hooks are parsed correctly."""
    print("Testing hook parsing...")

    # Test attention hook (all heads)
    parsed = parse_hook(('blocks.1.attn.hook_pattern', my_hook))
    assert parsed.component_type == 'attention', f"Expected 'attention', got '{parsed.component_type}'"
    assert parsed.layer_idx == 1, f"Expected layer 1, got {parsed.layer_idx}"
    assert parsed.function_name == 'my_hook', f"Expected 'my_hook', got '{parsed.function_name}'"
    assert parsed.head_indices is None, f"Expected None head_indices, got {parsed.head_indices}"
    print("  ✓ Attention hook (all heads) parsed correctly")

    # Test attention hook with specific heads
    parsed = parse_hook(('blocks.1.attn.hook_pattern', my_hook, {'heads': [0, 2]}))
    assert parsed.component_type == 'attention', f"Expected 'attention', got '{parsed.component_type}'"
    assert parsed.layer_idx == 1, f"Expected layer 1, got {parsed.layer_idx}"
    assert parsed.head_indices == [0, 2], f"Expected [0, 2], got {parsed.head_indices}"
    print("  ✓ Attention hook (specific heads) parsed correctly")

    # Test MLP hook
    parsed = parse_hook(('blocks.2.mlp.hook_post', ablation_hook))
    assert parsed.component_type == 'mlp', f"Expected 'mlp', got '{parsed.component_type}'"
    assert parsed.layer_idx == 2, f"Expected layer 2, got {parsed.layer_idx}"
    assert parsed.function_name == 'ablation_hook', f"Expected 'ablation_hook', got '{parsed.function_name}'"
    print("  ✓ MLP hook parsed correctly")

    # Test LN1 hook
    parsed = parse_hook(('blocks.0.ln1.hook_normalized', my_hook))
    assert parsed.component_type == 'ln1', f"Expected 'ln1', got '{parsed.component_type}'"
    assert parsed.layer_idx == 0, f"Expected layer 0, got {parsed.layer_idx}"
    print("  ✓ LN1 hook parsed correctly")

    # Test LN2 hook
    parsed = parse_hook(('blocks.1.ln2.hook_normalized', my_hook))
    assert parsed.component_type == 'ln2', f"Expected 'ln2', got '{parsed.component_type}'"
    assert parsed.layer_idx == 1, f"Expected layer 1, got {parsed.layer_idx}"
    print("  ✓ LN2 hook parsed correctly")

    # Test embed hook
    parsed = parse_hook(('hook_embed', my_hook))
    assert parsed.component_type == 'embed', f"Expected 'embed', got '{parsed.component_type}'"
    assert parsed.layer_idx is None, f"Expected None, got {parsed.layer_idx}"
    print("  ✓ Embed hook parsed correctly")

    # Test ln_final hook
    parsed = parse_hook(('ln_final.hook_normalized', my_hook))
    assert parsed.component_type == 'ln_final', f"Expected 'ln_final', got '{parsed.component_type}'"
    print("  ✓ LN final hook parsed correctly")

    print("All hook parsing tests passed!\n")


def test_color_assignment():
    """Test that colors are assigned correctly."""
    print("Testing color assignment...")

    hooks = [
        ('blocks.1.attn.hook_pattern', my_hook),
        ('blocks.2.mlp.hook_post', ablation_hook),
        ('blocks.0.ln1.hook_normalized', my_hook),  # Same function as first
    ]

    parsed = [parse_hook(h) for h in hooks]
    colors = assign_colors(parsed)

    # Should have 2 unique colors (for my_hook and ablation_hook)
    assert len(colors) == 2, f"Expected 2 colors, got {len(colors)}"
    assert 'my_hook' in colors, "Expected 'my_hook' in colors"
    assert 'ablation_hook' in colors, "Expected 'ablation_hook' in colors"

    # Colors should be different
    assert colors['my_hook'] != colors['ablation_hook'], "Colors should be different"
    print("  ✓ Colors assigned correctly")
    print(f"  ✓ my_hook -> {colors['my_hook']}")
    print(f"  ✓ ablation_hook -> {colors['ablation_hook']}")

    print("All color assignment tests passed!\n")


def test_process_hooks():
    """Test the full hook processing."""
    print("Testing process_hooks...")

    hooks = [
        ('blocks.1.attn.hook_pattern', my_hook),
        ('blocks.2.mlp.hook_post', ablation_hook),
    ]

    result = process_hooks(hooks)

    assert 'hooks' in result, "Expected 'hooks' in result"
    assert 'legend' in result, "Expected 'legend' in result"
    assert len(result['hooks']) == 2, f"Expected 2 hooks, got {len(result['hooks'])}"
    assert len(result['legend']) == 2, f"Expected 2 legend items, got {len(result['legend'])}"

    print("  ✓ process_hooks returns correct structure")
    print("All process_hooks tests passed!\n")


def test_visualization_with_hooks():
    """Test creating a visualization with hooks (all heads)."""
    print("Testing visualization with hooks (all heads)...")

    hooks = [
        ('blocks.1.attn.hook_pattern', my_hook),  # All heads in layer 1
        ('blocks.2.mlp.hook_post', ablation_hook),
        ('blocks.0.ln1.hook_normalized', my_hook),
    ]

    viz = visualize('gpt2-small', max_layers=4, hooks=hooks)
    viz.save_html('test_hooks_all_heads.html')

    print("  ✓ Visualization created successfully")
    print("  ✓ Saved to test_hooks_all_heads.html")
    print("\nOpen test_hooks_all_heads.html in a browser to verify:")
    print("  - ALL Layer 1 attention heads are highlighted (yellow)")
    print("  - Layer 2 MLP is highlighted (green)")
    print("  - Layer 0 pre-attention LN is highlighted (yellow)")
    print("\nAll visualization tests passed!\n")


def test_visualization_with_specific_heads():
    """Test creating a visualization with head-specific hooks."""
    print("Testing visualization with head-specific hooks...")

    hooks = [
        # Only highlight heads 0 and 2 in layer 1
        ('blocks.1.attn.hook_pattern', my_hook, {'heads': [0, 2]}),
        # Highlight head 5 in layer 2 with a different function
        ('blocks.2.attn.hook_pattern', ablation_hook, {'heads': [5]}),
        ('blocks.3.mlp.hook_post', ablation_hook),
    ]

    viz = visualize('gpt2-small', max_layers=4, hooks=hooks)
    viz.save_html('test_hooks.html')

    print("  ✓ Visualization created successfully")
    print("  ✓ Saved to test_hooks.html")
    print("\nOpen test_hooks.html in a browser to verify:")
    print("  - Layer 1: Only heads 0 and 2 are highlighted (yellow)")
    print("  - Layer 1: Heads 1 and 11 (last) are NOT highlighted")
    print("  - Layer 2: Only head 5 is highlighted (green) - but may not be visible")
    print("    (head 5 is between h1 and h11 which are shown as '...')")
    print("  - Layer 3 MLP is highlighted (green)")
    print("  - Legend shows 'my_hook' and 'ablation_hook' with their colors")
    print("\nAll head-specific visualization tests passed!\n")


def test_no_hooks():
    """Test that visualization works without hooks."""
    print("Testing visualization without hooks...")

    viz = visualize('gpt2-small', max_layers=4)
    viz.save_html('test_no_hooks.html')

    print("  ✓ Visualization without hooks created successfully")
    print("  ✓ Saved to test_no_hooks.html")
    print("All no-hooks tests passed!\n")


if __name__ == '__main__':
    test_hook_parsing()
    test_color_assignment()
    test_process_hooks()
    test_visualization_with_hooks()
    test_visualization_with_specific_heads()
    test_no_hooks()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)

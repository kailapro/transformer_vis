# Transformer Visualization Library

An interactive visualization library for transformer model architectures, compatible with [TransformerLens](https://github.com/neelnanda-io/TransformerLens). Styled after the diagrams in [Anthropic's Transformer Circuits publications](https://transformer-circuits.pub/2021/framework/index.html).

## Features

- Interactive SVG visualizations that render inline in Jupyter notebooks
- Hover to expand attention heads and MLP details
- Expandable/collapsible layers for large models
- Bottom-to-top data flow matching standard transformer circuit diagrams
- Shows individual attention heads (h₀, h₁, ..., hₙ₋₁)
- Residual stream with addition points clearly marked
- **Hook visualization**: Highlight components with TransformerLens hooks attached, with support for head-specific highlighting

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd mech_interp_visuals

# Install dependencies
pip install transformer_lens  # Optional, only needed for loading models directly
```

## Quick Start

### Basic Usage (Recommended)

The simplest way to visualize a transformer architecture:

```python
from transformer_viz import visualize

# Visualize a pretrained model by name
visualize("gpt2")
```

This will display an interactive diagram directly in your Jupyter notebook.

### From a TransformerLens Model

```python
from transformer_lens import HookedTransformer
from transformer_viz import visualize

# Load a model with TransformerLens
model = HookedTransformer.from_pretrained("gpt2-small")

# Visualize it
visualize(model)
```

### From a Custom Configuration

```python
from transformer_viz import visualize

# Define your own architecture
config = {
    "model_name": "my-transformer",
    "n_layers": 6,
    "d_model": 512,
    "n_heads": 8,
    "d_head": 64,
    "d_mlp": 2048,
    "d_vocab": 50257
}

visualize(config)
```

## Detailed Usage

### InteractiveTransformerViz Class

For more control, use the `InteractiveTransformerViz` class directly:

```python
from transformer_viz import InteractiveTransformerViz

viz = InteractiveTransformerViz()

# Load from a TransformerLens model
viz.from_model(model)

# Or from a pretrained name
viz.from_pretrained("gpt2-medium")

# Or from a config dict
viz.from_dict(config)

# Render with options
viz.render(max_layers=8, width=600)

# Display in notebook
viz.show()

# Or save to HTML file
viz.save_html("architecture.html")
```

### Options

- `max_layers`: Maximum number of layers to show initially (default: 8). Click "Show N more layers" to expand.
- `width`: Width of the visualization in pixels (default: 600)

### Supported Pretrained Models

The following model names are supported out of the box:

| Model Name | Layers | d_model | Heads | d_mlp |
|------------|--------|---------|-------|-------|
| `gpt2` / `gpt2-small` | 12 | 768 | 12 | 3072 |
| `gpt2-medium` | 24 | 1024 | 16 | 4096 |
| `gpt2-large` | 36 | 1280 | 20 | 5120 |
| `gpt2-xl` | 48 | 1600 | 25 | 6400 |
| `gpt-neo-125m` | 12 | 768 | 12 | 3072 |
| `pythia-70m` | 6 | 512 | 8 | 2048 |
| `pythia-160m` | 12 | 768 | 12 | 3072 |
| `pythia-410m` | 24 | 1024 | 16 | 4096 |
| `attn-only-1l` | 1 | 512 | 8 | 0 |
| `attn-only-2l` | 2 | 512 | 8 | 0 |

## Examples

### Visualizing GPT-2 in a Notebook

```python
from transformer_viz import visualize

# Basic visualization
visualize("gpt2")

# Show only first 4 layers
visualize("gpt2", max_layers=4)

# Wider visualization
visualize("gpt2", width=800)
```

### Comparing Model Architectures

```python
from transformer_viz import visualize

# Small model
visualize("pythia-70m", max_layers=6)

# Larger model
visualize("gpt2-medium", max_layers=8)
```

### Working with TransformerLens

```python
from transformer_lens import HookedTransformer
from transformer_viz import visualize

# Load and visualize
model = HookedTransformer.from_pretrained("gpt2-small")
visualize(model)

# You can also access the model's config directly
from transformer_viz import InteractiveTransformerViz

viz = InteractiveTransformerViz()
viz.from_config(model.cfg)
viz.render()
viz.show()
```

### Saving as HTML

```python
from transformer_viz import visualize

viz = visualize("gpt2", max_layers=4)
viz.save_html("gpt2_architecture.html")
```

## Hook Visualization

Visualize which components have TransformerLens hooks attached by highlighting them in the diagram with a color-coded legend.

### Basic Hook Visualization

```python
from transformer_viz import visualize

def my_hook(tensor, hook):
    # Your hook logic here
    return tensor

def ablation_hook(tensor, hook):
    return tensor * 0

# Pass hooks as a list of (hook_name, hook_function) tuples
viz = visualize('gpt2-small', hooks=[
    ('blocks.1.attn.hook_pattern', my_hook),
    ('blocks.2.mlp.hook_post', ablation_hook),
])
```

The hook names follow TransformerLens conventions (see [TransformerLens documentation](https://neelnanda-io.github.io/TransformerLens/generated/code/transformer_lens.hook_points.html) for details). Common hook points include:
- `hook_pattern` - Attention pattern matrix after softmax, shape `[batch, n_heads, seq_q, seq_k]`
- `hook_post` - MLP activations after the activation function
- `hook_q`, `hook_k`, `hook_v` - Query, key, value vectors
- `hook_result` - Attention output before projection

This will:
- Highlight Layer 1 attention in one color
- Highlight Layer 2 MLP in another color
- Display a legend mapping colors to function names

### Head-Specific Highlighting

For attention hooks, you can specify which heads to highlight. This is useful when your hook only modifies specific attention heads:

```python
def head_ablation(tensor, hook):
    # Zero out only heads 0 and 2
    tensor[:, [0, 2], :, :] = 0
    return tensor

# Highlight only heads 0 and 2 in layer 1
viz = visualize('gpt2-small', hooks=[
    ('blocks.1.attn.hook_pattern', head_ablation, {'heads': [0, 2]}),
])
```

The third element of the tuple is an options dict with:
- `heads`: List of head indices to highlight (omit to highlight all heads)

### Supported Hook Points

| Hook Pattern | Component Highlighted |
|-------------|----------------------|
| `blocks.{L}.attn.*` | Attention heads in layer L |
| `blocks.{L}.mlp.*` | MLP block in layer L |
| `blocks.{L}.ln1.*` | Pre-attention LayerNorm in layer L |
| `blocks.{L}.ln2.*` | Pre-MLP LayerNorm in layer L |
| `blocks.{L}.hook_resid_*` | Residual stream in layer L |
| `hook_embed` | Embedding layer |
| `hook_pos_embed` | Positional embedding |
| `ln_final.*` | Final LayerNorm |

### Viewing Hooked Heads

When you hover over an attention layer, the expanded panel shows all heads in a grid. Hooked heads are highlighted with their assigned color, making it easy to see which heads are affected even when the main diagram only shows h₀, h₁, ..., hₙ₋₁.

## Diagram Components

The visualization shows:

- **Tokens**: Input token sequence
- **Embed**: Token embedding layer
- **Residual Stream**: Dashed vertical line on the right showing the residual connection
- **Attention Heads**: Individual heads shown as h₀, h₁, ..., hₙ₋₁ (hover to see all heads)
- **MLP**: Feed-forward network block (hover to see dimensions)
- **+ Circles**: Points where outputs are added back to the residual stream
- **Unembed**: Output projection layer
- **Logits**: Final output logits

## API Reference

### `visualize(model_or_config, max_layers=None, width=600, config=None, hooks=None)`

Quick function to visualize a transformer.

**Arguments:**
- `model_or_config`: HookedTransformer, config dict, or model name string
- `max_layers`: Maximum layers to show initially (default: 8)
- `width`: Width in pixels (default: 600)
- `config`: Optional VisualizationConfig
- `hooks`: Optional list of hook specifications to highlight. Each spec is either:
  - `(hook_name, hook_function)` - highlights all heads for attention hooks
  - `(hook_name, hook_function, {'heads': [...]})` - highlights specific heads

**Returns:** InteractiveTransformerViz instance

### `InteractiveTransformerViz`

Main visualization class.

**Methods:**
- `from_model(model)`: Load from TransformerLens HookedTransformer
- `from_config(cfg)`: Load from TransformerLens config
- `from_dict(config)`: Load from dictionary
- `from_pretrained(model_name)`: Load from pretrained model name
- `render(max_layers=None, width=600, hooks=None)`: Render the visualization with optional hook highlighting
- `show()`: Display in Jupyter notebook
- `save_html(filepath)`: Save as standalone HTML file

### `ModelArchitecture`

Dataclass containing model architecture specifications.

**Fields:**
- `n_layers`: Number of transformer layers
- `d_model`: Model dimension (residual stream width)
- `n_heads`: Number of attention heads
- `d_head`: Dimension per attention head
- `d_mlp`: MLP hidden dimension (0 for attention-only models)
- `d_vocab`: Vocabulary size
- `n_ctx`: Context length (optional)
- `model_name`: Name of the model

## License

MIT

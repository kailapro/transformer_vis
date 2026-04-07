"""
Interactive transformer architecture visualization for Jupyter notebooks.

Renders as HTML/SVG with JavaScript for hover interactions,
styled after the Anthropic Transformer Circuits diagrams.
"""

import html
import json
import uuid
from typing import Optional, Any, Dict, List, Tuple, Callable
from dataclasses import dataclass

from .model_adapter import ModelArchitecture, TransformerLensAdapter
from .config import VisualizationConfig
from .hook_parser import process_hooks


def _generate_html(
    arch: ModelArchitecture,
    config: VisualizationConfig,
    max_layers: Optional[int] = None,
    width: int = 600,
    height: Optional[int] = None,
    initially_collapsed: bool = True,
    hooks: Optional[List[Tuple[str, Callable]]] = None,
) -> str:
    """Generate the interactive HTML/SVG visualization."""

    viz_id = f"transformer_viz_{uuid.uuid4().hex[:8]}"

    total_layers = arch.n_layers
    default_max = 8
    initial_layers = min(total_layers, max_layers if max_layers else default_max)
    can_expand = total_layers > initial_layers

    # Calculate dimensions - flow is bottom to top
    layer_height = 280  # Increased to accommodate LayerNorm blocks
    header_height = 220  # Space for logits + final LN at top
    footer_height = 160  # Space for tokens/embed + pos embed at bottom

    # Layout constants - blocks on LEFT, residual on RIGHT
    residual_x = 450  # Residual stream x position (right side) - moved right for LN
    block_center_x = 200  # Center of blocks (left side)
    block_width = 200
    block_height = 40
    ln_width = 40  # LayerNorm block width
    ln_height = 25  # LayerNorm block height
    head_size = 50
    head_gap = 8
    add_circle_radius = 14

    # Colors - warm tan/beige like the Anthropic diagram
    block_color = "#E8D5B7"  # Warm tan
    block_border = "#C4A77D"  # Darker tan border
    text_color = "#4A4A4A"
    residual_color = "#888888"
    add_circle_color = "#FFFFFF"
    add_circle_border = "#CCCCCC"

    html_template = f'''
<div id="{viz_id}" class="transformer-viz-container" style="font-family: 'Times New Roman', serif; position: relative;">
  <style>
    #{viz_id} {{
      background: white;
      border-radius: 4px;
      padding: 20px;
      display: inline-block;
    }}
    #{viz_id} .block {{
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    #{viz_id} .block:hover {{
      filter: brightness(0.95);
    }}
    #{viz_id} .block-label {{
      font-size: 14px;
      fill: {text_color};
      pointer-events: none;
      font-family: 'Times New Roman', serif;
    }}
    #{viz_id} .block-label-italic {{
      font-size: 16px;
      font-style: italic;
      fill: {text_color};
      pointer-events: none;
      font-family: 'Times New Roman', serif;
    }}
    #{viz_id} .dim-label {{
      font-size: 13px;
      font-style: italic;
      fill: {text_color};
      pointer-events: none;
      font-family: 'Times New Roman', serif;
    }}
    #{viz_id} .residual-label {{
      font-size: 13px;
      font-style: italic;
      fill: {residual_color};
      font-family: 'Times New Roman', serif;
    }}
    #{viz_id} .layer-label {{
      font-size: 12px;
      fill: {text_color};
      font-family: 'Times New Roman', serif;
    }}
    #{viz_id} .title {{
      font-size: 16px;
      font-weight: normal;
      fill: {text_color};
      font-family: 'Times New Roman', serif;
    }}
    #{viz_id} .residual-stream {{
      stroke: {residual_color};
      stroke-width: 1.5;
      stroke-dasharray: 4,4;
      fill: none;
    }}
    #{viz_id} .flow-line {{
      stroke: {text_color};
      stroke-width: 1.5;
      fill: none;
    }}
    #{viz_id} .flow-arrow {{
      stroke: {text_color};
      stroke-width: 1.5;
      fill: none;
      marker-end: url(#{viz_id}-arrowhead);
    }}
    #{viz_id} .add-circle {{
      fill: {add_circle_color};
      stroke: {add_circle_border};
      stroke-width: 1.5;
    }}
    #{viz_id} .add-symbol {{
      font-size: 16px;
      fill: {text_color};
      text-anchor: middle;
      dominant-baseline: central;
      font-family: 'Times New Roman', serif;
    }}
    #{viz_id} .ln-block {{
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    #{viz_id} .ln-block:hover {{
      filter: brightness(0.95);
    }}
    #{viz_id} .ln-label {{
      font-size: 11px;
      fill: {text_color};
      pointer-events: none;
      font-family: 'Times New Roman', serif;
    }}
    #{viz_id} .attention-expanded {{
      display: none;
      position: absolute;
      background: white;
      border: 1px solid {block_border};
      border-radius: 4px;
      padding: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      z-index: 100;
    }}
    #{viz_id} .attention-expanded.visible {{
      display: block;
    }}
    #{viz_id} .head-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      max-width: 280px;
    }}
    #{viz_id} .attention-head {{
      width: 36px;
      height: 36px;
      border-radius: 4px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      font-family: 'Times New Roman', serif;
      font-style: italic;
      cursor: pointer;
      transition: transform 0.15s ease;
      background: {block_color};
      border: 1px solid {block_border};
      color: {text_color};
    }}
    #{viz_id} .attention-head:hover {{
      transform: scale(1.1);
    }}
    #{viz_id} .expanded-title {{
      font-size: 14px;
      color: {text_color};
      margin-bottom: 8px;
      font-family: 'Times New Roman', serif;
    }}
    #{viz_id} .expanded-dims {{
      font-size: 12px;
      color: {residual_color};
      margin-top: 8px;
      font-family: 'Times New Roman', serif;
    }}
    #{viz_id} .mlp-expanded {{
      display: none;
      position: absolute;
      background: white;
      min-width: 360px;
      border: 1px solid {block_border};
      border-radius: 4px;
      padding: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      z-index: 100;
    }}
    #{viz_id} .mlp-expanded.visible {{
      display: block;
    }}
    #{viz_id} .residual-expanded {{
      display: none;
      position: absolute;
      background: white;
      border: 1px solid {block_border};
      border-radius: 4px;
      padding: 12px;
      min-width: 240px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      z-index: 100;
    }}
    #{viz_id} .residual-expanded.visible {{
      display: block;
    }}
    #{viz_id} .ln-expanded {{
      display: none;
      position: absolute;
      background: white;
      border: 1px solid {block_border};
      border-radius: 4px;
      padding: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      z-index: 100;
    }}
    #{viz_id} .ln-expanded.visible {{
      display: block;
    }}
    #{viz_id} .tokens-expanded {{
      display: none;
      position: absolute;
      background: white;
      border: 1px solid {block_border};
      border-radius: 4px;
      padding: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      z-index: 100;
      min-width: 220px;
    }}
    #{viz_id} .tokens-expanded.visible {{
      display: block;
    }}
    #{viz_id} .embed-expanded {{
      display: none;
      position: absolute;
      background: white;
      border: 1px solid {block_border};
      border-radius: 4px;
      padding: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      z-index: 100;
      min-width: 220px;
    }}
    #{viz_id} .embed-expanded.visible {{
      display: block;
    }}
    #{viz_id} .pos-embed-expanded {{
      display: none;
      position: absolute;
      background: white;
      border: 1px solid {block_border};
      border-radius: 4px;
      padding: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      z-index: 100;
    }}
    #{viz_id} .pos-embed-expanded.visible {{
      display: block;
    }}
    #{viz_id} .residual-hover {{
      cursor: pointer;
    }}
    #{viz_id} .residual-hover:hover .residual-stream {{
      stroke-width: 2.5;
    }}
    #{viz_id} .expand-btn {{
      cursor: pointer;
    }}
    #{viz_id} .expand-btn:hover rect {{
      fill: #d0c0a0;
    }}
    #{viz_id} .layer-box {{
      fill: none;
      stroke: #ddd;
      stroke-width: 1;
      stroke-dasharray: 4,4;
      rx: 8;
    }}
    #{viz_id} .layer-label-bg {{
      font-size: 11px;
      fill: #999;
      font-family: 'Times New Roman', serif;
    }}
    #{viz_id} .internal-diagram {{
      margin-top: 10px;
      padding: 10px;
      background: #fafafa;
      border-radius: 4px;
      border: 1px solid #e0e0e0;
    }}
    #{viz_id} .internal-flow {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      flex-wrap: nowrap;
      font-size: 12px;
      font-family: 'Times New Roman', serif;
    }}
    #{viz_id} .internal-box {{
      padding: 6px 10px;
      background: #E8D5B7;
      border: 1px solid #C4A77D;
      border-radius: 4px;
      font-size: 11px;
    }}
    #{viz_id} .internal-op {{
      padding: 4px 8px;
      background: #ffffff;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 11px;
      font-style: italic;
    }}
    #{viz_id} .internal-arrow {{
      color: #888;
    }}
    #{viz_id} .dim-info {{
      font-size: 10px;
      color: #888;
      text-align: center;
      margin-top: 2px;
    }}
    #{viz_id} .block.highlighted rect {{
      stroke-width: 2;
    }}
    #{viz_id} .ln-block.highlighted rect {{
      stroke-width: 2;
    }}
    #{viz_id} .learn-mode-btn {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 5px 10px;
      font-family: 'Times New Roman', serif;
      font-size: 12px;
      cursor: pointer;
      color: #666;
    }}
    #{viz_id} .learn-mode-btn:hover {{
      background: #f5f5f5;
      border-color: #C4A77D;
    }}
    #{viz_id} .learn-mode-btn.active {{
      background: #E8D5B7;
      border-color: #C4A77D;
      color: #4A4A4A;
    }}
    #{viz_id} .learn-content {{
      display: none;
      font-size: 12px;
      color: #666;
      font-family: 'Times New Roman', serif;
      margin-top: 8px;
      line-height: 1.5;
      border-top: 1px solid #eee;
      padding-top: 8px;
      max-width: 240px;
    }}
    #{viz_id}.learn-mode .learn-content {{
      display: block;
    }}
    #{viz_id} .learn-toggle-btn {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 5px 10px;
      font-family: 'Times New Roman', serif;
      font-size: 12px;
      cursor: pointer;
      color: #666;
    }}
    #{viz_id} .learn-toggle-btn:hover {{
      background: #f5f5f5;
      border-color: #C4A77D;
    }}
    #{viz_id} .learn-panel {{
      display: none;
      position: absolute;
      top: 50px;
      right: 10px;
      width: 260px;
      background: white;
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 12px 14px;
      font-family: 'Times New Roman', serif;
      font-size: 12px;
      z-index: 50;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      max-height: 80vh;
      overflow-y: auto;
    }}
    #{viz_id} .learn-panel.visible {{
      display: block;
    }}
    #{viz_id} .learn-panel h3 {{
      margin: 0 0 10px 0;
      font-size: 13px;
      font-weight: bold;
      color: {text_color};
      border-bottom: 1px solid #eee;
      padding-bottom: 6px;
    }}
    #{viz_id} .learn-entry {{
      margin: 7px 0;
      line-height: 1.5;
    }}
    #{viz_id} .learn-term {{
      font-weight: bold;
      color: {text_color};
    }}
    #{viz_id} .learn-swatch {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 2px;
      border: 1px solid #aaa;
      vertical-align: middle;
      margin-right: 3px;
    }}
    #{viz_id} .learn-def {{
      color: #666;
    }}
    #{viz_id} .learn-section {{
      font-size: 10px;
      font-weight: bold;
      text-transform: uppercase;
      color: #aaa;
      letter-spacing: 0.5px;
      margin-top: 12px;
      margin-bottom: 4px;
    }}
    #{viz_id} .top-legend {{
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 7px 12px;
      border-bottom: 1px solid #eee;
      margin-bottom: 8px;
      font-family: 'Times New Roman', serif;
      font-size: 12px;
      color: {text_color};
      flex-wrap: wrap;
    }}
    #{viz_id} .legend-color-item {{
      display: flex;
      align-items: center;
      gap: 5px;
    }}
    #{viz_id} .legend-swatch {{
      display: inline-block;
      width: 13px;
      height: 13px;
      border-radius: 3px;
      flex-shrink: 0;
    }}
    #{viz_id} .legend-divider {{
      width: 1px;
      height: 14px;
      background: #ddd;
      margin: 0 2px;
    }}
    #{viz_id} .legend-hook-label {{
      font-family: monospace;
      font-size: 11px;
    }}
    #{viz_id}-hook-items {{
      display: flex;
      flex-direction: column;
      gap: 3px;
      border-left: 1px solid #ddd;
      padding-left: 10px;
      margin-left: 4px;
    }}
  </style>

  <div class="top-legend">
    <div class="legend-color-item">
      <span class="legend-swatch" style="background:#E8D5B7; border:1px solid #C4A77D;"></span>
      <span>Learned</span>
    </div>
    <div class="legend-color-item">
      <span class="legend-swatch" style="background:#E0E0E0; border:1px solid #B0B0B0;"></span>
      <span>I/O</span>
    </div>
    <div class="legend-color-item">
      <span class="legend-swatch" style="background:#FFFFFF; border:1px solid #CCCCCC;"></span>
      <span>Operations</span>
    </div>
    <div id="{viz_id}-hook-items"></div>
    <div style="flex:1"></div>
    <button class="learn-mode-btn" id="{viz_id}-learn-mode-btn" onclick="toggleLearnMode('{viz_id}')">Learn Mode: OFF</button>
    <button class="learn-toggle-btn" onclick="toggleLearnPanel('{viz_id}')">? Learn More</button>
  </div>
  <div id="{viz_id}-svg-container">
  </div>
  <div id="{viz_id}-learn-panel" class="learn-panel">
    <h3>Component Guide</h3>
    <div class="learn-section">Colors</div>
    <div class="learn-entry">
      <span class="learn-swatch" style="background:#E8D5B7; border-color:#C4A77D;"></span>
      <span class="learn-term">Tan</span> <span class="learn-def">— learned parameters (weights that are trained)</span>
    </div>
    <div class="learn-entry">
      <span class="learn-swatch" style="background:#E0E0E0; border-color:#B0B0B0;"></span>
      <span class="learn-term">Gray</span> <span class="learn-def">— I/O: tokens (model input), logits (model output), residual stream (I/O between layers)</span>
    </div>
    <div class="learn-entry">
      <span class="learn-swatch" style="background:#FFFFFF; border-color:#CCCCCC;"></span>
      <span class="learn-term">White</span> <span class="learn-def">— pure operations: ⊕ addition (no learned parameters, just x + y)</span>
    </div>
    <div class="learn-section">Components</div>
    <div class="learn-entry">
      <span class="learn-term">Tokens</span>
      <span class="learn-def">— the input sequence. Each token is a piece of text (a word, subword chunk, or character) from the vocabulary of {arch.d_vocab:,} items.</span>
    </div>
    <div class="learn-entry">
      <span class="learn-swatch" style="background:#E8D5B7; border-color:#C4A77D;"></span>
      <span class="learn-term">Embed</span>
      <span class="learn-def">— maps each token ID to a {arch.d_model}-dim vector (a learned lookup table).</span>
    </div>
    {'<div class="learn-entry"><span class="learn-swatch" style="background:#E8D5B7; border-color:#C4A77D;"></span><span class="learn-term">Pos Embed</span><span class="learn-def"> — adds learned position information to each token\'s vector so the model knows word order.</span></div>' if arch.has_positional_embedding and arch.pos_embed_type != 'rotary' else ''}
    <div class="learn-entry">
      <span class="learn-swatch" style="background:none; border: 1.5px dashed #888; border-radius:0;"></span>
      <span class="learn-term">Residual stream</span>
      <span class="learn-def">— a {arch.d_model}-dim vector at each position that flows through all layers. Every component reads from it and writes back to it via addition (⊕).</span>
    </div>
    <div class="learn-entry">
      <span class="learn-swatch" style="background:#E8D5B7; border-color:#C4A77D;"></span>
      <span class="learn-term">LN (LayerNorm)</span>
      <span class="learn-def">— normalizes the residual stream before each major component. Has small learned scale/bias parameters.</span>
    </div>
    <div class="learn-entry">
      <span class="learn-swatch" style="background:#E8D5B7; border-color:#C4A77D;"></span>
      <span class="learn-term">Attention heads (h₀…h{arch.n_heads - 1})</span>
      <span class="learn-def">— {arch.n_heads} parallel heads per layer. Each learns to route information between token positions using queries (Q), keys (K), and values (V). Each head works in {arch.d_head} dimensions.</span>
    </div>
    <div class="learn-entry">
      <span class="learn-term">⊕</span>
      <span class="learn-def">— addition point. The output of a component is added back into the residual stream, not replacing it.</span>
    </div>
    {'<div class="learn-entry"><span class="learn-swatch" style="background:#E8D5B7; border-color:#C4A77D;"></span><span class="learn-term">MLP</span><span class="learn-def"> — a two-layer feed-forward network applied at each position independently. Expands to ' + str(arch.d_mlp) + ' dims then projects back to ' + str(arch.d_model) + '. Thought to store factual knowledge.</span></div>' if arch.d_mlp > 0 else ''}
    <div class="learn-entry">
      <span class="learn-swatch" style="background:#E8D5B7; border-color:#C4A77D;"></span>
      <span class="learn-term">Unembed</span>
      <span class="learn-def">— projects the final residual stream vector to logits over {arch.d_vocab_out or arch.d_vocab:,} vocabulary items.</span>
    </div>
    <div class="learn-entry">
      <span class="learn-term">Logits</span>
      <span class="learn-def">— raw output scores over the vocabulary. Apply softmax to get next-token probabilities.</span>
    </div>
    <div class="learn-section">Dimensions</div>
    <div class="learn-entry"><span class="learn-term">d_model</span> = {arch.d_model} <span class="learn-def">— width of the residual stream</span></div>
    <div class="learn-entry"><span class="learn-term">n_layers</span> = {arch.n_layers} <span class="learn-def">— number of transformer blocks</span></div>
    <div class="learn-entry"><span class="learn-term">n_heads</span> = {arch.n_heads} <span class="learn-def">— attention heads per layer</span></div>
    <div class="learn-entry"><span class="learn-term">d_head</span> = {arch.d_head} <span class="learn-def">— dimension per attention head</span></div>
    {'<div class="learn-entry"><span class="learn-term">d_mlp</span> = ' + str(arch.d_mlp) + ' <span class="learn-def">— MLP hidden dimension (' + str(arch.d_mlp // arch.d_model) + '× d_model)</span></div>' if arch.d_mlp > 0 else ''}
  </div>
'''

    # Add expanded panels for all layers
    for layer_idx in range(total_layers):
        html_template += f'''
  <div id="{viz_id}-attn-expanded-{layer_idx}" class="attention-expanded">
    <div class="expanded-title">Layer {layer_idx} Attention ({arch.attn_type.replace('-', ' ').title()})</div>
    <div class="expanded-dims">
      {arch.n_heads} heads{f' ({arch.n_key_value_heads} KV heads)' if arch.n_key_value_heads and arch.n_key_value_heads != arch.n_heads else ''} × d_head={arch.d_head}
      {f'<br><em>+ RoPE</em>' if arch.pos_embed_type == 'rotary' else ''}
    </div>
    <div class="internal-diagram">
      <div class="internal-flow">
        <span class="internal-box">x</span>
        <span class="internal-arrow">→</span>
        <span class="internal-box">W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub></span>
        <span class="internal-arrow">→</span>
        <span class="internal-box">Q, K, V</span>
        {f'<span class="internal-arrow">→</span><span class="internal-op">RoPE</span>' if arch.pos_embed_type == 'rotary' else ''}
      </div>
      <div class="dim-info">[{arch.d_model}] → [{arch.n_heads}×{arch.d_head}]</div>
      <div class="internal-flow" style="margin-top: 8px;">
        <span class="internal-op">QK<sup>T</sup>/√d</span>
        <span class="internal-arrow">→</span>
        <span class="internal-op">softmax</span>
        <span class="internal-arrow">→</span>
        <span class="internal-box">A</span>
        <span class="internal-arrow">→</span>
        <span class="internal-op">A·V</span>
        <span class="internal-arrow">→</span>
        <span class="internal-box">W<sub>O</sub></span>
      </div>
      <div class="dim-info">[{arch.n_heads}×{arch.d_head}] → [{arch.d_model}]</div>
    </div>
    <div class="learn-content">
      Each head learns to route information between token positions. The attention pattern (A) tells each token how much to "look at" every other token. One head might track the subject of a verb; another might match opening and closing brackets.
    </div>
    <div class="head-grid" style="margin-top: 10px;">
'''
        for h in range(arch.n_heads):
            html_template += f'''      <div class="attention-head" data-layer="{layer_idx}" data-head="{h}">h<sub>{h}</sub></div>
'''
        html_template += f'''    </div>
  </div>

  <div id="{viz_id}-mlp-expanded-{layer_idx}" class="mlp-expanded">
    <div class="expanded-title">Layer {layer_idx} MLP{' (Gated)' if arch.mlp_type == 'gated' else ''}</div>
    <div class="expanded-dims">
      {arch.d_model} → {arch.d_mlp} → {arch.d_model}
    </div>
    <div class="internal-diagram">
      <div class="internal-flow">
        <span class="internal-box">x</span>
        <span class="internal-arrow">→</span>
        {'<span class="internal-box">W<sub>gate</sub>, W<sub>up</sub></span><span class="internal-arrow">→</span><span class="internal-op">' + arch.activation.upper() + ' ⊙</span>' if arch.mlp_type == 'gated' else '<span class="internal-box">W<sub>in</sub></span><span class="internal-arrow">→</span><span class="internal-op">' + arch.activation.upper() + '</span>'}
        <span class="internal-arrow">→</span>
        <span class="internal-box">W<sub>{'down' if arch.mlp_type == 'gated' else 'out'}</sub></span>
        <span class="internal-arrow">→</span>
        <span class="internal-box">y</span>
      </div>
      <div class="dim-info">[{arch.d_model}] → [{arch.d_mlp}] → [{arch.d_model}]</div>
    </div>
    <div class="learn-content">
      A two-layer feed-forward network applied at each token position independently. Expands to {arch.d_mlp} dims, applies a non-linear transformation ({arch.activation.upper()}), then projects back to {arch.d_model}. Thought to store factual knowledge — individual neurons often fire for specific concepts like "capital cities" or "past tense verbs."
    </div>
  </div>

  <div id="{viz_id}-ln-attn-expanded-{layer_idx}" class="ln-expanded">
    <div class="expanded-title">Layer {layer_idx} Pre-Attention LayerNorm</div>
    <div class="expanded-dims">
      Rescales the {arch.d_model}-dim residual stream vector to mean=0, var=1,
      then applies learned scale (γ) and bias (β) to restore useful signal.
    </div>
    <div class="learn-content">
      Stabilizes training by pulling activations back to a consistent range before the attention heads read from the residual stream.
    </div>
  </div>

  <div id="{viz_id}-ln-mlp-expanded-{layer_idx}" class="ln-expanded">
    <div class="expanded-title">Layer {layer_idx} Pre-MLP LayerNorm</div>
    <div class="expanded-dims">
      Rescales the {arch.d_model}-dim residual stream vector to mean=0, var=1,
      then applies learned scale (γ) and bias (β) to restore useful signal.
    </div>
    <div class="learn-content">
      Stabilizes training by pulling activations back to a consistent range before the MLP reads from the residual stream.
    </div>
  </div>
'''

    # Add residual stream panel
    html_template += f'''
  <div id="{viz_id}-residual-expanded" class="residual-expanded">
    <div class="expanded-title">Residual Stream</div>
    <div class="expanded-dims">
      d_model = {arch.d_model}
    </div>
    <div class="learn-content">
      The shared {arch.d_model}-dim vector at each token position that flows through all layers. Every component reads from it and adds its output back to it — nothing is ever replaced, only accumulated. Think of it as the I/O bus between layers.
    </div>
  </div>

  <div id="{viz_id}-ln-final-expanded" class="ln-expanded">
    <div class="expanded-title">Final LayerNorm</div>
    <div class="expanded-dims">
      Rescales the {arch.d_model}-dim residual stream vector to mean=0, var=1,
      then applies learned scale (γ) and bias (β) to restore useful signal.
    </div>
    <div class="learn-content">
      Applied once before the unembedding layer. Same function as the per-layer LayerNorms — normalizes activations before the final projection to vocabulary logits.
    </div>
  </div>

  <div id="{viz_id}-tokens-expanded" class="tokens-expanded">
    <div class="expanded-title">Tokens</div>
    <div class="expanded-dims">
      The input sequence — up to n_ctx={arch.n_ctx} tokens
    </div>
    <div class="learn-content">
      Each token is a piece of text (a word, subword, or character) from the model's vocabulary of {arch.d_vocab:,} items. The model never sees raw text — it sees integer IDs that index into the vocabulary. These IDs are what get passed to the embedding layer.
    </div>
  </div>

  <div id="{viz_id}-embed-expanded" class="embed-expanded">
    <div class="expanded-title">Token Embedding</div>
    <div class="expanded-dims">
      Lookup table: one learned vector per token<br>
      d_vocab × d_model<br>
      <span style="font-size:14px;">{arch.d_vocab:,} × {arch.d_model}</span>
    </div>
    <div class="learn-content">
      A lookup table with one learned vector per token. During training the model learns which "fingerprint" best captures each token's meaning, grammar role, and typical contexts.
    </div>
  </div>

  <div id="{viz_id}-unembed-expanded" class="embed-expanded">
    <div class="expanded-title">Unembedding</div>
    <div class="expanded-dims">
      d_model × d_vocab<br>
      <span style="font-size:14px;">{arch.d_model} × {arch.d_vocab_out or arch.d_vocab:,}</span>
    </div>
    <div class="learn-content">
      The reverse of the embedding — projects the final residual stream vector into a score for every token in the vocabulary. A linear layer with no activation function.
    </div>
  </div>

  <div id="{viz_id}-logits-expanded" class="embed-expanded">
    <div class="expanded-title">Logits</div>
    <div class="expanded-dims">
      {arch.d_vocab_out or arch.d_vocab:,} scores (one per vocabulary token)
    </div>
    <div class="learn-content">
      Raw output scores over the entire vocabulary. Apply softmax to convert to probabilities — the highest-scoring token is the model's predicted next token.
    </div>
  </div>

  <div id="{viz_id}-pos-embed-expanded" class="pos-embed-expanded">
    <div class="expanded-title">Positional Embedding</div>
    <div class="expanded-dims">
      {f'n_ctx × d_model<br><span style="font-size:14px;">{arch.n_ctx} × {arch.d_model}</span>' if arch.pos_embed_type == 'learned' else f'Applied in attention (RoPE)' if arch.pos_embed_type == 'rotary' else f'Type: {arch.pos_embed_type}'}
    </div>
    <div class="learn-content">
      Adds a learned position signal to each token. Without it, the model can't distinguish word order — "dog bites man" would look identical to "man bites dog."
    </div>
  </div>
'''

    html_template += f'''
</div>

<script>
(function() {{
  const vizId = '{viz_id}';
  const totalLayers = {total_layers};
  const initialLayers = {initial_layers};
  const canExpand = {str(can_expand).lower()};
  const layerHeight = {layer_height};
  const headerHeight = {header_height};
  const footerHeight = {footer_height};
  const width = {width};
  const residualX = {residual_x};
  const blockCenterX = {block_center_x};
  const blockWidth = {block_width};
  const blockHeight = {block_height};
  const headSize = {head_size};
  const headGap = {head_gap};
  const addCircleRadius = {add_circle_radius};
  const hasMLP = {str(arch.d_mlp > 0).lower()};
  const nHeads = {arch.n_heads};

  const archInfo = {{
    modelName: '{html.escape(arch.model_name)}',
    dModel: {arch.d_model},
    nLayers: {arch.n_layers},
    nHeads: {arch.n_heads},
    dHead: {arch.d_head},
    dMlp: {arch.d_mlp},
    dVocab: {arch.d_vocab},
    dVocabOut: {arch.d_vocab_out or arch.d_vocab},
    hasPosEmbed: {str(arch.has_positional_embedding).lower()},
    posEmbedType: '{arch.pos_embed_type}',
    activation: '{arch.activation}',
    mlpType: '{arch.mlp_type}',
    attnType: '{arch.attn_type}'
  }};

  const hookData = {json.dumps(process_hooks(hooks) if hooks else {'hooks': [], 'legend': []})};

  // Helper function to get highlight color for a component (non-attention)
  function getComponentColor(layerIdx, componentType, defaultColor) {{
    const hook = hookData.hooks.find(h =>
      h.layer === layerIdx && h.component === componentType);
    return hook ? hook.color : defaultColor;
  }}

  // Helper to check if a component has a hook (non-attention)
  function hasHook(layerIdx, componentType) {{
    return hookData.hooks.some(h =>
      h.layer === layerIdx && h.component === componentType);
  }}

  // Helper to get highlight color for a specific attention head
  function getHeadColor(layerIdx, headIdx, defaultColor) {{
    const hook = hookData.hooks.find(h =>
      h.layer === layerIdx &&
      h.component === 'attention' &&
      (h.heads === null || h.heads.includes(headIdx)));
    return hook ? hook.color : defaultColor;
  }}

  // Helper to check if a specific attention head has a hook
  function headHasHook(layerIdx, headIdx) {{
    return hookData.hooks.some(h =>
      h.layer === layerIdx &&
      h.component === 'attention' &&
      (h.heads === null || h.heads.includes(headIdx)));
  }}

  let isExpanded = false;
  let pinnedPanelId = null;

  function generateSVG(numLayers, showEllipsis, hiddenCount) {{
    // Extra space at top for unembed/logits
    const topPadding = 180;
    const totalHeight = topPadding + numLayers * layerHeight + footerHeight + (showEllipsis ? 50 : 0);

    let svg = `
    <svg width="${{width}}" height="${{totalHeight}}" viewBox="0 0 ${{width}} ${{totalHeight}}">
      <defs>
        <marker id="${{vizId}}-arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#4A4A4A" />
        </marker>
      </defs>
    `;

    // Calculate Y positions (bottom to top, so we start from totalHeight)
    let currentY = totalHeight - footerHeight;

    // === BOTTOM: Tokens, Embed, and Positional Embedding ===
    const tokensY = totalHeight - 40;
    const embedY = totalHeight - 95;
    const posEmbedAddY = embedY - blockHeight/2 - 30;  // Where pos embed is added
    const embedOutY = posEmbedAddY - 20;

    // Tokens block (centered on residual stream)
    svg += `
      <g class="block"
         onmouseenter="showPanel('${{vizId}}', 'tokens-expanded', this)"
         onmouseleave="hidePanel('${{vizId}}', 'tokens-expanded')">
        <rect x="${{residualX - blockWidth/2}}" y="${{tokensY - blockHeight/2}}"
              width="${{blockWidth}}" height="${{blockHeight}}"
              rx="4" fill="#E0E0E0" stroke="#B0B0B0" stroke-width="1" />
        <text x="${{residualX}}" y="${{tokensY + 4}}"
              text-anchor="middle" class="block-label">tokens</text>
      </g>
    `;

    // Arrow from tokens to embed
    svg += `
      <line x1="${{residualX}}" y1="${{tokensY - blockHeight/2}}"
            x2="${{residualX}}" y2="${{embedY + blockHeight/2 + 2}}"
            class="flow-arrow" />
    `;

    // Embed block (centered on residual stream)
    svg += `
      <g class="block"
         onmouseenter="showPanel('${{vizId}}', 'embed-expanded', this)"
         onmouseleave="hidePanel('${{vizId}}', 'embed-expanded')">
        <rect x="${{residualX - blockWidth/2}}" y="${{embedY - blockHeight/2}}"
              width="${{blockWidth}}" height="${{blockHeight}}"
              rx="4" fill="#E8D5B7" stroke="#C4A77D" stroke-width="1" />
        <text x="${{residualX}}" y="${{embedY + 4}}"
              text-anchor="middle" class="block-label">embed</text>
      </g>
    `;


    // Positional embedding block (only for models with learned positional embeddings)
    const posEmbedX = blockCenterX;
    if (archInfo.hasPosEmbed) {{
      // Arrow from embed up to + circle
      svg += `
        <line x1="${{residualX}}" y1="${{embedY - blockHeight/2}}"
              x2="${{residualX}}" y2="${{posEmbedAddY + addCircleRadius + 2}}"
              class="flow-line" />
      `;
      svg += `
        <g class="block"
           onmouseenter="showPanel('${{vizId}}', 'pos-embed-expanded', this)"
           onmouseleave="hidePanel('${{vizId}}', 'pos-embed-expanded')">
          <rect x="${{posEmbedX - 70}}" y="${{posEmbedAddY - 15}}"
                width="140" height="30"
                rx="4" fill="#E8D5B7" stroke="#C4A77D" stroke-width="1" />
          <text x="${{posEmbedX}}" y="${{posEmbedAddY + 4}}"
                text-anchor="middle" class="block-label">pos_embed</text>
        </g>
      `;

      // Arrow from pos_embed to + circle
      svg += `
        <line x1="${{posEmbedX + 70}}" y1="${{posEmbedAddY}}"
              x2="${{residualX - addCircleRadius - 2}}" y2="${{posEmbedAddY}}"
              class="flow-line" />
      `;

      // + circle for positional embedding addition
      svg += `
        <circle cx="${{residualX}}" cy="${{posEmbedAddY}}" r="${{addCircleRadius}}" class="add-circle" />
        <text x="${{residualX}}" y="${{posEmbedAddY}}" class="add-symbol">+</text>
      `;

      // Arrow from + circle up to residual stream start
      svg += `
        <line x1="${{residualX}}" y1="${{posEmbedAddY - addCircleRadius}}"
              x2="${{residualX}}" y2="${{embedOutY}}"
              class="flow-arrow" />
      `;
    }} else {{
      // For RoPE/ALiBi models: direct arrow from embed to residual stream
      svg += `
        <line x1="${{residualX}}" y1="${{embedY - blockHeight/2}}"
              x2="${{residualX}}" y2="${{embedOutY}}"
              class="flow-arrow" />
      `;
    }}

    // x_0 label
    svg += `
      <text x="${{residualX + 15}}" y="${{embedOutY + 4}}" class="residual-label">x<tspan baseline-shift="sub" font-size="10">0</tspan></text>
    `;

    // Start of residual stream (dashed vertical line)
    const residualStartY = embedOutY;
    let residualEndY = headerHeight + 30;

    svg += `
      <g class="residual-hover"
         onmousemove="showResidual('${{vizId}}', event)"
         onmouseleave="hideResidual('${{vizId}}')">
        <line x1="${{residualX}}" y1="${{residualStartY}}" x2="${{residualX}}" y2="${{residualEndY}}"
              stroke="transparent" stroke-width="20" />
        <line x1="${{residualX}}" y1="${{residualStartY}}" x2="${{residualX}}" y2="${{residualEndY}}"
              class="residual-stream" />
      </g>
    `;

    // Track + circle positions
    let addCircles = [];

    // Current position moving up from bottom
    currentY = embedOutY - 40;

    // LayerNorm dimensions
    const lnWidth = 40;
    const lnHeight = 25;

    // === LAYERS (bottom to top) ===
    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {{

      // Calculate layer box dimensions first
      const layerBoxPadding = 15;
      const layerBoxBottom = currentY + 10;  // Bottom of this layer's box

      // === PRE-ATTENTION LAYERNORM + ATTENTION HEADS ===
      const headsY = currentY - 90;  // Moved up to make room for LN

      // Calculate heads layout - show h0, h1, ..., h(n-1)
      const showFirstHeads = 2;  // Show h0, h1
      const hasMoreHeads = nHeads > 3;
      const numHeadBoxes = hasMoreHeads ? 4 : nHeads;  // h0, h1, ..., h(last) or all if <=3
      const totalHeadsWidth = numHeadBoxes * headSize + (numHeadBoxes - 1) * headGap;
      const headsStartX = blockCenterX - totalHeadsWidth / 2;
      const headsEndX = headsStartX + totalHeadsWidth;

      // Y positions for this attention section
      const lnAttnY = headsY + headSize + 45;  // LN position
      const attnInputY = headsY + headSize + 15;  // Where horizontal line to heads is
      const attnOutputY = headsY - 20;  // Where arrows go up to
      const attnAddY = attnOutputY;  // Where + circle is

      addCircles.push({{ x: residualX, y: attnAddY }});

      // Arrow from residual down to LN level
      svg += `
        <line x1="${{residualX}}" y1="${{lnAttnY + 25}}"
              x2="${{residualX}}" y2="${{lnAttnY}}"
              class="flow-line" />
      `;

      // LN position (between residual and heads)
      const lnAttnX = residualX - 60;

      // Arrow from residual to LN
      svg += `
        <line x1="${{residualX}}" y1="${{lnAttnY}}"
              x2="${{lnAttnX + lnWidth/2 + 2}}" y2="${{lnAttnY}}"
              class="flow-arrow" />
      `;

      // Pre-attention LayerNorm block
      const ln1Color = getComponentColor(layerIdx, 'ln1', '#E8D5B7');
      const ln1Highlighted = hasHook(layerIdx, 'ln1');
      svg += `
        <g class="ln-block${{ln1Highlighted ? ' highlighted' : ''}}"
           data-component-id="layer-${{layerIdx}}-ln1"
           onmouseenter="showExpanded('${{vizId}}', 'ln-attn', ${{layerIdx}}, this)"
           onmouseleave="hideExpanded('${{vizId}}', 'ln-attn', ${{layerIdx}})">
          <rect x="${{lnAttnX - lnWidth/2}}" y="${{lnAttnY - lnHeight/2}}"
                width="${{lnWidth}}" height="${{lnHeight}}"
                rx="3" fill="${{ln1Color}}" stroke="${{ln1Highlighted ? '#666' : '#C4A77D'}}" stroke-width="1" />
          <text x="${{lnAttnX}}" y="${{lnAttnY + 4}}"
                text-anchor="middle" class="ln-label">LN</text>
        </g>
      `;

      // Arrow from LN down to horizontal attention input line
      svg += `
        <line x1="${{lnAttnX}}" y1="${{lnAttnY - lnHeight/2}}"
              x2="${{lnAttnX}}" y2="${{attnInputY}}"
              class="flow-line" />
        <line x1="${{lnAttnX}}" y1="${{attnInputY}}"
              x2="${{headsStartX + headSize/2}}" y2="${{attnInputY}}"
              class="flow-line" />
      `;

      // Draw attention heads: h0, h1, ..., h(n-1)
      // If more than 3 heads, show: h0, h1, ..., h(n-1)
      const headsToShow = hasMoreHeads
        ? [0, 1, '...', nHeads - 1]  // h0, h1, ..., h(last)
        : Array.from({{length: nHeads}}, (_, i) => i);  // all heads

      // Check if any hidden heads (behind '...') have hooks
      let dotsHighlighted = false;
      let dotsHookColor = '#E8D5B7';
      if (hasMoreHeads) {{
        for (let h = 2; h < nHeads - 1; h++) {{
          const c = getHeadColor(layerIdx, h, null);
          if (c) {{ dotsHighlighted = true; dotsHookColor = c; break; }}
        }}
      }}

      for (let i = 0; i < headsToShow.length; i++) {{
        const headIdx = headsToShow[i];
        const headX = headsStartX + i * (headSize + headGap);
        const headCenterX = headX + headSize/2;
        const isDots = headIdx === '...';

        // Get per-head color and highlight status
        const headColor = isDots ? dotsHookColor : getHeadColor(layerIdx, headIdx, '#E8D5B7');
        const headHighlighted = isDots ? dotsHighlighted : headHasHook(layerIdx, headIdx);

        // Arrow down into head (drops from the horizontal line)
        svg += `
          <line x1="${{headCenterX}}" y1="${{attnInputY}}"
                x2="${{headCenterX}}" y2="${{headsY + headSize + 2}}"
                class="flow-arrow" />
        `;

        // Extend horizontal line to next head (except for last)
        if (i < headsToShow.length - 1) {{
          const nextCenterX = headsStartX + (i + 1) * (headSize + headGap) + headSize/2;
          svg += `
            <line x1="${{headCenterX}}" y1="${{attnInputY}}"
                  x2="${{nextCenterX}}" y2="${{attnInputY}}"
                  class="flow-line" />
          `;
        }}

        // Head box
        if (isDots) {{
          svg += `
            <g class="block${{dotsHighlighted ? ' highlighted' : ''}}" data-type="attention" data-layer="${{layerIdx}}"
               data-component-id="layer-${{layerIdx}}-attention-dots"
               onmouseenter="showExpanded('${{vizId}}', 'attn', ${{layerIdx}}, this)"
               onmouseleave="hideExpanded('${{vizId}}', 'attn', ${{layerIdx}})">
              <rect x="${{headX}}" y="${{headsY}}"
                    width="${{headSize}}" height="${{headSize}}"
                    rx="4" fill="${{dotsHookColor}}" stroke="${{dotsHighlighted ? '#666' : '#C4A77D'}}" stroke-width="1" />
              <text x="${{headCenterX}}" y="${{headsY + headSize/2 + 5}}"
                    text-anchor="middle" class="block-label">...</text>
            </g>
          `;
        }} else {{
          svg += `
            <g class="block${{headHighlighted ? ' highlighted' : ''}}" data-type="attention" data-layer="${{layerIdx}}" data-head="${{headIdx}}"
               data-component-id="layer-${{layerIdx}}-attention-head-${{headIdx}}"
               onmouseenter="showExpanded('${{vizId}}', 'attn', ${{layerIdx}}, this)"
               onmouseleave="hideExpanded('${{vizId}}', 'attn', ${{layerIdx}})">
              <rect x="${{headX}}" y="${{headsY}}"
                    width="${{headSize}}" height="${{headSize}}"
                    rx="4" fill="${{headColor}}" stroke="${{headHighlighted ? '#666' : '#C4A77D'}}" stroke-width="1" />
              <text x="${{headCenterX}}" y="${{headsY + headSize/2 + 5}}"
                    text-anchor="middle" class="block-label-italic">h<tspan baseline-shift="sub" font-size="10">${{headIdx}}</tspan></text>
            </g>
          `;
        }}

        // Arrow up from head
        svg += `
          <line x1="${{headCenterX}}" y1="${{headsY}}"
                x2="${{headCenterX}}" y2="${{attnOutputY}}"
                class="flow-line" />
        `;
      }}

      // Horizontal merge line at top, then right to + circle
      svg += `
        <line x1="${{headsStartX + headSize/2}}" y1="${{attnOutputY}}"
              x2="${{headsEndX - headSize/2}}" y2="${{attnOutputY}}"
              class="flow-line" />
        <line x1="${{headsEndX - headSize/2}}" y1="${{attnOutputY}}"
              x2="${{residualX - addCircleRadius - 2}}" y2="${{attnAddY}}"
              class="flow-line" />
      `;

      // x_i+1 label
      svg += `
        <text x="${{residualX + 20}}" y="${{attnAddY + 5}}" class="residual-label">x<tspan baseline-shift="sub" font-size="10">${{layerIdx*2 + 1}}</tspan></text>
      `;

      currentY = attnAddY - 25;

      // === PRE-MLP LAYERNORM + MLP ===
      if (hasMLP) {{
        const mlpY = currentY - 90;  // Increased gap to accommodate LN
        const mlpOutputY = mlpY - 20;
        const mlpAddY = mlpOutputY;

        addCircles.push({{ x: residualX, y: mlpAddY }});

        // LN position for MLP
        const lnMlpY = mlpY + blockHeight + 45;
        const lnMlpX = residualX - 60;

        // Arrow from residual down to LN level
        svg += `
          <line x1="${{residualX}}" y1="${{lnMlpY + 25}}"
                x2="${{residualX}}" y2="${{lnMlpY}}"
                class="flow-line" />
        `;

        // Arrow from residual to LN
        svg += `
          <line x1="${{residualX}}" y1="${{lnMlpY}}"
                x2="${{lnMlpX + lnWidth/2 + 2}}" y2="${{lnMlpY}}"
                class="flow-arrow" />
        `;

        // Pre-MLP LayerNorm block
        const ln2Color = getComponentColor(layerIdx, 'ln2', '#E8D5B7');
        const ln2Highlighted = hasHook(layerIdx, 'ln2');
        svg += `
          <g class="ln-block${{ln2Highlighted ? ' highlighted' : ''}}"
             data-component-id="layer-${{layerIdx}}-ln2"
             onmouseenter="showExpanded('${{vizId}}', 'ln-mlp', ${{layerIdx}}, this)"
             onmouseleave="hideExpanded('${{vizId}}', 'ln-mlp', ${{layerIdx}})">
            <rect x="${{lnMlpX - lnWidth/2}}" y="${{lnMlpY - lnHeight/2}}"
                  width="${{lnWidth}}" height="${{lnHeight}}"
                  rx="3" fill="${{ln2Color}}" stroke="${{ln2Highlighted ? '#666' : '#C4A77D'}}" stroke-width="1" />
            <text x="${{lnMlpX}}" y="${{lnMlpY + 4}}"
                  text-anchor="middle" class="ln-label">LN</text>
          </g>
        `;

        // Arrow from LN up to MLP
        const mlpInputY = mlpY + blockHeight + 15;
        svg += `
          <line x1="${{lnMlpX}}" y1="${{lnMlpY - lnHeight/2}}"
                x2="${{lnMlpX}}" y2="${{mlpInputY}}"
                class="flow-line" />
          <line x1="${{lnMlpX}}" y1="${{mlpInputY}}"
                x2="${{blockCenterX}}" y2="${{mlpInputY}}"
                class="flow-line" />
          <line x1="${{blockCenterX}}" y1="${{mlpInputY}}"
                x2="${{blockCenterX}}" y2="${{mlpY + blockHeight + 2}}"
                class="flow-arrow" />
        `;

        // MLP block
        const mlpColor = getComponentColor(layerIdx, 'mlp', '#E8D5B7');
        const mlpHighlighted = hasHook(layerIdx, 'mlp');
        svg += `
          <g class="block${{mlpHighlighted ? ' highlighted' : ''}}" data-type="mlp" data-layer="${{layerIdx}}"
             data-component-id="layer-${{layerIdx}}-mlp"
             onmouseenter="showExpanded('${{vizId}}', 'mlp', ${{layerIdx}}, this)"
             onmouseleave="hideExpanded('${{vizId}}', 'mlp', ${{layerIdx}})">
            <rect x="${{blockCenterX - blockWidth/2}}" y="${{mlpY}}"
                  width="${{blockWidth}}" height="${{blockHeight}}"
                  rx="4" fill="${{mlpColor}}" stroke="${{mlpHighlighted ? '#666' : '#C4A77D'}}" stroke-width="1" />
            <text x="${{blockCenterX - 15}}" y="${{mlpY + blockHeight/2 + 5}}"
                  text-anchor="middle" class="block-label">MLP</text>
            <text x="${{blockCenterX + 30}}" y="${{mlpY + blockHeight/2 + 5}}"
                  text-anchor="middle" class="block-label-italic">m</text>
          </g>
        `;

        // Arrow up from MLP (centered), then right to + circle
        svg += `
          <line x1="${{blockCenterX}}" y1="${{mlpY}}"
                x2="${{blockCenterX}}" y2="${{mlpOutputY}}"
                class="flow-line" />
          <line x1="${{blockCenterX}}" y1="${{mlpOutputY}}"
                x2="${{residualX - addCircleRadius - 2}}" y2="${{mlpAddY}}"
                class="flow-line" />
        `;

        // x_i+2 label
        svg += `
          <text x="${{residualX + 20}}" y="${{mlpAddY + 5}}" class="residual-label">x<tspan baseline-shift="sub" font-size="10">${{layerIdx*2 + 2}}</tspan></text>
        `;

        currentY = mlpAddY - 25;
      }} else {{
        currentY = attnAddY - 25;
      }}

      // Draw layer box (dashed rectangle around the layer)
      const layerBoxTop = currentY + 5;  // Top of this layer's box
      const layerBoxLeft = 50;
      const layerBoxRight = residualX + 80;
      const layerBoxWidth = layerBoxRight - layerBoxLeft;
      const layerBoxHeight = layerBoxBottom - layerBoxTop;

      svg += `
        <rect x="${{layerBoxLeft}}" y="${{layerBoxTop}}"
              width="${{layerBoxWidth}}" height="${{layerBoxHeight}}"
              class="layer-box" />
        <text x="${{layerBoxLeft + 8}}" y="${{layerBoxTop + 15}}"
              class="layer-label-bg">Layer ${{layerIdx}}</text>
      `;
    }}

    // Ellipsis / expand button (positioned between layers and unembed)
    if (showEllipsis) {{
      const expandBtnY = currentY - 30;
      svg += `
        <g class="expand-btn" onclick="toggleExpand('${{vizId}}')">
          <rect x="${{blockCenterX - 80}}" y="${{expandBtnY}}" width="160" height="30"
                rx="4" fill="#E8D5B7" stroke="#C4A77D" stroke-width="1" />
          <text x="${{blockCenterX}}" y="${{expandBtnY + 20}}" text-anchor="middle"
                fill="#4A4A4A" font-size="12" style="font-family: 'Times New Roman', serif;">
            ${{isExpanded ? '▼ Collapse' : '▲ Show ' + hiddenCount + ' more layers'}}
          </text>
        </g>
      `;
      currentY = expandBtnY - 20;
    }}

    // === TOP: Final LN, Unembed and Logits ===
    // Position based on currentY to avoid overlap
    const finalLnY = currentY - 50;
    const unembedY = finalLnY - 55;
    const logitsY = unembedY - 55;

    // Continue residual stream line up to final LN level
    svg += `
      <line x1="${{residualX}}" y1="${{currentY}}"
            x2="${{residualX}}" y2="${{finalLnY + lnHeight/2 + 5}}"
            class="residual-stream" />
    `;

    // x_-1 label (final residual state)
    svg += `
      <text x="${{residualX + 15}}" y="${{finalLnY + lnHeight/2 + 10}}" class="residual-label">x<tspan baseline-shift="sub" font-size="10">-1</tspan></text>
    `;

    // Final LayerNorm block (centered on residual stream)
    svg += `
      <g class="ln-block"
         onmouseenter="showPanel('${{vizId}}', 'ln-final-expanded', this)"
         onmouseleave="hidePanel('${{vizId}}', 'ln-final-expanded')">
        <rect x="${{residualX - lnWidth/2}}" y="${{finalLnY - lnHeight/2}}"
              width="${{lnWidth}}" height="${{lnHeight}}"
              rx="3" fill="#E8D5B7" stroke="#C4A77D" stroke-width="1" />
        <text x="${{residualX}}" y="${{finalLnY + 4}}"
              text-anchor="middle" class="ln-label">LN</text>
      </g>
    `;

    // Arrow from final LN to unembed
    svg += `
      <line x1="${{residualX}}" y1="${{finalLnY - lnHeight/2}}"
            x2="${{residualX}}" y2="${{unembedY + blockHeight/2 + 2}}"
            class="flow-arrow" />
    `;

    // Unembed block (centered on residual stream)
    svg += `
      <g class="block"
         onmouseenter="showPanel('${{vizId}}', 'unembed-expanded', this)"
         onmouseleave="hidePanel('${{vizId}}', 'unembed-expanded')">
        <rect x="${{residualX - blockWidth/2}}" y="${{unembedY - blockHeight/2}}"
              width="${{blockWidth}}" height="${{blockHeight}}"
              rx="4" fill="#E8D5B7" stroke="#C4A77D" stroke-width="1" />
        <text x="${{residualX}}" y="${{unembedY + 4}}"
              text-anchor="middle" class="block-label">unembed</text>
      </g>
    `;

    // Arrow from unembed to logits
    svg += `
      <line x1="${{residualX}}" y1="${{unembedY - blockHeight/2}}"
            x2="${{residualX}}" y2="${{logitsY + blockHeight/2 + 2}}"
            class="flow-arrow" />
    `;

    // Logits block (centered on residual stream)
    svg += `
      <g class="block"
         onmouseenter="showPanel('${{vizId}}', 'logits-expanded', this)"
         onmouseleave="hidePanel('${{vizId}}', 'logits-expanded')">
        <rect x="${{residualX - blockWidth/2}}" y="${{logitsY - blockHeight/2}}"
              width="${{blockWidth}}" height="${{blockHeight}}"
              rx="4" fill="#E0E0E0" stroke="#B0B0B0" stroke-width="1" />
        <text x="${{residualX}}" y="${{logitsY + 4}}"
              text-anchor="middle" class="block-label">logits</text>
      </g>
    `;

    // Draw all + circles on top
    for (const circle of addCircles) {{
      svg += `
        <circle cx="${{circle.x}}" cy="${{circle.y}}" r="${{addCircleRadius}}" class="add-circle" />
        <text x="${{circle.x}}" y="${{circle.y}}" class="add-symbol">+</text>
      `;
    }}

    svg += '</svg>';
    return svg;
  }}

  function render() {{
    const container = document.getElementById(vizId + '-svg-container');
    if (isExpanded) {{
      container.innerHTML = generateSVG(totalLayers, false, 0);
    }} else {{
      const hiddenCount = totalLayers - initialLayers;
      container.innerHTML = generateSVG(initialLayers, canExpand && hiddenCount > 0, hiddenCount);
    }}

    // Populate hook color items into the top legend
    const hookItems = document.getElementById(vizId + '-hook-items');
    if (hookItems && hookData.legend.length > 0) {{
      let html = '';
      for (const item of hookData.legend) {{
        html += `
          <div class="legend-color-item">
            <span class="legend-swatch" style="background:${{item.color}}; border:1px solid #aaa;"></span>
            <span class="legend-hook-label">${{item.function}}</span>
          </div>
        `;
      }}
      hookItems.innerHTML = html;
    }}

    // Highlight heads in expanded attention panels
    const allHeadDivs = document.querySelectorAll('#' + vizId + ' .attention-head[data-layer][data-head]');
    allHeadDivs.forEach(headDiv => {{
      const layerIdx = parseInt(headDiv.getAttribute('data-layer'));
      const headIdx = parseInt(headDiv.getAttribute('data-head'));
      const color = getHeadColor(layerIdx, headIdx, null);
      if (color) {{
        headDiv.style.background = color;
        headDiv.style.borderColor = '#666';
      }} else {{
        // Reset to default
        headDiv.style.background = '';
        headDiv.style.borderColor = '';
      }}
    }});
  }}

  function toggleExpand(id) {{
    if (id !== vizId) return;
    isExpanded = !isExpanded;
    render();
  }}

  function showExpanded(id, type, layer, element) {{
    if (id !== vizId) return;
    const panel = document.getElementById(id + '-' + type + '-expanded-' + layer);
    if (panel) {{
      const rect = element.getBoundingClientRect();
      const container = document.getElementById(id).getBoundingClientRect();
      panel.style.left = (rect.right - container.left + 10) + 'px';
      panel.style.top = (rect.top - container.top) + 'px';
      panel.classList.add('visible');
    }}
  }}

  function hideExpanded(id, type, layer) {{
    if (id !== vizId) return;
    const panel = document.getElementById(id + '-' + type + '-expanded-' + layer);
    if (panel && panel.id !== pinnedPanelId) {{
      panel.classList.remove('visible');
    }}
  }}

  function showResidual(id, event) {{
    if (id !== vizId) return;
    const panel = document.getElementById(id + '-residual-expanded');
    if (panel) {{
      const container = document.getElementById(id).getBoundingClientRect();
      panel.style.left = (event.clientX - container.left + 15) + 'px';
      panel.style.top = (event.clientY - container.top - 20) + 'px';
      panel.classList.add('visible');
    }}
  }}

  function hideResidual(id) {{
    if (id !== vizId) return;
    const panel = document.getElementById(id + '-residual-expanded');
    if (panel && panel.id !== pinnedPanelId) {{
      panel.classList.remove('visible');
    }}
  }}

  function showPanel(id, panelId, element) {{
    if (id !== vizId) return;
    const panel = document.getElementById(id + '-' + panelId);
    if (panel) {{
      const rect = element.getBoundingClientRect();
      const container = document.getElementById(id).getBoundingClientRect();
      panel.style.left = (rect.right - container.left + 10) + 'px';
      panel.style.top = (rect.top - container.top) + 'px';
      panel.classList.add('visible');
    }}
  }}

  function hidePanel(id, panelId) {{
    if (id !== vizId) return;
    const panel = document.getElementById(id + '-' + panelId);
    if (panel && panel.id !== pinnedPanelId) {{
      panel.classList.remove('visible');
    }}
  }}

  function toggleLearnMode(id) {{
    if (id !== vizId) return;
    const container = document.getElementById(id);
    const btn = document.getElementById(id + '-learn-mode-btn');
    const isOn = container.classList.toggle('learn-mode');
    if (btn) {{
      btn.textContent = isOn ? 'Learn Mode: ON' : 'Learn Mode: OFF';
      btn.classList.toggle('active', isOn);
    }}
  }}

  function toggleLearnPanel(id) {{
    if (id !== vizId) return;
    const panel = document.getElementById(id + '-learn-panel');
    if (panel) panel.classList.toggle('visible');
  }}

  // Close learn panel when clicking outside it
  document.addEventListener('click', function(e) {{
    const panel = document.getElementById(vizId + '-learn-panel');
    const btn = e.target.closest('.learn-toggle-btn');
    if (!panel) return;
    if (!btn && !panel.contains(e.target)) {{
      panel.classList.remove('visible');
    }}
  }});

  // Click-to-pin: clicking an interactive element keeps its popup open
  document.getElementById(vizId).addEventListener('click', function(e) {{
    // Ignore clicks on the learn panel itself or its toggle button
    if (e.target.closest('.learn-panel') || e.target.closest('.learn-toggle-btn')) return;

    const interactive = e.target.closest('.block, .ln-block, .residual-hover');
    if (interactive) {{
      // Find the popup that is currently visible
      const visible = document.querySelector(
        ['attention-expanded', 'mlp-expanded', 'residual-expanded', 'ln-expanded', 'pos-embed-expanded', 'embed-expanded', 'unembed-expanded', 'logits-expanded', 'tokens-expanded']
          .map(cls => '#' + vizId + ' .' + cls + '.visible').join(', ')
      );
      if (!visible) return;
      if (visible.id === pinnedPanelId) {{
        // Already pinned — unpin (will close on next mouseleave)
        pinnedPanelId = null;
      }} else {{
        // Unpin old panel
        if (pinnedPanelId) {{
          const old = document.getElementById(pinnedPanelId);
          if (old) old.classList.remove('visible');
        }}
        pinnedPanelId = visible.id;
      }}
    }} else {{
      // Click outside any component — unpin and close
      if (pinnedPanelId) {{
        const panel = document.getElementById(pinnedPanelId);
        if (panel) panel.classList.remove('visible');
        pinnedPanelId = null;
      }}
    }}
  }});

  window.showExpanded = showExpanded;
  window.hideExpanded = hideExpanded;
  window.toggleExpand = toggleExpand;
  window.showResidual = showResidual;
  window.hideResidual = hideResidual;
  window.showPanel = showPanel;
  window.hidePanel = hidePanel;
  window.toggleLearnPanel = toggleLearnPanel;
  window.toggleLearnMode = toggleLearnMode;

  render();
}})();
</script>
'''

    return html_template


class InteractiveTransformerViz:
    """Interactive transformer architecture visualization."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.arch: Optional[ModelArchitecture] = None
        self._html: Optional[str] = None

    def from_model(self, model: Any) -> "InteractiveTransformerViz":
        """Load from TransformerLens HookedTransformer."""
        self.arch = TransformerLensAdapter.from_hooked_transformer(model)
        return self

    def from_config(self, cfg: Any) -> "InteractiveTransformerViz":
        """Load from TransformerLens config."""
        self.arch = TransformerLensAdapter.from_config(cfg)
        return self

    def from_dict(self, config: dict) -> "InteractiveTransformerViz":
        """Load from dictionary."""
        self.arch = TransformerLensAdapter.from_dict(config)
        return self

    def from_pretrained(self, model_name: str) -> "InteractiveTransformerViz":
        """Load from pretrained model name."""
        self.arch = TransformerLensAdapter.from_pretrained_name(model_name)
        return self

    def render(
        self,
        max_layers: Optional[int] = None,
        width: int = 600,
        height: Optional[int] = None,
        hooks: Optional[List[Tuple[str, Callable]]] = None,
    ) -> "InteractiveTransformerViz":
        """Render the visualization.

        Args:
            max_layers: Maximum layers to show initially
            width: Width in pixels
            height: Height in pixels (auto-calculated if None)
            hooks: List of (hook_name, hook_function) tuples to highlight
        """
        if self.arch is None:
            raise ValueError("No architecture loaded. Use from_model(), from_dict(), etc.")

        self._html = _generate_html(
            self.arch,
            self.config,
            max_layers=max_layers,
            width=width,
            height=height,
            hooks=hooks,
        )
        return self

    def _repr_html_(self) -> str:
        """Jupyter notebook display hook."""
        if self._html is None:
            self.render()
        return self._html

    def show(self):
        """Display in Jupyter notebook."""
        from IPython.display import display, HTML
        if self._html is None:
            self.render()
        display(HTML(self._html))

    def save_html(self, filepath: str):
        """Save as standalone HTML file."""
        if self._html is None:
            self.render()

        full_html = f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{self.arch.model_name} Architecture</title>
</head>
<body style="margin: 20px; background: #f9f9f9;">
{self._html}
</body>
</html>'''

        with open(filepath, 'w') as f:
            f.write(full_html)


def visualize(
    model_or_config,
    max_layers: Optional[int] = None,
    width: int = 600,
    config: Optional[VisualizationConfig] = None,
    hooks: Optional[List[Tuple[str, Callable]]] = None,
) -> InteractiveTransformerViz:
    """
    Quick function to visualize a transformer.

    Args:
        model_or_config: HookedTransformer, config dict, or model name string
        max_layers: Maximum layers to show initially (default 8, click to expand)
        width: Width in pixels
        config: Optional visualization config
        hooks: List of (hook_name, hook_function) tuples to highlight in the diagram.
               Components with hooks are color-coded and a legend shows the mapping.

    Returns:
        InteractiveTransformerViz instance (displays automatically in Jupyter)

    Example:
        >>> def my_hook(tensor, hook):
        ...     return tensor
        >>> viz = visualize('gpt2-small', hooks=[
        ...     ('blocks.1.attn.hook_pattern', my_hook),
        ...     ('blocks.2.mlp.hook_post', ablation_fn),
        ... ])
    """
    viz = InteractiveTransformerViz(config)

    if isinstance(model_or_config, str):
        viz.from_pretrained(model_or_config)
    elif isinstance(model_or_config, dict):
        viz.from_dict(model_or_config)
    else:
        viz.from_model(model_or_config)

    viz.render(max_layers=max_layers, width=width, hooks=hooks)
    return viz

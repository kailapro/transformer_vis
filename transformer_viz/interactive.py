"""
Interactive transformer architecture visualization for Jupyter notebooks.

Renders as HTML/SVG with JavaScript for hover interactions,
styled after the Anthropic Transformer Circuits diagrams.
"""

import html
import json
import uuid
from typing import Optional, Any, Dict, List
from dataclasses import dataclass

from .model_adapter import ModelArchitecture, TransformerLensAdapter
from .config import VisualizationConfig


def _generate_html(
    arch: ModelArchitecture,
    config: VisualizationConfig,
    max_layers: Optional[int] = None,
    width: int = 600,
    height: Optional[int] = None,
    initially_collapsed: bool = True,
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
    add_circle_color = "#E8D5B7"
    add_circle_border = "#A89070"

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
      flex-wrap: wrap;
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
      background: #e8e8e8;
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
  </style>

  <div id="{viz_id}-svg-container">
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
    <div class="head-grid" style="margin-top: 10px;">
'''
        for h in range(arch.n_heads):
            html_template += f'''      <div class="attention-head">h<sub>{h}</sub></div>
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
  </div>

  <div id="{viz_id}-ln-attn-expanded-{layer_idx}" class="ln-expanded">
    <div class="expanded-title">Layer {layer_idx} Pre-Attention LayerNorm</div>
    <div class="expanded-dims">
      Normalizes to d_model = {arch.d_model}
    </div>
  </div>

  <div id="{viz_id}-ln-mlp-expanded-{layer_idx}" class="ln-expanded">
    <div class="expanded-title">Layer {layer_idx} Pre-MLP LayerNorm</div>
    <div class="expanded-dims">
      Normalizes to d_model = {arch.d_model}
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
  </div>

  <div id="{viz_id}-ln-final-expanded" class="ln-expanded">
    <div class="expanded-title">Final LayerNorm</div>
    <div class="expanded-dims">
      Normalizes to d_model = {arch.d_model}
    </div>
  </div>

  <div id="{viz_id}-pos-embed-expanded" class="pos-embed-expanded">
    <div class="expanded-title">Positional Embedding ({arch.pos_embed_type.title()})</div>
    <div class="expanded-dims">
      {f'n_ctx = {arch.n_ctx} × d_model = {arch.d_model}' if arch.pos_embed_type == 'learned' else f'Applied in attention (RoPE)' if arch.pos_embed_type == 'rotary' else f'Type: {arch.pos_embed_type}'}
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

  let isExpanded = false;

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
      <g class="block">
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
      <g class="block">
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
      svg += `
        <g class="ln-block"
           onmouseenter="showExpanded('${{vizId}}', 'ln-attn', ${{layerIdx}}, this)"
           onmouseleave="hideExpanded('${{vizId}}', 'ln-attn', ${{layerIdx}})">
          <rect x="${{lnAttnX - lnWidth/2}}" y="${{lnAttnY - lnHeight/2}}"
                width="${{lnWidth}}" height="${{lnHeight}}"
                rx="3" fill="#D4E5D4" stroke="#9AB89A" stroke-width="1" />
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

      for (let i = 0; i < headsToShow.length; i++) {{
        const headIdx = headsToShow[i];
        const headX = headsStartX + i * (headSize + headGap);
        const headCenterX = headX + headSize/2;
        const isDots = headIdx === '...';

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
            <g class="block" data-type="attention" data-layer="${{layerIdx}}"
               onmouseenter="showExpanded('${{vizId}}', 'attn', ${{layerIdx}}, this)"
               onmouseleave="hideExpanded('${{vizId}}', 'attn', ${{layerIdx}})">
              <rect x="${{headX}}" y="${{headsY}}"
                    width="${{headSize}}" height="${{headSize}}"
                    rx="4" fill="#E8D5B7" stroke="#C4A77D" stroke-width="1" />
              <text x="${{headCenterX}}" y="${{headsY + headSize/2 + 5}}"
                    text-anchor="middle" class="block-label">...</text>
            </g>
          `;
        }} else {{
          svg += `
            <g class="block" data-type="attention" data-layer="${{layerIdx}}" data-head="${{headIdx}}"
               onmouseenter="showExpanded('${{vizId}}', 'attn', ${{layerIdx}}, this)"
               onmouseleave="hideExpanded('${{vizId}}', 'attn', ${{layerIdx}})">
              <rect x="${{headX}}" y="${{headsY}}"
                    width="${{headSize}}" height="${{headSize}}"
                    rx="4" fill="#E8D5B7" stroke="#C4A77D" stroke-width="1" />
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
        svg += `
          <g class="ln-block"
             onmouseenter="showExpanded('${{vizId}}', 'ln-mlp', ${{layerIdx}}, this)"
             onmouseleave="hideExpanded('${{vizId}}', 'ln-mlp', ${{layerIdx}})">
            <rect x="${{lnMlpX - lnWidth/2}}" y="${{lnMlpY - lnHeight/2}}"
                  width="${{lnWidth}}" height="${{lnHeight}}"
                  rx="3" fill="#D4E5D4" stroke="#9AB89A" stroke-width="1" />
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
        svg += `
          <g class="block" data-type="mlp" data-layer="${{layerIdx}}"
             onmouseenter="showExpanded('${{vizId}}', 'mlp', ${{layerIdx}}, this)"
             onmouseleave="hideExpanded('${{vizId}}', 'mlp', ${{layerIdx}})">
            <rect x="${{blockCenterX - blockWidth/2}}" y="${{mlpY}}"
                  width="${{blockWidth}}" height="${{blockHeight}}"
                  rx="4" fill="#E8D5B7" stroke="#C4A77D" stroke-width="1" />
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
              rx="3" fill="#D4E5D4" stroke="#9AB89A" stroke-width="1" />
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
      <g class="block">
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
      <g class="block">
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
    if (panel) {{
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
    if (panel) {{
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
    if (panel) {{
      panel.classList.remove('visible');
    }}
  }}

  window.showExpanded = showExpanded;
  window.hideExpanded = hideExpanded;
  window.toggleExpand = toggleExpand;
  window.showResidual = showResidual;
  window.hideResidual = hideResidual;
  window.showPanel = showPanel;
  window.hidePanel = hidePanel;

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
    ) -> "InteractiveTransformerViz":
        """Render the visualization."""
        if self.arch is None:
            raise ValueError("No architecture loaded. Use from_model(), from_dict(), etc.")

        self._html = _generate_html(
            self.arch,
            self.config,
            max_layers=max_layers,
            width=width,
            height=height,
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
) -> InteractiveTransformerViz:
    """
    Quick function to visualize a transformer.

    Args:
        model_or_config: HookedTransformer, config dict, or model name string
        max_layers: Maximum layers to show initially (default 8, click to expand)
        width: Width in pixels
        config: Optional visualization config

    Returns:
        InteractiveTransformerViz instance (displays automatically in Jupyter)
    """
    viz = InteractiveTransformerViz(config)

    if isinstance(model_or_config, str):
        viz.from_pretrained(model_or_config)
    elif isinstance(model_or_config, dict):
        viz.from_dict(model_or_config)
    else:
        viz.from_model(model_or_config)

    viz.render(max_layers=max_layers, width=width)
    return viz

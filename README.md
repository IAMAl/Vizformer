# Vizformer

A Streamlit app that visualizes the per-module compute, parameter, and memory footprint of an encoder–decoder Transformer (Vaswani et al.). Adjust hyperparameters in the sidebar and watch the table and charts update live.

## Run

```bash
pip install streamlit numpy pandas plotly
streamlit run vizformer.py
```

## What it models

A standard Post-LN Transformer (forward-pass only):

- **Encoder block** — MHA → Skip → LayerNorm → FFN → Skip → LayerNorm
- **Decoder block** — Masked MHA → Skip → LayerNorm → Cross-Attention → Skip → LayerNorm → FFN → Skip → LayerNorm
- Embedding layers + sinusoidal positional encoding
- Output projection to vocabulary

Each row in the module table reports input/output shapes, weight/bias counts, FLOPs, and memory broken down into weight / bias / activation / KV-cache bytes.

## Sidebar parameters

| Group | Parameter |
| --- | --- |
| Shape | `batch_size`, `L_s`, `L_t`, `d_model`, `d_ff`, `h`, `N_enc`, `N_dec`, `V_src`, `V_tgt` |
| Precision | `float64 / float32 / float16 / int8` |
| KV cache | enable per-decoder-layer; sequence length `0`–`2²⁰` |
| FlashAttention | enable + `tile_m / tile_n / tile_k` |
| Reduction modes | Softmax / LayerNorm: `Sequential O(n)` or `Parallel O(log n)` |

Parameter sets can be saved/loaded as JSON from the sidebar.

## Charts

Five charts under the **Graphs** section, each filterable by Encoder / Decoder / Both and switchable between Linear and Log Y-axis:

1. **Weights / Biases** — per-module parameter counts (words)
2. **Computations** — per-module operation counts (ops)
3. **Memory Usage** — stacked weight / bias / activation / KV cache bytes
4. **Input vs Output Tensor Elements** — words
5. **Aggregated Element Count** — input breakdown (Activation + Weights + Biases + KV Cache) vs Output, words

Graph height has three levels (Small 0.5×, Medium 1×, Large 1.5×).

## Caveats

- **Forward pass only.** Backward and optimizer state are not modeled (~3× extra for training).
- **KV cache row** represents the minimum bytes to hold K and V for `kv_seq_length` tokens at the configured precision (no padding, no metadata, no paged-attention overhead).
- **Activation/computation** assumes training-mode shapes (full `L_t × L_t` self-attention). It does not reduce to inference-mode `1 × kv_seq_length` shapes when KV cache is enabled.
- **Cross-attention** counts K/V projections per decoder layer; in practice they are typically computed once on the encoder output and reused.

## License

See `LICENSE`.

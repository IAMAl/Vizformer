import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import base64
import re
import math

# パース系ユーティリティ
def parse_int_value(x):
    if isinstance(x, str):
        x = x.replace('(SingleTile)', '')
        x = x.replace(',', '')
        x = x.strip()
        m = re.search(r'(\d+)', x)
        if m:
            return int(m.group(1))
        else:
            return 0
    else:
        return int(x)

def parse_float_value(x):
    if isinstance(x, str):
        x = x.replace('(SingleTile)', '')
        x = x.replace(',', '')
        x = x.strip()
        m = re.search(r'(\d+(\.\d+)?)', x)
        if m:
            return float(m.group(1))
        else:
            return 0.0
    else:
        return float(x)

def format_memory_size(bytes_val):
    if bytes_val >= 1024**3:
        return f"{bytes_val/(1024**3):.2f} GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val/(1024**2):.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val/1024:.2f} KB"
    else:
        return f"{bytes_val} B"

def calculate_elements(shape_str):
    # シェイプ文字列から括弧内の各数値またはパラメータ名を抽出し、各次元の積を計算する
    tuples = re.findall(r'\(([^)]+)\)', shape_str)
    total = 0
    for tup in tuples:
        parts = tup.split(',')
        product = 1
        for elem in parts:
            elem = elem.strip()
            if elem.isdigit():
                product *= int(elem)
            else:
                val = st.session_state['params'].get(elem, 1)
                product *= int(val)
        total += product
    return total

# FlashAttention 関連
# m = query tile rows, n = key/value tile rows, k = head dim (d_k)
def flash_single_tile_comp(m, n, k):
    # QK^T: 2mnk, online softmax over n keys per row of m queries: ~5mn,
    # softmax×V: 2mnk
    return 2*m*n*k + 5*m*n + 2*m*n*k

def flash_single_tile_params(d_model, tile_k):
    # one head's worth of slice through one projection (Q, K or V): d_model × d_k
    return d_model * tile_k

def flash_single_tile_qkv_data(m, n, k):
    # Q tile: (m,k), K tile: (n,k), V tile: (n,k)
    return (m*k) + (n*k) + (n*k)

def flash_single_tile_kv_cache(n, k):
    # K and V tiles read into SRAM: each (n,k)
    return 2*n*k

# デフォルトパラメータ
DEFAULT_PARAMS = {
    "batch_size": 32,
    "L_s": 50,
    "L_t": 50,
    "d_model": 512,
    "N_enc": 1,
    "N_dec": 1,
    "h": 8,
    "d_ff": 2048,
    "V_src": 30000,
    "V_tgt": 30000,
    "dtype_str": "float32 (4bytes)",
    "use_kv_cache": False,
    "use_flash_attention": False,
    "tile_size_m": 128,
    "tile_size_n": 128,
    "tile_size_k": 64,
    "kv_cache_layers": [1],
    "kv_seq_length": 50
}

# パラメータ保存／読み込み関数
def save_parameters(params):
    json_str = json.dumps(params, ensure_ascii=False, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="parameters.json">Download Params</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

def load_parameters():
    f = st.sidebar.file_uploader("Upload param file (.json)", type=["json"])
    if f:
        try:
            data = f.read().decode()
            pr = json.loads(data)
            return pr
        except Exception as e:
            st.sidebar.error(f"Failed to load: {e}")
    return None

# Streamlit ページ設定
st.set_page_config(layout="wide")
st.title("Transformer")
st.write("")
st.write("version: 1.5")

# サイドバー：パラメータ設定
st.sidebar.header("Model Params")
if 'params' not in st.session_state:
    st.session_state['params'] = DEFAULT_PARAMS.copy()

if st.sidebar.button("Reset to Default"):
    st.session_state['params'] = DEFAULT_PARAMS.copy()

loaded = load_parameters()
if loaded:
    st.session_state['params'] = {**DEFAULT_PARAMS, **loaded}
    st.sidebar.success("Params loaded successfully")

p = st.session_state['params']

p["batch_size"] = st.sidebar.number_input("Batch Size", 1, 999999, value=p["batch_size"])
p["L_s"] = st.sidebar.number_input("Source Seq (L_s)", 1, 999999, value=p["L_s"])
p["L_t"] = st.sidebar.number_input("Target Seq (L_t)", 1, 999999, value=p["L_t"])
p["d_model"] = st.sidebar.number_input("d_model", 1, 999999, value=p["d_model"])
p["N_enc"] = st.sidebar.number_input("Encoder Layers (N_enc)", 1, 9999, value=p["N_enc"])
p["N_dec"] = st.sidebar.number_input("Decoder Layers (N_dec)", 1, 9999, value=p["N_dec"])
p["h"] = st.sidebar.number_input("Heads (h)", 1, 9999, value=p["h"])
if p["d_model"] % p["h"] != 0:
    st.sidebar.warning(
        f"d_model ({p['d_model']}) is not divisible by h ({p['h']}); per-head dim will be truncated to {p['d_model'] // p['h']}."
    )
p["d_ff"] = st.sidebar.number_input("FeedForward Dim (d_ff)", 1, 999999, value=p["d_ff"])
p["V_src"] = st.sidebar.number_input("V_src (Source Vocab)", 1, 999999, value=p["V_src"])
p["V_tgt"] = st.sidebar.number_input("V_tgt (Target Vocab)", 1, 999999, value=p["V_tgt"])

dtype_map = {
    "float64 (8bytes)":8,
    "float32 (4bytes)":4,
    "float16 (2bytes)":2,
    "int8 (1byte)":1
}
all_dtypes = list(dtype_map.keys())
dtype_idx = all_dtypes.index(p["dtype_str"]) if p["dtype_str"] in all_dtypes else all_dtypes.index(DEFAULT_PARAMS["dtype_str"])
p["dtype_str"] = st.sidebar.selectbox("Data Type", all_dtypes, index=dtype_idx)

p["use_kv_cache"] = st.sidebar.checkbox("Use KV Cache (per head)", value=p["use_kv_cache"])
if p["use_kv_cache"]:
    available_kv_layers = list(range(1, p["N_dec"] + 1))
    saved_kv_layers = [l for l in p.get("kv_cache_layers", available_kv_layers) if l in available_kv_layers]
    if not saved_kv_layers:
        saved_kv_layers = available_kv_layers
    p["kv_cache_layers"] = st.sidebar.multiselect(
        "KV Cache Decoder Layers (1-indexed)",
        options=available_kv_layers,
        default=saved_kv_layers
    )
    p["kv_seq_length"] = st.sidebar.number_input(
        "KV Cache Sequence Length",
        min_value=0, max_value=2**20,
        value=p.get("kv_seq_length", p["L_t"]),
        help="Number of tokens stored in the cache (context + generated). Per-layer KV bytes = batch × kv_seq_length × d_model × 2 × precision."
    )

p["use_flash_attention"] = st.sidebar.checkbox("Use FlashAttention", value=p["use_flash_attention"])

if p["use_flash_attention"]:
    st.sidebar.subheader("FlashAttention Tile Sizes")
    p["tile_size_m"] = st.sidebar.number_input("tile_size_m", 1, 9999, value=p["tile_size_m"])
    p["tile_size_n"] = st.sidebar.number_input("tile_size_n", 1, 9999, value=p["tile_size_n"])
    p["tile_size_k"] = st.sidebar.number_input("tile_size_k", 1, 9999, value=p["tile_size_k"])

# ★ サイドバー：ソフトマックスの演算モード選択 ★
softmax_mode = st.sidebar.radio("Softmax Reduction Mode", ("Sequential (O(n))", "Parallel (O(log n))"))

# ★ サイドバー：LayerNormの演算モード選択 ★
ln_mode = st.sidebar.radio("LayerNorm Reduction Mode", ("Sequential (O(n))", "Parallel (O(log n))"))

if st.sidebar.button("Save Params"):
    save_parameters(p)

st.session_state['params'] = p

# モジュール情報作成用ユーティリティ
def mkrow(
    name,
    input_shape="",
    output_shape="",
    wnum=0,
    bnum=0,
    comp=0,
    wmem=0,
    bmem=0,
    actmem=0,
    kvcache=0,
    qkv=0,
    tile_ex=1,
    reuse_c=1
):
    precision = dtype_map[p["dtype_str"]]
    kv_elems = 0
    if kvcache > 0 and precision > 0:
        kv_elems = kvcache // precision
    return {
        "Module": name,
        "Input Shape": str(input_shape),
        "Output Shape": str(output_shape),
        "Weights (#)": f"{wnum:,}" if wnum else "0",
        "Biases (#)": f"{bnum:,}" if bnum else "0",
        "Computations": f"{comp:,}" if comp else "0",
        "Weight Memory": format_memory_size(wmem),
        "Bias Memory": format_memory_size(bmem),
        "Activation Memory": format_memory_size(actmem),
        "KV Cache Memory": format_memory_size(kvcache),
        "QKV Data (#)": f"{qkv:,}" if qkv else "0",
        "KV Cache (#)": f"{kv_elems:,}" if kv_elems else "0",
        "Total Memory": format_memory_size(wmem + bmem + actmem + kvcache),
        "Tile Executions": tile_ex,
        "Data Reuse Count": reuse_c
    }

# calc_details() 内で各モジュールの計算量・メモリを計算
def calc_details():
    param = st.session_state['params']
    batch_size = param["batch_size"]
    L_s = param["L_s"]
    L_t = param["L_t"]
    d_model = param["d_model"]
    N_enc = param["N_enc"]
    N_dec = param["N_dec"]
    h = param["h"]
    d_ff = param["d_ff"]
    V_src = param["V_src"]
    V_tgt = param["V_tgt"]
    precision = dtype_map[param["dtype_str"]]
    use_kv = param["use_kv_cache"]
    use_flash = param["use_flash_attention"]
    tile_m = param["tile_size_m"]
    tile_n = param["tile_size_n"]
    tile_k = param["tile_size_k"]

    if ln_mode.startswith("Sequential"):
        ln_cost = 5 * d_model
    else:
        ln_cost = 3 * d_model + 2 * math.log(d_model, 2)

    def ln_comp_enc():
        return ln_cost * batch_size * L_s
    def ln_comp_dec():
        return ln_cost * batch_size * L_t

    kv_layers_set = set(param.get("kv_cache_layers", list(range(1, N_dec + 1)))) if use_kv else set()

    rows = []
    total_w_mem = 0
    total_b_mem = 0
    total_act_mem = 0
    total_kv_mem = 0
    total_mem = 0
    total_comp_no_flash = 0

    # Encoder Input Embedding
    w_enc_in = V_src * d_model
    w_enc_in_mem = w_enc_in * precision
    act_enc_in = batch_size * L_s * d_model * precision
    total_w_mem += w_enc_in_mem
    total_act_mem += act_enc_in
    total_mem += (w_enc_in_mem + act_enc_in)
    rows.append(
        mkrow(
            name="Encoder Input Embedding",
            input_shape=f"({batch_size},{L_s})",
            output_shape=f"({batch_size},{L_s},{d_model})",
            wnum=w_enc_in,
            wmem=w_enc_in_mem,
            actmem=act_enc_in
        )
    )

    pe_enc_comp = batch_size * L_s * d_model
    total_comp_no_flash += pe_enc_comp
    rows.append(
        mkrow(
            name="Encoder Positional Encoding",
            input_shape=f"({batch_size},{L_s},{d_model})",
            output_shape=f"({batch_size},{L_s},{d_model})",
            comp=pe_enc_comp
        )
    )

    for i in range(N_enc):
        layer_idx = i + 1
        w_mha = d_model * d_model * 3
        b_mha = d_model * 3
        w_mha_mem = w_mha * precision
        b_mha_mem = b_mha * precision
        comp_mha = batch_size * L_s * d_model * d_model * 3 * 2
        qkv_count = batch_size * L_s * d_model * 3
        qkv_mem = qkv_count * precision
        total_comp_no_flash += comp_mha
        total_w_mem += w_mha_mem
        total_b_mem += b_mha_mem
        total_act_mem += qkv_mem
        total_mem += (w_mha_mem + b_mha_mem + qkv_mem)
        rows.append(
            mkrow(
                name=f"Encoder Layer {layer_idx} - MHA Linear Projections",
                input_shape=f"({batch_size},{L_s},{d_model})",
                output_shape=f"QKV:({batch_size},{L_s},{3*d_model})",
                wnum=w_mha, bnum=b_mha, comp=comp_mha,
                wmem=w_mha_mem, bmem=b_mha_mem, actmem=qkv_mem,
                qkv=qkv_count
            )
        )
        comp_qk = batch_size * h * L_s * L_s * (d_model // h) * 2
        attn_score_mem = batch_size * h * L_s * L_s * precision
        total_comp_no_flash += comp_qk
        total_act_mem += attn_score_mem
        total_mem += attn_score_mem
        rows.append(
            mkrow(
                name=f"Encoder Layer {layer_idx} - Attention QK^T",
                comp=comp_qk,
                actmem=attn_score_mem
            )
        )
        comp_scale = batch_size * h * L_s * L_s
        total_comp_no_flash += comp_scale
        rows.append(
            mkrow(
                name=f"Encoder Layer {layer_idx} - Attention Scaling",
                comp=comp_scale
            )
        )
        if softmax_mode.startswith("Sequential"):
            cost_soft = 5 * L_s
        else:
            cost_soft = 3 * L_s + 2 * math.log(L_s, 2)
        comp_soft = batch_size * h * L_s * cost_soft
        total_comp_no_flash += comp_soft
        rows.append(
            mkrow(
                name=f"Encoder Layer {layer_idx} - Attention Softmax",
                comp=comp_soft
            )
        )
        comp_av = batch_size * h * L_s * L_s * (d_model // h) * 2
        av_ct = batch_size * L_s * d_model
        av_mem = av_ct * precision
        total_comp_no_flash += comp_av
        total_act_mem += av_mem
        total_mem += av_mem
        rows.append(
            mkrow(
                name=f"Encoder Layer {layer_idx} - Attention Softmax×V",
                comp=comp_av,
                output_shape=f"Context:({batch_size},{L_s},{d_model})",
                actmem=av_mem
            )
        )
        w_out = d_model * d_model
        b_out = d_model
        w_out_mem = w_out * precision
        b_out_mem = b_out * precision
        comp_out = batch_size * L_s * d_model * d_model * 2
        out_mha_act = batch_size * L_s * d_model * precision
        total_comp_no_flash += comp_out
        total_w_mem += w_out_mem
        total_b_mem += b_out_mem
        total_act_mem += out_mha_act
        total_mem += (w_out_mem + b_out_mem + out_mha_act)
        rows.append(
            mkrow(
                name=f"Encoder Layer {layer_idx} - MHA Output Linear",
                input_shape=f"({batch_size},{L_s},{d_model})",
                output_shape=f"({batch_size},{L_s},{d_model})",
                wnum=w_out, bnum=b_out, comp=comp_out,
                wmem=w_out_mem, bmem=b_out_mem, actmem=out_mha_act
            )
        )
        skip_mem = batch_size * L_s * d_model * precision
        skip_comp = batch_size * L_s * d_model
        total_act_mem += skip_mem
        total_mem += skip_mem
        total_comp_no_flash += skip_comp
        rows.append(
            mkrow(
                name=f"Encoder Layer {layer_idx} - Skip Connection",
                comp=skip_comp,
                actmem=skip_mem
            )
        )
        ln_w2 = d_model
        ln_b2 = d_model
        ln_w2_mem = ln_w2 * precision
        ln_b2_mem = ln_b2 * precision
        ln_act2 = batch_size * L_s * d_model * precision
        comp_ln2 = ln_comp_enc()
        total_comp_no_flash += comp_ln2
        total_w_mem += ln_w2_mem
        total_b_mem += ln_b2_mem
        total_act_mem += ln_act2
        total_mem += (ln_w2_mem + ln_b2_mem + ln_act2)
        rows.append(
            mkrow(
                name=f"Encoder Layer {layer_idx} - MHA LayerNorm",
                wnum=ln_w2, bnum=ln_b2, comp=comp_ln2,
                wmem=ln_w2_mem, bmem=ln_b2_mem, actmem=ln_act2
            )
        )
        w_ffn1 = d_model * d_ff
        b_ffn1 = d_ff
        w_ffn1_mem = w_ffn1 * precision
        b_ffn1_mem = b_ffn1 * precision
        comp_ffn1 = batch_size * L_s * d_model * d_ff * 2 + batch_size * L_s * d_ff
        ffn1_mem = batch_size * L_s * d_ff * precision
        total_comp_no_flash += comp_ffn1
        total_w_mem += w_ffn1_mem
        total_b_mem += b_ffn1_mem
        total_act_mem += ffn1_mem
        total_mem += (w_ffn1_mem + b_ffn1_mem + ffn1_mem)
        rows.append(
            mkrow(
                name=f"Encoder Layer {layer_idx} - FFN Layer 1 + Activation",
                input_shape=f"({batch_size},{L_s},{d_model})",
                output_shape=f"({batch_size},{L_s},{d_ff})",
                wnum=w_ffn1, bnum=b_ffn1, comp=comp_ffn1,
                wmem=w_ffn1_mem, bmem=b_ffn1_mem, actmem=ffn1_mem
            )
        )
        w_ffn2 = d_ff * d_model
        b_ffn2 = d_model
        w_ffn2_mem = w_ffn2 * precision
        b_ffn2_mem = b_ffn2 * precision
        comp_ffn2 = batch_size * L_s * d_ff * d_model * 2
        ffn2_act_mem = batch_size * L_s * d_model * precision
        total_comp_no_flash += comp_ffn2
        total_w_mem += w_ffn2_mem
        total_b_mem += b_ffn2_mem
        total_act_mem += ffn2_act_mem
        total_mem += (w_ffn2_mem + b_ffn2_mem + ffn2_act_mem)
        rows.append(
            mkrow(
                name=f"Encoder Layer {layer_idx} - FFN Layer 2",
                input_shape=f"({batch_size},{L_s},{d_ff})",
                output_shape=f"({batch_size},{L_s},{d_model})",
                wnum=w_ffn2, bnum=b_ffn2, comp=comp_ffn2,
                wmem=w_ffn2_mem, bmem=b_ffn2_mem, actmem=ffn2_act_mem
            )
        )
        skip_ffn = batch_size * L_s * d_model * precision
        skip_ffn_comp = batch_size * L_s * d_model
        total_act_mem += skip_ffn
        total_mem += skip_ffn
        total_comp_no_flash += skip_ffn_comp
        rows.append(
            mkrow(
                name=f"Encoder Layer {layer_idx} - FFN Skip Connection",
                comp=skip_ffn_comp,
                actmem=skip_ffn
            )
        )
        ln_w3 = d_model
        ln_b3 = d_model
        ln_w3_mem = ln_w3 * precision
        ln_b3_mem = ln_b3 * precision
        ln_act3 = batch_size * L_s * d_model * precision
        comp_ln3 = ln_comp_enc()
        total_comp_no_flash += comp_ln3
        total_w_mem += ln_w3_mem
        total_b_mem += ln_b3_mem
        total_act_mem += ln_act3
        total_mem += (ln_w3_mem + ln_b3_mem + ln_act3)
        rows.append(
            mkrow(
                name=f"Encoder Layer {layer_idx} - FFN LayerNorm",
                wnum=ln_w3, bnum=ln_b3, comp=comp_ln3,
                wmem=ln_w3_mem, bmem=ln_b3_mem, actmem=ln_act3
            )
        )

    w_dec_in = V_tgt * d_model
    w_dec_in_mem = w_dec_in * precision
    dec_in_act = batch_size * L_t * d_model * precision
    total_w_mem += w_dec_in_mem
    total_act_mem += dec_in_act
    total_mem += (w_dec_in_mem + dec_in_act)
    rows.append(
        mkrow(
            name="Decoder Input Embedding",
            input_shape=f"({batch_size},{L_t})",
            output_shape=f"({batch_size},{L_t},{d_model})",
            wnum=w_dec_in,
            bnum=0,
            wmem=w_dec_in_mem,
            bmem=0,
            actmem=dec_in_act
        )
    )

    pe_dec_comp = batch_size * L_t * d_model
    total_comp_no_flash += pe_dec_comp
    rows.append(
        mkrow(
            name="Decoder Positional Encoding",
            input_shape=f"({batch_size},{L_t},{d_model})",
            output_shape=f"({batch_size},{L_t},{d_model})",
            comp=pe_dec_comp
        )
    )

    for dec_i in range(N_dec):
        layer_name = dec_i + 1
        w_mha_dec = d_model * d_model * 3
        b_mha_dec = d_model * 3
        w_mha_dec_mem = w_mha_dec * precision
        b_mha_dec_mem = b_mha_dec * precision
        comp_mha_dec = batch_size * L_t * d_model * d_model * 3 * 2
        qkv_ct_dec = batch_size * L_t * d_model * 3
        qkv_mem_dec = qkv_ct_dec * precision
        total_comp_no_flash += comp_mha_dec
        total_w_mem += w_mha_dec_mem
        total_b_mem += b_mha_dec_mem
        total_act_mem += qkv_mem_dec
        total_mem += (w_mha_dec_mem + b_mha_dec_mem + qkv_mem_dec)
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Masked MHA Linear Projections",
                input_shape=f"({batch_size},{L_t},{d_model})",
                output_shape=f"QKV:({batch_size},{L_t},{3*d_model})",
                wnum=w_mha_dec, bnum=b_mha_dec,
                comp=comp_mha_dec,
                wmem=w_mha_dec_mem, bmem=b_mha_dec_mem,
                actmem=qkv_mem_dec,
                qkv=qkv_ct_dec
            )
        )
        comp_qk_dec = batch_size * h * L_t * L_t * ((d_model // h)) * 2
        qk_dec_mem = batch_size * h * L_t * L_t * precision
        total_comp_no_flash += comp_qk_dec
        total_act_mem += qk_dec_mem
        total_mem += qk_dec_mem
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Masked Attention QK^T",
                comp=comp_qk_dec,
                actmem=qk_dec_mem
            )
        )
        comp_scale_dec = batch_size * h * L_t * L_t
        total_comp_no_flash += comp_scale_dec
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Masked Attention Scaling",
                comp=comp_scale_dec
            )
        )
        if softmax_mode.startswith("Sequential"):
            cost_soft_dec = 5 * L_t
        else:
            cost_soft_dec = 3 * L_t + 2 * math.log(L_t, 2)
        comp_soft_dec = batch_size * h * L_t * cost_soft_dec
        total_comp_no_flash += comp_soft_dec
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Masked Attention Softmax",
                comp=comp_soft_dec
            )
        )
        comp_av_dec = batch_size * h * L_t * L_t * ((d_model // h)) * 2
        av_dec_ct = batch_size * L_t * d_model
        av_dec_mem = av_dec_ct * precision
        total_comp_no_flash += comp_av_dec
        total_act_mem += av_dec_mem
        total_mem += av_dec_mem
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Masked Attention Softmax×V",
                comp=comp_av_dec,
                output_shape=f"Context:({batch_size},{L_t},{d_model})",
                actmem=av_dec_mem
            )
        )
        if use_kv and (dec_i + 1) in kv_layers_set:
            d_k = d_model // h
            if d_k > 0:
                kv_seq_length = param.get("kv_seq_length", L_t)
                kv_count_per_layer = batch_size * h * kv_seq_length * d_k * 2
                kv_bytes = kv_count_per_layer * precision
                total_mem += kv_bytes
                total_kv_mem += kv_bytes
                rows.append(
                    mkrow(
                        name=f"Decoder Layer {layer_name} - KV Cache (persistent)",
                        input_shape=f"K,V:({batch_size},{h},{kv_seq_length},{d_k})",
                        output_shape=f"K,V:({batch_size},{h},{kv_seq_length},{d_k})",
                        kvcache=kv_bytes
                    )
                )
        w_out_dec = d_model * d_model
        b_out_dec = d_model
        w_out_dec_mem = w_out_dec * precision
        b_out_dec_mem = b_out_dec * precision
        comp_out_dec = batch_size * L_t * d_model * d_model * 2
        out_mha_dec_act = batch_size * L_t * d_model * precision
        total_comp_no_flash += comp_out_dec
        total_w_mem += w_out_dec_mem
        total_b_mem += b_out_dec_mem
        total_act_mem += out_mha_dec_act
        total_mem += (w_out_dec_mem + b_out_dec_mem + out_mha_dec_act)
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Masked MHA Output Linear",
                input_shape=f"({batch_size},{L_t},{d_model})",
                output_shape=f"({batch_size},{L_t},{d_model})",
                wnum=w_out_dec, bnum=b_out_dec,
                comp=comp_out_dec,
                wmem=w_out_dec_mem, bmem=b_out_dec_mem,
                actmem=out_mha_dec_act
            )
        )
        skip_mha = batch_size * L_t * d_model * precision
        skip_mha_comp = batch_size * L_t * d_model
        total_act_mem += skip_mha
        total_mem += skip_mha
        total_comp_no_flash += skip_mha_comp
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Skip Connection (Masked MHA)",
                comp=skip_mha_comp,
                actmem=skip_mha
            )
        )
        ln_w5 = d_model
        ln_b5 = d_model
        ln_w5_mem = ln_w5 * precision
        ln_b5_mem = ln_b5 * precision
        ln_act5 = batch_size * L_t * d_model * precision
        comp_ln5 = ln_comp_dec()
        total_comp_no_flash += comp_ln5
        total_w_mem += ln_w5_mem
        total_b_mem += ln_b5_mem
        total_act_mem += ln_act5
        total_mem += (ln_w5_mem + ln_b5_mem + ln_act5)
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Masked MHA LayerNorm",
                wnum=ln_w5, bnum=ln_b5, comp=comp_ln5,
                wmem=ln_w5_mem, bmem=ln_b5_mem, actmem=ln_act5
            )
        )
        # ── Cross-Attention (Q from decoder, K,V from encoder output) ──
        w_cross = d_model * d_model * 3
        b_cross = d_model * 3
        w_cross_mem = w_cross * precision
        b_cross_mem = b_cross * precision
        # Q proj on L_t, K/V proj on L_s (encoder output)
        comp_cross_qkv = batch_size * (L_t + 2 * L_s) * d_model * d_model * 2
        qkv_ct_cross = batch_size * (L_t + 2 * L_s) * d_model
        qkv_mem_cross = qkv_ct_cross * precision
        total_comp_no_flash += comp_cross_qkv
        total_w_mem += w_cross_mem
        total_b_mem += b_cross_mem
        total_act_mem += qkv_mem_cross
        total_mem += (w_cross_mem + b_cross_mem + qkv_mem_cross)
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Cross-Attention Linear Projections",
                input_shape=f"Q-in:({batch_size},{L_t},{d_model});KV-in:({batch_size},{L_s},{d_model})",
                output_shape=f"Q:({batch_size},{L_t},{d_model});K:({batch_size},{L_s},{d_model});V:({batch_size},{L_s},{d_model})",
                wnum=w_cross, bnum=b_cross,
                comp=comp_cross_qkv,
                wmem=w_cross_mem, bmem=b_cross_mem,
                actmem=qkv_mem_cross,
                qkv=qkv_ct_cross
            )
        )
        comp_qk_cross = batch_size * h * L_t * L_s * (d_model // h) * 2
        qk_cross_mem = batch_size * h * L_t * L_s * precision
        total_comp_no_flash += comp_qk_cross
        total_act_mem += qk_cross_mem
        total_mem += qk_cross_mem
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Cross-Attention QK^T",
                comp=comp_qk_cross,
                actmem=qk_cross_mem
            )
        )
        comp_scale_cross = batch_size * h * L_t * L_s
        total_comp_no_flash += comp_scale_cross
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Cross-Attention Scaling",
                comp=comp_scale_cross
            )
        )
        if softmax_mode.startswith("Sequential"):
            cost_soft_cross = 5 * L_s
        else:
            cost_soft_cross = 3 * L_s + 2 * math.log(L_s, 2)
        comp_soft_cross = batch_size * h * L_t * cost_soft_cross
        total_comp_no_flash += comp_soft_cross
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Cross-Attention Softmax",
                comp=comp_soft_cross
            )
        )
        comp_av_cross = batch_size * h * L_t * L_s * (d_model // h) * 2
        av_cross_ct = batch_size * L_t * d_model
        av_cross_mem = av_cross_ct * precision
        total_comp_no_flash += comp_av_cross
        total_act_mem += av_cross_mem
        total_mem += av_cross_mem
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Cross-Attention Softmax×V",
                comp=comp_av_cross,
                output_shape=f"Context:({batch_size},{L_t},{d_model})",
                actmem=av_cross_mem
            )
        )
        w_out_cross = d_model * d_model
        b_out_cross = d_model
        w_out_cross_mem = w_out_cross * precision
        b_out_cross_mem = b_out_cross * precision
        comp_out_cross = batch_size * L_t * d_model * d_model * 2
        out_cross_act = batch_size * L_t * d_model * precision
        total_comp_no_flash += comp_out_cross
        total_w_mem += w_out_cross_mem
        total_b_mem += b_out_cross_mem
        total_act_mem += out_cross_act
        total_mem += (w_out_cross_mem + b_out_cross_mem + out_cross_act)
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Cross-Attention Output Linear",
                input_shape=f"({batch_size},{L_t},{d_model})",
                output_shape=f"({batch_size},{L_t},{d_model})",
                wnum=w_out_cross, bnum=b_out_cross, comp=comp_out_cross,
                wmem=w_out_cross_mem, bmem=b_out_cross_mem,
                actmem=out_cross_act
            )
        )
        skip_cross_mem = batch_size * L_t * d_model * precision
        skip_cross_comp = batch_size * L_t * d_model
        total_act_mem += skip_cross_mem
        total_mem += skip_cross_mem
        total_comp_no_flash += skip_cross_comp
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Skip Connection (Cross-Attention)",
                comp=skip_cross_comp,
                actmem=skip_cross_mem
            )
        )
        ln_w_cross = d_model
        ln_b_cross = d_model
        ln_w_cross_mem = ln_w_cross * precision
        ln_b_cross_mem = ln_b_cross * precision
        ln_act_cross = batch_size * L_t * d_model * precision
        comp_ln_cross = ln_comp_dec()
        total_comp_no_flash += comp_ln_cross
        total_w_mem += ln_w_cross_mem
        total_b_mem += ln_b_cross_mem
        total_act_mem += ln_act_cross
        total_mem += (ln_w_cross_mem + ln_b_cross_mem + ln_act_cross)
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - Cross-Attention LayerNorm",
                wnum=ln_w_cross, bnum=ln_b_cross, comp=comp_ln_cross,
                wmem=ln_w_cross_mem, bmem=ln_b_cross_mem, actmem=ln_act_cross
            )
        )
        w_ffn1d = d_ff * d_model
        b_ffn1d = d_ff
        w_ffn1d_mem = w_ffn1d * precision
        b_ffn1d_mem = b_ffn1d * precision
        comp_ffn1d = batch_size * L_t * d_model * d_ff * 2 + batch_size * L_t * d_ff
        ffn1d_act = batch_size * L_t * d_ff * precision
        total_comp_no_flash += comp_ffn1d
        total_w_mem += w_ffn1d_mem
        total_b_mem += b_ffn1d_mem
        total_act_mem += ffn1d_act
        total_mem += (w_ffn1d_mem + b_ffn1d_mem + ffn1d_act)
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - FFN Layer 1 + Activation",
                input_shape=f"({batch_size},{L_t},{d_model})",
                output_shape=f"({batch_size},{L_t},{d_ff})",
                wnum=w_ffn1d, bnum=b_ffn1d, comp=comp_ffn1d,
                wmem=w_ffn1d_mem, bmem=b_ffn1d_mem, actmem=ffn1d_act
            )
        )
        w_ffn2d = d_ff * d_model
        b_ffn2d = d_model
        w_ffn2d_mem = w_ffn2d * precision
        b_ffn2d_mem = b_ffn2d * precision
        comp_ffn2d = batch_size * L_t * d_ff * d_model * 2
        ffn2d_act_mem = batch_size * L_t * d_model * precision
        total_comp_no_flash += comp_ffn2d
        total_w_mem += w_ffn2d_mem
        total_b_mem += b_ffn2d_mem
        total_act_mem += ffn2d_act_mem
        total_mem += (w_ffn2d_mem + b_ffn2d_mem + ffn2d_act_mem)
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - FFN Layer 2",
                input_shape=f"({batch_size},{L_t},{d_ff})",
                output_shape=f"({batch_size},{L_t},{d_model})",
                wnum=w_ffn2d, bnum=b_ffn2d, comp=comp_ffn2d,
                wmem=w_ffn2d_mem, bmem=b_ffn2d_mem, actmem=ffn2d_act_mem
            )
        )
        skip_ffn_d = batch_size * L_t * d_model * precision
        skip_ffn_d_comp = batch_size * L_t * d_model
        total_act_mem += skip_ffn_d
        total_mem += skip_ffn_d
        total_comp_no_flash += skip_ffn_d_comp
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - FFN Skip Connection",
                comp=skip_ffn_d_comp,
                actmem=skip_ffn_d
            )
        )
        ln_w6 = d_model
        ln_b6 = d_model
        ln_w6_mem = ln_w6 * precision
        ln_b6_mem = ln_b6 * precision
        ln_act6 = batch_size * L_t * d_model * precision
        comp_ln6 = ln_comp_dec()
        total_comp_no_flash += comp_ln6
        total_w_mem += ln_w6_mem
        total_b_mem += ln_b6_mem
        total_act_mem += ln_act6
        total_mem += (ln_w6_mem + ln_b6_mem + ln_act6)
        rows.append(
            mkrow(
                name=f"Decoder Layer {layer_name} - FFN LayerNorm",
                wnum=ln_w6, bnum=ln_b6, comp=comp_ln6,
                wmem=ln_w6_mem, bmem=ln_b6_mem, actmem=ln_act6
            )
        )

    w_out_lin = d_model * V_tgt
    b_out_lin = V_tgt
    w_out_lin_mem = w_out_lin * precision
    b_out_lin_mem = b_out_lin * precision
    comp_out_lin = batch_size * L_t * d_model * V_tgt * 2
    out_lin_act = batch_size * L_t * V_tgt * precision
    total_comp_no_flash += comp_out_lin
    total_w_mem += w_out_lin_mem
    total_b_mem += b_out_lin_mem
    total_act_mem += out_lin_act
    total_mem += (w_out_lin_mem + b_out_lin_mem + out_lin_act)
    rows.append(
        mkrow(
            name="Output Linear",
            input_shape=f"({batch_size},{L_t},{d_model})",
            output_shape=f"({batch_size},{L_t},{V_tgt})",
            wnum=w_out_lin, bnum=b_out_lin, comp=comp_out_lin,
            wmem=w_out_lin_mem, bmem=b_out_lin_mem,
            actmem=out_lin_act
        )
    )

    total_comp_with_flash = 0
    if param["use_flash_attention"]:
        comp_tile = flash_single_tile_comp(tile_m, tile_n, tile_k)
        param_tile = flash_single_tile_params(d_model, tile_k)
        qkv_tile = flash_single_tile_qkv_data(tile_m, tile_n, tile_k)
        kv_tile = flash_single_tile_kv_cache(tile_n, tile_k)
        qkv_mem = qkv_tile * precision
        kv_mem = kv_tile * precision
        total_comp_with_flash = comp_tile
        rows.append(
            mkrow(
                name="WithFlash Single-Tile Summary",
                input_shape=f"(tile_m={tile_m}, tile_k={tile_k})",
                output_shape=f"(tile_m={tile_m}, tile_n={tile_n})",
                wnum=param_tile, bnum=0,
                comp=comp_tile,
                wmem=0, bmem=0,
                actmem=qkv_mem,
                kvcache=kv_mem,
                qkv=qkv_tile
            )
        )

    rows.append(
        mkrow(
            name="Computation Summary",
            comp=total_comp_no_flash
        )
    )

    return (
        rows,
        total_mem,
        total_w_mem,
        total_b_mem,
        total_act_mem,
        total_kv_mem,
        total_comp_no_flash,
        total_comp_with_flash
    )

(module_list,
 total_mem,
 total_w_mem,
 total_b_mem,
 total_act_mem,
 total_kv_mem,
 total_comp_no_flash,
 total_comp_with_flash) = calc_details()

df = pd.DataFrame(module_list)

def classify_mod_type(mname):
    low = mname.lower()
    if "encoder" in low:
        return "Encoder"
    elif "decoder" in low or "output linear" in low:
        return "Decoder"
    else:
        return "Other"

df["Module Type"] = df["Module"].apply(classify_mod_type)
df["Input Elements"] = df["Input Shape"].apply(calculate_elements)
df["Output Elements"] = df["Output Shape"].apply(calculate_elements)

st.header("Module Details")
tabs = st.tabs(["All Modules", "Encoder", "Decoder", "Other"])
tab_all, tab_enc, tab_dec, tab_oth = tabs

with tab_all:
    st.subheader("All Modules")
    st.table(df)
with tab_enc:
    st.subheader("Encoder Modules")
    st.table(df[df["Module Type"]=="Encoder"])
with tab_dec:
    st.subheader("Decoder Modules")
    st.table(df[df["Module Type"]=="Decoder"])
with tab_oth:
    st.subheader("Other (Summaries / SingleTile)")
    st.table(df[df["Module Type"]=="Other"])

st.write(f"**Total Memory**: {format_memory_size(total_mem)}")
st.write(f"- Weight Memory: {format_memory_size(total_w_mem)}")
st.write(f"- Bias Memory: {format_memory_size(total_b_mem)}")
st.write(f"- Activation Memory: {format_memory_size(total_act_mem)}")
if p["use_kv_cache"]:
    st.write(f"- KV Cache Memory (per head allocated): {format_memory_size(total_kv_mem)}")

st.write(f"**Without Flash** total computations: {int(total_comp_no_flash):,}")
if p["use_flash_attention"]:
    st.write(f"**With Flash (SingleTile)** computations: {int(total_comp_with_flash):,}")

st.header("Graphs")
if "show_graphs" not in st.session_state:
    st.session_state["show_graphs"] = True

if "graph_filters" not in st.session_state:
    st.session_state["graph_filters"] = {
        "Weights (#)": True,
        "Biases (#)": True,
        "Computations": True,
        "Weight Memory": True,
        "Bias Memory": True,
        "Activation Memory": True,
        "KV Cache Memory": True,
        "Input/Output Elements": True,
        "Show Input Embedding": True,
        "Show Output Linear": True
    }

col1, col2 = st.columns(2)
with col1:
    if st.button("Toggle Graph Visibility"):
        st.session_state["show_graphs"] = not st.session_state["show_graphs"]
with col2:
    if st.button("Reset Graph Filters"):
        st.session_state["graph_filters"] = {
            "Weights (#)": True,
            "Biases (#)": True,
            "Computations": True,
            "Weight Memory": True,
            "Bias Memory": True,
            "Activation Memory": True,
            "KV Cache Memory": True,
            "Input/Output Elements": True,
            "Show Input Embedding": True,
            "Show Output Linear": True
        }

if st.session_state["show_graphs"]:
    st.subheader("Graph Filters")
    with st.expander("Configure Graph Filters"):
        st.session_state["graph_filters"]["Weights (#)"] = st.checkbox(
            "Weights (#)", value=st.session_state["graph_filters"]["Weights (#)"]
        )
        st.session_state["graph_filters"]["Biases (#)"] = st.checkbox(
            "Biases (#)", value=st.session_state["graph_filters"]["Biases (#)"]
        )
        st.session_state["graph_filters"]["Computations"] = st.checkbox(
            "Computations", value=st.session_state["graph_filters"]["Computations"]
        )
        st.session_state["graph_filters"]["Weight Memory"] = st.checkbox(
            "Weight Memory", value=st.session_state["graph_filters"]["Weight Memory"]
        )
        st.session_state["graph_filters"]["Bias Memory"] = st.checkbox(
            "Bias Memory", value=st.session_state["graph_filters"]["Bias Memory"]
        )
        st.session_state["graph_filters"]["Activation Memory"] = st.checkbox(
            "Activation Memory", value=st.session_state["graph_filters"]["Activation Memory"]
        )
        st.session_state["graph_filters"]["KV Cache Memory"] = st.checkbox(
            "KV Cache Memory", value=st.session_state["graph_filters"]["KV Cache Memory"]
        )
        st.session_state["graph_filters"]["Input/Output Elements"] = st.checkbox(
            "Input/Output Elements", value=st.session_state["graph_filters"]["Input/Output Elements"]
        )
        st.session_state["graph_filters"]["Show Input Embedding"] = st.checkbox(
            "Include Input Embedding in Graphs",
            value=st.session_state["graph_filters"]["Show Input Embedding"]
        )
        st.session_state["graph_filters"]["Show Output Linear"] = st.checkbox(
            "Include Output Linear in Graphs",
            value=st.session_state["graph_filters"]["Show Output Linear"]
        )

    display_scope = st.radio("Display Modules for Graph:", ("Encoder Only", "Decoder Only", "Both"), index=2)
    axis_scale = st.radio("Y-axis Scale", ("Linear", "Log"), index=0)
    graph_height_level = st.radio("Graph Height", ("Small", "Medium", "Large"), index=1)
    graph_height = {"Small": 450, "Medium": 900, "Large": 1350}[graph_height_level]

    def filter_df_for_graph(dfin):
        if display_scope == "Encoder Only":
            df_f = dfin[dfin["Module Type"] == "Encoder"]
        elif display_scope == "Decoder Only":
            df_f = dfin[dfin["Module Type"] == "Decoder"]
        else:
            df_f = dfin[dfin["Module Type"].isin(["Encoder", "Decoder"])]
        if not st.session_state["graph_filters"]["Show Input Embedding"]:
            df_f = df_f[~df_f["Module"].str.contains("Encoder Input Embedding")]
            df_f = df_f[~df_f["Module"].str.contains("Decoder Input Embedding")]
        if not st.session_state["graph_filters"]["Show Output Linear"]:
            df_f = df_f[df_f["Module"] != "Output Linear"]
        return df_f

    def set_axis_scale(fig):
        if axis_scale == "Log":
            fig.update_yaxes(type="log")
        fig.update_layout(height=graph_height)
        return fig

    def replace_zeros_for_log(values):
        if axis_scale != "Log":
            return values
        if isinstance(values, pd.Series):
            return values.replace(0, 0.1)
        return [v if v != 0 else 0.1 for v in values]

    if st.session_state["graph_filters"]["Weights (#)"] or st.session_state["graph_filters"]["Biases (#)"]:
        st.subheader("Weights and Biases (#)")
        wdf = df[["Module", "Weights (#)", "Biases (#)", "Module Type"]].copy()
        wdf["Weights (#)"] = wdf["Weights (#)"].apply(parse_int_value)
        wdf["Biases (#)"] = wdf["Biases (#)"].apply(parse_int_value)
        wdf = filter_df_for_graph(wdf)
        wdf_melted = pd.melt(
            wdf,
            id_vars=["Module", "Module Type"],
            value_vars=["Weights (#)", "Biases (#)"],
            var_name="ParamType",
            value_name="Count"
        )
        active = []
        if st.session_state["graph_filters"]["Weights (#)"]:
            active.append("Weights (#)")
        if st.session_state["graph_filters"]["Biases (#)"]:
            active.append("Biases (#)")
        wdf_melted = wdf_melted[wdf_melted["ParamType"].isin(active)]
        if len(wdf_melted) > 0:
            wdf_melted["Count"] = replace_zeros_for_log(wdf_melted["Count"])
            fig = px.bar(
                wdf_melted, x="Module", y="Count", color="ParamType",
                title="Weights/Biases (#) per Module",
                labels={"Count": "Count (words)", "Module": "Module"},
                barmode="group"
            )
            fig = set_axis_scale(fig)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No modules for Weights/Biases (#).")

    if st.session_state["graph_filters"]["Computations"]:
        st.subheader("Computations")
        cdf = df[["Module", "Computations", "Module Type"]].copy()
        cdf["Computations"] = cdf["Computations"].apply(parse_float_value)
        cdf = filter_df_for_graph(cdf)
        if len(cdf) > 0:
            cdf["Computations"] = replace_zeros_for_log(cdf["Computations"])
            fig = px.bar(
                cdf, x="Module", y="Computations", color="Module Type",
                title="Computations per Module",
                labels={"Computations": "Computations (ops)", "Module": "Module"}
            )
            fig = set_axis_scale(fig)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No modules for Computations under current filter.")

    mem_cols = ["Weight Memory", "Bias Memory", "Activation Memory", "KV Cache Memory"]
    sel_mems = []
    if st.session_state["graph_filters"]["Weight Memory"]:
        sel_mems.append("Weight Memory")
    if st.session_state["graph_filters"]["Bias Memory"]:
        sel_mems.append("Bias Memory")
    if st.session_state["graph_filters"]["Activation Memory"]:
        sel_mems.append("Activation Memory")
    if st.session_state["graph_filters"]["KV Cache Memory"]:
        sel_mems.append("KV Cache Memory")

    if len(sel_mems) > 0:
        st.subheader("Memory Usage")
        mdf = df[["Module", "Module Type"] + mem_cols].copy()
        mdf = filter_df_for_graph(mdf)
        def parse_mem(s):
            s = s.strip()
            if "GB" in s:
                return float(s.replace(" GB", "")) * (1024**3)
            elif "MB" in s:
                return float(s.replace(" MB", "")) * (1024**2)
            elif "KB" in s:
                return float(s.replace(" KB", "")) * 1024
            elif "B" in s:
                return float(s.replace(" B", ""))
            else:
                return 0
        for c in mem_cols:
            mdf[c + "_bytes"] = mdf[c].apply(parse_mem)
        melted = pd.melt(
            mdf,
            id_vars=["Module", "Module Type"],
            value_vars=[c + "_bytes" for c in mem_cols],
            var_name="MemType",
            value_name="Memory (bytes)"
        )
        melted["MemType"] = melted["MemType"].str.replace("_bytes", "")
        melted = melted[melted["MemType"].isin(sel_mems)]
        if len(melted) > 0:
            melted["Memory (bytes)"] = replace_zeros_for_log(melted["Memory (bytes)"])
            fig = px.bar(
                melted,
                x="Module", y="Memory (bytes)", color="MemType",
                title="Memory Usage per Module",
                labels={"Memory (bytes)": "Memory (bytes)", "Module": "Module"},
                barmode="stack"
            )
            fig = set_axis_scale(fig)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No memory columns matched the filter.")

    if st.session_state["graph_filters"]["Input/Output Elements"]:
        st.subheader("Input vs Output Tensor Elements")
        iodf = df[["Module", "Input Elements", "Output Elements", "Module Type"]].copy()
        iodf = filter_df_for_graph(iodf)
        if len(iodf) > 0:
            iodf["Input Elements"] = iodf["Input Elements"].astype(int)
            iodf["Output Elements"] = iodf["Output Elements"].astype(int)
            iodf_melted = pd.melt(
                iodf,
                id_vars=["Module", "Module Type"],
                value_vars=["Input Elements", "Output Elements"],
                var_name="Side",
                value_name="Element Count"
            )
            if len(iodf_melted) > 0:
                iodf_melted["Element Count"] = replace_zeros_for_log(iodf_melted["Element Count"])
                fig = px.bar(iodf_melted, x="Module", y="Element Count", color="Side",
                            title="Input vs Output Element Count per Module",
                            labels={"Element Count": "Element Count (words)", "Module": "Module", "Side": "Side"},
                            barmode="group")
                fig = set_axis_scale(fig)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No Input/Output elements matched filter.")
        else:
            st.info("No modules for Input/Output elements under current filter.")

    # ── Aggregated Element Count（入力側内訳 vs 出力） ──
    # Computation Summary 行は除外
    df_agg = df[df["Module"] != "Computation Summary"].copy()
    df_agg = filter_df_for_graph(df_agg)
    # 数値化：入力側の内訳
    df_agg["Input_Activation"] = df_agg["Input Elements"]
    df_agg["Weights_Count"] = df_agg["Weights (#)"].apply(parse_int_value)
    df_agg["Biases_Count"] = df_agg["Biases (#)"].apply(parse_int_value)
    df_agg["KV_Cache_Count"] = df_agg["KV Cache (#)"].apply(parse_int_value)
    # 出力側はそのまま
    df_agg["Output_Activation"] = df_agg["Output Elements"]
    # グループ化して各モジュール毎に合計を算出
    df_grouped = df_agg.groupby("Module").agg({
        "Input_Activation": "sum",
        "Weights_Count": "sum",
        "Biases_Count": "sum",
        "KV_Cache_Count": "sum",
        "Output_Activation": "sum"
    }).reset_index()
    modules = df_grouped["Module"].tolist()
    # 数値軸の x 座標は、各モジュールのインデックスを用い、入力側は index - 0.2、出力側は index + 0.2 とする
    module_indices = list(range(len(modules)))
    input_x = [i - 0.2 for i in module_indices]
    output_x = [i + 0.2 for i in module_indices]
    input_activation = replace_zeros_for_log(df_grouped["Input_Activation"].tolist())
    weights = replace_zeros_for_log(df_grouped["Weights_Count"].tolist())
    biases = replace_zeros_for_log(df_grouped["Biases_Count"].tolist())
    cache = replace_zeros_for_log(df_grouped["KV_Cache_Count"].tolist())
    output_vals = replace_zeros_for_log(df_grouped["Output_Activation"].tolist())

    fig_agg = go.Figure()
    # 入力側：各内訳を積み上げ表示（同じ x 値）
    fig_agg.add_trace(go.Bar(
        x=input_x,
        y=input_activation,
        name="Activation",
        marker_color="blue"
    ))
    fig_agg.add_trace(go.Bar(
        x=input_x,
        y=weights,
        name="Weights",
        marker_color="red"
    ))
    fig_agg.add_trace(go.Bar(
        x=input_x,
        y=biases,
        name="Biases",
        marker_color="green"
    ))
    fig_agg.add_trace(go.Bar(
        x=input_x,
        y=cache,
        name="KV Cache",
        marker_color="orange"
    ))
    # 出力側：単独表示
    fig_agg.add_trace(go.Bar(
        x=output_x,
        y=output_vals,
        name="Output",
        marker_color="purple"
    ))
    fig_agg.update_layout(
        barmode="stack",
        title="Aggregated Element Count per Module (Input Breakdown vs Output)",
        xaxis=dict(
            tickmode="array",
            tickvals=[(i - 0.2 + i + 0.2) / 2 for i in module_indices],
            ticktext=modules,
            title="Module"
        ),
        yaxis_title="Element Count (words)"
    )
    fig_agg = set_axis_scale(fig_agg)
    st.header("Aggregated Element Count (Input Breakdown vs Output)")
    st.plotly_chart(fig_agg, width='stretch')

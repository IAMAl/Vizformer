import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os
import json  # パラメータ保存・読み込み用
import base64  # ファイルダウンロード用
import re  # 正規表現用

# バージョン情報
VERSION = "0.01.00"

# ページ設定
st.set_page_config(layout="wide")  # 表示幅を広げる

# タイトルと説明
st.title(f"Transformerモデルのパラメータ可視化ツール")

st.write(f"""
Version: {VERSION}
""")

st.write(f"""
このツールでは、Transformerモデルの主要なパラメータを調整し、
モデルの各モジュールの詳細な入出力テンソルサイズ、重みとバイアスの数、計算量、
および詳細なメモリ使用量の内訳を可視化します。
""")

# デフォルトのパラメータ値を設定
default_params = {
    'batch_size': 32,
    'L_s': 50,
    'L_t': 50,
    'd_model': 512,
    'N_enc': 6,
    'N_dec': 6,
    'h': 8,
    'd_ff': 2048,
    'V_src': 30000,
    'V_tgt': 30000,
    'dtype_str': 'float32 (4バイト)',
    'use_kv_cache': False,
    'use_flash_attention': False,
    'tile_size_m': 128,
    'tile_size_n': 128,
    'tile_size_k': 64
}

# パラメータの保存・読み込み用関数
def save_parameters(params):
    json_str = json.dumps(params, ensure_ascii=False, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="parameters.json">パラメータをダウンロード</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

def load_parameters():
    uploaded_file = st.sidebar.file_uploader("パラメータファイルを選択してください", type=["json"])
    if uploaded_file is not None:
        try:
            json_str = uploaded_file.read().decode()
            params = json.loads(json_str)
            return params
        except Exception as e:
            st.sidebar.error(f"パラメータの読み込みに失敗しました: {e}")
    return None

# サイドバーでパラメータの入力
st.sidebar.header("モデルパラメータの設定")

# パラメータのリセットボタン
if st.sidebar.button("パラメータをリセット"):
    st.session_state['params'] = default_params.copy()
else:
    # セッションステートにパラメータがない場合、デフォルト値を設定
    if 'params' not in st.session_state:
        st.session_state['params'] = default_params.copy()

# パラメータの読み込み
loaded_params = load_parameters()
if loaded_params:
    st.session_state['params'] = loaded_params
    st.sidebar.success("パラメータを読み込みました。")

params = st.session_state['params']

# パラメータ入力
batch_size = st.sidebar.number_input("バッチサイズ (batch_size)", min_value=1, value=params['batch_size'])
L_s = st.sidebar.number_input("ソースシーケンス長 (L_s)", min_value=1, value=params['L_s'])
L_t = st.sidebar.number_input("ターゲットシーケンス長 (L_t)", min_value=1, value=params['L_t'])
d_model = st.sidebar.number_input("埋め込み次元数 (d_model)", min_value=1, value=params['d_model'])
N_enc = st.sidebar.number_input("エンコーダのレイヤー数 (N_enc)", min_value=1, value=params['N_enc'])
N_dec = st.sidebar.number_input("デコーダのレイヤー数 (N_dec)", min_value=1, value=params['N_dec'])
h = st.sidebar.number_input("アテンションヘッド数 (h)", min_value=1, value=params['h'])
d_ff = st.sidebar.number_input("フィードフォワードネットワークの次元数 (d_ff)", min_value=1, value=params['d_ff'])
V_src = st.sidebar.number_input("ソース語彙サイズ (V_src)", min_value=1, value=params['V_src'])
V_tgt = st.sidebar.number_input("ターゲット語彙サイズ (V_tgt)", min_value=1, value=params['V_tgt'])

# データ型の選択
dtype_options = {
    'float64 (8バイト)': 8,
    'float32 (4バイト)': 4,
    'float16 (2バイト)': 2,
    'int8 (1バイト)': 1
}
dtype_str = st.sidebar.selectbox("データ型の選択", options=list(dtype_options.keys()), index=list(dtype_options.keys()).index(params['dtype_str']))
precision = dtype_options[dtype_str]

# KVキャッシュの有無
use_kv_cache = st.sidebar.checkbox("デコーダでKVキャッシュを使用する", value=params['use_kv_cache'])

# FlashAttentionの有無
use_flash_attention = st.sidebar.checkbox("FlashAttentionを使用する", value=params['use_flash_attention'])

# タイルサイズの設定（FlashAttention用）
if use_flash_attention:
    st.sidebar.subheader("タイルサイズの設定（FlashAttention）")
    tile_size_m = st.sidebar.number_input("タイルサイズ（M次元）", min_value=1, value=params['tile_size_m'])
    tile_size_n = st.sidebar.number_input("タイルサイズ（N次元）", min_value=1, value=params['tile_size_n'])
    tile_size_k = st.sidebar.number_input("タイルサイズ（K次元）", min_value=1, value=params['tile_size_k'])
else:
    tile_size_m = tile_size_n = tile_size_k = None

# 現在のパラメータをセッションステートに保存
st.session_state['params'] = {
    'batch_size': batch_size,
    'L_s': L_s,
    'L_t': L_t,
    'd_model': d_model,
    'N_enc': N_enc,
    'N_dec': N_dec,
    'h': h,
    'd_ff': d_ff,
    'V_src': V_src,
    'V_tgt': V_tgt,
    'dtype_str': dtype_str,
    'use_kv_cache': use_kv_cache,
    'use_flash_attention': use_flash_attention,
    'tile_size_m': tile_size_m if tile_size_m else 128,
    'tile_size_n': tile_size_n if tile_size_n else 128,
    'tile_size_k': tile_size_k if tile_size_k else 64
}

# パラメータの保存ボタン
if st.sidebar.button("パラメータを保存"):
    save_parameters(st.session_state['params'])

# メモリ使用量の単位変換
def format_memory_size(bytes_size):
    if bytes_size >= 1024 ** 3:
        return f"{bytes_size / (1024 ** 3):.2f} GB"
    elif bytes_size >= 1024 ** 2:
        return f"{bytes_size / (1024 ** 2):.2f} MB"
    elif bytes_size >= 1024:
        return f"{bytes_size / 1024:.2f} KB"
    else:
        return f"{bytes_size} B"

# FlashAttentionの計算量を計算する関数
def calculate_computation_with_flash_attention(batch_size, h, seq_len_q, seq_len_k, d_k, tile_size_m, tile_size_n, tile_size_k):
    """FlashAttentionの計算量を計算"""
    # タイルの数を計算
    num_tiles_m = np.ceil(seq_len_q / tile_size_m)
    num_tiles_n = np.ceil(seq_len_k / tile_size_n)
    num_tiles_k = np.ceil(d_k / tile_size_k)

    # 1タイルあたりの計算量
    computation_per_tile = (
        2 * tile_size_m * tile_size_n * tile_size_k +  # QK^T
        tile_size_m * tile_size_n +                    # Softmax
        2 * tile_size_m * tile_size_n * tile_size_k    # AV
    )

    # 全タイルの合計計算量
    total_computation = (
        batch_size * h *
        num_tiles_m * num_tiles_n * num_tiles_k *
        computation_per_tile
    )

    return total_computation

# タイルの実行回数とデータ再利用回数を計算する関数
def calculate_tile_execution(m_len, n_len, k_len, tile_size_m, tile_size_n, tile_size_k):
    """タイルの実行回数とデータ再利用回数を計算

    Returns:
        tuple: (total_tiles, avg_reuse)
        - total_tiles: 実行する必要があるタイルの総数
        - avg_reuse: 各データの平均再利用回数（整数）
    """
    # タイル分割数を計算
    num_tiles_m = int(np.ceil(m_len / tile_size_m))
    num_tiles_n = int(np.ceil(n_len / tile_size_n))
    num_tiles_k = int(np.ceil(k_len / tile_size_k))

    # 総タイル数
    total_tiles = num_tiles_m * num_tiles_n * num_tiles_k

    # データ再利用回数の計算
    q_reuse = num_tiles_n * num_tiles_k
    k_reuse = num_tiles_m * num_tiles_k
    v_reuse = num_tiles_m * num_tiles_n

    # 平均再利用回数を整数に丸める
    avg_reuse = int(np.round((q_reuse + k_reuse + v_reuse) / 3))

    return total_tiles, avg_reuse

# グローバルに calculate_elements 関数を定義
def calculate_elements(size_str):
    # 正規表現で全てのタプルを抽出
    tuples = re.findall(r'\(([^)]+)\)', size_str)
    total_elements = 0
    for tup in tuples:
        # 各要素を分割し、数値か変数かを判断
        elements = tup.split(',')
        product = 1
        for elem in elements:
            elem = elem.strip()
            if elem.isdigit():
                product *= int(elem)
            else:
                # 変数名の場合、セッションステートから値を取得
                value = st.session_state['params'].get(elem, 1)
                product *= int(value)
        total_elements += product
    return total_elements

# 各モジュールの計算
def calculate_module_details():
    module_details = []
    total_memory = 0
    total_weight_memory = 0
    total_bias_memory = 0
    total_activation_memory = 0
    total_kv_cache_memory = 0
    total_computation_without_flash = 0
    total_computation_with_flash = 0

    # メモリ使用量の計算のためのヘルパー関数
    def calc_tensor_size(shape):
        num_elements = np.prod(shape)
        return num_elements * precision  # バイト数

    # レイヤーノーマライゼーションのパラメータ数とメモリ
    def layer_norm_params(size):
        weight_size = size  # gamma
        bias_size = size    # beta
        weight_memory = weight_size * precision
        bias_memory = bias_size * precision
        return weight_size, bias_size, weight_memory, bias_memory

    # エンコーダ入力エンベッディング層
    input_size = (batch_size, L_s)
    output_size = (batch_size, L_s, d_model)
    weight_size = V_src * d_model
    bias_size = 0
    weight_memory = weight_size * precision
    bias_memory = bias_size * precision
    activation_memory = calc_tensor_size(output_size)
    total_weight_memory += weight_memory
    total_bias_memory += bias_size
    total_activation_memory += activation_memory
    total_memory += weight_memory + bias_memory + activation_memory

    encoder_embedding = {
        'モジュール': 'エンコーダ入力エンベッディング',
        '入力サイズ': f'{input_size}',
        '出力サイズ': f'{output_size}',
        '入力テンソル形状: バッチサイズ': batch_size,
        '入力テンソル形状: シーケンス長': L_s,
        '出力テンソル形状: バッチサイズ': batch_size,
        '出力テンソル形状: シーケンス長': L_s,
        '出力テンソル形状: 埋め込み次元': d_model,
        '重みの数': f"{weight_size:,}",
        'バイアスの数': f"{bias_size:,}",
        '計算量': f"{0:,}",  # エンベディングはルックアップなので計算量は考慮しない
        '重みメモリ': format_memory_size(weight_memory),
        'バイアスメモリ': format_memory_size(bias_memory),
        'アクティベーションメモリ': format_memory_size(activation_memory),
        'KVキャッシュメモリ': format_memory_size(0),
        '合計メモリ': format_memory_size(weight_memory + bias_memory + activation_memory),
        'タイル実行回数': "N/A",
        'データ再利用回数': "N/A"
    }
    module_details.append(encoder_embedding)

    # エンコーダレイヤー
    for layer in range(N_enc):
        # レイヤーノーマライゼーション
        ln_weight_size, ln_bias_size, ln_weight_memory, ln_bias_memory = layer_norm_params(d_model)

        # MHAのQ、K、Vの線形変換
        module_name = f'エンコーダレイヤー{layer + 1} - MHA Linear Projections'
        weight_size = d_model * d_model * 3
        bias_size = d_model * 3
        weight_memory = weight_size * precision
        bias_memory = bias_size * precision

        computation = batch_size * L_s * d_model * d_model * 3 * 2  # 2倍はforwardとbackward

        # アクティベーションメモリ
        qkv_size = calc_tensor_size((batch_size, L_s, d_model * 3))

        total_weight_memory += weight_memory + ln_weight_memory
        total_bias_memory += bias_size + ln_bias_size
        total_activation_memory += qkv_size
        total_memory += weight_memory + bias_memory + qkv_size + ln_weight_memory + ln_bias_memory

        total_computation_without_flash += computation

        # タイル実行回数とデータ再利用回数の計算
        total_tiles, avg_reuse = calculate_tile_execution(
            m_len=d_model,
            n_len=d_ff,
            k_len=d_model,
            tile_size_m=tile_size_m if use_flash_attention else d_model,
            tile_size_n=tile_size_n if use_flash_attention else d_ff,
            tile_size_k=tile_size_k if use_flash_attention else d_model
        )

        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'({batch_size}, {L_s}, {d_model})',
            '出力サイズ': f'QKV:({batch_size}, {L_s}, {d_model * 3})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': L_s,
            '入力テンソル形状: 埋め込み次元': d_model,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_s,
            '出力テンソル形状: 埋め込み次元': d_model * 3,
            '重みの数': f"{weight_size + ln_weight_size:,}",
            'バイアスの数': f"{bias_size + ln_bias_size:,}",
            '計算量': f"{int(computation):,}",
            '重みメモリ': format_memory_size(weight_memory + ln_weight_memory),
            'バイアスメモリ': format_memory_size(bias_memory + ln_bias_memory),
            'アクティベーションメモリ': format_memory_size(qkv_size),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(weight_memory + bias_memory + qkv_size + ln_weight_memory + ln_bias_memory),
            'タイル実行回数': total_tiles,
            'データ再利用回数': avg_reuse
        })

        # スケールドドットプロダクトアテンションを細分化
        # QK^T
        module_name = f'エンコーダレイヤー{layer + 1} - Attention QK^T'
        computation_qk = batch_size * h * L_s * L_s * (d_model // h) * 2
        total_computation_without_flash += computation_qk
        attn_scores_size = calc_tensor_size((batch_size, h, L_s, L_s))
        total_activation_memory += attn_scores_size
        total_memory += attn_scores_size

        # タイル実行回数とデータ再利用回数の計算
        total_tiles_qk, avg_reuse_qk = calculate_tile_execution(
            m_len=L_s,
            n_len=L_s,
            k_len=(d_model // h),
            tile_size_m=tile_size_m if use_flash_attention else L_s,
            tile_size_n=tile_size_n if use_flash_attention else L_s,
            tile_size_k=tile_size_k if use_flash_attention else (d_model // h)
        )

        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'Q:({batch_size}, {h}, {L_s}, d_k), K:({batch_size}, {h}, {L_s}, d_k)',
            '出力サイズ': f'Scores:({batch_size}, {h}, {L_s}, {L_s})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': L_s,
            '入力テンソル形状: 埋め込み次元': d_model,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_s,
            '出力テンソル形状: 埋め込み次元': L_s,
            '重みの数': f"0",
            'バイアスの数': f"0",
            '計算量': f"{int(computation_qk):,}",
            '重みメモリ': format_memory_size(0),
            'バイアスメモリ': format_memory_size(0),
            'アクティベーションメモリ': format_memory_size(attn_scores_size),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(attn_scores_size),
            'タイル実行回数': total_tiles_qk,
            'データ再利用回数': avg_reuse_qk
        })

        # スケーリング
        module_name = f'エンコーダレイヤー{layer + 1} - Attention Scaling'
        computation_scaling = batch_size * h * L_s * L_s
        total_computation_without_flash += computation_scaling

        # スケーリングはタイル処理が不要なため、"N/A"とします
        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'Scores:({batch_size}, {h}, {L_s}, {L_s})',
            '出力サイズ': f'Scaled Scores:({batch_size}, {h}, {L_s}, {L_s})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': L_s,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_s,
            '出力テンソル形状: 埋め込み次元': L_s,
            '重みの数': f"0",
            'バイアスの数': f"0",
            '計算量': f"{int(computation_scaling):,}",
            '重みメモリ': format_memory_size(0),
            'バイアスメモリ': format_memory_size(0),
            'アクティベーションメモリ': format_memory_size(0),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(0),
            'タイル実行回数': "N/A",
            'データ再利用回数': "N/A"
        })

        # Softmax
        module_name = f'エンコーダレイヤー{layer + 1} - Attention Softmax'
        computation_softmax = batch_size * h * L_s * L_s * 5  # 近似的にSoftmaxの計算コストを乗算5回分とする
        total_computation_without_flash += computation_softmax

        # Softmaxもタイル処理が不要なため、"N/A"とします
        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'Scaled Scores:({batch_size}, {h}, {L_s}, {L_s})',
            '出力サイズ': f'Attention Weights:({batch_size}, {h}, {L_s}, {L_s})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': L_s,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_s,
            '出力テンソル形状: 埋め込み次元': L_s,
            '重みの数': f"0",
            'バイアスの数': f"0",
            '計算量': f"{int(computation_softmax):,}",
            '重みメモリ': format_memory_size(0),
            'バイアスメモリ': format_memory_size(0),
            'アクティベーションメモリ': format_memory_size(0),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(0),
            'タイル実行回数': "N/A",
            'データ再利用回数': "N/A"
        })

        # SoftmaxとVの掛け算
        module_name = f'エンコーダレイヤー{layer + 1} - Attention Softmax×V'
        computation_av = batch_size * h * L_s * L_s * (d_model // h) * 2
        total_computation_without_flash += computation_av
        attn_output_size = calc_tensor_size((batch_size, L_s, d_model))
        total_activation_memory += attn_output_size
        total_memory += attn_output_size

        # タイル実行回数とデータ再利用回数の計算
        total_tiles_av, avg_reuse_av = calculate_tile_execution(
            m_len=L_s,
            n_len=L_s,
            k_len=(d_model // h),
            tile_size_m=tile_size_m if use_flash_attention else L_s,
            tile_size_n=tile_size_n if use_flash_attention else L_s,
            tile_size_k=tile_size_k if use_flash_attention else (d_model // h)
        )

        # KVキャッシュメモリ
        if use_kv_cache:
            kv_cache_size = batch_size * L_s * d_model * 2  # KとV
            kv_cache_memory = kv_cache_size * precision * N_enc  # エンコーダの全レイヤー分
            if layer == 0:
                total_kv_cache_memory = kv_cache_memory  # 合計に加算
        else:
            kv_cache_memory = 0

        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'Attention Weights:({batch_size}, {h}, {L_s}, {L_s}), V:({batch_size}, {h}, {L_s}, d_v)',
            '出力サイズ': f'Context:({batch_size}, {L_s}, {d_model})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': L_s,
            '入力テンソル形状: 埋め込み次元': d_model,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_s,
            '出力テンソル形状: 埋め込み次元': d_model,
            '重みの数': f"0",
            'バイアスの数': f"0",
            '計算量': f"{int(computation_av):,}",
            '重みメモリ': format_memory_size(0),
            'バイアスメモリ': format_memory_size(0),
            'アクティベーションメモリ': format_memory_size(attn_output_size),
            'KVキャッシュメモリ': format_memory_size(kv_cache_memory) if layer == 0 else format_memory_size(0),
            '合計メモリ': format_memory_size(attn_output_size + (kv_cache_memory if layer == 0 else 0)),
            'タイル実行回数': total_tiles_av,
            'データ再利用回数': avg_reuse_av
        })

        # MHAの最終線形層
        module_name = f'エンコーダレイヤー{layer + 1} - MHA Output Linear'
        weight_size = d_model * d_model
        bias_size = d_model
        weight_memory = weight_size * precision
        bias_memory = bias_size * precision

        computation = batch_size * L_s * d_model * d_model * 2  # 2倍はforwardとbackward

        total_weight_memory += weight_memory
        total_bias_memory += bias_size
        total_memory += weight_memory + bias_size * precision

        total_computation_without_flash += computation

        # タイル実行回数とデータ再利用回数の計算
        total_tiles_output, avg_reuse_output = calculate_tile_execution(
            m_len=d_model,
            n_len=d_model,
            k_len=d_model,
            tile_size_m=tile_size_m if use_flash_attention else d_model,
            tile_size_n=tile_size_n if use_flash_attention else d_model,
            tile_size_k=tile_size_k if use_flash_attention else d_model
        )

        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'({batch_size}, {L_s}, {d_model})',
            '出力サイズ': f'({batch_size}, {L_s}, {d_model})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': L_s,
            '入力テンソル形状: 埋め込み次元': d_model,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_s,
            '出力テンソル形状: 埋め込み次元': d_model,
            '重みの数': f"{weight_size:,}",
            'バイアスの数': f"{bias_size:,}",
            '計算量': f"{int(computation):,}",
            '重みメモリ': format_memory_size(weight_memory),
            'バイアスメモリ': format_memory_size(bias_memory),
            'アクティベーションメモリ': format_memory_size(0),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(weight_memory + bias_memory),
            'タイル実行回数': total_tiles_output,
            'データ再利用回数': avg_reuse_output
        })

        # スキップ接続とレイヤーノーマライゼーション
        module_name = f'エンコーダレイヤー{layer + 1} - Skip Connection and LayerNorm'
        ln_weight_size, ln_bias_size, ln_weight_memory, ln_bias_memory = layer_norm_params(d_model)

        skip_connection_memory = calc_tensor_size((batch_size, L_s, d_model))

        total_bias_memory += ln_bias_memory
        total_weight_memory += ln_weight_memory
        total_activation_memory += skip_connection_memory
        total_memory += ln_weight_memory + ln_bias_memory + skip_connection_memory

        # スキップ接続はタイル処理が不要なため、"N/A"とします
        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'',
            '出力サイズ': f'',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': L_s,
            '入力テンソル形状: 埋め込み次元': d_model,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_s,
            '出力テンソル形状: 埋め込み次元': d_model,
            '重みの数': f"{ln_weight_size:,}",
            'バイアスの数': f"{ln_bias_size:,}",
            '計算量': f"{0}",
            '重みメモリ': format_memory_size(ln_weight_memory),
            'バイアスメモリ': format_memory_size(ln_bias_memory),
            'アクティベーションメモリ': format_memory_size(skip_connection_memory),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(ln_weight_memory + ln_bias_memory + skip_connection_memory),
            'タイル実行回数': "N/A",
            'データ再利用回数': "N/A"
        })

        # FFNの最初の線形層と活性化関数
        module_name = f'エンコーダレイヤー{layer + 1} - FFN Layer 1 + Activation'
        weight_size = d_model * d_ff
        bias_size = d_ff
        weight_memory = weight_size * precision
        bias_memory = bias_size * precision

        computation = batch_size * L_s * d_model * d_ff * 2  # 2倍はforwardとbackward

        hidden_layer_size = calc_tensor_size((batch_size, L_s, d_ff))

        total_weight_memory += weight_memory
        total_bias_memory += bias_size
        total_activation_memory += hidden_layer_size
        total_memory += weight_memory + bias_size * precision + hidden_layer_size

        total_computation_without_flash += computation

        # タイル実行回数とデータ再利用回数の計算
        total_tiles_ffn1, avg_reuse_ffn1 = calculate_tile_execution(
            m_len=d_model,
            n_len=d_ff,
            k_len=d_model,
            tile_size_m=tile_size_m if use_flash_attention else d_model,
            tile_size_n=tile_size_n if use_flash_attention else d_ff,
            tile_size_k=tile_size_k if use_flash_attention else d_model
        )

        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'({batch_size}, {L_s}, {d_model})',
            '出力サイズ': f'({batch_size}, {L_s}, {d_ff})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': L_s,
            '入力テンソル形状: 埋め込み次元': d_model,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_s,
            '重みの数': f"{weight_size:,}",
            'バイアスの数': f"{bias_size:,}",
            '計算量': f"{int(computation):,}",
            '重みメモリ': format_memory_size(weight_memory),
            'バイアスメモリ': format_memory_size(bias_size * precision),
            'アクティベーションメモリ': format_memory_size(hidden_layer_size),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(weight_memory + bias_size * precision + hidden_layer_size),
            'タイル実行回数': total_tiles_ffn1,
            'データ再利用回数': avg_reuse_ffn1
        })

        # FFNの2番目の線形層
        module_name = f'エンコーダレイヤー{layer + 1} - FFN Layer 2'
        weight_size = d_ff * d_model
        bias_size = d_model
        weight_memory = weight_size * precision
        bias_memory = bias_size * precision

        computation = batch_size * L_s * d_ff * d_model * 2  # 2倍はforwardとbackward

        total_weight_memory += weight_size * precision
        total_bias_memory += bias_size * precision
        total_memory += weight_memory + bias_size * precision

        total_computation_without_flash += computation

        # タイル実行回数とデータ再利用回数の計算
        total_tiles_ffn2, avg_reuse_ffn2 = calculate_tile_execution(
            m_len=d_ff,
            n_len=d_model,
            k_len=d_ff,
            tile_size_m=tile_size_m if use_flash_attention else d_ff,
            tile_size_n=tile_size_n if use_flash_attention else d_model,
            tile_size_k=tile_size_k if use_flash_attention else d_ff
        )

        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'({batch_size}, {L_s}, {d_ff})',
            '出力サイズ': f'({batch_size}, {L_s}, {d_model})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': L_s,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_s,
            '出力テンソル形状: 埋め込み次元': d_model,
            '重みの数': f"{weight_size:,}",
            'バイアスの数': f"{bias_size:,}",
            '計算量': f"{int(computation):,}",
            '重みメモリ': format_memory_size(weight_memory),
            'バイアスメモリ': format_memory_size(bias_size * precision),
            'アクティベーションメモリ': format_memory_size(0),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(weight_memory + bias_size * precision),
            'タイル実行回数': total_tiles_ffn2,
            'データ再利用回数': avg_reuse_ffn2
        })

        # スキップ接続とレイヤーノーマライゼーション（FFN後）
        module_name = f'エンコーダレイヤー{layer + 1} - FFN Skip Connection and LayerNorm'
        ln_weight_size, ln_bias_size, ln_weight_memory, ln_bias_memory = layer_norm_params(d_model)

        skip_connection_memory = calc_tensor_size((batch_size, L_s, d_model))

        total_bias_memory += ln_bias_memory
        total_weight_memory += ln_weight_memory
        total_activation_memory += skip_connection_memory
        total_memory += ln_weight_memory + ln_bias_memory + skip_connection_memory

        # スキップ接続はタイル処理が不要なため、"N/A"とします
        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'',
            '出力サイズ': f'',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': L_s,
            '入力テンソル形状: 埋め込み次元': d_model,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_s,
            '出力テンソル形状: 埋め込み次元': d_model,
            '重みの数': f"{ln_weight_size:,}",
            'バイアスの数': f"{ln_bias_size:,}",
            '計算量': f"{0}",
            '重みメモリ': format_memory_size(ln_weight_memory),
            'バイアスメモリ': format_memory_size(ln_bias_memory),
            'アクティベーションメモリ': format_memory_size(skip_connection_memory),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(ln_weight_memory + ln_bias_memory + skip_connection_memory),
            'タイル実行回数': "N/A",
            'データ再利用回数': "N/A"
        })

    # デコーダ入力エンベッディング層
    input_size = (batch_size, L_t)
    output_size = (batch_size, L_t, d_model)
    weight_size = V_tgt * d_model
    bias_size = 0
    weight_memory = weight_size * precision
    bias_memory = bias_size * precision
    activation_memory = calc_tensor_size(output_size)
    total_weight_memory += weight_memory
    total_bias_memory += bias_size
    total_activation_memory += activation_memory
    total_memory += weight_memory + bias_size * precision + activation_memory

    decoder_embedding = {
        'モジュール': 'デコーダ入力エンベッディング',
        '入力サイズ': f'{input_size}',
        '出力サイズ': f'{output_size}',
        '入力テンソル形状: バッチサイズ': batch_size,
        '入力テンソル形状: シーケンス長': L_t,
        '出力テンソル形状: バッチサイズ': batch_size,
        '出力テンソル形状: シーケンス長': L_t,
        '出力テンソル形状: 埋め込み次元': d_model,
        '重みの数': f"{weight_size:,}",
        'バイアスの数': f"{bias_size:,}",
        '計算量': f"{0:,}",
        '重みメモリ': format_memory_size(weight_memory),
        'バイアスメモリ': format_memory_size(bias_memory),
        'アクティベーションメモリ': format_memory_size(activation_memory),
        'KVキャッシュメモリ': format_memory_size(0),
        '合計メモリ': format_memory_size(weight_memory + bias_memory + activation_memory),
        'タイル実行回数': "N/A",
        'データ再利用回数': "N/A"
    }
    module_details.append(decoder_embedding)

    # デコーダレイヤー
    for layer in range(N_dec):
        # レイヤーノーマライゼーション
        ln_weight_size, ln_bias_size, ln_weight_memory, ln_bias_memory = layer_norm_params(d_model)

        # マスクドMHAのQ、K、Vの線形変換
        module_name = f'Decoder Layer {layer + 1} - Masked MHA Linear Projections'
        weight_size = d_model * d_model * 3
        bias_size = d_model * 3
        weight_memory = weight_size * precision
        bias_memory = bias_size * precision

        computation = batch_size * L_t * d_model * d_model * 3 * 2

        qkv_size = calc_tensor_size((batch_size, L_t, d_model * 3))

        total_weight_memory += weight_memory + ln_weight_memory
        total_bias_memory += bias_size + ln_bias_size
        total_activation_memory += qkv_size
        total_memory += weight_memory + bias_memory + qkv_size + ln_weight_memory + ln_bias_memory

        total_computation_without_flash += computation

        # タイル実行回数とデータ再利用回数の計算
        total_tiles_qkv, avg_reuse_qkv = calculate_tile_execution(
            m_len=d_model,
            n_len=d_model,
            k_len=d_model,
            tile_size_m=tile_size_m if use_flash_attention else d_model,
            tile_size_n=tile_size_n if use_flash_attention else d_model,
            tile_size_k=tile_size_k if use_flash_attention else d_model
        )

        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'({batch_size}, {L_t}, {d_model})',
            '出力サイズ': f'QKV:({batch_size}, {L_t}, {d_model * 3})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': L_t,
            '入力テンソル形状: 埋め込み次元': d_model,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_t,
            '出力テンソル形状: 埋め込み次元': d_model * 3,
            '重みの数': f"{weight_size + ln_weight_size:,}",
            'バイアスの数': f"{bias_size + ln_bias_size:,}",
            '計算量': f"{int(computation):,}",
            '重みメモリ': format_memory_size(weight_memory + ln_weight_memory),
            'バイアスメモリ': format_memory_size(bias_memory + ln_bias_memory),
            'アクティベーションメモリ': format_memory_size(qkv_size),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(weight_memory + bias_memory + qkv_size + ln_weight_memory + ln_bias_memory),
            'タイル実行回数': total_tiles_qkv,
            'データ再利用回数': avg_reuse_qkv
        })

        # マスクドスケールドドットプロダクトアテンションを細分化
        # QK^T
        module_name = f'Decoder Layer {layer + 1} - Masked Attention QK^T'
        if use_kv_cache:
            seq_len_q = 1
            seq_len_k = L_t
        else:
            seq_len_q = L_t
            seq_len_k = L_t

        computation_qk = batch_size * h * seq_len_q * seq_len_k * (d_model // h) * 2
        total_computation_without_flash += computation_qk
        attn_scores_size = calc_tensor_size((batch_size, h, seq_len_q, seq_len_k))
        total_activation_memory += attn_scores_size
        total_memory += attn_scores_size

        # タイル実行回数とデータ再利用回数の計算
        total_tiles_qk_dec, avg_reuse_qk_dec = calculate_tile_execution(
            m_len=seq_len_q,
            n_len=seq_len_k,
            k_len=(d_model // h),
            tile_size_m=tile_size_m if use_flash_attention else seq_len_q,
            tile_size_n=tile_size_n if use_flash_attention else seq_len_k,
            tile_size_k=tile_size_k if use_flash_attention else (d_model // h)
        )

        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'Q:({batch_size}, {h}, {seq_len_q}, d_k), K:({batch_size}, {h}, {seq_len_k}, d_k)',
            '出力サイズ': f'Scores:({batch_size}, {h}, {seq_len_q}, {seq_len_k})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': seq_len_q,
            '入力テンソル形状: 埋め込み次元': d_model,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': seq_len_q,
            '出力テンソル形状: 埋め込み次元': seq_len_k,
            '重みの数': f"0",
            'バイアスの数': f"0",
            '計算量': f"{int(computation_qk):,}",
            '重みメモリ': format_memory_size(0),
            'バイアスメモリ': format_memory_size(0),
            'アクティベーションメモリ': format_memory_size(attn_scores_size),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(attn_scores_size),
            'タイル実行回数': total_tiles_qk_dec,
            'データ再利用回数': avg_reuse_qk_dec
        })

        # スケーリング
        module_name = f'Decoder Layer {layer + 1} - Masked Attention Scaling'
        computation_scaling = batch_size * h * seq_len_q * seq_len_k
        total_computation_without_flash += computation_scaling

        # スケーリングはタイル処理が不要なため、"N/A"とします
        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'Scores:({batch_size}, {h}, {seq_len_q}, {seq_len_k})',
            '出力サイズ': f'Scaled Scores:({batch_size}, {h}, {seq_len_q}, {seq_len_k})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': seq_len_q,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': seq_len_q,
            '出力テンソル形状: 埋め込み次元': seq_len_k,
            '重みの数': f"0",
            'バイアスの数': f"0",
            '計算量': f"{int(computation_scaling):,}",
            '重みメモリ': format_memory_size(0),
            'バイアスメモリ': format_memory_size(0),
            'アクティベーションメモリ': format_memory_size(0),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(0),
            'タイル実行回数': "N/A",
            'データ再利用回数': "N/A"
        })

        # Softmax
        module_name = f'Decoder Layer {layer + 1} - Masked Attention Softmax'
        computation_softmax = batch_size * h * seq_len_q * seq_len_k * 5  # 近似的にSoftmaxの計算コストを乗算5回分とする
        total_computation_without_flash += computation_softmax

        # Softmaxもタイル処理が不要なため、"N/A"とします
        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'Scaled Scores:({batch_size}, {h}, {seq_len_q}, {seq_len_k})',
            '出力サイズ': f'Attention Weights:({batch_size}, {h}, {seq_len_q}, {seq_len_k})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': seq_len_q,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': seq_len_q,
            '出力テンソル形状: 埋め込み次元': seq_len_k,
            '重みの数': f"0",
            'バイアスの数': f"0",
            '計算量': f"{int(computation_softmax):,}",
            '重みメモリ': format_memory_size(0),
            'バイアスメモリ': format_memory_size(0),
            'アクティベーションメモリ': format_memory_size(0),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(0),
            'タイル実行回数': "N/A",
            'データ再利用回数': "N/A"
        })

        # SoftmaxとVの掛け算
        module_name = f'Decoder Layer {layer + 1} - Masked Attention Softmax×V'
        computation_av = batch_size * h * seq_len_q * seq_len_k * (d_model // h) * 2
        total_computation_without_flash += computation_av
        attn_output_size = calc_tensor_size((batch_size, L_t, d_model))
        total_activation_memory += attn_output_size
        total_memory += attn_output_size

        # タイル実行回数とデータ再利用回数の計算
        total_tiles_av_dec, avg_reuse_av_dec = calculate_tile_execution(
            m_len=L_t,
            n_len=L_t,
            k_len=(d_model // h),
            tile_size_m=tile_size_m if use_flash_attention else L_t,
            tile_size_n=tile_size_n if use_flash_attention else L_t,
            tile_size_k=tile_size_k if use_flash_attention else (d_model // h)
        )

        # KVキャッシュメモリ
        if use_kv_cache:
            kv_cache_size = batch_size * seq_len_k * d_model * 2  # KとV
            kv_cache_memory = kv_cache_size * precision * N_dec  # デコーダの全レイヤー分
            if layer == 0:
                total_kv_cache_memory = kv_cache_memory  # 合計に加算
        else:
            kv_cache_memory = 0

        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'Attention Weights:({batch_size}, {h}, {seq_len_q}, {seq_len_k}), V:({batch_size}, {h}, {seq_len_k}, d_v)',
            '出力サイズ': f'Context:({batch_size}, {L_t}, {d_model})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': seq_len_q,
            '入力テンソル形状: 埋め込み次元': d_model,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_t,
            '出力テンソル形状: 埋め込み次元': d_model,
            '重みの数': f"0",
            'バイアスの数': f"0",
            '計算量': f"{int(computation_av):,}",
            '重みメモリ': format_memory_size(0),
            'バイアスメモリ': format_memory_size(0),
            'アクティベーションメモリ': format_memory_size(attn_output_size),
            'KVキャッシュメモリ': format_memory_size(kv_cache_memory) if layer == 0 else format_memory_size(0),
            '合計メモリ': format_memory_size(attn_output_size + (kv_cache_memory if layer == 0 else 0)),
            'タイル実行回数': total_tiles_av_dec,
            'データ再利用回数': avg_reuse_av_dec
        })

        # MHAの最終線形層
        module_name = f'Decoder Layer {layer + 1} - Masked MHA Output Linear'
        weight_size = d_model * d_model
        bias_size = d_model
        weight_memory = weight_size * precision
        bias_memory = bias_size * precision

        computation = batch_size * L_t * d_model * d_model * 2

        total_weight_memory += weight_memory
        total_bias_memory += bias_size
        total_memory += weight_memory + bias_size * precision

        total_computation_without_flash += computation

        # タイル実行回数とデータ再利用回数の計算
        total_tiles_output_dec, avg_reuse_output_dec = calculate_tile_execution(
            m_len=d_model,
            n_len=d_model,
            k_len=d_model,
            tile_size_m=tile_size_m if use_flash_attention else d_model,
            tile_size_n=tile_size_n if use_flash_attention else d_model,
            tile_size_k=tile_size_k if use_flash_attention else d_model
        )

        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'({batch_size}, {L_t}, {d_model})',
            '出力サイズ': f'({batch_size}, {L_t}, {d_model})',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': L_t,
            '入力テンソル形状: 埋め込み次元': d_model,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_t,
            '出力テンソル形状: 埋め込み次元': d_model,
            '重みの数': f"{weight_size:,}",
            'バイアスの数': f"{bias_size:,}",
            '計算量': f"{int(computation):,}",
            '重みメモリ': format_memory_size(weight_memory),
            'バイアスメモリ': format_memory_size(bias_memory),
            'アクティベーションメモリ': format_memory_size(0),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(weight_memory + bias_memory),
            'タイル実行回数': total_tiles_output_dec,
            'データ再利用回数': avg_reuse_output_dec
        })

        # スキップ接続とレイヤーノーマライゼーション（Encoder-Decoder Attention後）
        module_name = f'Decoder Layer {layer + 1} - Skip Connection and LayerNorm (After Encoder-Decoder Attention)'
        ln_weight_size, ln_bias_size, ln_weight_memory, ln_bias_memory = layer_norm_params(d_model)

        skip_connection_memory = calc_tensor_size((batch_size, L_t, d_model))

        total_bias_memory += ln_bias_memory
        total_weight_memory += ln_weight_memory
        total_activation_memory += skip_connection_memory
        total_memory += ln_weight_memory + ln_bias_memory + skip_connection_memory

        # スキップ接続はタイル処理が不要なため、"N/A"とします
        module_details.append({
            'モジュール': module_name,
            '入力サイズ': f'',
            '出力サイズ': f'',
            '入力テンソル形状: バッチサイズ': batch_size,
            '入力テンソル形状: シーケンス長': L_t,
            '入力テンソル形状: 埋め込み次元': d_model,
            '出力テンソル形状: バッチサイズ': batch_size,
            '出力テンソル形状: シーケンス長': L_t,
            '出力テンソル形状: 埋め込み次元': d_model,
            '重みの数': f"{ln_weight_size:,}",
            'バイアスの数': f"{ln_bias_size:,}",
            '計算量': f"{0}",
            '重みメモリ': format_memory_size(ln_weight_memory),
            'バイアスメモリ': format_memory_size(ln_bias_memory),
            'アクティベーションメモリ': format_memory_size(skip_connection_memory),
            'KVキャッシュメモリ': format_memory_size(0),
            '合計メモリ': format_memory_size(ln_weight_memory + ln_bias_memory + skip_connection_memory),
            'タイル実行回数': "N/A",
            'データ再利用回数': "N/A"
        })

    # 出力線形変換層
    input_size = (batch_size, L_t, d_model)
    output_size = (batch_size, L_t, V_tgt)
    weight_size = d_model * V_tgt
    bias_size = V_tgt
    computation = batch_size * L_t * d_model * V_tgt * 2  # 2倍はforwardとbackward
    weight_memory = weight_size * precision
    bias_memory = bias_size * precision
    activation_memory = calc_tensor_size(output_size)
    total_weight_memory += weight_memory
    total_bias_memory += bias_size * precision
    total_activation_memory += activation_memory
    total_memory += weight_memory + bias_size * precision + activation_memory

    total_computation_without_flash += computation

    # タイル実行回数とデータ再利用回数の計算
    total_tiles_out, avg_reuse_out = calculate_tile_execution(
        m_len=d_model,
        n_len=V_tgt,
        k_len=d_model,
        tile_size_m=tile_size_m if use_flash_attention else d_model,
        tile_size_n=tile_size_n if use_flash_attention else V_tgt,
        tile_size_k=tile_size_k if use_flash_attention else d_model
    )

    output_linear = {
        'モジュール': '出力線形変換',
        '入力サイズ': f'{input_size}',
        '出力サイズ': f'{output_size}',
        '入力テンソル形状: バッチサイズ': batch_size,
        '入力テンソル形状: シーケンス長': L_t,
        '入力テンソル形状: 埋め込み次元': d_model,
        '出力テンソル形状: バッチサイズ': batch_size,
        '出力テンソル形状: シーケンス長': L_t,
        '重みの数': f"{weight_size:,}",
        'バイアスの数': f"{bias_size:,}",
        '計算量': f"{int(computation):,}",
        '重みメモリ': format_memory_size(weight_memory),
        'バイアスメモリ': format_memory_size(bias_size * precision),
        'アクティベーションメモリ': format_memory_size(activation_memory),
        'KVキャッシュメモリ': format_memory_size(0),
        '合計メモリ': format_memory_size(weight_memory + bias_size * precision + activation_memory),
        'タイル実行回数': total_tiles_out,
        'データ再利用回数': avg_reuse_out
    }
    module_details.append(output_linear)

    return (module_details, total_memory, total_weight_memory, total_bias_memory,
            total_activation_memory, total_kv_cache_memory,
            total_computation_without_flash, total_computation_with_flash)

# 計算結果を取得
(module_details, total_memory, total_weight_memory, total_bias_memory,
 total_activation_memory, total_kv_cache_memory,
 total_computation_without_flash, total_computation_with_flash) = calculate_module_details()

# データフレームに変換
df = pd.DataFrame(module_details)

# モジュールタイプの追加
def get_module_type(module_name):
    if 'エンコーダ' in module_name or 'Encoder' in module_name:
        return 'エンコーダ'
    elif 'デコーダ' in module_name or 'Decoder' in module_name:
        return 'デコーダ'
    else:
        return 'その他'

df['モジュールタイプ'] = df['モジュール'].apply(get_module_type)

# 入出力テンソルサイズの計算
df['入力要素数'] = df['入力サイズ'].apply(calculate_elements)
df['出力要素数'] = df['出力サイズ'].apply(calculate_elements)

# 結果の表示
st.header("各モジュールの詳細")

# テーブルの表示（数値のフォーマットを適用）
st.table(df)

# 合計メモリ使用量の表示
st.write(f"**推定合計メモリ使用量**: {format_memory_size(total_memory)}")
st.write(f"- 重みのメモリ合計: {format_memory_size(total_weight_memory)}")
st.write(f"- バイアスのメモリ合計: {format_memory_size(total_bias_memory)}")
st.write(f"- アクティベーションのメモリ合計: {format_memory_size(total_activation_memory)}")
if use_kv_cache:
    st.write(f"- KVキャッシュのメモリ合計: {format_memory_size(total_kv_cache_memory)}")

# FlashAttentionの有無による比較
if use_flash_attention:
    # FlashAttentionが有効な場合の計算量を計算
    flash_computation = calculate_computation_with_flash_attention(
        batch_size=batch_size,
        h=h,
        seq_len_q=L_s,
        seq_len_k=L_s,
        d_k=(d_model // h),
        tile_size_m=tile_size_m,
        tile_size_n=tile_size_n,
        tile_size_k=tile_size_k
    )
    total_computation_with_flash = flash_computation
    # FlashAttention使用時の計算量を表示
    st.header("FlashAttentionによる最適化の効果")
    reduction_ratio = (1 - total_computation_with_flash / total_computation_without_flash) * 100
    st.write(f"**計算量の削減率**: {reduction_ratio:.2f}%")
    st.write(f"- FlashAttention未使用時の総計算量: {int(total_computation_without_flash):,}")
    st.write(f"- FlashAttention使用時の総計算量: {int(total_computation_with_flash):,}")

# グラフの表示

# グラフの非表示・リセット機能の追加
if 'show_graphs' not in st.session_state:
    st.session_state['show_graphs'] = True

if 'graph_filters' not in st.session_state:
    st.session_state['graph_filters'] = {
        '重みの数': True,
        'バイアスの数': True,
        '計算量': True,
        '重みメモリ': True,
        'バイアスメモリ': True,
        'アクティベーションメモリ': True,
        'KVキャッシュメモリ': True,
        '入力サイズ/出力サイズ（要素数）': True
    }

# グラフの表示・非表示ボタン
col1, col2 = st.columns(2)
with col1:
    if st.button("グラフを表示/非表示", key='toggle_graphs'):
        st.session_state['show_graphs'] = not st.session_state['show_graphs']
with col2:
    if st.button("グラフフィルターをリセット", key='reset_graph_filters'):
        st.session_state['graph_filters'] = {
            '重みの数': True,
            'バイアスの数': True,
            '計算量': True,
            '重みメモリ': True,
            'バイアスメモリ': True,
            'アクティベーションメモリ': True,
            'KVキャッシュメモリ': True,
            '入力サイズ/出力サイズ（要素数）': True
        }

if st.session_state['show_graphs']:
    st.subheader("グラフフィルターの設定")
    with st.expander("グラフフィルターを設定"):
        st.session_state['graph_filters']['重みの数'] = st.checkbox("重みの数", value=st.session_state['graph_filters']['重みの数'], key='filter_weights')
        st.session_state['graph_filters']['バイアスの数'] = st.checkbox("バイアスの数", value=st.session_state['graph_filters']['バイアスの数'], key='filter_biases')
        st.session_state['graph_filters']['計算量'] = st.checkbox("計算量", value=st.session_state['graph_filters']['計算量'], key='filter_computation')
        st.session_state['graph_filters']['重みメモリ'] = st.checkbox("重みメモリ", value=st.session_state['graph_filters']['重みメモリ'], key='filter_weight_memory')
        st.session_state['graph_filters']['バイアスメモリ'] = st.checkbox("バイアスメモリ", value=st.session_state['graph_filters']['バイアスメモリ'], key='filter_bias_memory')
        st.session_state['graph_filters']['アクティベーションメモリ'] = st.checkbox("アクティベーションメモリ", value=st.session_state['graph_filters']['アクティベーションメモリ'], key='filter_activation_memory')
        st.session_state['graph_filters']['KVキャッシュメモリ'] = st.checkbox("KVキャッシュメモリ", value=st.session_state['graph_filters']['KVキャッシュメモリ'], key='filter_kv_cache_memory')
        st.session_state['graph_filters']['入力サイズ/出力サイズ（要素数）'] = st.checkbox("入力サイズ/出力サイズ（要素数）", value=st.session_state['graph_filters']['入力サイズ/出力サイズ（要素数）'], key='filter_tensor_sizes')

    # 重みの数のグラフ
    if st.session_state['graph_filters']['重みの数'] or st.session_state['graph_filters']['バイアスの数']:
        st.subheader("重みとバイアスの数")
        weights_df = df[['モジュール', '重みの数', 'バイアスの数', 'モジュールタイプ']]
        weights_df_melted = pd.melt(weights_df, id_vars=['モジュール', 'モジュールタイプ'],
                                    value_vars=['重みの数', 'バイアスの数'],
                                    var_name='パラメータタイプ', value_name='数')
        weights_df_melted['数'] = weights_df_melted['数'].apply(lambda x: int(x.replace(',', '')))
        # フィルタリング
        weights_df_melted = weights_df_melted[weights_df_melted['パラメータタイプ'].isin(
            [k for k, v in st.session_state['graph_filters'].items() if v and k in ['重みの数', 'バイアスの数']])]
        fig_weights = px.bar(weights_df_melted, x='モジュール', y='数', color='パラメータタイプ',
                             title='各モジュールの重みとバイアスの数', labels={'数': '数', 'モジュール': 'モジュール名'},
                             barmode='group')
        st.plotly_chart(fig_weights, use_container_width=True)

    # 計算量のグラフ
    if st.session_state['graph_filters']['計算量']:
        st.subheader("計算量")
        computation_df = df[['モジュール', '計算量', 'モジュールタイプ']]
        computation_df['計算量'] = computation_df['計算量'].apply(lambda x: int(x.replace(',', '')))
        fig_computation = px.bar(computation_df, x='モジュール', y='計算量', color='モジュールタイプ',
                                 title='各モジュールの計算量', labels={'計算量': '計算量', 'モジュール': 'モジュール名'})
        st.plotly_chart(fig_computation, use_container_width=True)

    # メモリ使用量のグラフ
    memory_types = ['重みメモリ', 'バイアスメモリ', 'アクティベーションメモリ', 'KVキャッシュメモリ']
    selected_memory_types = [k for k, v in st.session_state['graph_filters'].items() if v and k in memory_types]
    if selected_memory_types:
        st.subheader("メモリ使用量")
        memory_df = df[['モジュール', 'モジュールタイプ'] + memory_types]

        # メモリサイズをバイト数に変換
        def parse_memory_size(size_str):
            if 'GB' in size_str:
                return float(size_str.replace(' GB', '')) * (1024 ** 3)
            elif 'MB' in size_str:
                return float(size_str.replace(' MB', '')) * (1024 ** 2)
            elif 'KB' in size_str:
                return float(size_str.replace(' KB', '')) * 1024
            elif 'B' in size_str:
                return float(size_str.replace(' B', ''))
            else:
                return 0

        for col in memory_types:
            memory_df[col + '_bytes'] = memory_df[col].apply(parse_memory_size)

        memory_df_melted = pd.melt(memory_df, id_vars=['モジュール', 'モジュールタイプ'],
                                   value_vars=[col + '_bytes' for col in memory_types],
                                   var_name='メモリタイプ', value_name='メモリ使用量 (バイト)')

        # メモリタイプの名前を修正
        memory_df_melted['メモリタイプ'] = memory_df_melted['メモリタイプ'].str.replace('_bytes', '')
        # フィルタリング
        memory_df_melted = memory_df_melted[memory_df_melted['メモリタイプ'].isin(selected_memory_types)]
        fig_memory = px.bar(memory_df_melted, x='モジュール', y='メモリ使用量 (バイト)', color='メモリタイプ',
                            title='各モジュールのメモリ使用量の内訳', labels={'メモリ使用量 (バイト)': 'メモリ使用量', 'モジュール': 'モジュール名'},
                            barmode='stack')
        st.plotly_chart(fig_memory, use_container_width=True)

    # 入出力テンソルサイズのグラフ
    if st.session_state['graph_filters']['入力サイズ/出力サイズ（要素数）']:
        st.subheader("入出力テンソルサイズ（要素数）")
        tensor_sizes_df = df[['モジュール', '入力要素数', '出力要素数', 'モジュールタイプ']]

        tensor_sizes_melted = pd.melt(tensor_sizes_df, id_vars=['モジュール', 'モジュールタイプ'],
                                      value_vars=['入力要素数', '出力要素数'],
                                      var_name='テンソルタイプ', value_name='要素数')

        fig_tensor_sizes = px.bar(tensor_sizes_melted, x='モジュール', y='要素数', color='テンソルタイプ',
                                  title='各モジュールの入出力テンソルサイズ（要素数）', labels={'要素数': '要素数', 'モジュール': 'モジュール名'},
                                  barmode='group')
        st.plotly_chart(fig_tensor_sizes, use_container_width=True)

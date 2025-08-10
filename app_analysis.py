"""
事前分析専用 Streamlit アプリ
合成データセット（GSM8K）の問題とHLEデータセットの類似性を視覚的に分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import umap
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
import pickle
from datetime import datetime

# ページ設定
st.set_page_config(
    page_title="HLE類似性分析ツール",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# セッション状態の初期化
if 'embeddings_cache' not in st.session_state:
    st.session_state.embeddings_cache = {}
if 'umap_projection' not in st.session_state:
    st.session_state.umap_projection = None
if 'hle_data' not in st.session_state:
    st.session_state.hle_data = None
if 'gsm8k_data' not in st.session_state:
    st.session_state.gsm8k_data = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

@st.cache_resource
def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """埋め込みモデルをロード（キャッシュ付き）"""
    return SentenceTransformer(model_name)

@st.cache_data
def generate_mock_hle_data(n_samples: int = 500) -> pd.DataFrame:
    """モックHLEデータセットを生成"""
    np.random.seed(42)
    subjects = ["数学", "物理", "化学", "生物", "歴史", "地理", "英語", "国語"]
    subject_templates = {
        "数学": ["方程式", "幾何学", "統計", "微積分", "代数"],
        "物理": ["力学", "電磁気", "熱力学", "波動", "量子"],
        "化学": ["有機化学", "無機化学", "分析", "反応", "構造"],
        "生物": ["細胞", "遺伝", "生態", "進化", "解剖"],
        "歴史": ["古代", "中世", "近代", "現代", "文化"],
        "地理": ["地形", "気候", "人口", "産業", "都市"],
        "英語": ["文法", "語彙", "読解", "作文", "会話"],
        "国語": ["漢字", "文法", "読解", "作文", "古文"]
    }
    
    data = []
    for i in range(n_samples):
        subject = np.random.choice(subjects)
        topic = np.random.choice(subject_templates[subject])
        difficulty = np.random.choice(["基礎", "標準", "応用"])
        
        question = f"{subject}の{topic}に関する{difficulty}問題: サンプル問題文 {i:04d}"
        data.append({
            "hle_id": f"HLE_{i:04d}",
            "subject": subject,
            "topic": topic,
            "difficulty": difficulty,
            "question_text": question
        })
    
    return pd.DataFrame(data)

@st.cache_data
def generate_mock_gsm8k_data(n_samples: int = 100) -> pd.DataFrame:
    """モックGSM8Kデータセットを生成"""
    np.random.seed(123)
    problem_types = ["算術", "代数", "幾何", "確率", "論理"]
    
    data = []
    for i in range(n_samples):
        problem_type = np.random.choice(problem_types)
        difficulty = np.random.randint(1, 6)
        
        question = f"GSM8K {problem_type}問題 (難易度{difficulty}): "
        if problem_type == "算術":
            a, b = np.random.randint(1, 100, 2)
            question += f"{a} + {b} の答えは？"
        elif problem_type == "代数":
            x = np.random.randint(1, 20)
            question += f"x + {x} = {x*2} のとき、x の値は？"
        elif problem_type == "幾何":
            side = np.random.randint(1, 20)
            question += f"一辺が {side} の正方形の面積は？"
        elif problem_type == "確率":
            n = np.random.randint(2, 7)
            question += f"サイコロを投げて {n} が出る確率は？"
        else:
            question += f"論理パズル問題 {i}"
        
        data.append({
            "gsm8k_id": f"GSM_{i:04d}",
            "problem_type": problem_type,
            "difficulty": difficulty,
            "question": question
        })
    
    return pd.DataFrame(data)

@st.cache_data
def compute_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """テキストの埋め込みを計算"""
    model = load_embedding_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    # L2正規化
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

@st.cache_data
def compute_umap_projection(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """UMAP投影を計算"""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='cosine',
        random_state=42
    )
    projection = reducer.fit_transform(embeddings)
    return projection, reducer

def find_top_k_similar(query_embedding: np.ndarray, corpus_embeddings: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    """コサイン類似度でトップK個の類似項目を検索"""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), corpus_embeddings)[0]
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return [(idx, similarities[idx]) for idx in top_k_indices]

def create_umap_plot(
    projection: np.ndarray,
    labels: List[str],
    colors: List[str],
    query_point: Optional[np.ndarray] = None,
    similar_indices: Optional[List[int]] = None,
    title: str = "HLE埋め込みのUMAP投影"
) -> go.Figure:
    """インタラクティブなUMAP可視化を作成"""
    
    # データフレーム作成
    df = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'label': labels,
        'color': colors
    })
    
    # カラーマップ
    unique_colors = df['color'].unique()
    color_map = {color: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)] 
                 for i, color in enumerate(unique_colors)}
    
    fig = go.Figure()
    
    # 各カテゴリごとにプロット
    for color in unique_colors:
        mask = df['color'] == color
        fig.add_trace(go.Scatter(
            x=df[mask]['x'],
            y=df[mask]['y'],
            mode='markers',
            name=color,
            marker=dict(
                size=6,
                color=color_map[color],
                opacity=0.6,
                line=dict(width=0)
            ),
            text=df[mask]['label'],
            hovertemplate='<b>%{text}</b><br>領域: ' + color + '<extra></extra>'
        ))
    
    # 類似項目をハイライト
    if similar_indices:
        fig.add_trace(go.Scatter(
            x=projection[similar_indices, 0],
            y=projection[similar_indices, 1],
            mode='markers',
            name='類似TOP5',
            marker=dict(
                size=12,
                color='red',
                symbol='star',
                line=dict(color='darkred', width=2)
            ),
            text=[f"類似度 #{i+1}" for i in range(len(similar_indices))],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    
    # クエリポイントを追加
    if query_point is not None:
        fig.add_trace(go.Scatter(
            x=[query_point[0]],
            y=[query_point[1]],
            mode='markers',
            name='選択した問題',
            marker=dict(
                size=15,
                color='black',
                symbol='diamond',
                line=dict(color='yellow', width=3)
            ),
            hovertemplate='<b>選択した問題</b><extra></extra>'
        ))
    
    # レイアウト設定
    fig.update_layout(
        title=title,
        xaxis_title="UMAP次元1",
        yaxis_title="UMAP次元2",
        hovermode='closest',
        width=900,
        height=700,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def main():
    st.title("🔬 HLE類似性分析ツール")
    st.markdown("""
    合成データセット（GSM8K）の問題とHLEデータセットの類似性を視覚的に分析します。
    問題文を埋め込みベクトル化し、コサイン類似度で類似性を評価します。
    """)
    
    # サイドバー設定
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # モデル選択
        model_name = st.selectbox(
            "埋め込みモデル",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ],
            help="埋め込みベクトル生成に使用するモデル"
        )
        
        # UMAP パラメータ
        st.subheader("UMAP設定")
        n_neighbors = st.slider("n_neighbors", 5, 50, 15, help="近傍点の数")
        min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, 0.05, help="最小距離")
        
        # データサイズ
        st.subheader("データサイズ")
        n_hle = st.slider("HLEサンプル数", 100, 1000, 500, 50)
        n_gsm8k = st.slider("GSM8Kサンプル数", 20, 200, 100, 10)
        
        # データ生成ボタン
        if st.button("📊 データ生成/更新", type="primary"):
            with st.spinner("データを生成中..."):
                st.session_state.hle_data = generate_mock_hle_data(n_hle)
                st.session_state.gsm8k_data = generate_mock_gsm8k_data(n_gsm8k)
                st.success("データ生成完了！")
    
    # メインコンテンツ
    tabs = st.tabs(["📍 類似性分析", "🗺️ UMAP可視化", "📊 データ探索", "📈 分析履歴"])
    
    with tabs[0]:
        st.header("類似性分析")
        
        # データチェック
        if st.session_state.hle_data is None or st.session_state.gsm8k_data is None:
            st.warning("⚠️ サイドバーから「データ生成/更新」をクリックしてください。")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # GSM8K問題選択
            gsm8k_options = st.session_state.gsm8k_data['gsm8k_id'].tolist()
            selected_id = st.selectbox("GSM8K問題を選択", gsm8k_options)
            
            if selected_id:
                selected_row = st.session_state.gsm8k_data[
                    st.session_state.gsm8k_data['gsm8k_id'] == selected_id
                ].iloc[0]
                
                st.info(f"**問題タイプ**: {selected_row['problem_type']} | **難易度**: {selected_row['difficulty']}")
                st.text_area("問題文", selected_row['question'], height=100, disabled=True)
        
        with col2:
            st.subheader("検索設定")
            top_k = st.number_input("類似TOP-K", min_value=1, max_value=20, value=5)
        
        if st.button("🔍 類似性分析実行"):
            with st.spinner("分析中..."):
                # 埋め込み計算
                hle_texts = st.session_state.hle_data['question_text'].tolist()
                hle_embeddings = compute_embeddings(hle_texts, model_name)
                
                query_text = selected_row['question']
                query_embedding = compute_embeddings([query_text], model_name)
                
                # 類似検索
                similar_items = find_top_k_similar(query_embedding, hle_embeddings, k=top_k)
                
                # 結果表示
                st.success(f"✅ 類似TOP-{top_k}を検出しました")
                
                results_data = []
                for rank, (idx, similarity) in enumerate(similar_items, 1):
                    hle_item = st.session_state.hle_data.iloc[idx]
                    results_data.append({
                        "順位": rank,
                        "HLE ID": hle_item['hle_id'],
                        "領域": hle_item['subject'],
                        "トピック": hle_item['topic'],
                        "難易度": hle_item['difficulty'],
                        "類似度": f"{similarity:.3f}",
                        "問題文": hle_item['question_text']
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # 分析履歴に追加
                analysis_record = {
                    "timestamp": datetime.now().isoformat(),
                    "gsm8k_id": selected_id,
                    "query": query_text,
                    "top_k": top_k,
                    "results": results_data
                }
                st.session_state.analysis_history.append(analysis_record)
                
                # CSVダウンロード
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 結果をCSVでダウンロード",
                    csv,
                    f"similarity_analysis_{selected_id}.csv",
                    "text/csv"
                )
    
    with tabs[1]:
        st.header("UMAP可視化")
        
        if st.session_state.hle_data is None:
            st.warning("⚠️ データを生成してください。")
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("可視化オプション")
            show_labels = st.checkbox("ラベル表示", value=False)
            highlight_query = st.checkbox("クエリ点を強調", value=True)
            
            if st.session_state.gsm8k_data is not None:
                query_options = ["なし"] + st.session_state.gsm8k_data['gsm8k_id'].tolist()
                selected_query = st.selectbox("投影する問題", query_options)
        
        with col1:
            if st.button("🗺️ UMAP投影を計算"):
                with st.spinner("UMAP投影を計算中..."):
                    # HLE埋め込み計算
                    hle_texts = st.session_state.hle_data['question_text'].tolist()
                    hle_embeddings = compute_embeddings(hle_texts, model_name)
                    
                    # UMAP投影
                    projection, reducer = compute_umap_projection(
                        hle_embeddings, 
                        n_neighbors=n_neighbors,
                        min_dist=min_dist
                    )
                    
                    # セッションに保存
                    st.session_state.umap_projection = {
                        'projection': projection,
                        'reducer': reducer,
                        'embeddings': hle_embeddings
                    }
                    
                    st.success("UMAP投影完了！")
        
        # 可視化
        if st.session_state.umap_projection is not None:
            projection = st.session_state.umap_projection['projection']
            labels = st.session_state.hle_data['hle_id'].tolist() if show_labels else [""] * len(projection)
            colors = st.session_state.hle_data['subject'].tolist()
            
            query_point = None
            similar_indices = None
            
            # クエリ点の投影
            if selected_query and selected_query != "なし":
                query_row = st.session_state.gsm8k_data[
                    st.session_state.gsm8k_data['gsm8k_id'] == selected_query
                ].iloc[0]
                
                query_embedding = compute_embeddings([query_row['question']], model_name)
                
                # 類似検索
                similar_items = find_top_k_similar(
                    query_embedding,
                    st.session_state.umap_projection['embeddings'],
                    k=5
                )
                similar_indices = [item[0] for item in similar_items]
                
                # UMAP空間への投影
                reducer = st.session_state.umap_projection['reducer']
                query_projection = reducer.transform(query_embedding)
                query_point = query_projection[0]
            
            # プロット作成
            fig = create_umap_plot(
                projection,
                labels,
                colors,
                query_point=query_point,
                similar_indices=similar_indices if highlight_query else None,
                title=f"HLE埋め込みのUMAP投影 (n={len(projection)})"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 統計情報
            with st.expander("📊 投影統計"):
                st.write(f"- 総データ点数: {len(projection)}")
                st.write(f"- 領域数: {len(st.session_state.hle_data['subject'].unique())}")
                st.write(f"- UMAP パラメータ: n_neighbors={n_neighbors}, min_dist={min_dist}")
    
    with tabs[2]:
        st.header("データ探索")
        
        data_type = st.radio("データセット選択", ["HLE", "GSM8K"], horizontal=True)
        
        if data_type == "HLE" and st.session_state.hle_data is not None:
            st.subheader("HLEデータセット")
            
            # フィルタ
            subjects = st.multiselect(
                "領域フィルタ",
                st.session_state.hle_data['subject'].unique(),
                default=[]
            )
            
            filtered_data = st.session_state.hle_data
            if subjects:
                filtered_data = filtered_data[filtered_data['subject'].isin(subjects)]
            
            st.dataframe(filtered_data, use_container_width=True)
            
            # 統計
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("総問題数", len(filtered_data))
            with col2:
                st.metric("領域数", filtered_data['subject'].nunique())
            with col3:
                st.metric("トピック数", filtered_data['topic'].nunique())
            
            # 分布
            fig = px.histogram(filtered_data, x='subject', title="領域別問題数分布")
            st.plotly_chart(fig, use_container_width=True)
        
        elif data_type == "GSM8K" and st.session_state.gsm8k_data is not None:
            st.subheader("GSM8Kデータセット")
            st.dataframe(st.session_state.gsm8k_data, use_container_width=True)
            
            # 統計
            col1, col2 = st.columns(2)
            with col1:
                st.metric("総問題数", len(st.session_state.gsm8k_data))
            with col2:
                st.metric("問題タイプ数", st.session_state.gsm8k_data['problem_type'].nunique())
            
            # 分布
            fig = px.histogram(st.session_state.gsm8k_data, x='problem_type', title="問題タイプ別分布")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.header("分析履歴")
        
        if not st.session_state.analysis_history:
            st.info("まだ分析履歴がありません。")
        else:
            # 履歴表示
            for i, record in enumerate(reversed(st.session_state.analysis_history[-10:]), 1):
                with st.expander(f"分析 #{i} - {record['gsm8k_id']} ({record['timestamp'][:19]})"):
                    st.write(f"**クエリ**: {record['query']}")
                    st.write(f"**TOP-K**: {record['top_k']}")
                    
                    results_df = pd.DataFrame(record['results'])
                    st.dataframe(results_df[['順位', 'HLE ID', '領域', '類似度']], use_container_width=True)
            
            # 履歴クリア
            if st.button("🗑️ 履歴をクリア"):
                st.session_state.analysis_history = []
                st.rerun()
    
    # フッター
    st.divider()
    st.caption("🔬 HLE類似性分析ツール v1.0 | 埋め込みベースの類似性分析")

if __name__ == "__main__":
    main()
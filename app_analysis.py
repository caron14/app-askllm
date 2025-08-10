"""
Unified HLE Similarity Analysis Tool
Automatically detects environment and uses appropriate embedding method
- Streamlit Cloud: TF-IDF + PCA (lightweight)
- Local: Sentence Transformers + UMAP (full features)
Supports both English and Japanese interfaces
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
from datetime import datetime
import os
import sys

# Detect if running on Streamlit Cloud first
IS_STREAMLIT_CLOUD = os.environ.get('STREAMLIT_RUNTIME_ENV') == 'cloud'

# Only try to import heavy dependencies if not on Streamlit Cloud
HAS_TRANSFORMERS = False
if not IS_STREAMLIT_CLOUD:
    try:
        from sentence_transformers import SentenceTransformer
        import umap
        HAS_TRANSFORMERS = True
    except ImportError:
        pass

# Update cloud detection to include missing dependencies
IS_STREAMLIT_CLOUD = IS_STREAMLIT_CLOUD or not HAS_TRANSFORMERS

# Initialize session state
if 'embeddings_cache' not in st.session_state:
    st.session_state.embeddings_cache = {}
if 'projection_data' not in st.session_state:
    st.session_state.projection_data = None
if 'hle_data' not in st.session_state:
    st.session_state.hle_data = None
if 'gsm8k_data' not in st.session_state:
    st.session_state.gsm8k_data = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'language' not in st.session_state:
    st.session_state.language = 'ja'

# Page configuration
st.set_page_config(
    page_title="HLE Similarity Analysis Tool / HLE類似性分析ツール",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleTfidfEmbedder:
    """Simple TF-IDF based embedder for lightweight deployment"""
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=384, ngram_range=(1, 2))
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """Fit vectorizer on corpus"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
    
    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """Convert texts to embedding vectors"""
        if isinstance(texts, str):
            texts = [texts]
        
        if not self.is_fitted:
            self.fit(texts)
        
        # Generate TF-IDF vectors
        embeddings = self.vectorizer.transform(texts).toarray()
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        return embeddings

@st.cache_resource
def load_embedding_model(model_name: str = None):
    """Load embedding model based on environment"""
    if IS_STREAMLIT_CLOUD or not HAS_TRANSFORMERS:
        return SimpleTfidfEmbedder()
    else:
        if model_name is None:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        return SentenceTransformer(model_name)

@st.cache_data
def generate_mock_hle_data(n_samples: int = 500, lang: str = 'ja') -> pd.DataFrame:
    """Generate mock HLE dataset"""
    np.random.seed(42)
    
    if lang == 'ja':
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
        difficulty_levels = ["基礎", "標準", "応用"]
    else:
        subjects = ["Mathematics", "Physics", "Chemistry", "Biology", "History", "Geography", "English", "Language"]
        subject_templates = {
            "Mathematics": ["Equations", "Geometry", "Statistics", "Calculus", "Algebra"],
            "Physics": ["Mechanics", "Electromagnetism", "Thermodynamics", "Waves", "Quantum"],
            "Chemistry": ["Organic", "Inorganic", "Analysis", "Reactions", "Structure"],
            "Biology": ["Cells", "Genetics", "Ecology", "Evolution", "Anatomy"],
            "History": ["Ancient", "Medieval", "Modern", "Contemporary", "Culture"],
            "Geography": ["Topography", "Climate", "Population", "Industry", "Cities"],
            "English": ["Grammar", "Vocabulary", "Reading", "Writing", "Conversation"],
            "Language": ["Characters", "Grammar", "Reading", "Writing", "Classical"]
        }
        difficulty_levels = ["Basic", "Standard", "Advanced"]
    
    data = []
    for i in range(n_samples):
        subject = np.random.choice(subjects)
        topics = subject_templates.get(subject, ["General"])
        topic = np.random.choice(topics)
        difficulty = np.random.choice(difficulty_levels)
        
        if lang == 'ja':
            question = f"{subject}の{topic}に関する{difficulty}問題: サンプル問題文 {i:04d}"
        else:
            question = f"{subject} {topic} {difficulty} problem: Sample problem text {i:04d}"
        
        data.append({
            "hle_id": f"HLE_{i:04d}",
            "subject": subject,
            "topic": topic,
            "difficulty": difficulty,
            "question_text": question
        })
    
    return pd.DataFrame(data)

@st.cache_data
def generate_mock_gsm8k_data(n_samples: int = 100, lang: str = 'ja') -> pd.DataFrame:
    """Generate mock GSM8K dataset"""
    np.random.seed(123)
    
    if lang == 'ja':
        problem_types = ["算術", "代数", "幾何", "確率", "論理"]
    else:
        problem_types = ["Arithmetic", "Algebra", "Geometry", "Probability", "Logic"]
    
    data = []
    for i in range(n_samples):
        problem_type = np.random.choice(problem_types)
        difficulty = np.random.randint(1, 6)
        
        if lang == 'ja':
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
        else:
            question = f"GSM8K {problem_type} problem (difficulty {difficulty}): "
            if problem_type == "Arithmetic":
                a, b = np.random.randint(1, 100, 2)
                question += f"What is {a} + {b}?"
            elif problem_type == "Algebra":
                x = np.random.randint(1, 20)
                question += f"If x + {x} = {x*2}, what is x?"
            elif problem_type == "Geometry":
                side = np.random.randint(1, 20)
                question += f"What is the area of a square with side {side}?"
            elif problem_type == "Probability":
                n = np.random.randint(2, 7)
                question += f"What is the probability of rolling {n} on a die?"
            else:
                question += f"Logic puzzle problem {i}"
        
        data.append({
            "gsm8k_id": f"GSM_{i:04d}",
            "problem_type": problem_type,
            "difficulty": difficulty,
            "question": question
        })
    
    return pd.DataFrame(data)

def compute_embeddings(texts: List[str], model) -> np.ndarray:
    """Compute text embeddings"""
    if IS_STREAMLIT_CLOUD:
        with st.spinner("Computing embeddings..." if st.session_state.language == 'en' else "埋め込みを計算中..."):
            embeddings = model.encode(texts)
    else:
        embeddings = model.encode(texts, show_progress_bar=True)
        # L2 normalization
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

@st.cache_data
def compute_projection(embeddings: np.ndarray, method: str = "auto", **kwargs):
    """Compute 2D projection using PCA or UMAP"""
    if method == "auto":
        method = "pca" if (IS_STREAMLIT_CLOUD or not HAS_TRANSFORMERS) else "umap"
    
    if method == "pca" or not HAS_TRANSFORMERS:
        reducer = PCA(n_components=2, random_state=42)
        projection = reducer.fit_transform(embeddings)
        method = "pca"  # Force PCA if UMAP not available
    else:  # umap
        n_neighbors = kwargs.get('n_neighbors', 15)
        min_dist = kwargs.get('min_dist', 0.1)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric='cosine',
            random_state=42
        )
        projection = reducer.fit_transform(embeddings)
    
    return projection, reducer, method

def find_top_k_similar(query_embedding: np.ndarray, corpus_embeddings: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    """Find top-K similar items by cosine similarity"""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), corpus_embeddings)[0]
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return [(idx, similarities[idx]) for idx in top_k_indices]

def create_projection_plot(
    projection: np.ndarray,
    labels: List[str],
    colors: List[str],
    query_point: Optional[np.ndarray] = None,
    similar_indices: Optional[List[int]] = None,
    title: str = None,
    method: str = "auto",
    lang: str = 'ja'
) -> go.Figure:
    """Create interactive 2D projection visualization"""
    
    if title is None:
        if method == "pca":
            title = f"PCA Projection (n={len(projection)})" if lang == 'en' else f"PCA投影 (n={len(projection)})"
        else:
            title = f"UMAP Projection (n={len(projection)})" if lang == 'en' else f"UMAP投影 (n={len(projection)})"
    
    # Create dataframe
    df = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'label': labels,
        'color': colors
    })
    
    # Color map
    unique_colors = df['color'].unique()
    color_map = {color: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)] 
                 for i, color in enumerate(unique_colors)}
    
    fig = go.Figure()
    
    # Plot each category
    for color in unique_colors:
        mask = df['color'] == color
        domain_text = "Domain" if lang == 'en' else "領域"
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
            hovertemplate=f'<b>%{{text}}</b><br>{domain_text}: {color}<extra></extra>'
        ))
    
    # Highlight similar items
    if similar_indices:
        similar_label = "Similar TOP5" if lang == 'en' else "類似TOP5"
        fig.add_trace(go.Scatter(
            x=projection[similar_indices, 0],
            y=projection[similar_indices, 1],
            mode='markers',
            name=similar_label,
            marker=dict(
                size=12,
                color='red',
                symbol='star',
                line=dict(color='darkred', width=2)
            ),
            text=[f"Similarity #{i+1}" if lang == 'en' else f"類似度 #{i+1}" for i in range(len(similar_indices))],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    
    # Add query point
    if query_point is not None:
        query_label = "Selected Problem" if lang == 'en' else "選択した問題"
        fig.add_trace(go.Scatter(
            x=[query_point[0]],
            y=[query_point[1]],
            mode='markers',
            name=query_label,
            marker=dict(
                size=15,
                color='black',
                symbol='diamond',
                line=dict(color='yellow', width=3)
            ),
            hovertemplate=f'<b>{query_label}</b><extra></extra>'
        ))
    
    # Layout settings
    if method == "pca":
        x_label = "Principal Component 1" if lang == 'en' else "主成分1"
        y_label = "Principal Component 2" if lang == 'en' else "主成分2"
    else:
        x_label = "UMAP Dimension 1" if lang == 'en' else "UMAP次元1"
        y_label = "UMAP Dimension 2" if lang == 'en' else "UMAP次元2"
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
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
    # Language selector in sidebar
    with st.sidebar:
        lang = st.selectbox(
            "Language / 言語",
            options=['ja', 'en'],
            format_func=lambda x: '🇯🇵 日本語' if x == 'ja' else '🇬🇧 English',
            index=0 if st.session_state.language == 'ja' else 1
        )
        if lang != st.session_state.language:
            st.session_state.language = lang
            st.rerun()
        st.markdown("---")
    
    lang = st.session_state.language
    
    # Title and description
    if lang == 'ja':
        st.title("🔬 HLE類似性分析ツール")
        if IS_STREAMLIT_CLOUD:
            st.markdown("""
            合成データセット（GSM8K）の問題とHLEデータセットの類似性を視覚的に分析します。
            
            **注意**: Streamlit Cloud用の軽量版モードで実行中です（TF-IDF + PCA）。
            """)
        else:
            st.markdown("""
            合成データセット（GSM8K）の問題とHLEデータセットの類似性を視覚的に分析します。
            問題文を埋め込みベクトル化し、コサイン類似度で類似性を評価します。
            """)
    else:
        st.title("🔬 HLE Similarity Analysis Tool")
        if IS_STREAMLIT_CLOUD:
            st.markdown("""
            Visually analyze the similarity between synthetic dataset (GSM8K) problems and HLE dataset.
            
            **Note**: Running in lightweight mode for Streamlit Cloud (TF-IDF + PCA).
            """)
        else:
            st.markdown("""
            Visually analyze the similarity between synthetic dataset (GSM8K) problems and HLE dataset.
            Evaluate similarity using embedding vectors and cosine similarity.
            """)
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ " + ("Settings" if lang == 'en' else "設定"))
        
        # Show mode info
        if IS_STREAMLIT_CLOUD or not HAS_TRANSFORMERS:
            mode_text = "Lightweight Mode (TF-IDF + PCA)" if lang == 'en' else "軽量モード (TF-IDF + PCA)"
            st.info(f"💡 {mode_text}")
            model_name = None
            n_neighbors = 15  # Default values for variables used later
            min_dist = 0.1
        else:
            # Model selection for full mode
            model_text = "Embedding Model" if lang == 'en' else "埋め込みモデル"
            model_help = "Model to use for generating embedding vectors" if lang == 'en' else "埋め込みベクトル生成に使用するモデル"
            model_name = st.selectbox(
                model_text,
                [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                ],
                help=model_help
            )
            
            # UMAP parameters
            umap_text = "UMAP Settings" if lang == 'en' else "UMAP設定"
            st.subheader(umap_text)
            n_neighbors_help = "Number of neighboring points" if lang == 'en' else "近傍点の数"
            min_dist_help = "Minimum distance" if lang == 'en' else "最小距離"
            n_neighbors = st.slider("n_neighbors", 5, 50, 15, help=n_neighbors_help)
            min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, 0.05, help=min_dist_help)
        
        # Data size
        data_size_text = "Data Size" if lang == 'en' else "データサイズ"
        st.subheader(data_size_text)
        
        hle_text = "HLE Samples" if lang == 'en' else "HLEサンプル数"
        gsm8k_text = "GSM8K Samples" if lang == 'en' else "GSM8Kサンプル数"
        
        if IS_STREAMLIT_CLOUD:
            n_hle = st.slider(hle_text, 100, 500, 300, 50)
            n_gsm8k = st.slider(gsm8k_text, 20, 100, 50, 10)
        else:
            n_hle = st.slider(hle_text, 100, 1000, 500, 50)
            n_gsm8k = st.slider(gsm8k_text, 20, 200, 100, 10)
        
        # Generate data button
        gen_text = "📊 Generate/Update Data" if lang == 'en' else "📊 データ生成/更新"
        if st.button(gen_text, type="primary"):
            gen_msg = "Generating data..." if lang == 'en' else "データを生成中..."
            with st.spinner(gen_msg):
                st.session_state.hle_data = generate_mock_hle_data(n_hle, lang)
                st.session_state.gsm8k_data = generate_mock_gsm8k_data(n_gsm8k, lang)
                complete_msg = "Data generation complete!" if lang == 'en' else "データ生成完了！"
                st.success(complete_msg)
    
    # Main content tabs
    if lang == 'ja':
        tab_labels = ["📍 類似性分析", "🗺️ 可視化", "📊 データ探索", "📈 分析履歴"]
    else:
        tab_labels = ["📍 Similarity Analysis", "🗺️ Visualization", "📊 Data Exploration", "📈 Analysis History"]
    
    tabs = st.tabs(tab_labels)
    
    # Tab 1: Similarity Analysis
    with tabs[0]:
        header_text = "Similarity Analysis" if lang == 'en' else "類似性分析"
        st.header(header_text)
        
        # Data check
        if st.session_state.hle_data is None or st.session_state.gsm8k_data is None:
            warn_text = "⚠️ Please click 'Generate/Update Data' in the sidebar." if lang == 'en' else "⚠️ サイドバーから「データ生成/更新」をクリックしてください。"
            st.warning(warn_text)
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # GSM8K problem selection
            select_text = "Select GSM8K Problem" if lang == 'en' else "GSM8K問題を選択"
            gsm8k_options = st.session_state.gsm8k_data['gsm8k_id'].tolist()
            selected_id = st.selectbox(select_text, gsm8k_options)
            
            if selected_id:
                selected_row = st.session_state.gsm8k_data[
                    st.session_state.gsm8k_data['gsm8k_id'] == selected_id
                ].iloc[0]
                
                type_text = "Problem Type" if lang == 'en' else "問題タイプ"
                diff_text = "Difficulty" if lang == 'en' else "難易度"
                st.info(f"**{type_text}**: {selected_row['problem_type']} | **{diff_text}**: {selected_row['difficulty']}")
                
                prob_text = "Problem Text" if lang == 'en' else "問題文"
                st.text_area(prob_text, selected_row['question'], height=100, disabled=True)
        
        with col2:
            search_text = "Search Settings" if lang == 'en' else "検索設定"
            st.subheader(search_text)
            topk_text = "Similar TOP-K" if lang == 'en' else "類似TOP-K"
            top_k = st.number_input(topk_text, min_value=1, max_value=20, value=5)
        
        run_text = "🔍 Run Similarity Analysis" if lang == 'en' else "🔍 類似性分析実行"
        if st.button(run_text):
            analyze_text = "Analyzing..." if lang == 'en' else "分析中..."
            with st.spinner(analyze_text):
                # Initialize model
                if IS_STREAMLIT_CLOUD:
                    model = load_embedding_model()
                    # Fit on entire corpus for TF-IDF
                    all_texts = (st.session_state.hle_data['question_text'].tolist() + 
                               st.session_state.gsm8k_data['question'].tolist())
                    model.fit(all_texts)
                else:
                    model = load_embedding_model(model_name)
                
                # Compute embeddings
                hle_texts = st.session_state.hle_data['question_text'].tolist()
                hle_embeddings = compute_embeddings(hle_texts, model)
                
                query_text = selected_row['question']
                query_embedding = compute_embeddings([query_text], model)
                
                # Similarity search
                similar_items = find_top_k_similar(query_embedding, hle_embeddings, k=top_k)
                
                # Display results
                success_text = f"✅ Detected TOP-{top_k} similar items" if lang == 'en' else f"✅ 類似TOP-{top_k}を検出しました"
                st.success(success_text)
                
                results_data = []
                for rank, (idx, similarity) in enumerate(similar_items, 1):
                    hle_item = st.session_state.hle_data.iloc[idx]
                    
                    if lang == 'ja':
                        results_data.append({
                            "順位": rank,
                            "HLE ID": hle_item['hle_id'],
                            "領域": hle_item['subject'],
                            "トピック": hle_item['topic'],
                            "難易度": hle_item['difficulty'],
                            "類似度": f"{similarity:.3f}",
                            "問題文": hle_item['question_text']
                        })
                    else:
                        results_data.append({
                            "Rank": rank,
                            "HLE ID": hle_item['hle_id'],
                            "Domain": hle_item['subject'],
                            "Topic": hle_item['topic'],
                            "Difficulty": hle_item['difficulty'],
                            "Similarity": f"{similarity:.3f}",
                            "Question": hle_item['question_text']
                        })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Add to history
                analysis_record = {
                    "timestamp": datetime.now().isoformat(),
                    "gsm8k_id": selected_id,
                    "query": query_text,
                    "top_k": top_k,
                    "results": results_data
                }
                st.session_state.analysis_history.append(analysis_record)
                
                # CSV download
                csv = results_df.to_csv(index=False)
                dl_text = "📥 Download Results as CSV" if lang == 'en' else "📥 結果をCSVでダウンロード"
                st.download_button(
                    dl_text,
                    csv,
                    f"similarity_analysis_{selected_id}.csv",
                    "text/csv"
                )
    
    # Tab 2: Visualization
    with tabs[1]:
        viz_method = "PCA" if IS_STREAMLIT_CLOUD else "UMAP"
        header_text = f"{viz_method} Visualization" if lang == 'en' else f"{viz_method}可視化"
        st.header(header_text)
        
        if st.session_state.hle_data is None:
            warn_text = "⚠️ Please generate data." if lang == 'en' else "⚠️ データを生成してください。"
            st.warning(warn_text)
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            opt_text = "Visualization Options" if lang == 'en' else "可視化オプション"
            st.subheader(opt_text)
            
            label_text = "Show Labels" if lang == 'en' else "ラベル表示"
            show_labels = st.checkbox(label_text, value=False)
            
            highlight_text = "Highlight Query Point" if lang == 'en' else "クエリ点を強調"
            highlight_query = st.checkbox(highlight_text, value=True)
            
            if st.session_state.gsm8k_data is not None:
                none_text = "None" if lang == 'en' else "なし"
                query_options = [none_text] + st.session_state.gsm8k_data['gsm8k_id'].tolist()
                proj_text = "Problem to Project" if lang == 'en' else "投影する問題"
                selected_query = st.selectbox(proj_text, query_options)
        
        with col1:
            compute_text = f"🗺️ Compute {viz_method} Projection" if lang == 'en' else f"🗺️ {viz_method}投影を計算"
            if st.button(compute_text):
                computing_text = f"Computing {viz_method} projection..." if lang == 'en' else f"{viz_method}投影を計算中..."
                with st.spinner(computing_text):
                    # Initialize model
                    if IS_STREAMLIT_CLOUD:
                        model = load_embedding_model()
                        all_texts = (st.session_state.hle_data['question_text'].tolist() + 
                                   st.session_state.gsm8k_data['question'].tolist())
                        model.fit(all_texts)
                    else:
                        model = load_embedding_model(model_name)
                    
                    # Compute HLE embeddings
                    hle_texts = st.session_state.hle_data['question_text'].tolist()
                    hle_embeddings = compute_embeddings(hle_texts, model)
                    
                    # Compute projection
                    if IS_STREAMLIT_CLOUD or not HAS_TRANSFORMERS:
                        projection, reducer, method = compute_projection(hle_embeddings, method="pca")
                    else:
                        projection, reducer, method = compute_projection(
                            hle_embeddings, 
                            method="umap",
                            n_neighbors=n_neighbors,
                            min_dist=min_dist
                        )
                    
                    # Save to session
                    st.session_state.projection_data = {
                        'projection': projection,
                        'reducer': reducer,
                        'embeddings': hle_embeddings,
                        'model': model,
                        'method': method
                    }
                    
                    complete_text = f"{viz_method} projection complete!" if lang == 'en' else f"{viz_method}投影完了！"
                    st.success(complete_text)
        
        # Visualization
        if st.session_state.projection_data is not None:
            projection = st.session_state.projection_data['projection']
            method = st.session_state.projection_data['method']
            labels = st.session_state.hle_data['hle_id'].tolist() if show_labels else [""] * len(projection)
            colors = st.session_state.hle_data['subject'].tolist()
            
            query_point = None
            similar_indices = None
            
            # Project query point
            none_text = "None" if lang == 'en' else "なし"
            if selected_query and selected_query != none_text:
                query_row = st.session_state.gsm8k_data[
                    st.session_state.gsm8k_data['gsm8k_id'] == selected_query
                ].iloc[0]
                
                model = st.session_state.projection_data['model']
                query_embedding = compute_embeddings([query_row['question']], model)
                
                # Similarity search
                similar_items = find_top_k_similar(
                    query_embedding,
                    st.session_state.projection_data['embeddings'],
                    k=5
                )
                similar_indices = [item[0] for item in similar_items]
                
                # Project to 2D space
                reducer = st.session_state.projection_data['reducer']
                query_projection = reducer.transform(query_embedding)
                query_point = query_projection[0]
            
            # Create plot
            fig = create_projection_plot(
                projection,
                labels,
                colors,
                query_point=query_point,
                similar_indices=similar_indices if highlight_query else None,
                method=method,
                lang=lang
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            stats_text = "📊 Projection Statistics" if lang == 'en' else "📊 投影統計"
            with st.expander(stats_text):
                points_text = "Total data points" if lang == 'en' else "総データ点数"
                domains_text = "Number of domains" if lang == 'en' else "領域数"
                
                st.write(f"- {points_text}: {len(projection)}")
                st.write(f"- {domains_text}: {len(st.session_state.hle_data['subject'].unique())}")
                
                if method == "pca" and IS_STREAMLIT_CLOUD:
                    explained_var = st.session_state.projection_data['reducer'].explained_variance_ratio_
                    var_text = "Explained variance" if lang == 'en' else "説明分散"
                    st.write(f"- {var_text}: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}")
                elif method == "umap" and HAS_TRANSFORMERS:
                    params_text = "UMAP parameters" if lang == 'en' else "UMAP パラメータ"
                    st.write(f"- {params_text}: n_neighbors={n_neighbors}, min_dist={min_dist}")
    
    # Tab 3: Data Exploration
    with tabs[2]:
        explore_text = "Data Exploration" if lang == 'en' else "データ探索"
        st.header(explore_text)
        
        select_text = "Select Dataset" if lang == 'en' else "データセット選択"
        data_type = st.radio(select_text, ["HLE", "GSM8K"], horizontal=True)
        
        if data_type == "HLE" and st.session_state.hle_data is not None:
            dataset_text = "HLE Dataset" if lang == 'en' else "HLEデータセット"
            st.subheader(dataset_text)
            
            # Filter
            filter_text = "Domain Filter" if lang == 'en' else "領域フィルタ"
            subjects = st.multiselect(
                filter_text,
                st.session_state.hle_data['subject'].unique(),
                default=[]
            )
            
            filtered_data = st.session_state.hle_data
            if subjects:
                filtered_data = filtered_data[filtered_data['subject'].isin(subjects)]
            
            st.dataframe(filtered_data, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                total_text = "Total Problems" if lang == 'en' else "総問題数"
                st.metric(total_text, len(filtered_data))
            with col2:
                domains_text = "Domains" if lang == 'en' else "領域数"
                st.metric(domains_text, filtered_data['subject'].nunique())
            with col3:
                topics_text = "Topics" if lang == 'en' else "トピック数"
                st.metric(topics_text, filtered_data['topic'].nunique())
            
            # Distribution
            dist_text = "Distribution by Domain" if lang == 'en' else "領域別問題数分布"
            fig = px.histogram(filtered_data, x='subject', title=dist_text)
            st.plotly_chart(fig, use_container_width=True)
        
        elif data_type == "GSM8K" and st.session_state.gsm8k_data is not None:
            dataset_text = "GSM8K Dataset" if lang == 'en' else "GSM8Kデータセット"
            st.subheader(dataset_text)
            st.dataframe(st.session_state.gsm8k_data, use_container_width=True)
            
            # Statistics
            col1, col2 = st.columns(2)
            with col1:
                total_text = "Total Problems" if lang == 'en' else "総問題数"
                st.metric(total_text, len(st.session_state.gsm8k_data))
            with col2:
                types_text = "Problem Types" if lang == 'en' else "問題タイプ数"
                st.metric(types_text, st.session_state.gsm8k_data['problem_type'].nunique())
            
            # Distribution
            dist_text = "Distribution by Problem Type" if lang == 'en' else "問題タイプ別分布"
            fig = px.histogram(st.session_state.gsm8k_data, x='problem_type', title=dist_text)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Analysis History
    with tabs[3]:
        history_text = "Analysis History" if lang == 'en' else "分析履歴"
        st.header(history_text)
        
        if not st.session_state.analysis_history:
            no_history_text = "No analysis history yet." if lang == 'en' else "まだ分析履歴がありません。"
            st.info(no_history_text)
        else:
            # Display history
            for i, record in enumerate(reversed(st.session_state.analysis_history[-10:]), 1):
                analysis_text = f"Analysis #{i}" if lang == 'en' else f"分析 #{i}"
                with st.expander(f"{analysis_text} - {record['gsm8k_id']} ({record['timestamp'][:19]})"):
                    query_text = "Query" if lang == 'en' else "クエリ"
                    st.write(f"**{query_text}**: {record['query']}")
                    st.write(f"**TOP-K**: {record['top_k']}")
                    
                    results_df = pd.DataFrame(record['results'])
                    if lang == 'ja':
                        display_cols = ['順位', 'HLE ID', '領域', '類似度']
                    else:
                        display_cols = ['Rank', 'HLE ID', 'Domain', 'Similarity']
                    
                    actual_cols = [col for col in display_cols if col in results_df.columns]
                    if actual_cols:
                        st.dataframe(results_df[actual_cols], use_container_width=True)
            
            # Clear history
            clear_text = "🗑️ Clear History" if lang == 'en' else "🗑️ 履歴をクリア"
            if st.button(clear_text):
                st.session_state.analysis_history = []
                st.rerun()
    
    # Footer
    st.divider()
    if IS_STREAMLIT_CLOUD:
        footer_text = "🔬 HLE Similarity Analysis Tool v2.0 | Lightweight Mode (TF-IDF)" if lang == 'en' else "🔬 HLE類似性分析ツール v2.0 | 軽量モード (TF-IDF)"
    else:
        footer_text = "🔬 HLE Similarity Analysis Tool v2.0 | Full Mode" if lang == 'en' else "🔬 HLE類似性分析ツール v2.0 | フルモード"
    st.caption(footer_text)

if __name__ == "__main__":
    main()
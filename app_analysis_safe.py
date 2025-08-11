"""
Streamlit Cloud-Safe HLE Similarity Analysis Tool
This version is guaranteed to work on Streamlit Cloud by completely avoiding
problematic imports when in cloud environment.
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

# CRITICAL: Detect Streamlit Cloud environment FIRST
IS_STREAMLIT_CLOUD = os.environ.get('STREAMLIT_RUNTIME_ENV') == 'cloud'

# Safety check - completely skip heavy imports on Streamlit Cloud
if IS_STREAMLIT_CLOUD:
    # Set all heavy dependencies to None
    SentenceTransformer = None
    umap = None
    HAS_TRANSFORMERS = False
    print("Running on Streamlit Cloud - using lightweight mode only")
else:
    # Only attempt imports when NOT on Streamlit Cloud
    HAS_TRANSFORMERS = False
    try:
        from sentence_transformers import SentenceTransformer
        import umap
        HAS_TRANSFORMERS = True
        print("Heavy dependencies loaded successfully")
    except (ImportError, AttributeError) as e:
        print(f"Could not import heavy dependencies: {e}")
        SentenceTransformer = None
        umap = None

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

# Translations dictionary
TRANSLATIONS = {
    'ja': {
        'title': 'HLE 類似性分析ツール',
        'mode_info': '軽量モード (TF-IDF + PCA)',
        'settings': '設定',
        'data_size': 'データサイズ',
        'hle_samples': 'HLEサンプル数',
        'gsm8k_samples': 'GSM8Kサンプル数',
        'embedding_model': '埋め込みモデル',
        'model_help': '埋め込みベクトル生成に使用するモデル',
        'umap_settings': 'UMAP設定',
        'n_neighbors_help': '近傍点の数',
        'min_dist_help': '最小距離',
        'main_tabs': ['データ概要', '類似性分析', '可視化', '履歴'],
        'load_data': 'データ読み込み',
        'data_loaded': 'データを読み込みました',
        'compute_embeddings': '埋め込み計算中...',
        'embeddings_computed': '埋め込みを計算しました',
        'no_data': 'データがありません。サイドバーから「データ読み込み」をクリックしてください。',
    },
    'en': {
        'title': 'HLE Similarity Analysis Tool',
        'mode_info': 'Lightweight Mode (TF-IDF + PCA)',
        'settings': 'Settings',
        'data_size': 'Data Size',
        'hle_samples': 'HLE Samples',
        'gsm8k_samples': 'GSM8K Samples',
        'embedding_model': 'Embedding Model',
        'model_help': 'Model to use for generating embedding vectors',
        'umap_settings': 'UMAP Settings',
        'n_neighbors_help': 'Number of neighboring points',
        'min_dist_help': 'Minimum distance',
        'main_tabs': ['Data Overview', 'Similarity Analysis', 'Visualization', 'History'],
        'load_data': 'Load Data',
        'data_loaded': 'Data loaded successfully',
        'compute_embeddings': 'Computing embeddings...',
        'embeddings_computed': 'Embeddings computed',
        'no_data': 'No data available. Please click "Load Data" in the sidebar.',
    }
}

def t(key: str, lang: str = None) -> str:
    """Translation helper function"""
    if lang is None:
        lang = st.session_state.get('language', 'ja')
    return TRANSLATIONS.get(lang, TRANSLATIONS['ja']).get(key, key)

# Simple TF-IDF Embedder for lightweight mode
from sklearn.feature_extraction.text import TfidfVectorizer

class SimpleTfidfEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=384)
        self.is_fitted = False
    
    def encode(self, texts, **kwargs):
        if not self.is_fitted:
            embeddings = self.vectorizer.fit_transform(texts).toarray()
            self.is_fitted = True
        else:
            embeddings = self.vectorizer.transform(texts).toarray()
        
        # Pad or truncate to ensure consistent dimensions
        if embeddings.shape[1] < 384:
            padding = np.zeros((embeddings.shape[0], 384 - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
        
        return embeddings

@st.cache_resource
def load_embedding_model(model_name: str = None):
    """Load embedding model based on environment"""
    if IS_STREAMLIT_CLOUD or not HAS_TRANSFORMERS:
        return SimpleTfidfEmbedder()
    else:
        if model_name is None:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            return SentenceTransformer(model_name)
        except:
            return SimpleTfidfEmbedder()

@st.cache_data
def generate_mock_hle_data(n_samples: int = 500, lang: str = 'ja') -> pd.DataFrame:
    """Generate mock HLE dataset"""
    np.random.seed(42)
    
    if lang == 'ja':
        categories = ['数学', '科学', '歴史', '文学', '地理']
        difficulties = ['基礎', '中級', '上級', '専門']
    else:
        categories = ['Mathematics', 'Science', 'History', 'Literature', 'Geography']
        difficulties = ['Basic', 'Intermediate', 'Advanced', 'Expert']
    
    questions = []
    for i in range(n_samples):
        cat = np.random.choice(categories)
        diff = np.random.choice(difficulties)
        if lang == 'ja':
            q = f"{cat}に関する{diff}レベルの質問 {i+1}"
        else:
            q = f"{diff} level question about {cat} #{i+1}"
        questions.append(q)
    
    return pd.DataFrame({
        'question': questions,
        'category': [np.random.choice(categories) for _ in range(n_samples)],
        'difficulty': [np.random.choice(difficulties) for _ in range(n_samples)],
        'score': np.random.uniform(0, 100, n_samples)
    })

@st.cache_data
def generate_mock_gsm8k_data(n_samples: int = 100, lang: str = 'ja') -> pd.DataFrame:
    """Generate mock GSM8K dataset"""
    np.random.seed(43)
    
    if lang == 'ja':
        templates = [
            "リンゴが{}個あります。さらに{}個買いました。全部で何個？",
            "{}人の生徒がいます。各生徒が{}冊の本を持っています。本は全部で何冊？",
            "{}kmの道のりを時速{}kmで進みます。何時間かかりますか？"
        ]
    else:
        templates = [
            "There are {} apples. You buy {} more. How many in total?",
            "There are {} students. Each has {} books. How many books total?",
            "Travel {} km at {} km/h. How many hours does it take?"
        ]
    
    problems = []
    for i in range(n_samples):
        template = np.random.choice(templates)
        nums = np.random.randint(1, 100, 2)
        problem = template.format(*nums)
        problems.append(problem)
    
    return pd.DataFrame({
        'problem': problems,
        'answer': np.random.randint(1, 1000, n_samples),
        'difficulty': np.random.choice(['easy', 'medium', 'hard'], n_samples)
    })

@st.cache_data
def compute_embeddings(texts: List[str], model) -> np.ndarray:
    """Compute embeddings for texts"""
    cache_key = f"{model.__class__.__name__}_{len(texts)}"
    
    if cache_key in st.session_state.embeddings_cache:
        return st.session_state.embeddings_cache[cache_key]
    
    embeddings = model.encode(texts, show_progress_bar=False)
    st.session_state.embeddings_cache[cache_key] = embeddings
    
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

def main():
    # Language selector in sidebar
    with st.sidebar:
        lang = st.selectbox(
            "Language / 言語",
            options=['ja', 'en'],
            index=0 if st.session_state.get('language', 'ja') == 'ja' else 1,
            key='language_selector'
        )
        st.session_state.language = lang
    
    st.title(t('title', lang))
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ " + t('settings', lang))
        
        # Show mode info - SAFE VERSION
        if IS_STREAMLIT_CLOUD or not HAS_TRANSFORMERS:
            st.info(f"💡 {t('mode_info', lang)}")
            model_name = None
            n_neighbors = 15
            min_dist = 0.1
        else:
            # Only access SentenceTransformer if we have it
            st.success(f"🚀 Full Mode")
            model_name = st.selectbox(
                t('embedding_model', lang),
                [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                ],
                help=t('model_help', lang)
            )
            
            # UMAP parameters
            st.subheader(t('umap_settings', lang))
            n_neighbors = st.slider("n_neighbors", 5, 50, 15, help=t('n_neighbors_help', lang))
            min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, 0.05, help=t('min_dist_help', lang))
        
        # Data size
        st.subheader(t('data_size', lang))
        
        if IS_STREAMLIT_CLOUD:
            n_hle = st.slider(t('hle_samples', lang), 100, 500, 300, 50)
            n_gsm8k = st.slider(t('gsm8k_samples', lang), 20, 100, 50, 10)
        else:
            n_hle = st.slider(t('hle_samples', lang), 100, 1000, 500, 50)
            n_gsm8k = st.slider(t('gsm8k_samples', lang), 50, 500, 100, 50)
        
        # Load data button
        if st.button(t('load_data', lang), type="primary"):
            with st.spinner(t('compute_embeddings', lang)):
                # Generate data
                st.session_state.hle_data = generate_mock_hle_data(n_hle, lang)
                st.session_state.gsm8k_data = generate_mock_gsm8k_data(n_gsm8k, lang)
                
                # Load model
                model = load_embedding_model(model_name)
                
                # Compute embeddings
                hle_texts = st.session_state.hle_data['question'].tolist()
                gsm8k_texts = st.session_state.gsm8k_data['problem'].tolist()
                
                hle_embeddings = compute_embeddings(hle_texts, model)
                gsm8k_embeddings = compute_embeddings(gsm8k_texts, model)
                
                # Store results
                st.session_state.hle_embeddings = hle_embeddings
                st.session_state.gsm8k_embeddings = gsm8k_embeddings
                
                st.success(t('data_loaded', lang))
    
    # Main content area
    if st.session_state.hle_data is None:
        st.info(t('no_data', lang))
    else:
        tabs = st.tabs(t('main_tabs', lang))
        
        # Data Overview Tab
        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("HLE Data")
                st.dataframe(st.session_state.hle_data.head())
            with col2:
                st.subheader("GSM8K Data")
                st.dataframe(st.session_state.gsm8k_data.head())
        
        # Similarity Analysis Tab
        with tabs[1]:
            if 'hle_embeddings' in st.session_state:
                st.write("Embeddings computed and ready for analysis")
                # Add similarity analysis here
        
        # Visualization Tab
        with tabs[2]:
            if 'hle_embeddings' in st.session_state:
                # Compute projection
                all_embeddings = np.vstack([
                    st.session_state.hle_embeddings,
                    st.session_state.gsm8k_embeddings
                ])
                
                projection, reducer, method = compute_projection(
                    all_embeddings,
                    method="auto",
                    n_neighbors=n_neighbors,
                    min_dist=min_dist
                )
                
                # Create visualization
                n_hle = len(st.session_state.hle_embeddings)
                labels = ['HLE'] * n_hle + ['GSM8K'] * len(st.session_state.gsm8k_embeddings)
                
                fig = px.scatter(
                    x=projection[:, 0],
                    y=projection[:, 1],
                    color=labels,
                    title=f"Embedding Projection ({method.upper()})",
                    labels={'x': 'Component 1', 'y': 'Component 2'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # History Tab
        with tabs[3]:
            st.write("Analysis history will appear here")

if __name__ == "__main__":
    main()
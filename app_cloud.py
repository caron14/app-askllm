"""
Streamlit Cloud-optimized version of HLE Quality Screener
This version uses smaller models and pre-computed data for cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
from typing import Optional
import requests

# Cloud-friendly configuration
CLOUD_CONFIG = {
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",  # Smaller model (384 dim)
    "use_mock_judge": True,  # Use mock judge instead of LLM
    "use_precomputed": True,  # Use pre-computed embeddings
    "max_display_items": 100
}

st.set_page_config(
    page_title="HLE Quality Screener (Demo)",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.mock_data = None
    st.session_state.results = []

def load_mock_data():
    """Load mock data for demo purposes"""
    if st.session_state.mock_data is None:
        # Create mock HLE data
        subjects = ["mathematics", "physics", "chemistry", "biology", "history"]
        mock_hle = []
        for i in range(50):
            mock_hle.append({
                "hle_id": f"hle_{i:04d}",
                "subject": subjects[i % len(subjects)],
                "question_text": f"Sample {subjects[i % len(subjects)]} question {i}: This is a placeholder question for demonstration purposes."
            })
        st.session_state.mock_data = pd.DataFrame(mock_hle)
    return st.session_state.mock_data

def mock_similarity_search(query_text: str, k: int = 5):
    """Mock similarity search for demo"""
    df = load_mock_data()
    # Simple keyword matching for demo
    scores = []
    query_lower = query_text.lower()
    for _, row in df.iterrows():
        score = 0
        if any(word in row['question_text'].lower() for word in query_lower.split()):
            score = np.random.uniform(0.6, 0.95)
        else:
            score = np.random.uniform(0.3, 0.6)
        scores.append(score)
    
    df['similarity'] = scores
    top_k = df.nlargest(k, 'similarity')
    
    results = []
    for rank, (_, row) in enumerate(top_k.iterrows(), 1):
        results.append({
            'rank': rank,
            'hle_id': row['hle_id'],
            'subject': row['subject'],
            'similarity': row['similarity'],
            'question_text': row['question_text']
        })
    return results

def mock_llm_judge(query_text: str, references: list):
    """Mock LLM judge for demo"""
    # Simple heuristic scoring
    base_score = np.random.uniform(0.5, 0.9)
    
    # Adjust based on similarity
    avg_sim = np.mean([ref['similarity'] for ref in references])
    adjusted_score = base_score * 0.7 + avg_sim * 0.3
    
    rationale = f"Based on similarity analysis, this item shows {'high' if adjusted_score > 0.7 else 'moderate'} relevance to the reference questions."
    
    return {
        'yes_probability': adjusted_score,
        'rationale': rationale
    }

def calculate_quality_score(askllm_score: float, sim_score: float, alpha: float = 0.8):
    """Calculate composite quality score"""
    return round(alpha * askllm_score * 100 + (1 - alpha) * sim_score * 100, 1)

def main():
    st.title("🔍 HLE Quality Screener (Cloud Demo)")
    st.markdown("""
    **Demo Version** - This is a lightweight demonstration using mock data and simplified models.
    
    For production use with full models, please run locally.
    """)
    
    st.warning("⚠️ This demo uses mock data and simplified scoring. Results are for demonstration only.")
    
    tabs = st.tabs(["🔎 Search & Score", "📊 About", "⚙️ Configuration"])
    
    with tabs[0]:
        st.header("Evaluate Synthetic Questions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query_text = st.text_area(
                "Enter question text to evaluate:",
                height=100,
                placeholder="Enter a math problem or any educational question..."
            )
            
            item_id = st.text_input("Item ID (optional)", value="demo_001")
        
        with col2:
            st.subheader("Parameters")
            k = st.slider("Number of similar items", 3, 10, 5)
            alpha = st.slider("Quality weight (α)", 0.0, 1.0, 0.8, 0.1)
        
        if st.button("🚀 Analyze", type="primary"):
            if not query_text:
                st.error("Please enter question text")
            else:
                with st.spinner("Analyzing..."):
                    # Mock similarity search
                    references = mock_similarity_search(query_text, k)
                    
                    # Mock LLM judge
                    judge_result = mock_llm_judge(query_text, references)
                    
                    # Calculate scores
                    sim_score = np.mean([ref['similarity'] for ref in references])
                    askllm_score = judge_result['yes_probability']
                    quality_score = calculate_quality_score(askllm_score, sim_score, alpha)
                    
                    # Store result
                    result = {
                        'item_id': item_id,
                        'query': query_text,
                        'quality_score': quality_score,
                        'askllm_score': round(askllm_score * 100, 1),
                        'sim_score': round(sim_score, 3),
                        'references': references
                    }
                    st.session_state.results.append(result)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Quality Score", f"{quality_score:.1f}")
                with col2:
                    st.metric("Judge Score", f"{round(askllm_score * 100, 1):.1f}")
                with col3:
                    st.metric("Similarity Score", f"{sim_score:.3f}")
                
                st.subheader("📋 Similar Reference Items")
                ref_df = pd.DataFrame(references)
                ref_df['similarity'] = ref_df['similarity'].round(3)
                st.dataframe(
                    ref_df[['rank', 'hle_id', 'subject', 'similarity', 'question_text']],
                    use_container_width=True
                )
                
                st.subheader("🤖 Judge Analysis")
                st.info(judge_result['rationale'])
                
                with st.expander("📄 Full Result JSON"):
                    st.json(result)
        
        # History
        if st.session_state.results:
            st.divider()
            st.subheader("📜 Analysis History")
            history_df = pd.DataFrame([
                {
                    'ID': r['item_id'],
                    'Quality': r['quality_score'],
                    'Judge': r['askllm_score'],
                    'Similarity': r['sim_score'],
                    'Query': r['query'][:50] + '...' if len(r['query']) > 50 else r['query']
                }
                for r in st.session_state.results[-5:]  # Last 5
            ])
            st.dataframe(history_df, use_container_width=True)
    
    with tabs[1]:
        st.header("About This Demo")
        
        st.markdown("""
        ### 🎯 Purpose
        This demonstration shows the concept of the HLE Quality Screener system, which evaluates 
        synthetic dataset items against reference questions using semantic similarity and quality scoring.
        
        ### 🔬 How It Works
        1. **Similarity Search**: Finds the most similar reference questions (using mock data in demo)
        2. **Quality Assessment**: Evaluates relevance using a judge model (simulated in demo)
        3. **Composite Scoring**: Combines similarity and judge scores with configurable weights
        
        ### 📊 Scoring Formula
        ```
        quality_score = α × judge_score + (1-α) × similarity_score
        ```
        
        ### ⚠️ Limitations of Demo
        - Uses mock data instead of real HLE dataset
        - Simplified similarity matching (no embeddings)
        - Mock judge instead of LLM
        - No persistence or batch processing
        
        ### 🚀 Full Version Features
        The complete local version includes:
        - Real embeddings with BAAI/bge-m3 or similar models
        - Qwen2.5-3B LLM judge for quality assessment
        - FAISS indexing for efficient similarity search
        - Batch processing with resume capability
        - UMAP visualizations
        - Full CLI interface
        
        ### 📦 Installation
        To run the full version locally:
        ```bash
        git clone <repo-url>
        cd hle-screener
        uv sync
        uv run python -m hle_screener.cli build-index
        uv run python -m hle_screener.cli serve
        ```
        """)
    
    with tabs[2]:
        st.header("Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demo Settings")
            st.json({
                "mode": "demo",
                "embedding_model": "mock",
                "judge_model": "mock",
                "data_source": "synthetic",
                "max_items": 50
            })
        
        with col2:
            st.subheader("Production Settings")
            st.json({
                "mode": "production",
                "embedding_model": "BAAI/bge-m3",
                "judge_model": "Qwen/Qwen2.5-3B-Instruct",
                "data_source": "HLE dataset",
                "index": "FAISS"
            })
        
        st.info("""
        💡 **Note**: This demo runs entirely in the browser with mock data. 
        For production use with real models and data, please deploy locally or on a GPU-enabled server.
        """)
        
        if st.button("Clear History"):
            st.session_state.results = []
            st.success("History cleared!")
            st.rerun()

    # Footer
    st.divider()
    st.caption("HLE Quality Screener Demo v0.1.0 | ⚠️ DEMO MODE - Not for production use")

if __name__ == "__main__":
    main()
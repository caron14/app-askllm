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
from i18n_cloud import t, language_selector

# Cloud-friendly configuration
CLOUD_CONFIG = {
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",  # Smaller model (384 dim)
    "use_mock_judge": True,  # Use mock judge instead of LLM
    "use_precomputed": True,  # Use pre-computed embeddings
    "max_display_items": 100
}

st.set_page_config(
    page_title=t("app_title"),
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
    # Add language selector in sidebar
    with st.sidebar:
        language_selector()
        st.markdown("---")
    
    st.title(t("main_title"))
    st.markdown(t("demo_note"))
    
    st.warning(t("warning_demo"))
    
    tabs = st.tabs([t("tab_search"), t("tab_about"), t("tab_config")])
    
    with tabs[0]:
        st.header(t("evaluate_header"))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query_text = st.text_area(
                t("enter_question"),
                height=100,
                placeholder=t("placeholder_question")
            )
            
            item_id = st.text_input(t("item_id"), value="demo_001")
        
        with col2:
            st.subheader(t("parameters"))
            k = st.slider(t("num_similar"), 3, 10, 5)
            alpha = st.slider(t("quality_weight"), 0.0, 1.0, 0.8, 0.1)
        
        if st.button(t("analyze_button"), type="primary"):
            if not query_text:
                st.error(t("error_no_text"))
            else:
                with st.spinner(t("info_analyzing")):
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
                    st.metric(t("quality_score"), f"{quality_score:.1f}")
                with col2:
                    st.metric(t("judge_score"), f"{round(askllm_score * 100, 1):.1f}")
                with col3:
                    st.metric(t("similarity_score"), f"{sim_score:.3f}")
                
                st.subheader(t("similar_refs"))
                ref_df = pd.DataFrame(references)
                ref_df['similarity'] = ref_df['similarity'].round(3)
                st.dataframe(
                    ref_df[['rank', 'hle_id', 'subject', 'similarity', 'question_text']],
                    use_container_width=True
                )
                
                st.subheader(t("judge_analysis"))
                st.info(judge_result['rationale'])
                
                with st.expander(t("full_json")):
                    st.json(result)
        
        # History
        if st.session_state.results:
            st.divider()
            st.subheader(t("analysis_history"))
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
        st.header(t("about_header"))
        
        st.markdown(t("about_purpose"))
        st.markdown(t("about_purpose_text"))
        
        st.markdown(t("about_how"))
        st.markdown(t("about_steps"))
        
        st.markdown(t("about_formula"))
        st.markdown("""
        ```
        quality_score = α × judge_score + (1-α) × similarity_score
        ```
        """)
        
        st.markdown(t("about_limitations"))
        st.markdown(t("limitations_list"))
        
        st.markdown(t("about_full"))
        st.markdown(t("full_features"))
        
        st.markdown(t("about_install"))
        st.markdown(t("install_text"))
    
    with tabs[2]:
        st.header(t("config_header"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(t("demo_settings"))
            st.json({
                "mode": "demo",
                "embedding_model": "mock",
                "judge_model": "mock",
                "data_source": "synthetic",
                "max_items": 50
            })
        
        with col2:
            st.subheader(t("prod_settings"))
            st.json({
                "mode": "production",
                "embedding_model": "BAAI/bge-m3",
                "judge_model": "Qwen/Qwen2.5-3B-Instruct",
                "data_source": "HLE dataset",
                "index": "FAISS"
            })
        
        st.info(t("note_demo"))
        
        if st.button(t("clear_history")):
            st.session_state.results = []
            st.success(t("history_cleared"))
            st.rerun()

    # Footer
    st.divider()
    st.caption(t("footer"))

if __name__ == "__main__":
    main()
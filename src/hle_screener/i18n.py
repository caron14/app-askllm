"""
Internationalization (i18n) module for Streamlit apps
Supports Japanese and English localization
"""

from typing import Dict, Any, Optional
import streamlit as st

TRANSLATIONS = {
    "en": {
        # Page titles and headers
        "app_title": "HLE Quality Screener",
        "app_title_demo": "HLE Quality Screener (Demo)",
        "app_title_analysis": "HLE Visual Similarity Analysis",
        "main_title": "🔍 HLE-like Quality Screener",
        "main_subtitle": "Evaluate synthetic dataset items against HLE reference questions",
        
        # Tab names
        "tab_search": "🔎 Search & Score",
        "tab_umap": "🗺️ UMAP Visualization",
        "tab_settings": "⚙️ Settings",
        "tab_batch": "📊 Batch Results",
        "tab_about": "📊 About",
        "tab_config": "⚙️ Configuration",
        
        # Search & Score section
        "search_header": "Search and Score Items",
        "input_type": "Input Type",
        "input_text": "Text",
        "input_id": "ID",
        "query_prompt": "Enter question text to evaluate:",
        "query_placeholder": "Enter a math problem or question...",
        "item_id_label": "Item ID (optional)",
        "item_id_required": "Enter item ID:",
        "search_params": "Search Parameters",
        "num_similar": "Number of similar HLE items",
        "score_button": "🚀 Score Item",
        "analyze_button": "🚀 Analyze",
        
        # Results section
        "quality_score": "Quality Score",
        "askllm_score": "Ask-LLM Score",
        "judge_score": "Judge Score",
        "similarity_score": "Similarity Score",
        "similar_items": "📋 Top Similar HLE Items",
        "similar_refs": "📋 Similar Reference Items",
        "judge_rationale": "🤖 Judge Rationale",
        "judge_analysis": "🤖 Judge Analysis",
        "full_json": "📄 Full Result JSON",
        "download_csv": "📥 Download Results as CSV",
        "download_csv_batch": "📥 Download as CSV",
        
        # UMAP section
        "umap_header": "UMAP Visualization",
        "umap_query": "Enter question text to project:",
        "umap_project": "📍 Project and Visualize",
        "umap_legend": "Legend",
        "umap_legend_hle": "**Colored dots**: HLE items colored by subject",
        "umap_legend_similar": "**⭐ Red stars**: Top-5 most similar HLE items",
        "umap_legend_query": "**◆ Black diamond**: Your query point",
        
        # Settings section
        "settings_header": "Settings",
        "model_config": "Model Configuration",
        "umap_params": "UMAP Parameters",
        "safety_config": "Safety Configuration",
        "reload_config": "🔄 Reload Configuration",
        
        # Batch results section
        "batch_header": "Batch Results Viewer",
        "select_file": "Select results file:",
        "total_items": "Total Items",
        "avg_quality": "Average Quality",
        "min_quality": "Min Quality",
        "max_quality": "Max Quality",
        
        # Error messages
        "error_no_index": "⚠️ No index found. Please run `uv run python -m hle_screener.cli build-index` first.",
        "error_no_text": "Please enter question text",
        "error_no_umap": "⚠️ No UMAP projection found. Please run `uv run python -m hle_screener.cli prep-umap` first.",
        
        # Warning messages
        "warning_eval_only": "⚠️ All HLE content is EVAL-ONLY. DO NOT use for training.",
        "warning_demo": "⚠️ This demo uses mock data and simplified scoring. Results are for demonstration only.",
        
        # Info messages
        "info_no_results": "No batch results found yet.",
        "info_no_dir": "No results directory found.",
        "info_loading": "Retrieving similar HLE items and scoring...",
        "info_projecting": "Projecting query and generating visualization...",
        "info_analyzing": "Analyzing...",
        
        # Demo/About section
        "about_purpose": "### 🎯 Purpose",
        "about_how": "### 🔬 How It Works",
        "about_formula": "### 📊 Scoring Formula",
        "about_limitations": "### ⚠️ Limitations of Demo",
        "about_full": "### 🚀 Full Version Features",
        "about_install": "### 📦 Installation",
        
        # Configuration
        "config_demo": "Demo Settings",
        "config_prod": "Production Settings",
        "clear_history": "Clear History",
        "history_cleared": "History cleared!",
        "analysis_history": "📜 Analysis History",
        
        # Language selector
        "language": "Language",
        
        # Footer
        "footer_version": "HLE Quality Screener v0.1.0",
        "footer_warning": "⚠️ EVAL-ONLY - DO NOT TRAIN",
        "footer_demo": "HLE Quality Screener Demo v0.1.0 | ⚠️ DEMO MODE - Not for production use",
    },
    
    "ja": {
        # Page titles and headers
        "app_title": "HLE品質スクリーナー",
        "app_title_demo": "HLE品質スクリーナー（デモ版）",
        "app_title_analysis": "HLE視覚的類似性分析",
        "main_title": "🔍 HLE品質評価システム",
        "main_subtitle": "HLE参照質問に対して合成データセット項目を評価",
        
        # Tab names
        "tab_search": "🔎 検索＆スコアリング",
        "tab_umap": "🗺️ UMAP可視化",
        "tab_settings": "⚙️ 設定",
        "tab_batch": "📊 バッチ結果",
        "tab_about": "📊 概要",
        "tab_config": "⚙️ 設定",
        
        # Search & Score section
        "search_header": "項目の検索とスコアリング",
        "input_type": "入力タイプ",
        "input_text": "テキスト",
        "input_id": "ID",
        "query_prompt": "評価する質問テキストを入力：",
        "query_placeholder": "数学問題や質問を入力してください...",
        "item_id_label": "項目ID（オプション）",
        "item_id_required": "項目IDを入力：",
        "search_params": "検索パラメータ",
        "num_similar": "類似HLE項目数",
        "score_button": "🚀 スコア計算",
        "analyze_button": "🚀 分析開始",
        
        # Results section
        "quality_score": "品質スコア",
        "askllm_score": "LLM評価スコア",
        "judge_score": "判定スコア",
        "similarity_score": "類似度スコア",
        "similar_items": "📋 上位類似HLE項目",
        "similar_refs": "📋 類似参照項目",
        "judge_rationale": "🤖 判定理由",
        "judge_analysis": "🤖 判定分析",
        "full_json": "📄 完全な結果（JSON）",
        "download_csv": "📥 結果をCSVでダウンロード",
        "download_csv_batch": "📥 CSVとしてダウンロード",
        
        # UMAP section
        "umap_header": "UMAP可視化",
        "umap_query": "投影する質問テキストを入力：",
        "umap_project": "📍 投影と可視化",
        "umap_legend": "凡例",
        "umap_legend_hle": "**色付きの点**: 科目別に色分けされたHLE項目",
        "umap_legend_similar": "**⭐ 赤い星**: 上位5つの類似HLE項目",
        "umap_legend_query": "**◆ 黒いダイヤ**: クエリポイント",
        
        # Settings section
        "settings_header": "設定",
        "model_config": "モデル設定",
        "umap_params": "UMAPパラメータ",
        "safety_config": "安全性設定",
        "reload_config": "🔄 設定を再読み込み",
        
        # Batch results section
        "batch_header": "バッチ結果ビューア",
        "select_file": "結果ファイルを選択：",
        "total_items": "総項目数",
        "avg_quality": "平均品質",
        "min_quality": "最小品質",
        "max_quality": "最大品質",
        
        # Error messages
        "error_no_index": "⚠️ インデックスが見つかりません。まず `uv run python -m hle_screener.cli build-index` を実行してください。",
        "error_no_text": "質問テキストを入力してください",
        "error_no_umap": "⚠️ UMAP投影が見つかりません。まず `uv run python -m hle_screener.cli prep-umap` を実行してください。",
        
        # Warning messages
        "warning_eval_only": "⚠️ すべてのHLEコンテンツは評価専用です。トレーニングには使用しないでください。",
        "warning_demo": "⚠️ このデモはモックデータと簡略化されたスコアリングを使用しています。結果はデモンストレーション目的のみです。",
        
        # Info messages
        "info_no_results": "バッチ結果はまだありません。",
        "info_no_dir": "結果ディレクトリが見つかりません。",
        "info_loading": "類似HLE項目を取得してスコアリング中...",
        "info_projecting": "クエリを投影して可視化を生成中...",
        "info_analyzing": "分析中...",
        
        # Demo/About section
        "about_purpose": "### 🎯 目的",
        "about_how": "### 🔬 仕組み",
        "about_formula": "### 📊 スコアリング式",
        "about_limitations": "### ⚠️ デモ版の制限",
        "about_full": "### 🚀 フルバージョンの機能",
        "about_install": "### 📦 インストール",
        
        # Configuration
        "config_demo": "デモ設定",
        "config_prod": "本番設定",
        "clear_history": "履歴をクリア",
        "history_cleared": "履歴がクリアされました！",
        "analysis_history": "📜 分析履歴",
        
        # Language selector
        "language": "言語",
        
        # Footer
        "footer_version": "HLE品質スクリーナー v0.1.0",
        "footer_warning": "⚠️ 評価専用 - トレーニングに使用禁止",
        "footer_demo": "HLE品質スクリーナー デモ v0.1.0 | ⚠️ デモモード - 本番使用不可",
    }
}

def get_language() -> str:
    """Get the current language setting from session state"""
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    return st.session_state.language

def set_language(lang: str):
    """Set the language in session state"""
    if lang in TRANSLATIONS:
        st.session_state.language = lang

def t(key: str, **kwargs) -> str:
    """
    Translate a key to the current language
    
    Args:
        key: Translation key
        **kwargs: Format arguments for string interpolation
    
    Returns:
        Translated string
    """
    lang = get_language()
    translations = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
    
    text = translations.get(key, key)
    
    # Apply any format arguments
    if kwargs:
        try:
            text = text.format(**kwargs)
        except:
            pass
    
    return text

def language_selector(key: str = "lang_selector"):
    """
    Create a language selector widget
    
    Args:
        key: Unique key for the selectbox widget
    """
    current_lang = get_language()
    
    languages = {
        "en": "🇬🇧 English",
        "ja": "🇯🇵 日本語"
    }
    
    # Find the index of current language
    lang_codes = list(languages.keys())
    current_index = lang_codes.index(current_lang) if current_lang in lang_codes else 0
    
    selected = st.selectbox(
        t("language"),
        options=lang_codes,
        format_func=lambda x: languages[x],
        index=current_index,
        key=key
    )
    
    if selected != current_lang:
        set_language(selected)
        st.rerun()

def get_all_translations(lang: str = None) -> Dict[str, str]:
    """
    Get all translations for a specific language
    
    Args:
        lang: Language code (defaults to current language)
    
    Returns:
        Dictionary of all translations
    """
    if lang is None:
        lang = get_language()
    
    return TRANSLATIONS.get(lang, TRANSLATIONS['en'])
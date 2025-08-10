"""
Internationalization (i18n) module for Cloud Streamlit app
Standalone version that doesn't require the main package
"""

from typing import Dict
import streamlit as st

TRANSLATIONS = {
    "en": {
        # Page titles
        "app_title": "HLE Quality Screener (Demo)",
        "main_title": "🔍 HLE Quality Screener (Cloud Demo)",
        "demo_note": """**Demo Version** - This is a lightweight demonstration using mock data and simplified models.
    
For production use with full models, please run locally.""",
        
        # Tabs
        "tab_search": "🔎 Search & Score",
        "tab_about": "📊 About",
        "tab_config": "⚙️ Configuration",
        
        # Search section
        "evaluate_header": "Evaluate Synthetic Questions",
        "enter_question": "Enter question text to evaluate:",
        "placeholder_question": "Enter a math problem or any educational question...",
        "item_id": "Item ID (optional)",
        "parameters": "Parameters",
        "num_similar": "Number of similar items",
        "quality_weight": "Quality weight (α)",
        "analyze_button": "🚀 Analyze",
        
        # Results
        "quality_score": "Quality Score",
        "judge_score": "Judge Score",
        "similarity_score": "Similarity Score",
        "similar_refs": "📋 Similar Reference Items",
        "judge_analysis": "🤖 Judge Analysis",
        "full_json": "📄 Full Result JSON",
        "analysis_history": "📜 Analysis History",
        
        # About section
        "about_header": "About This Demo",
        "about_purpose": "### 🎯 Purpose",
        "about_purpose_text": """This demonstration shows the concept of the HLE Quality Screener system, which evaluates 
synthetic dataset items against reference questions using semantic similarity and quality scoring.""",
        "about_how": "### 🔬 How It Works",
        "about_steps": """1. **Similarity Search**: Finds the most similar reference questions (using mock data in demo)
2. **Quality Assessment**: Evaluates relevance using a judge model (simulated in demo)
3. **Composite Scoring**: Combines similarity and judge scores with configurable weights""",
        "about_formula": "### 📊 Scoring Formula",
        "about_limitations": "### ⚠️ Limitations of Demo",
        "limitations_list": """- Uses mock data instead of real HLE dataset
- Simplified similarity matching (no embeddings)
- Mock judge instead of LLM
- No persistence or batch processing""",
        "about_full": "### 🚀 Full Version Features",
        "full_features": """The complete local version includes:
- Real embeddings with BAAI/bge-m3 or similar models
- Qwen2.5-3B LLM judge for quality assessment
- FAISS indexing for efficient similarity search
- Batch processing with resume capability
- UMAP visualizations
- Full CLI interface""",
        "about_install": "### 📦 Installation",
        "install_text": """To run the full version locally:
```bash
git clone <repo-url>
cd hle-screener
uv sync
uv run python -m hle_screener.cli build-index
uv run python -m hle_screener.cli serve
```""",
        
        # Configuration
        "config_header": "Configuration",
        "demo_settings": "Demo Settings",
        "prod_settings": "Production Settings",
        "clear_history": "Clear History",
        "history_cleared": "History cleared!",
        
        # Messages
        "warning_demo": "⚠️ This demo uses mock data and simplified scoring. Results are for demonstration only.",
        "error_no_text": "Please enter question text",
        "info_analyzing": "Analyzing...",
        "note_demo": """💡 **Note**: This demo runs entirely in the browser with mock data. 
For production use with real models and data, please deploy locally or on a GPU-enabled server.""",
        
        # Language
        "language": "Language",
        
        # Footer
        "footer": "HLE Quality Screener Demo v0.1.0 | ⚠️ DEMO MODE - Not for production use",
    },
    
    "ja": {
        # Page titles
        "app_title": "HLE品質スクリーナー（デモ版）",
        "main_title": "🔍 HLE品質スクリーナー（クラウドデモ）",
        "demo_note": """**デモ版** - モックデータと簡略化されたモデルを使用した軽量デモンストレーションです。
    
本番環境でフルモデルを使用するには、ローカルで実行してください。""",
        
        # Tabs
        "tab_search": "🔎 検索＆スコア",
        "tab_about": "📊 概要",
        "tab_config": "⚙️ 設定",
        
        # Search section
        "evaluate_header": "合成質問の評価",
        "enter_question": "評価する質問テキストを入力：",
        "placeholder_question": "数学問題や教育的な質問を入力してください...",
        "item_id": "項目ID（オプション）",
        "parameters": "パラメータ",
        "num_similar": "類似項目数",
        "quality_weight": "品質重み（α）",
        "analyze_button": "🚀 分析",
        
        # Results
        "quality_score": "品質スコア",
        "judge_score": "判定スコア",
        "similarity_score": "類似度スコア",
        "similar_refs": "📋 類似参照項目",
        "judge_analysis": "🤖 判定分析",
        "full_json": "📄 完全な結果（JSON）",
        "analysis_history": "📜 分析履歴",
        
        # About section
        "about_header": "このデモについて",
        "about_purpose": "### 🎯 目的",
        "about_purpose_text": """このデモンストレーションは、意味的類似性と品質スコアリングを使用して、
参照質問に対して合成データセット項目を評価するHLE品質スクリーナーシステムの概念を示しています。""",
        "about_how": "### 🔬 仕組み",
        "about_steps": """1. **類似性検索**: 最も類似した参照質問を見つける（デモではモックデータを使用）
2. **品質評価**: 判定モデルを使用して関連性を評価（デモではシミュレート）
3. **複合スコアリング**: 設定可能な重みで類似性スコアと判定スコアを組み合わせる""",
        "about_formula": "### 📊 スコアリング式",
        "about_limitations": "### ⚠️ デモの制限",
        "limitations_list": """- 実際のHLEデータセットの代わりにモックデータを使用
- 簡略化された類似性マッチング（埋め込みなし）
- LLMの代わりにモック判定
- 永続性やバッチ処理なし""",
        "about_full": "### 🚀 フルバージョンの機能",
        "full_features": """完全なローカルバージョンには以下が含まれます：
- BAAI/bge-m3または類似モデルによる実際の埋め込み
- 品質評価のためのQwen2.5-3B LLM判定
- 効率的な類似性検索のためのFAISSインデックス
- 再開機能付きバッチ処理
- UMAP可視化
- 完全なCLIインターフェース""",
        "about_install": "### 📦 インストール",
        "install_text": """フルバージョンをローカルで実行するには：
```bash
git clone <repo-url>
cd hle-screener
uv sync
uv run python -m hle_screener.cli build-index
uv run python -m hle_screener.cli serve
```""",
        
        # Configuration
        "config_header": "設定",
        "demo_settings": "デモ設定",
        "prod_settings": "本番設定",
        "clear_history": "履歴をクリア",
        "history_cleared": "履歴がクリアされました！",
        
        # Messages
        "warning_demo": "⚠️ このデモはモックデータと簡略化されたスコアリングを使用しています。結果はデモンストレーション目的のみです。",
        "error_no_text": "質問テキストを入力してください",
        "info_analyzing": "分析中...",
        "note_demo": """💡 **注記**: このデモはブラウザ内でモックデータを使用して完全に実行されます。
実際のモデルとデータを使用した本番環境では、ローカルまたはGPU対応サーバーでデプロイしてください。""",
        
        # Language
        "language": "言語",
        
        # Footer
        "footer": "HLE品質スクリーナー デモ v0.1.0 | ⚠️ デモモード - 本番使用不可",
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
    """Translate a key to the current language"""
    lang = get_language()
    translations = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
    text = translations.get(key, key)
    
    if kwargs:
        try:
            text = text.format(**kwargs)
        except:
            pass
    
    return text

def language_selector(key: str = "lang_selector"):
    """Create a language selector widget"""
    current_lang = get_language()
    
    languages = {
        "en": "🇬🇧 English",
        "ja": "🇯🇵 日本語"
    }
    
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
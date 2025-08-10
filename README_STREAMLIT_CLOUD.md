# Streamlit Cloud Deployment Guide

## 🚨 重要な制限事項

現在のレポジトリは **Streamlit Cloud での完全な動作は困難** です。理由：

### 主な課題

1. **メモリ制限** (1GB RAM)
   - Qwen2.5-3B モデル: 約 6GB 必要
   - BAAI/bge-m3 埋め込みモデル: 約 2GB 必要
   - FAISS インデックス: データ量に依存

2. **モデルのダウンロード**
   - 初回起動時に大規模モデルのダウンロードが必要
   - タイムアウトの可能性

3. **データセットアクセス**
   - HLE データセットへのアクセス制限
   - 評価専用データの取り扱い

## ✅ Streamlit Cloud で可能なこと

### デモ版の公開

`app_cloud.py` を使用した **軽量デモ版** は公開可能です：

```bash
# Streamlit Cloud の設定
Main file path: app_cloud.py
Python version: 3.11
```

#### デモ版の機能
- ✅ モックデータでの動作デモ
- ✅ UI/UX の体験
- ✅ スコアリングロジックの説明
- ✅ 軽量な類似度計算

#### デモ版の制限
- ❌ 実際の ML モデル使用不可
- ❌ 本格的な埋め込み計算不可
- ❌ FAISS インデックス使用不可
- ❌ バッチ処理機能なし

## 🚀 推奨デプロイメント方法

### 1. ローカル実行（推奨）
```bash
git clone <repo-url>
cd hle-screener
uv sync
uv run python -m hle_screener.cli build-index
uv run python -m hle_screener.cli serve
```

### 2. GPU 対応クラウドサービス

#### Google Colab
```python
!git clone <repo-url>
!cd hle-screener && pip install -e .
!python -m hle_screener.cli serve --port 8501
```

#### AWS EC2 / GCP Compute Engine
- インスタンスタイプ: g4dn.xlarge 以上（GPU付き）
- メモリ: 16GB 以上
- ストレージ: 50GB 以上

#### Hugging Face Spaces
- Gradio インターフェースへの変換が必要
- GPU スペースの利用（有料）

### 3. API サービス化

FastAPI でラップして軽量 API として公開：

```python
from fastapi import FastAPI
from src.hle_screener import score_single_item

app = FastAPI()

@app.post("/score")
async def score(text: str):
    # 事前構築済みインデックスを使用
    result = score_single_item(...)
    return result
```

## 📝 Streamlit Cloud デモ版のセットアップ

1. **GitHub リポジトリの準備**
```bash
git add app_cloud.py requirements.txt .streamlit/
git commit -m "Add Streamlit Cloud demo version"
git push origin main
```

2. **Streamlit Cloud での設定**
- https://share.streamlit.io/ にアクセス
- "New app" をクリック
- リポジトリを選択
- Main file: `app_cloud.py`
- Deploy!

3. **環境変数（必要に応じて）**
```
DEMO_MODE=true
MAX_ITEMS=50
```

## 🔧 プロダクション向け代替案

### Docker コンテナ化
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8501
CMD ["streamlit", "run", "src/hle_screener/app_streamlit.py"]
```

### Kubernetes デプロイメント
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hle-screener
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: app
        image: hle-screener:latest
        resources:
          requests:
            memory: "8Gi"
            nvidia.com/gpu: 1
```

## 📊 コスト比較

| プラットフォーム | 月額コスト | GPU | 適用性 |
|--------------|---------|-----|------|
| Streamlit Cloud | 無料 | ❌ | デモのみ |
| Google Colab Pro | $10 | ✅ | 開発/テスト |
| AWS EC2 g4dn | ~$400 | ✅ | プロダクション |
| ローカル | $0 | 依存 | 開発/小規模 |

## 🎯 結論

- **デモ/プレゼン用**: Streamlit Cloud で `app_cloud.py` を使用
- **開発/テスト**: ローカル環境または Google Colab
- **プロダクション**: GPU 対応クラウドまたはオンプレミス

Streamlit Cloud は優れたプラットフォームですが、ML モデルを使用する本アプリケーションには
リソース制限があるため、デモ版のみの公開となります。
import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pickle
from typing import List, Optional, Tuple, Dict, Any
import logging

from .schema import HLEItem, RetrievalResult, UMAPPoint
from .embed import EmbeddingClient

logger = logging.getLogger(__name__)


class UMAPVisualizer:
    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'cosine',
        random_state: int = 42
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.reducer = None
        self.embeddings_2d = None
        self.metadata = None
    
    def fit_transform(
        self,
        embeddings: np.ndarray,
        metadata: List[HLEItem]
    ) -> np.ndarray:
        logger.info(f"Computing UMAP projection for {len(embeddings)} points")
        
        self.reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            n_components=2
        )
        
        self.embeddings_2d = self.reducer.fit_transform(embeddings)
        self.metadata = metadata
        
        logger.info("UMAP projection computed successfully")
        return self.embeddings_2d
    
    def transform_query(self, query_embedding: np.ndarray) -> np.ndarray:
        if self.reducer is None:
            raise ValueError("UMAP reducer not fitted yet")
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        return self.reducer.transform(query_embedding)
    
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'reducer': self.reducer,
            'embeddings_2d': self.embeddings_2d,
            'metadata': [item.dict() for item in self.metadata] if self.metadata else None,
            'params': {
                'n_neighbors': self.n_neighbors,
                'min_dist': self.min_dist,
                'metric': self.metric,
                'random_state': self.random_state
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved UMAP visualization data to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'UMAPVisualizer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        viz = cls(**data['params'])
        viz.reducer = data['reducer']
        viz.embeddings_2d = data['embeddings_2d']
        viz.metadata = [HLEItem(**item) for item in data['metadata']] if data['metadata'] else None
        
        logger.info(f"Loaded UMAP visualization from {path}")
        return viz
    
    def create_plot(
        self,
        query_point: Optional[np.ndarray] = None,
        highlighted_indices: Optional[List[int]] = None,
        title: str = "HLE Embeddings UMAP Projection"
    ) -> go.Figure:
        if self.embeddings_2d is None or self.metadata is None:
            raise ValueError("No UMAP projection available")
        
        df = pd.DataFrame({
            'x': self.embeddings_2d[:, 0],
            'y': self.embeddings_2d[:, 1],
            'subject': [item.subject for item in self.metadata],
            'hle_id': [item.hle_id for item in self.metadata],
            'question': [item.question_text[:100] + "..." if len(item.question_text) > 100 
                        else item.question_text for item in self.metadata]
        })
        
        color_map = px.colors.qualitative.Set3
        subjects = df['subject'].unique()
        subject_colors = {subj: color_map[i % len(color_map)] 
                         for i, subj in enumerate(subjects)}
        
        fig = go.Figure()
        
        for subject in subjects:
            mask = df['subject'] == subject
            fig.add_trace(go.Scatter(
                x=df[mask]['x'],
                y=df[mask]['y'],
                mode='markers',
                name=subject,
                marker=dict(
                    size=6,
                    color=subject_colors[subject],
                    opacity=0.6
                ),
                text=df[mask]['question'],
                hovertemplate='<b>%{text}</b><br>Subject: ' + subject + '<extra></extra>'
            ))
        
        if highlighted_indices:
            highlighted_df = df.iloc[highlighted_indices]
            fig.add_trace(go.Scatter(
                x=highlighted_df['x'],
                y=highlighted_df['y'],
                mode='markers',
                name='Top-5 Similar',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='star',
                    line=dict(color='darkred', width=2)
                ),
                text=highlighted_df['question'],
                hovertemplate='<b>Top-5 Similar</b><br>%{text}<extra></extra>'
            ))
        
        if query_point is not None:
            fig.add_trace(go.Scatter(
                x=[query_point[0]],
                y=[query_point[1]],
                mode='markers',
                name='Query',
                marker=dict(
                    size=15,
                    color='black',
                    symbol='diamond',
                    line=dict(color='yellow', width=3)
                ),
                hovertemplate='<b>Query Point</b><extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            hovermode='closest',
            width=900,
            height=700,
            template='plotly_white'
        )
        
        return fig


def compute_umap_projection(
    embeddings: np.ndarray,
    metadata: List[HLEItem],
    config: Dict[str, Any],
    save_path: Optional[Path] = None
) -> UMAPVisualizer:
    viz = UMAPVisualizer(
        n_neighbors=config.get('umap_n_neighbors', 15),
        min_dist=config.get('umap_min_dist', 0.1),
        metric='cosine',
        random_state=config.get('seed', 42)
    )
    
    viz.fit_transform(embeddings, metadata)
    
    if save_path:
        viz.save(save_path)
    
    return viz


def project_and_visualize(
    query_text: str,
    top5_results: List[RetrievalResult],
    viz: UMAPVisualizer,
    embedding_client: EmbeddingClient
) -> go.Figure:
    query_embedding = embedding_client.encode(query_text, show_progress=False)
    query_2d = viz.transform_query(query_embedding)[0]
    
    highlighted_indices = []
    for result in top5_results:
        for i, item in enumerate(viz.metadata):
            if item.hle_id == result.hle_id:
                highlighted_indices.append(i)
                break
    
    fig = viz.create_plot(
        query_point=query_2d,
        highlighted_indices=highlighted_indices,
        title=f"UMAP Projection with Query and Top-5 Similar HLE Items"
    )
    
    return fig
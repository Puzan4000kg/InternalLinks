"""
Internal Link Opportunity Analyzer
--------------------------------
Install dependencies:
pip install streamlit sentence-transformers keybert yake rake-nltk summa hdbscan umap-learn networkx python-louvain gensim pyvis plotly scikit-learn spacy trafilatura pandas numpy tqdm
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
from datetime import datetime
import json
from typing import List, Dict, Tuple, Optional, Union
import re
import warnings
from collections import Counter, defaultdict
import itertools
import tempfile
import base64
from io import StringIO

# NLP
import spacy
import yake
from rake_nltk import Rake
from summa import keywords as summa_keywords
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import trafilatura

# Clustering & ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan
import umap
import networkx as nx
try:
    import community  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    warnings.warn("python-louvain not installed, Louvain clustering will be disabled")

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit config and styling
st.set_page_config(
    page_title="Internal Link Analyzer",
    page_icon="🔗",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
.main {max-width: 1200px;}
.stProgress .st-bo {height: 20px;}
.reportview-container .main .block-container {max-width: 1200px;}
</style>
""", unsafe_allow_html=True)

# Cache the spaCy and sentence transformer models
@st.cache_resource
def load_spacy_model(lang: str):
    if lang == "en":
        return spacy.load("en_core_web_sm")
    elif lang == "ru":
        return spacy.load("ru_core_news_sm")
    else:
        return spacy.load("en_core_web_sm")  # fallback

@st.cache_resource
def load_sentence_transformer(model_name: str):
    return SentenceTransformer(model_name)

@st.cache_resource
def load_keybert():
    return KeyBERT()

# Sample texts for demo mode
SAMPLE_TEXTS = {
    "AI Ethics": """
    Artificial Intelligence ethics is a complex field addressing the moral implications of AI systems.
    Machine learning algorithms raise questions about bias, fairness, and transparency.
    Neural networks need careful oversight to ensure responsible AI development.
    Deep learning models must be designed with privacy and security in mind.
    """,
    "Climate Change": """
    Global warming is causing significant changes to Earth's climate systems.
    Greenhouse gas emissions continue to rise despite international agreements.
    Renewable energy adoption is crucial for reducing carbon footprint.
    Climate change impacts include rising sea levels and extreme weather events.
    """,
    "Digital Privacy": """
    Data protection regulations help safeguard personal information online.
    Encryption technologies play a vital role in maintaining digital privacy.
    User consent and transparency are key principles in privacy frameworks.
    Cybersecurity measures must evolve to address emerging privacy threats.
    """
}

class InternalLinkAnalyzer:
    def __init__(self, lang: str = "en", random_seed: int = 42):
        self.lang = lang
        self.random_seed = random_seed
        self.nlp = load_spacy_model(lang)
        np.random.seed(random_seed)
        
    def preprocess_text(self, text: str) -> str:
        doc = self.nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and token.text.strip()
        ]
        return " ".join(tokens)

    def extract_keywords_yake(self, text: str, top_n: int = 50) -> List[Tuple[str, float]]:
        try:
            kw_extractor = yake.KeywordExtractor(
                lan=self.lang,
                n=3,
                dedupLim=0.7,
                windowsSize=3,
                top=top_n
            )
            keywords = kw_extractor.extract_keywords(text)
            return [(kw, score) for kw, score in keywords]
        except Exception as e:
            logger.warning(f"YAKE extraction failed: {str(e)}")
            return []

    def extract_keywords_rake(self, text: str, top_n: int = 50) -> List[Tuple[str, float]]:
        try:
            rake = Rake(max_length=3)
            rake.extract_keywords_from_text(text)
            keywords = rake.get_ranked_phrases_with_scores()
            return [(kw, score) for score, kw in keywords[:top_n]]
        except Exception as e:
            logger.warning(f"RAKE extraction failed: {str(e)}")
            return []

    def extract_keywords_textrank(self, text: str, top_n: int = 50) -> List[Tuple[str, float]]:
        try:
            keywords = summa_keywords.keywords(text, ratio=0.3, split=True)
            # Normalize scores to [0,1]
            scores = np.linspace(1, 0.1, len(keywords))
            return list(zip(keywords[:top_n], scores[:top_n]))
        except Exception as e:
            logger.warning(f"TextRank extraction failed: {str(e)}")
            return []

    def extract_keywords_tfidf(self, texts: List[str], top_n: int = 50) -> List[List[Tuple[str, float]]]:
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=1000,
                stop_words="english" if self.lang == "en" else None
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            keywords_per_doc = []
            for doc_idx in range(len(texts)):
                scores = tfidf_matrix[doc_idx].toarray()[0]
                sorted_idx = np.argsort(scores)[::-1]
                keywords = [
                    (feature_names[idx], scores[idx])
                    for idx in sorted_idx[:top_n]
                ]
                keywords_per_doc.append(keywords)
            return keywords_per_doc
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {str(e)}")
            return [[] for _ in texts]

    def extract_keywords_keybert(
        self, text: str, top_n: int = 50
    ) -> List[Tuple[str, float]]:
        try:
            kw_model = load_keybert()
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words="english" if self.lang == "en" else None,
                top_n=top_n
            )
            return keywords
        except Exception as e:
            logger.warning(f"KeyBERT extraction failed: {str(e)}")
            return []

    def extract_keywords_spacy(self, text: str, top_n: int = 50) -> List[Tuple[str, float]]:
        try:
            doc = self.nlp(text)
            noun_chunks = list(doc.noun_chunks)
            # Score based on chunk length and frequency
            chunk_scores = defaultdict(float)
            for chunk in noun_chunks:
                chunk_text = chunk.text.lower()
                chunk_scores[chunk_text] += len(chunk.text.split()) * 0.1
            
            sorted_chunks = sorted(
                chunk_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_chunks[:top_n]
        except Exception as e:
            logger.warning(f"spaCy extraction failed: {str(e)}")
            return []

def main():
    st.title("🔗 Internal Link Opportunity Analyzer")
    st.markdown("""
    Discover internal linking opportunities by analyzing and clustering keyphrases across your articles.
    This tool helps you build a semantic network of your content and suggests relevant internal links.
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Input mode
        input_mode = st.selectbox(
            "Input Mode",
            ["Paste article texts", "Upload CSV", "Fetch by URLs", "Demo Mode"]
        )
        
        # Language selection
        lang = st.selectbox("Language", ["English", "Russian", "Auto"], index=0)
        lang_code = {"English": "en", "Russian": "ru", "Auto": "en"}[lang]
        
        # Model selection
        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
            index=0
        )
        
        # Method toggles
        st.subheader("Enable/Disable Methods")
        use_yake = st.checkbox("YAKE", value=True)
        use_rake = st.checkbox("RAKE", value=True)
        use_textrank = st.checkbox("TextRank", value=True)
        use_tfidf = st.checkbox("TF-IDF", value=True)
        use_keybert = st.checkbox("KeyBERT", value=True)
        use_spacy = st.checkbox("spaCy Noun Chunks", value=True)
        
        # Clustering toggles
        use_agglomerative = st.checkbox("Agglomerative", value=True)
        use_hdbscan = st.checkbox("HDBSCAN", value=True)
        use_spectral = st.checkbox("Spectral", value=True)
        use_louvain = st.checkbox("Louvain", value=LOUVAIN_AVAILABLE)
        use_bertopic = st.checkbox("BERTopic", value=True)
        
        # Parameters
        st.subheader("Parameters")
        top_n_keyphrases = st.number_input(
            "Top-N keyphrases per method",
            min_value=10,
            max_value=200,
            value=50
        )
        distance_threshold = st.slider(
            "Distance threshold (Agglomerative)",
            min_value=0.1,
            max_value=0.9,
            value=0.4
        )
        min_topic_size = st.number_input(
            "Min topic size",
            min_value=2,
            max_value=20,
            value=5
        )
        random_seed = st.number_input(
            "Random seed",
            min_value=1,
            max_value=9999,
            value=42
        )
        
        run_button = st.button("🚀 Run Analysis")
        
    # Main area
    if input_mode == "Demo Mode":
        texts = list(SAMPLE_TEXTS.values())
        titles = list(SAMPLE_TEXTS.keys())
        urls = [f"https://example.com/{title.lower().replace(' ', '-')}" for title in titles]
    elif input_mode == "Paste article texts":
        text_input = st.text_area(
            "Paste your articles (one per line, separate with ---)",
            height=200
        )
        if text_input:
            texts = [t.strip() for t in text_input.split("---")]
            titles = [f"Article {i+1}" for i in range(len(texts))]
            urls = [f"https://example.com/article-{i+1}" for i in range(len(texts))]
    elif input_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            required_cols = ["url", "title"]
            if not all(col in df.columns for col in required_cols):
                st.error("CSV must contain 'url' and 'title' columns!")
                return
            urls = df["url"].tolist()
            titles = df["title"].tolist()
            texts = df["content"].tolist() if "content" in df.columns else titles
    elif input_mode == "Fetch by URLs":
        url_input = st.text_area(
            "Enter URLs (one per line)",
            height=200
        )
        if url_input:
            urls = [url.strip() for url in url_input.split("\n") if url.strip()]
            
            progress_text = "Fetching articles..."
            progress_bar = st.progress(0)
            
            texts = []
            titles = []
            for i, url in enumerate(urls):
                try:
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded:
                        text = trafilatura.extract(downloaded)
                        title = trafilatura.extract_metadata(downloaded).title
                        texts.append(text)
                        titles.append(title)
                    else:
                        st.warning(f"Could not fetch {url}")
                        texts.append("")
                        titles.append(f"Article {i+1}")
                except Exception as e:
                    st.error(f"Error fetching {url}: {str(e)}")
                    texts.append("")
                    titles.append(f"Article {i+1}")
                progress_bar.progress((i + 1) / len(urls))
            
            progress_bar.empty()

    # Initialize analyzer and run analysis
    if run_button and 'texts' in locals() and texts:
        analyzer = InternalLinkAnalyzer(lang=lang_code, random_seed=random_seed)
        
        # Progress tracking
        progress_placeholder = st.empty()
        status = st.empty()
        timeline = []
        start_time = time.time()
        
        def update_progress(step: str, progress: float):
            status.text(f"Step: {step}")
            progress_placeholder.progress(progress)
            timeline.append({
                "step": step,
                "time": time.time() - start_time
            })
        
        try:
            # 1. Preprocessing
            update_progress("Preprocessing texts", 0.1)
            processed_texts = [analyzer.preprocess_text(text) for text in texts]
            
            # 2. Keyword Extraction
            update_progress("Extracting keywords", 0.2)
            all_keywords = defaultdict(list)
            
            if use_yake:
                for text in processed_texts:
                    all_keywords["yake"].extend(analyzer.extract_keywords_yake(text, top_n_keyphrases))
            
            if use_rake:
                for text in processed_texts:
                    all_keywords["rake"].extend(analyzer.extract_keywords_rake(text, top_n_keyphrases))
            
            if use_textrank:
                for text in processed_texts:
                    all_keywords["textrank"].extend(analyzer.extract_keywords_textrank(text, top_n_keyphrases))
            
            if use_tfidf:
                tfidf_keywords = analyzer.extract_keywords_tfidf(processed_texts, top_n_keyphrases)
                for doc_keywords in tfidf_keywords:
                    all_keywords["tfidf"].extend(doc_keywords)
            
            if use_keybert:
                for text in processed_texts:
                    all_keywords["keybert"].extend(analyzer.extract_keywords_keybert(text, top_n_keyphrases))
            
            if use_spacy:
                for text in processed_texts:
                    all_keywords["spacy"].extend(analyzer.extract_keywords_spacy(text, top_n_keyphrases))
            
            # Deduplicate and merge keywords
            unique_keywords = set()
            for method_keywords in all_keywords.values():
                unique_keywords.update(kw for kw, _ in method_keywords)
            
            # 3. Embeddings
            update_progress("Computing embeddings", 0.4)
            model = load_sentence_transformer(embedding_model)
            keyword_embeddings = model.encode(list(unique_keywords), show_progress_bar=False)
            
            # Normalize embeddings
            keyword_embeddings = keyword_embeddings / np.linalg.norm(keyword_embeddings, axis=1)[:, np.newaxis]
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(keyword_embeddings)
            
            # 4. Clustering
            update_progress("Clustering keyphrases", 0.6)
            clustering_results = {}
            
            if use_agglomerative:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold,
                    affinity="precomputed",
                    linkage="average"
                )
                distances = 1 - similarity_matrix
                clustering_results["agglomerative"] = clustering.fit_predict(distances)
            
            if use_hdbscan:
                clustering = hdbscan.HDBSCAN(
                    min_cluster_size=min_topic_size,
                    metric="precomputed"
                )
                distances = 1 - similarity_matrix
                clustering_results["hdbscan"] = clustering.fit_predict(distances)
            
            if use_spectral:
                clustering = SpectralClustering(
                    n_clusters=max(len(texts) // 2, 2),
                    affinity="precomputed",
                    random_state=random_seed
                )
                clustering_results["spectral"] = clustering.fit_predict(similarity_matrix)
            
            # 5. Visualization
            update_progress("Generating visualizations", 0.8)
            
            # UMAP for dimensionality reduction
            reducer = umap.UMAP(random_state=random_seed)
            embedding_2d = reducer.fit_transform(keyword_embeddings)
            
            # Scatter plot
            fig_scatter = px.scatter(
                x=embedding_2d[:, 0],
                y=embedding_2d[:, 1],
                color=clustering_results["agglomerative"] if use_agglomerative else None,
                hover_name=list(unique_keywords),
                title="Keyphrase Clusters (UMAP projection)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Heatmap of similarity matrix
            fig_heatmap = px.imshow(
                similarity_matrix,
                title="Keyphrase Similarity Matrix"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 6. Internal Link Suggestions
            update_progress("Generating link suggestions", 0.9)
            
            suggestions = []
            for i, (source_url, source_title, source_text) in enumerate(zip(urls, titles, texts)):
                # Get source document keywords
                source_keywords = set()
                for method_keywords in all_keywords.values():
                    source_keywords.update(
                        kw for kw, _ in method_keywords
                        if kw.lower() in source_text.lower()
                    )
                
                # Find source cluster (using agglomerative clustering)
                if use_agglomerative:
                    source_cluster = Counter(
                        clustering_results["agglomerative"][
                            list(unique_keywords).index(kw)
                        ]
                        for kw in source_keywords
                        if kw in unique_keywords
                    ).most_common(1)[0][0]
                    
                    # Find target articles in same cluster
                    for j, (target_url, target_title, target_text) in enumerate(zip(urls, titles, texts)):
                        if i != j:  # Avoid self-links
                            target_keywords = set()
                            for method_keywords in all_keywords.values():
                                target_keywords.update(
                                    kw for kw, _ in method_keywords
                                    if kw.lower() in target_text.lower()
                                )
                            
                            # Find keywords that are:
                            # 1. In target cluster
                            # 2. Present in target text
                            # 3. Not strongly present in source text
                            candidate_anchors = []
                            for kw in target_keywords:
                                if kw in unique_keywords:
                                    kw_idx = list(unique_keywords).index(kw)
                                    if (
                                        clustering_results["agglomerative"][kw_idx] == source_cluster
                                        and kw.lower() not in source_text.lower()
                                    ):
                                        candidate_anchors.append(kw)
                            
                            if candidate_anchors:
                                # Score anchors by similarity to target document
                                anchor_scores = []
                                for anchor in candidate_anchors:
                                    anchor_idx = list(unique_keywords).index(anchor)
                                    score = np.mean([
                                        similarity_matrix[anchor_idx][
                                            list(unique_keywords).index(kw)
                                        ]
                                        for kw in target_keywords
                                        if kw in unique_keywords
                                    ])
                                    anchor_scores.append((anchor, score))
                                
                                # Get top anchor
                                best_anchor, confidence = max(anchor_scores, key=lambda x: x[1])
                                
                                suggestions.append({
                                    "source_url": source_url,
                                    "source_title": source_title,
                                    "cluster": source_cluster,
                                    "anchor_text": best_anchor,
                                    "target_url": target_url,
                                    "target_title": target_title,
                                    "confidence": confidence
                                })
            
            # Display results
            update_progress("Completed!", 1.0)
            
            # Summary metrics
            st.header("📊 Analysis Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Articles", len(texts))
            with col2:
                st.metric("Unique Keywords", len(unique_keywords))
            with col3:
                st.metric("Suggested Links", len(suggestions))
            
            # Timeline visualization
            fig_timeline = go.Figure()
            for t in timeline:
                fig_timeline.add_trace(go.Scatter(
                    x=[t["time"]],
                    y=[t["step"]],
                    mode="markers+text",
                    text=[f"{t['time']:.1f}s"],
                    textposition="top center"
                ))
            fig_timeline.update_layout(
                title="Analysis Timeline",
                xaxis_title="Time (seconds)",
                showlegend=False
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Link suggestions table
            st.header("🔗 Internal Link Suggestions")
            df_suggestions = pd.DataFrame(suggestions)
            st.dataframe(
                df_suggestions[[
                    "source_title", "anchor_text",
                    "target_title", "confidence"
                ]].style.format({
                    "confidence": "{:.2f}"
                })
            )
            
            # Export results
            if st.button("📥 Download Results"):
                csv = df_suggestions.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="internal_links.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.exception("Analysis failed")

if __name__ == "__main__":
    main() 
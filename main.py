import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sentence_transformers import SentenceTransformer
import plotly.express as px
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()


class EmbeddingModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def get_embeddings(self, chunks):
        if isinstance(self.model, SentenceTransformer):
            return self.model.encode(chunks)
        return np.array([self.model.embed_query(chunk) for chunk in chunks])


class EmbeddingVisualizer:
    def __init__(self):
        self.embedding_models = {
            'MiniLM-L6': EmbeddingModel(
                'MiniLM-L6',
                SentenceTransformer('all-MiniLM-L6-v2')
            ),
            'OpenAI': EmbeddingModel(
                'OpenAI',
                OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            ),
            'JINA': EmbeddingModel(
                'JINA',
                SentenceTransformer('jinaai/jina-embeddings-v2-base-en')
            ),
            # 'Nomic': EmbeddingModel(
            #     'Nomic',
            #     SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
            # )
        }

    def chunk_text(self, text, chunk_size):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def compute_embeddings(self, chunks, model_name):
        model = self.embedding_models[model_name]
        return model.get_embeddings(chunks)

    def reduce_dimensions(self, embeddings, method='pca', n_components=3):
        if len(embeddings) < 2:
            raise ValueError(
                "At least two chunks are required for dimensionality reduction.")

        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            if len(embeddings) <= 5:  # Check for minimum samples for t-SNE
                raise ValueError(
                    "Perplexity must be less than n_samples. Increase the number of input samples.")
            reducer = TSNE(n_components=n_components, random_state=42,
                           perplexity=min(30, len(embeddings) - 1))
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        return reducer.fit_transform(embeddings)

    def create_3d_plot(self, reduced_embeddings, chunks, method_name, model_name):
        df = pd.DataFrame(
            reduced_embeddings,
            columns=['Dim1', 'Dim2', 'Dim3']
        )
        # Change: Use the actual text as labels instead of "Chunk X"
        df['Text'] = chunks
        title = "3D Embedding Visualization" + \
            str(model_name) + " using " + str(method_name)

        fig = px.scatter_3d(
            df,
            x='Dim1',
            y='Dim2',
            z='Dim3',
            text='Text',  # Change: Use Text column for point labels
            hover_data=['Text'],
            title=title,
            labels={
                'Dim1': f'{method_name} Component 1',
                'Dim2': f'{method_name} Component 2',
                'Dim3': f'{method_name} Component 3'
            }
        )

        fig.update_traces(
            marker=dict(size=8),
            textposition='top center'
        )

        # Change: Adjust text settings for better readability
        fig.update_traces(
            textfont=dict(size=10),  # Adjust text size
            textposition='top center'
        )

        fig.update_layout(
            scene=dict(
                xaxis_title=f'{method_name} Component 1',
                yaxis_title=f'{method_name} Component 2',
                zaxis_title=f'{method_name} Component 3'
            ),
            width=800,
            height=600,
            showlegend=False  # Hide legend since we're showing text directly
        )

        return fig


def main():
    st.set_page_config(layout="wide")
    st.title("3D Embedding Visualizer - Model Comparison")

    # Initialize the visualizer
    visualizer = EmbeddingVisualizer()

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Text input
        input_text = st.text_area(
            "Input Text",
            value="Minha terra tem palmeiras, Onde canta o sabiá; As aves que aqui gorjeiam, Não gorjeiam como lá.",
            height=150
        )

        # Model selection
        model_1 = st.selectbox(
            "First Embedding Model",
            options=list(visualizer.embedding_models.keys()),
            index=0,
            key="model_1"
        )

        model_2 = st.selectbox(
            "Second Embedding Model",
            options=list(visualizer.embedding_models.keys()),
            index=1,
            key="model_2"
        )

        # Dimensionality reduction method
        reduction_method = st.selectbox(
            "Dimensionality Reduction Method",
            options=['PCA', 'TSNE', 'UMAP'],
            index=0
        )

        # Chunk size
        chunk_size = st.slider(
            "Chunk size (number of words):",
            min_value=1,
            max_value=50,
            value=10
        )

        # Generate button
        generate_button = st.button("Generate Visualizations")

    # Main content area
    if generate_button:
        try:
            # Process text into chunks
            chunks = visualizer.chunk_text(input_text, chunk_size)

            if len(chunks) < 2:
                st.error(
                    "You need at least two chunks for visualization. Try reducing the chunk size.")
                return

            st.write(f"Generated {len(chunks)} chunks.")

            # Create two columns for the visualizations
            col1, col2 = st.columns(2)

            # Process first model
            with col1:
                embeddings_1 = visualizer.compute_embeddings(chunks, model_1)
                reduced_embeddings_1 = visualizer.reduce_dimensions(
                    embeddings_1,
                    method=reduction_method.lower()
                )
                fig_1 = visualizer.create_3d_plot(
                    reduced_embeddings_1, chunks, reduction_method, model_1)
                st.plotly_chart(fig_1, use_container_width=True)

            # Process second model
            with col2:
                embeddings_2 = visualizer.compute_embeddings(chunks, model_2)
                reduced_embeddings_2 = visualizer.reduce_dimensions(
                    embeddings_2,
                    method=reduction_method.lower()
                )
                fig_2 = visualizer.create_3d_plot(
                    reduced_embeddings_2, chunks, reduction_method, model_2)
                st.plotly_chart(fig_2, use_container_width=True)

            # Display chunks for reference
            with st.expander("View Text Chunks"):
                for i, chunk in enumerate(chunks, 1):
                    st.text(f"Chunk {i}: {chunk}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

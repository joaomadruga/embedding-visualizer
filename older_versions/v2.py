import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sentence_transformers import SentenceTransformer
import plotly.express as px
import pandas as pd
import numpy as np


class EmbeddingVisualizer:
    def __init__(self, is_local=True):
        self.is_local = is_local
        self.model = self.load_model()

    def load_model(self):
        if self.is_local:
            return SentenceTransformer('all-MiniLM-L6-v2')
        return OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key="your-open-ai-key"
        )

    def chunk_text(self, text, chunk_size):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def compute_embeddings(self, chunks):
        if self.is_local:
            return self.model.encode(chunks)
        return np.array([self.model.embed_query(chunk) for chunk in chunks])

    def reduce_dimensions(self, embeddings, method='pca', n_components=3):
        if len(embeddings) < 2:
            raise ValueError(
                "At least two chunks are required for dimensionality reduction.")

        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        return reducer.fit_transform(embeddings)

    def create_3d_plot(self, reduced_embeddings, chunks, method_name):
        df = pd.DataFrame(
            reduced_embeddings,
            columns=['Dim1', 'Dim2', 'Dim3']
        )
        df['Chunk'] = [f'Chunk {i+1}' for i in range(len(chunks))]
        df['Text'] = chunks

        fig = px.scatter_3d(
            df,
            x='Dim1',
            y='Dim2',
            z='Dim3',
            text='Chunk',
            hover_data=['Text'],
            title=f'3D Embedding Visualization using {method_name}',
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

        fig.update_layout(
            scene=dict(
                xaxis_title=f'{method_name} Component 1',
                yaxis_title=f'{method_name} Component 2',
                zaxis_title=f'{method_name} Component 3'
            ),
            width=800,
            height=600
        )

        return fig


def main():
    st.set_page_config(layout="wide")
    st.title("3D Embedding Visualizer")

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
        generate_button = st.button("Generate Visualization")

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

            # Compute embeddings
            embeddings = visualizer.compute_embeddings(chunks)

            # Reduce dimensions
            reduced_embeddings = visualizer.reduce_dimensions(
                embeddings,
                method=reduction_method.lower()
            )

            # Create and display the plot
            fig = visualizer.create_3d_plot(
                reduced_embeddings, chunks, reduction_method)
            st.plotly_chart(fig, use_container_width=True)

            # Display chunks for reference
            with st.expander("View Text Chunks"):
                for i, chunk in enumerate(chunks, 1):
                    st.text(f"Chunk {i}: {chunk}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

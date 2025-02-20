import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np

is_local = True

# Load a pre-trained model for embeddings


@st.cache_resource
def load_model():
    if is_local:
        return SentenceTransformer('all-MiniLM-L6-v2')
    return OpenAIEmbeddings(model="text-embedding-ada-002", api_key="your-open-ai-key")

# Split text into chunks


def chunk_text(text, chunk_size):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Generate embeddings for the chunks


def compute_embeddings(chunks, model):
    if is_local:
        return model.encode(chunks)

    embeddings = []
    for chunk in chunks:
        embeddings.append(model.embed_query(chunk))
    return embeddings

# Reduce dimensions for visualization


def reduce_dimensions(embeddings):
    if len(embeddings) < 2:
        raise ValueError(
            "At least two chunks are required to perform dimensionality reduction.")
    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)


# Poems for tabs
poems = {
    "Tab 1": "Minha terra tem palmeiras, Onde canta o sabiá; As aves que aqui gorjeiam, Não gorjeiam como lá.",
    "Tab 2": "As pessoas sensíveis não são capazes De matar galinhas Porém são capazes De comer galinhas.",
    "Tab 3": "O poeta é um fingidor. Finge tão completamente Que chega a fingir que é dor A dor que deveras sente."
}

# Streamlit app
st.title("Embedding Visualizer with Tabs")

# Tabs for poems
tabs = st.tabs(list(poems.keys()))

# Load model
model = load_model()

# Iterate through tabs
for tab_name, tab_content in zip(poems.keys(), tabs):
    with tab_content:
        st.subheader(tab_name)
        input_text = poems[tab_name]
        st.text_area("Poem:", input_text, height=150)

        # Chunk size slider
        chunk_size = st.slider(f"Chunk size for {
                               tab_name} (number of words):", min_value=1, max_value=50, value=10, key=f"slider_{tab_name}")

        # Calculate embeddings and plot
        if st.button(f"Generate Embeddings for {tab_name}", key=f"button_{tab_name}"):
            # Process text into chunks
            chunks = chunk_text(input_text, chunk_size)
            st.write(f"Generated {len(chunks)} chunks.")

            if len(chunks) < 2:
                st.error(
                    "You need at least two chunks for visualization. Try reducing the chunk size.")
            else:
                # Compute embeddings
                embeddings = compute_embeddings(chunks, model)

                # Reduce dimensions for visualization
                reduced_embeddings = reduce_dimensions(embeddings)

                # Plot the points
                fig, ax = plt.subplots()
                ax.scatter(reduced_embeddings[:, 0],
                           reduced_embeddings[:, 1], alpha=0.7)

                for i, chunk in enumerate(chunks):
                    ax.annotate(f"Chunk {
                                i+1}", (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.6)

                ax.set_title("Embedding Cloud")
                ax.set_xlabel("PCA Component 1")
                ax.set_ylabel("PCA Component 2")
                st.pyplot(fig)

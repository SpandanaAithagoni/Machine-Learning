import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")

st.title("News Topic Discovery Dashboard")

st.markdown("""
This system uses Hierarchical Clustering to automatically group similar news articles based on textual similarity.

Discover hidden themes without defining categories upfront.
""")

st.sidebar.header("Dataset & Clustering Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

def load_csv(file):
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(file, encoding=enc)
        except:
            continue
    st.error("Unable to read file. Unsupported encoding.")
    st.stop()

if uploaded_file:
    df = load_csv(uploaded_file)
else:
    st.warning("Upload a CSV file to continue")
    st.stop()

text_columns = df.select_dtypes(include=['object']).columns.tolist()

if not text_columns:
    st.error("No text column detected in dataset.")
    st.stop()

text_column = st.sidebar.selectbox("Select Text Column", text_columns)

st.sidebar.subheader("TF-IDF Settings")

max_features = st.sidebar.slider("Maximum TF-IDF Features", 100, 2000, 1000)

use_stopwords = st.sidebar.checkbox("Use English Stopwords", value=True)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1,1)
elif ngram_option == "Bigrams":
    ngram_range = (2,2)
else:
    ngram_range = (1,2)

st.sidebar.subheader("Hierarchical Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

dendro_sample_size = st.sidebar.slider(
    "Number of Articles for Dendrogram",
    20, min(200, len(df)), 50
)

stop_words = "english" if use_stopwords else None

vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words=stop_words,
    ngram_range=ngram_range
)

X = vectorizer.fit_transform(df[text_column].astype(str))

if st.sidebar.button("Generate Dendrogram"):

    st.subheader("Dendrogram")

    subset_size = min(dendro_sample_size, X.shape[0])
    subset_indices = np.random.choice(range(X.shape[0]), size=subset_size, replace=False)

    X_subset = X[subset_indices].toarray()

    linked = linkage(X_subset, method=linkage_method)

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linked, ax=ax)
    ax.set_ylabel("Distance")
    ax.set_xlabel("Article Index")
    st.pyplot(fig)

    st.info("Inspect large vertical gaps to decide the number of clusters.")

st.sidebar.subheader("Apply Clustering")

n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

if st.sidebar.button("Apply Clustering"):

    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            metric="euclidean"
        )
    except:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            affinity="euclidean"
        )

    cluster_labels = model.fit_predict(X.toarray())

    df["Cluster"] = cluster_labels

    st.success("Clustering Applied Successfully")

    st.subheader("Cluster Visualization (PCA Projection)")

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.toarray())

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(
        X_reduced[:,0],
        X_reduced[:,1],
        c=cluster_labels,
        cmap="tab10"
    )
    ax2.set_xlabel("PCA Component 1")
    ax2.set_ylabel("PCA Component 2")
    st.pyplot(fig2)

    st.subheader("Cluster Summary")

    feature_names = vectorizer.get_feature_names_out()
    summary_data = []

    for cluster in range(n_clusters):

        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_size = len(cluster_indices)

        cluster_tfidf = X[cluster_indices].mean(axis=0)
        top_indices = np.argsort(cluster_tfidf.A1)[::-1][:10]
        top_keywords = [feature_names[i] for i in top_indices]

        representative_article = df.iloc[cluster_indices[0]][text_column][:200]

        summary_data.append([
            cluster,
            cluster_size,
            ", ".join(top_keywords),
            representative_article
        ])

    summary_df = pd.DataFrame(
        summary_data,
        columns=["Cluster ID", "Number of Articles", "Top Keywords", "Sample Article"]
    )

    st.dataframe(summary_df)

    st.subheader("Silhouette Score")

    sil_score = silhouette_score(X.toarray(), cluster_labels)

    st.metric("Silhouette Score", round(sil_score, 3))

    if sil_score > 0.5:
        st.success("Clusters are well separated.")
    elif sil_score > 0:
        st.warning("Clusters moderately overlap.")
    else:
        st.error("Poor clustering structure.")

    st.subheader("Business Interpretation")

    for cluster in range(n_clusters):
        keywords = summary_df.iloc[cluster]["Top Keywords"].split(",")[:3]
        st.markdown(
            f"Cluster {cluster}: Articles mainly discuss themes related to {', '.join(keywords)}. "
            "These can be used for automatic tagging, recommendations, and content organization."
        )

    st.info(
        "Articles grouped in the same cluster share similar vocabulary and themes. "
        "These clusters can be used for automatic tagging, recommendations, and content organization."
    )

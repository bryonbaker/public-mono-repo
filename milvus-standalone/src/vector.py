# Milvus client for vector database operations
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
# SentenceTransformer for generating text embeddings
from sentence_transformers import SentenceTransformer
# Cosine similarity metric for embedding comparison
from sklearn.metrics.pairwise import cosine_similarity
# NumPy for numerical computations
import numpy as np
# Matplotlib for visualisation
import matplotlib.pyplot as plt
# PCA for dimensionality reduction
from sklearn.decomposition import PCA

def main():
    # Connect to Milvus first
    milvus_url = "http://localhost:19530"
    connections.connect(alias="my-test", uri=milvus_url)

    # Define the collection name
    collection_name = "vectordb_collection"

    # If the collection already exists, drop it for a clean start (optional)
    if utility.has_collection(collection_name, using="my-test"):
        print(f"⚠️ Collection '{collection_name}' already exists. Dropping for clean run.")
        utility.drop_collection(collection_name, using="my-test")

    # Define schema after connecting
    id_field = FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=False
    )

    # Embed a test sentence so we can find the vector dimension that is needed to define the schema.
    embedding_model = "all-MiniLM-L6-v2"
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedding_model)
    test_text="test sentence"
    test_embedding = model.encode(test_text)
    embedding_dim = len(test_embedding)

    print(f"📊 EMBEDDING ANALYSIS:")
    print(f"   • Input text: '{test_text}'")
    print(f"   • Embedding dimensions: {embedding_dim}")
    print(f"   • Data type: {type(test_embedding)}")
    print(f"   • Sample values: {test_embedding[:10]}")
    print(f"   • Value range: [{min(test_embedding):.4f}, {max(test_embedding):.4f}]")

    embedding_field = FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=embedding_dim
    )

    doc_field = FieldSchema(
        name="metadata",
        dtype=DataType.VARCHAR,
        max_length=256
    )

    schema = CollectionSchema(
        fields=[id_field, embedding_field, doc_field],
        description="Milvus shakeout test",
        enable_dynamic_field=False
    )

    # ✅ Create the collection now
    collection = Collection(
        name=collection_name,
        schema=schema,
        using="my-test",
        shards_num=2
    )

    print(f"📚 Collection list: {utility.list_collections(using='my-test')}")

    # Step 4: Encode real sentences and explore similarity
    print("\n🔍 SEMANTIC SIMILARITY TEST:")
    sentences = [
        "The weather is beautiful today.",
        "Today has gorgeous weather.",
        "I love programming in Python.",
        "Dogs are better than cats."
    ]

    embeddings = model.encode(sentences)

    for sentence, vector in zip(sentences, embeddings):
        print(f"Embedding for '{sentence}' (first 5 values):\n{vector[:5]}\n")

    cos_sim = cosine_similarity([embeddings[0]], embeddings[1:])
    print("Similarity (weather vs weather):", cos_sim[0][0])
    print("Similarity (weather vs programming):", cos_sim[0][1])
    print("Similarity (weather vs dogs & cats):", cos_sim[0][2])

    if __name__ == "__main__":
        plt.title("PCA Projection of Embeddings")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.tight_layout()
        plt.savefig("embedding_pca_plot.png")  # Saves to file instead of displaying
        print("Plot saved to embedding_pca_plot.png")
    else:
        # Step 5: Visualise embeddings with PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        plt.scatter(reduced[:, 0], reduced[:, 1])
        for i, sentence in enumerate(sentences):
            plt.annotate(sentence, (reduced[i, 0], reduced[i, 1]))
        plt.title("PCA Projection of Embeddings")
        plt.show()

    # Create a list of dictionaries for DB insertion
    data = [
        {"id": i, "embedding": vec.tolist(), "metadata": sentences[i]}  # Ensure vector is a list. Store the original sentence as metadata.
        for i, vec in enumerate(embeddings)
    ]

    # Insert the vectors into the collection
    collection.insert(data=data)

    # OPTIONAL: Create index if not already created
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        },
        index_name="idx"
    )

    # Commit the update to the database
    collection.flush()
    collection.load()

  # 🔎 Now demonstrate vector DB retrieval with semantic search
    print("\n🔁 VECTOR DATABASE RETRIEVAL DEMO")
    query = "what is the weather today?"
    query_vector = model.encode([query])

    # Search DB using the embedded query
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param={"metric_type": "COSINE"},
        limit=3,                           # Return at most three results
        output_fields=["embedding","metadata"]        # optional if you want to view the matched vectors
    )

    # Mapping from id to original input text (stored in memory from `sentences`)
    id_to_text = {i: sentence for i, sentence in enumerate(sentences)}

    # Print matches
    print(f"\n📌 Query term: '{query}'\n")
    print("📥 Top three matches returned by vector DB:\n")
    for match in results[0]:
        match_id = match.id
        score = match.score
        matched_vector = match.entity.get("embedding")
        metadata = match.entity.get("metadata")
        matched_text = id_to_text.get(match_id, "[Unknown]")

        print(f"🆔 ID: {match_id}")
        print(f"📄 Metadata: {metadata}")
        print(f"🧠 Text: {matched_text}")
        print(f"📏 Cosine Similarity Score: {score:.4f}")
        print(f"📊 First 5 vector values: {matched_vector[:5]}\n")

    print("✅ Only semantically similar entries (e.g. weather-related) were returned.")
    print("❌ Unrelated entries like 'I love programming in Python.' were filtered out due to low similarity.")
  

    # Close the Milvus connection
    collection.release()
    utility.drop_collection(collection_name, using="my-test")

if __name__ == "__main__":
    main()

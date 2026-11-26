"""Embed local corpus and upsert it into a Pinecone namespace."""
from langchain_openai import OpenAIEmbeddings
from config import MODEL, index_client, settings
from corpus import load_chunks


def ingest():
    cfg = settings()
    index = index_client()
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("No supported documents found in data/docs")
    embeddings = OpenAIEmbeddings(model=MODEL, dimensions=1536)
    for start in range(0, len(chunks), 64):
        batch = chunks[start:start + 64]
        vectors = embeddings.embed_documents([doc.page_content for doc in batch])
        index.upsert(vectors=[{"id": doc.metadata["chunk_id"], "values": vector,
                               "metadata": doc.metadata} for doc, vector in zip(batch, vectors)],
                     namespace=cfg["namespace"])
    print(f"Upserted {len(chunks)} chunks into {cfg['index_name']} / {cfg['namespace']}")


if __name__ == "__main__":
    ingest()

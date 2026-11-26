"""Hybrid retrieval: BM25 over local corpus + dense Pinecone search."""
import argparse
import re
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from pydantic import ConfigDict, Field
from config import MODEL, index_client, settings
from corpus import load_chunks


def tokenize(text):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class PineconeRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: object = Field(exclude=True)
    embeddings: object = Field(exclude=True)
    namespace: str
    top_k: int = 10

    def _get_relevant_documents(self, query, *, run_manager):
        vector = self.embeddings.embed_query(query)
        result = self.index.query(vector=vector, top_k=self.top_k,
                                  include_metadata=True, namespace=self.namespace)
        return [Document(page_content=match.metadata["text"], metadata=dict(match.metadata))
                for match in result.matches if match.metadata and "text" in match.metadata]


class RAGSystem:
    def __init__(self, top_k=5, namespace=None):
        cfg = settings()
        self.top_k = top_k
        self.namespace = namespace or cfg["namespace"]
        self.chunks = load_chunks()
        if not self.chunks:
            raise RuntimeError("No local documents found for BM25")
        lexical = BM25Retriever.from_documents(self.chunks, preprocess_func=tokenize, k=max(10, top_k * 2))
        semantic = PineconeRetriever(index=index_client(), embeddings=OpenAIEmbeddings(model=MODEL, dimensions=1536),
                                     namespace=self.namespace, top_k=max(10, top_k * 2))
        self.retriever = EnsembleRetriever(retrievers=[lexical, semantic], weights=[0.5, 0.5],
                                           id_key="chunk_id")

    def retrieve(self, query):
        docs = self.retriever.invoke(query)
        unique = []
        seen = set()
        for doc in docs:
            document_id = doc.metadata["document_id"]
            if document_id not in seen:
                unique.append(doc)
                seen.add(document_id)
            if len(unique) == self.top_k:
                break
        return unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--namespace")
    args = parser.parse_args()
    for rank, doc in enumerate(RAGSystem(namespace=args.namespace).retrieve(args.query), 1):
        print(f"{rank}. {doc.metadata['document_id']} | {doc.metadata['source']} | page {doc.metadata['page']}")
        print(doc.page_content[:300], "\n")

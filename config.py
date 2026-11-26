"""Shared configuration and Pinecone index validation."""
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
DIMENSION = 1536
MODEL = "text-embedding-3-small"


def settings():
    missing = [key for key in ("PINECONE_API_KEY", "OPENAI_API_KEY", "INDEX_NAME") if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Complete .env: missing {', '.join(missing)}")
    return {
        "index_name": os.environ["INDEX_NAME"],
        "namespace": os.getenv("PINECONE_NAMESPACE", "technical-docs"),
        "cloud": os.getenv("PINECONE_CLOUD", "aws"),
        "region": os.getenv("PINECONE_REGION", "us-east-1"),
    }


def index_client(create=False):
    cfg = settings()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    name = cfg["index_name"]
    if not pc.has_index(name):
        if not create:
            raise RuntimeError(f"Index {name!r} does not exist. Run: python init_index.py")
        pc.create_index(name=name, dimension=DIMENSION, metric="cosine",
                        spec=ServerlessSpec(cloud=cfg["cloud"], region=cfg["region"]))
    info = pc.describe_index(name)
    if info.dimension != DIMENSION:
        raise RuntimeError(f"Index dimension is {info.dimension}; expected {DIMENSION} for {MODEL}")
    if not info.spec.serverless:
        raise RuntimeError("Expected a Pinecone Serverless index")
    return pc.Index(name)

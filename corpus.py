"""Load PDF, Markdown, and JSON documents into reproducible token-sized chunks."""
import hashlib
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DATA_DIR = Path(__file__).parent / "data" / "docs"


def load_documents(root=DATA_DIR):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    documents = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".pdf", ".json"}:
            continue
        source = path.relative_to(root).as_posix()
        category = path.parent.name if path.parent != root else "general"
        if path.suffix.lower() == ".pdf":
            pages = [(i, page.extract_text() or "") for i, page in enumerate(PdfReader(path).pages, 1)]
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            records = data if isinstance(data, list) else [data]
            pages = [(i, record.get("text", json.dumps(record, ensure_ascii=False)) if isinstance(record, dict) else str(record)) for i, record in enumerate(records, 1)]
        else:
            pages = [(1, path.read_text(encoding="utf-8"))]
        for page, content in pages:
            if content.strip():
                documents.append(Document(page_content=content.strip(), metadata={
                    "document_id": path.stem, "source": source, "page": page, "category": category,
                }))
    return documents


def load_chunks(root=DATA_DIR):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-small", chunk_size=650, chunk_overlap=90,
    )
    chunks = splitter.split_documents(load_documents(root))
    counters = {}
    for chunk in chunks:
        m = chunk.metadata
        key = (m["source"], m["page"])
        m["chunk_index"] = counters.get(key, 0)
        counters[key] = m["chunk_index"] + 1
        m["chunk_id"] = hashlib.sha256(
            f"{m['source']}:{m['page']}:{m['chunk_index']}:{chunk.page_content}".encode()
        ).hexdigest()[:32]
        m["text"] = chunk.page_content
    return chunks

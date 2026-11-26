"""Source-level Precision@k and Recall@k for a labeled golden set."""
import json
from pathlib import Path
from rag_system import RAGSystem


def evaluate(path=Path("data/golden_set.json"), k=5):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    rag = RAGSystem(top_k=k)
    precisions, recalls = [], []
    for case in cases:
        expected = set(case.get("documentos_relevantes", [case["documento_id_esperado"]]))
        found = [doc.metadata["document_id"] for doc in rag.retrieve(case["pregunta"])]
        hits = len(expected.intersection(found))
        precision = hits / k
        recall = hits / len(expected)
        precisions.append(precision)
        recalls.append(recall)
        print(f"{case['pregunta']}\n  expected={sorted(expected)} found={found} P@{k}={precision:.2f} R@{k}={recall:.2f}")
    print(f"\nMean Precision@{k}: {sum(precisions)/len(cases):.3f}")
    print(f"Mean Recall@{k}:    {sum(recalls)/len(cases):.3f}")


if __name__ == "__main__":
    evaluate()

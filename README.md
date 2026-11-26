# Pre-entrega 4: RAG híbrido con Pinecone

Servicio de recuperación de documentos técnicos de Python. La ingesta genera embeddings con `text-embedding-3-small` (1536 dimensiones) y guarda cada chunk y sus metadatos en Pinecone Serverless. `RAGSystem` combina BM25 local con búsqueda vectorial de Pinecone mediante `EnsembleRetriever`, y devuelve cinco documentos fuente distintos.

## Preparación

Requiere Python 3.10 o superior, una cuenta de Pinecone y una clave de OpenAI con acceso a embeddings. Desde la raíz del repo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Completar `.env` con `PINECONE_API_KEY`, `OPENAI_API_KEY` e `INDEX_NAME`. El archivo real está excluido de Git. `PINECONE_NAMESPACE` separa el corpus de otros datos; cloud y región se pueden modificar antes de crear el índice.

## Replicar el índice y ejecutar

```bash
python init_index.py
python ingest.py
python rag_system.py "¿Cómo busco archivos recursivamente con pathlib?"
python evaluate.py
```

`init_index.py` crea el índice Serverless en AWS us-east-1 si no existe y comprueba su dimensión. Si el índice ya existe con una dimensión distinta, detiene el proceso con un error explícito. `ingest.py` lee Markdown, PDF y JSON bajo `data/docs/`, divide en chunks de 650 tokens con solapamiento de 90 usando `RecursiveCharacterTextSplitter`, y sube vectores al namespace configurado. Cada vector incluye `document_id`, `chunk_id`, `source`, `page`, `category`, `chunk_index` y el texto original en `text`. Los IDs son deterministas para permitir reingestas por upsert. Si se elimina o modifica un documento, se deben borrar sus IDs anteriores del índice antes de reingerirlo.

BM25 se construye con los mismos archivos locales al arrancar el recuperador. Por ello, el directorio `data/docs/` debe acompañar al servicio y corresponder al corpus ingerido. Pinecone devuelve texto desde la metadata y no requiere otra base de datos. Los resultados se fusionan con pesos 0.5/0.5 y se deduplican por documento fuente.

## Evaluación

`data/golden_set.json` contiene cinco preguntas con un documento fuente relevante cada una. `evaluate.py` imprime los cinco IDs recuperados por pregunta, Precision@5 y Recall@5, más las medias. Se calcula a nivel de documento fuente: `Precision@5 = relevantes_recuperados / 5` y `Recall@5 = relevantes_recuperados / relevantes_esperados`. Como cada caso tiene **un** documento relevante, la Precision@5 máxima de este benchmark es **0.20**, incluso con Recall@5 = 1.00. El archivo también admite `documentos_relevantes` para etiquetar varios documentos útiles y evaluar la precisión de manera más completa.

Las métricas reales se imprimen al ejecutar el script con claves válidas e ingesta completa; el repositorio no incluye un resultado numérico inventado. La evaluación genera consultas a OpenAI y Pinecone y puede incurrir en costos.

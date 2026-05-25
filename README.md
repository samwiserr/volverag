# VolveRAG

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-ready-orange.svg)
![RAG](https://img.shields.io/badge/RAG-hybrid%20%2B%20structured-green.svg)
![CI](https://img.shields.io/badge/tests-passing-brightgreen.svg)

VolveRAG is a Streamlit-ready Retrieval-Augmented Generation application for querying Volve petrophysical reports with natural language. It combines deterministic structured lookups for exact numeric answers with hybrid semantic/keyword retrieval for narrative questions.

The current app uses:

- **Groq** for LLM calls
- **Hugging Face / Sentence Transformers** for local embeddings
- **ChromaDB** for vector search
- **BM25 + RRF** for hybrid retrieval
- **Structured JSON caches** for well picks, petrophysical parameters, evaluation parameters, and numeric facts

## Quick Start

```bash
git clone https://github.com/samwiserr/volverag.git
cd volverag/advanced_rag

pip install -r requirements.txt

cp .env.example .env
# Add GROQ_API_KEY and keep LLM_PROVIDER=groq

python scripts/build_sota.py --documents-path "../spwla_volve-main" --no-contextual --no-raptor
streamlit run web_app.py
```

For Streamlit Community Cloud, use the prebuilt release assets instead of building on the server:

```toml
VECTORSTORE_URL = "https://github.com/samwiserr/volverag/releases/download/v2.0.1-sota/vectorstore.zip"
PDFS_URL = "https://github.com/samwiserr/volverag/releases/download/v2.0.1-sota/pdfs.zip"
GROQ_API_KEY = "gsk_..."
LLM_PROVIDER = "groq"
EMBEDDING_PROVIDER = "huggingface"
LOCAL_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
```

## Repository Structure

```text
.
├── advanced_rag/          # Main application
│   ├── src/               # RAG graph, tools, loaders, processors
│   ├── web_app/           # Streamlit app and UI logic
│   ├── scripts/           # Build and evaluation scripts
│   └── README.md          # Detailed developer documentation
├── DATA_POLICY.md         # Data handling policy
├── EXTERNAL_TOOLS.md      # Optional local dependencies
└── LICENSE
```

## Example Queries

- `What is the water saturation value of Hugin formation in 15/9-F-5?`
- `What formations are present in 15/9-F-5?`
- `What is the porosity for Hugin in 15/9-F-5?`
- `Show the evaluation parameters for Hugin in 15/9-F-5.`

Validated structured lookup example:

```text
15/9-F-5 / Hugin
SW   = 0.216
PHIF = 0.22
N/G  = 0.889
```

## Documentation

- [Detailed developer README](advanced_rag/README.md)
- [Setup Guide](advanced_rag/SETUP.md)
- [Data Policy](DATA_POLICY.md)
- [External Tools](EXTERNAL_TOOLS.md)

## Data Policy

The Volve source dataset, generated vectorstores, PDF bundles, and cache files are intentionally excluded from Git. Build them locally or download the release assets configured above.

## License

See [LICENSE](LICENSE).





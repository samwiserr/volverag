# VolveRAG

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-ready-orange.svg)
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)

Retrieval-Augmented Generation (RAG) system for querying Volve petrophysical reports using natural language. Built with LangGraph, OpenAI GPT-4o, and advanced retrieval techniques.

## 📖 Overview

VolveRAG enables natural language querying of petrophysical reports from the Volve field dataset. Ask questions about wells, formations, petrophysical parameters, and more - the system understands your queries and provides accurate answers with source citations.

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/samwiserr/VolveRAG.git
cd VolveRAG/advanced_rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Download Volve dataset separately (outside repository)
# Place it at: ../spwla_volve-main/ (or configure your path)
# See DATA_POLICY.md for details

# 5. Build index
python -m src.main --build-index --documents-path ../spwla_volve-main

# 6. Run web UI
streamlit run web_app.py
```

## 📁 Repository Structure

```
VolveRAG/
├── advanced_rag/          # ✅ Main application (use this)
│   ├── src/               # Source code
│   ├── web_app.py         # Streamlit UI
│   ├── README.md          # 📖 Main documentation
│   └── requirements.txt   # Dependencies
├── DATA_POLICY.md         # Data handling policy
├── EXTERNAL_TOOLS.md      # External dependencies
└── LICENSE                # MIT License
```

## 📚 Documentation

- **[Main README](advanced_rag/README.md)** - Complete documentation, features, and usage guide
- **[Setup Guide](advanced_rag/SETUP.md)** - Detailed installation instructions
- **[Data Policy](DATA_POLICY.md)** - What should/shouldn't be committed
- **[External Tools](EXTERNAL_TOOLS.md)** - Required and optional dependencies
- **[Contributing](advanced_rag/CONTRIBUTING.md)** - How to contribute

## ✨ Key Features

- **Natural Language Querying**: Ask questions in plain English
- **Deterministic Fact Retrieval**: 100% accurate structured lookups
- **Hybrid Retrieval**: Semantic + keyword search for better results
- **Source Citations**: Every answer includes exact page numbers
- **Stateful Chat**: Maintains conversation context
- **Entity Disambiguation**: Handles typos and ambiguous queries

## 🔍 Example Queries

- "What formations are in well 15/9-F-5?"
- "What is the porosity for Hugin in 15/9-F-5?"
- "What is the depth of Sleipner formation in 15/9-19A?"
- "List all formations and their properties"

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Volve dataset (download separately - see [DATA_POLICY.md](DATA_POLICY.md))

## 🤝 Contributing

See [CONTRIBUTING.md](advanced_rag/CONTRIBUTING.md) for guidelines.

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Main Documentation**: [advanced_rag/README.md](advanced_rag/README.md)
- **GitHub Repository**: https://github.com/samwiserr/VolveRAG

---

**Note**: All development and usage should be done in the `advanced_rag/` directory. See the [main README](advanced_rag/README.md) for complete documentation.





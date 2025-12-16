# Setup Guide

## Step-by-Step Installation

### 1. Prerequisites

- Python 3.8 or higher
- pip package manager
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### 2. Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/volve-rag.git
cd volve-rag/advanced_rag

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 4. Prepare Documents

Place your Volve petrophysical reports in a directory. The system expects:
- PDF files (petrophysical reports)
- DOC/DOCX files (LFP reports)
- `Well_picks_Volve_v1.dat` (formation picks data)

Example structure:
```
spwla_volve-main/
├── 15_9-F-5/
│   └── PETROPHYSICAL_REPORT_1.PDF
├── 15_9-F-15/
│   └── PETROPHYSICAL_REPORT_1.PDF
└── Well_picks_Volve_v1.dat
```

### 5. Build the Index

```bash
python -m src.main --build-index --documents-path ../spwla_volve-main
```

This will:
- Process all documents (may take 10-30 minutes)
- Extract text and parse tables
- Create embeddings
- Build vector store
- Generate caches

### 6. Run the Application

**Web UI:**
```bash
streamlit run web_app.py
```

**CLI:**
```bash
python -m src.main --query "What formations are in well 15/9-F-5?"
```

## Troubleshooting

### Issue: "OPENAI_API_KEY not set"
**Solution:** Make sure your `.env` file exists and contains `OPENAI_API_KEY=sk-...`

### Issue: "Vector store not found"
**Solution:** Run `python -m src.main --build-index` first

### Issue: Import errors
**Solution:** Make sure all dependencies are installed: `pip install -r requirements.txt`

### Issue: Document processing fails
**Solution:** Check that documents are in the correct format and path

### Issue: Missing dependencies (antiword, tesseract)
**Solution:** 
- For `.doc` files: Install antiword (see `ANTIWORD_INSTALLATION.md`)
- For OCR: Install Tesseract OCR

## Optional: Install Additional Tools

### For .doc file support (Windows)
```powershell
# Run the installation script
.\install_antiword.ps1
```

### For OCR support
- **Windows**: Download Tesseract from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt-get install tesseract-ocr`
- **Mac**: `brew install tesseract`


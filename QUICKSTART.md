# Quick Start Guide - Volve Wells RAG System

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Web Frontend

**Windows:**
```bash
run_frontend.bat
```

**Linux/Mac:**
```bash
chmod +x run_frontend.sh
./run_frontend.sh
```

**Or directly:**
```bash
streamlit run src/frontend.py
```

### 3. Initialize the System

1. The web interface will open in your browser at `http://localhost:8501`
2. In the sidebar, click **"🔄 Initialize System"** to load the existing index
   - OR click **"🔨 Rebuild Index"** if this is your first time (takes longer)

### 4. Start Querying!

Try these example queries:
- "Which wells have high porosity?"
- "Find wells with permeability > 100 mD"
- "Show me wells in the Hugin formation"
- "Which wells have low water saturation?"

## 📋 Command Line Alternative

If you prefer the command line:

```bash
# Build index (first time)
python src/main.py --rebuild

# Query the system
python src/main.py --query "Which wells have high porosity?"

# Interactive mode
python src/main.py --interactive
```

## 🎨 Frontend Features

- **Modern UI**: Clean, sleek interface with gradient styling
- **Natural Language Queries**: Ask questions in plain English
- **Structured Queries**: Use filters like "porosity > 0.2"
- **Semantic Search**: Find similar wells based on characteristics
- **Interactive Results**: Expandable result cards with detailed information
- **Export Results**: Download results as CSV or JSON
- **Example Queries**: Quick access to common queries
- **Search Settings**: Adjustable number of results and search types

## 🛠️ Troubleshooting

### "System Not Initialized" Error
- Make sure you've clicked "Initialize System" in the sidebar
- Check that the vector database exists in `vector_db/` directory
- If not, click "Rebuild Index" to create it

### "No LAS files found"
- Ensure the Volve data is in `spwla_volve-main/` directory
- Check that LAS files exist in the well folders

### Port Already in Use
- Streamlit uses port 8501 by default
- Change it: `streamlit run src/frontend.py --server.port 8502`

## 📚 More Information

See `README.md` for detailed documentation and API usage.


## How to Handle Data

### For Users

1. **Download the Volve dataset separately** from the official Equinor source
2. **Place it outside the repository** (e.g., `../spwla_volve-main/` or `~/volve-data/`)
3. **Reference it via path** in configuration or command-line arguments
4. **Never commit it** to Git

### For Developers

- The `.gitignore` file is configured to exclude all data files
- If you accidentally commit data files, remove them with:
  ```bash
  git rm --cached -r spwla_volve-main/
  git commit -m "Remove data files from repository"
  ```

## Why This Policy?

### Repository Size
- Data files are large (potentially GBs) and dramatically slow down:
  - `git clone` operations
  - `git pull` updates
  - CI/CD pipeline execution
  - Repository browsing on GitHub

### Licensing & Distribution
- The Volve dataset has specific distribution terms from Equinor
- Committing it to a public repository may violate these terms
- Users should obtain the dataset directly from the official source

### Version Control Efficiency
- Binary files (PDFs, LAS, etc.) don't benefit from Git's diff/merge
- Git is designed for text-based source code, not large binary datasets
- Large files bloat repository history

### CI/CD Performance
- Automated tests and builds fail or timeout with large repositories
- GitHub Actions has size limits and timeouts

## Expected Workflow

1. **Clone the repository** (code only, fast)
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Download Volve dataset** separately (from official source)
4. **Place dataset** at expected path (e.g., `../spwla_volve-main/`)
5. **Build index**: `python -m src.main --build-index --documents-path ../spwla_volve-main`
6. **Run application**: `streamlit run web_app.py`




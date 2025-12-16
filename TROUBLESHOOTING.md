# Troubleshooting Guide

## Blank Screen Issue

If you see a blank screen when launching the frontend:

### Solution 1: Check the Console/Terminal
- Look at the terminal where you ran `streamlit run src/frontend.py`
- Check for error messages
- Common errors:
  - Import errors
  - Missing dependencies
  - Path issues

### Solution 2: Initialize the System
1. The frontend should show a welcome screen even if the system isn't initialized
2. Click **"🔄 Initialize System"** in the sidebar
3. If that fails, click **"🔨 Rebuild Index"** (this will take longer but creates everything from scratch)

### Solution 3: Check Vector Database
- The vector database should be in `vector_db/` directory
- If it doesn't exist or is corrupted, rebuild the index

### Solution 4: Clear Browser Cache
- Sometimes browser cache can cause issues
- Try:
  - Hard refresh: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
  - Clear browser cache
  - Try incognito/private mode

### Solution 5: Check Port
- Streamlit uses port 8501 by default
- If it's in use, try:
  ```bash
  streamlit run src/frontend.py --server.port 8502
  ```

## Common Errors

### "Error initializing RAG system"
- **Cause**: Vector database doesn't exist or is corrupted
- **Solution**: Click "Rebuild Index" in the sidebar

### "No LAS files found"
- **Cause**: LAS files not in the correct location
- **Solution**: Ensure files are in `spwla_volve-main/` directory

### "ChromaDB metadata error"
- **Cause**: This was a bug that has been fixed
- **Solution**: Rebuild the index to use the fixed version

### "Module not found"
- **Cause**: Missing dependencies
- **Solution**: Run `pip install -r requirements.txt`

## Getting Help

1. Check the terminal/console output for detailed error messages
2. Make sure all dependencies are installed
3. Try rebuilding the index
4. Check that LAS files are in the correct location


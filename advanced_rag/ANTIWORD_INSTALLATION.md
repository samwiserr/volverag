# Antiword Installation Guide for Windows

## Why Antiword is Needed
Antiword is required to extract text from legacy `.doc` files (old Microsoft Word format). Without it, 2 documents will be skipped.

## Manual Installation Steps

### Option 1: Download from Official Site (Recommended)

1. **Visit the official antiword website:**
   - Go to: http://www.winfield.demon.nl/
   - Or search for "antiword winfield demon" in your browser

2. **Download antiword:**
   - Download `antiword-0.37.zip` (or latest version)
   - Save to your Downloads folder

3. **Extract the files:**
   - Extract the zip file to: `C:\Users\samue\antiword\`
   - You should have `antiword.exe` in that folder

4. **Add to PATH:**
   - Press `Win + X` and select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "User variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\Users\samue\antiword`
   - Click OK on all dialogs

5. **Verify installation:**
   - Open a NEW PowerShell window
   - Run: `antiword --version`
   - You should see version information

### Option 2: Using Chocolatey (If Available)

If you have Chocolatey package manager installed:

```powershell
choco install antiword
```

### Option 3: Alternative Download Sources

Try these alternative sources if the official site is unavailable:

1. **GitHub releases:**
   - Search GitHub for "antiword windows" or "antiword releases"
   - Look for pre-compiled Windows binaries

2. **Archive.org:**
   - Search archive.org for "antiword windows"
   - May have archived versions

## After Installation

1. **Restart your terminal/PowerShell** (required for PATH changes)

2. **Verify antiword works:**
   ```powershell
   antiword --version
   ```

3. **Rebuild the RAG index:**
   ```powershell
   python -m src.main_rag_system --rebuild
   ```

## Current Status

- ✅ 35 PDF documents processed successfully
- ❌ 2 .doc files skipped (waiting for antiword installation)

Once antiword is installed, all 37 documents will be processed!

## Troubleshooting

**If antiword is installed but not detected:**
- Make sure you restarted PowerShell after adding to PATH
- Check that `antiword.exe` exists in the PATH directory
- Try running `antiword --version` manually to verify it works
- The system checks these locations automatically:
  - System PATH
  - `C:\Program Files\antiword\`
  - `C:\Users\samue\antiword\`
  - `%LOCALAPPDATA%\antiword\`


# External Tools Required

This document lists external tools and dependencies needed to run VolveRAG.

## Required Tools

### Python 3.8+
- **Purpose**: Core runtime environment
- **Installation**: 
  - Download from [python.org](https://www.python.org/downloads/)
  - Or use package manager: `brew install python3` (Mac), `apt-get install python3` (Linux)
- **Verification**: `python --version` or `python3 --version`

### Git
- **Purpose**: Version control (for cloning the repository)
- **Installation**: 
  - Download from [git-scm.com](https://git-scm.com/downloads)
  - Or use package manager: `brew install git` (Mac), `apt-get install git` (Linux)
- **Verification**: `git --version`

### pip (Python Package Manager)
- **Purpose**: Installing Python dependencies
- **Usually comes with**: Python 3.4+
- **Verification**: `pip --version` or `pip3 --version`

## Optional Tools (for Enhanced Features)

### antiword (Windows)
- **Purpose**: Parsing legacy `.doc` files (not DOCX)
- **When needed**: If you have `.doc` files in your Volve dataset
- **Installation**:
  - **Windows**: Run `advanced_rag/install_antiword.ps1` (PowerShell script)
  - **Manual**: Download from [winfield.demon.nl](http://www.winfield.demon.nl/)
  - Extract to a directory in your PATH
- **Verification**: `antiword --version`
- **Note**: DOCX files are handled by `python-docx` library (no external tool needed)

### Tesseract OCR
- **Purpose**: Optical Character Recognition for scanned PDFs
- **When needed**: If PDFs contain scanned images instead of text layers
- **Installation**:
  - **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
  - **Linux**: `sudo apt-get install tesseract-ocr`
  - **Mac**: `brew install tesseract`
- **Verification**: `tesseract --version`
- **Note**: Most modern PDFs have text layers and don't need OCR

### Poppler
- **Purpose**: PDF processing utilities (if using `pdf2image` or similar)
- **When needed**: If you're using PDF-to-image conversion features
- **Installation**:
  - **Windows**: Download from [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)
  - **Linux**: `sudo apt-get install poppler-utils`
  - **Mac**: `brew install poppler`
- **Verification**: `pdftoppm -h` or `pdfinfo -h`
- **Note**: Not required for basic PDF text extraction (PyMuPDF handles this)

## Platform-Specific Notes

### Windows
- Use PowerShell or Command Prompt
- For `.doc` files, install `antiword` (see above)
- Tesseract and Poppler are optional

### Linux
- Most tools available via package manager (`apt-get`, `yum`, etc.)
- May need to install development headers for some Python packages

### macOS
- Use Homebrew (`brew`) for most tools
- May need Xcode Command Line Tools: `xcode-select --install`

## Verification Checklist

Before running VolveRAG, verify you have:

- [ ] Python 3.8+ installed
- [ ] pip installed
- [ ] Git installed (for cloning)
- [ ] OpenAI API key (for embeddings and LLM)
- [ ] Volve dataset downloaded (separately, outside repo)
- [ ] (Optional) antiword installed (if using `.doc` files)
- [ ] (Optional) Tesseract installed (if PDFs are scanned)
- [ ] (Optional) Poppler installed (if using PDF-to-image features)

## Troubleshooting

### "antiword: command not found"
- Install antiword (see above)
- Ensure it's in your system PATH
- Or skip `.doc` files (use DOCX instead)

### "Tesseract not found"
- Only needed for scanned PDFs
- Most PDFs work without it
- Install if you encounter OCR errors

### "Poppler not found"
- Only needed for PDF-to-image conversion
- Not required for basic text extraction
- Install if you need image rendering features

## Questions?

If you encounter issues with external tools:
1. Check if the tool is actually required for your use case
2. Verify installation with the verification commands above
3. Check PATH environment variable
4. Open an issue on GitHub with error details





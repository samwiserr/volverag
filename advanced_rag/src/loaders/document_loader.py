"""
Document loader with LangChain integration and fallback to existing processor.
Handles .doc and .dat files that LangChain can't process natively.
"""
import logging
from pathlib import Path
from typing import List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
)

# Fallback processor for .doc and .dat files
try:
    from ..processors.advanced_document_processor import AdvancedDocumentProcessor, ExtractionResult
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False
    AdvancedDocumentProcessor = None
    ExtractionResult = None

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Document loader that uses LangChain loaders first,
    then falls back to existing processor for .doc and .dat files.
    """
    
    def __init__(self):
        """Initialize the document loader."""
        self.fallback_processor = None
        if FALLBACK_AVAILABLE:
            try:
                self.fallback_processor = AdvancedDocumentProcessor()
                logger.info("[OK] Fallback processor available for .doc and .dat files")
            except Exception as e:
                logger.warning(f"Could not initialize fallback processor: {e}")
    
    def load_documents(self, directory: Path) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of LangChain Document objects
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        documents = []
        supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.dat'}
        
        # Find all supported documents
        document_files = []
        for ext in supported_extensions:
            document_files.extend(directory.glob(f'**/*{ext}'))
        
        # Filter out Readme files
        document_files = [
            f for f in document_files 
            if not f.name.lower() in ['readme.md', 'readme.txt']
        ]
        
        logger.info(f"Found {len(document_files)} documents to process")
        
        for file_path in document_files:
            try:
                loaded = self._load_single_document(file_path)
                if not loaded:
                    continue

                # Normalize to list
                loaded_docs: List[Document] = loaded if isinstance(loaded, list) else [loaded]
                documents.extend(loaded_docs)
                logger.info(f"[OK] Loaded: {file_path.name} ({len(loaded_docs)} document(s))")
            except Exception as e:
                logger.error(f"[ERROR] Failed to load {file_path.name}: {e}")
        
        logger.info(f"[OK] Successfully loaded {len(documents)} documents")
        return documents
    
    def _load_single_document(self, file_path: Path) -> Optional[Union[Document, List[Document]]]:
        """
        Load a single document, trying LangChain first, then fallback.
        
        Args:
            file_path: Path to document file
            
        Returns:
            LangChain Document or None if loading fails
        """
        ext = file_path.suffix.lower()
        
        # Use fallback for .doc and .dat files
        if ext in {'.doc', '.dat'}:
            return self._load_with_fallback(file_path)
        
        # Try LangChain loaders for other formats
        try:
            return self._try_langchain_loader(file_path)
        except Exception as e:
            logger.warning(f"LangChain loader failed for {file_path.name}: {e}")
            # Try fallback if available
            if self.fallback_processor:
                return self._load_with_fallback(file_path)
            raise
    
    def _try_langchain_loader(self, file_path: Path) -> List[Document]:
        """Try to load document using LangChain loaders (returns one Document per page when applicable)."""
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            # Try PyMuPDF first (better), then PyPDF
            try:
                loader = PyMuPDFLoader(str(file_path))
            except:
                loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            return docs or []
        
        elif ext == '.docx':
            # Try UnstructuredWordDocumentLoader first, then Docx2txtLoader
            try:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            except:
                loader = Docx2txtLoader(str(file_path))
            docs = loader.load()
            return docs or []
        
        elif ext in {'.txt', '.md'}:
            loader = TextLoader(str(file_path), encoding='utf-8')
            docs = loader.load()
            if not docs:
                return []

            # Add metadata for well picks detection to all docs
            is_well_picks = "Well_picks" in file_path.name or "well_picks" in file_path.name.lower()
            if is_well_picks:
                for d in docs:
                    d.metadata['document_type'] = 'well_picks'
                    d.metadata['is_formation_data'] = True
                    d.metadata['is_well_picks'] = True

            return docs
        
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    def _load_with_fallback(self, file_path: Path) -> Document:
        """
        Load document using fallback processor (for .doc and .dat files).
        
        Args:
            file_path: Path to document file
            
        Returns:
            LangChain Document
        """
        if not self.fallback_processor:
            raise RuntimeError("Fallback processor not available")
        
        # Use existing processor
        result: ExtractionResult = self.fallback_processor._process_single_document(file_path)
        
        if not result or not result.text:
            raise ValueError(f"Failed to extract text from {file_path.name}")
        
        # Convert to LangChain Document
        # Add document type metadata for better retrieval
        is_well_picks = "Well_picks" in file_path.name or "well_picks" in file_path.name.lower()
        is_formation_data = is_well_picks or ".dat" in file_path.suffix.lower()
        
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size': result.metadata.file_size,
            'page_count': result.metadata.page_count,
            'extraction_method': result.metadata.extraction_method,
            'document_type': 'well_picks' if is_well_picks else 'petrophysical_report',
            'is_formation_data': is_formation_data,
            'is_well_picks': is_well_picks,
        }
        
        return Document(
            page_content=result.text,
            metadata=metadata
        )


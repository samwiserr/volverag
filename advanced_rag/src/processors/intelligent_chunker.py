import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import numpy as np

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    chunk_id: int
    start_char: int
    end_char: int
    token_count: int
    sentence_count: int
    section_header: Optional[str] = None
    confidence_score: float = 1.0

@dataclass
class ChunkingResult:
    """Result of document chunking."""
    chunks: List[TextChunk]
    original_text: str
    metadata: Dict[str, Any]

class IntelligentChunker:
    """
    Advanced text chunking optimized for petrophysical documents.
    Uses semantic understanding, section detection, and quality metrics.
    """

    def __init__(self,
                 chunk_size: int = 512,
                 overlap: int = 50,
                 preserve_sections: bool = True):
        """
        Initialize the intelligent chunker.

        Args:
            chunk_size: Target tokens per chunk
            overlap: Token overlap between chunks
            preserve_sections: Whether to preserve document sections
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.preserve_sections = preserve_sections

        # Initialize NLTK
        try:
            # Try punkt_tab first (NLTK 3.9+)
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                # Fall back to punkt for older NLTK versions
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    # Download punkt_tab (primary) and punkt (fallback)
                    nltk.download('punkt_tab', quiet=True)
                    try:
                        nltk.download('punkt', quiet=True)
                    except:
                        pass  # punkt may not be needed if punkt_tab works
        except Exception as e:
            pass  # NLTK will be handled by sent_tokenize if available

        # Petrophysical document patterns
        self.section_patterns = {
            'executive_summary': re.compile(r'(?i)executive\s+summary|summary'),
            'introduction': re.compile(r'(?i)introduction|overview'),
            'methodology': re.compile(r'(?i)methodology|methods|approach'),
            'data_analysis': re.compile(r'(?i)data\s+analysis|analysis'),
            'results': re.compile(r'(?i)results|findings'),
            'conclusions': re.compile(r'(?i)conclusion|discussion'),
            'recommendations': re.compile(r'(?i)recommendation|advice'),
            'appendices': re.compile(r'(?i)appendix|appendices'),
            'references': re.compile(r'(?i)reference|bibliography'),
            'tables': re.compile(r'(?i)table\s+\d+|table\s+of\s+contents'),
            'figures': re.compile(r'(?i)figure\s+\d+|list\s+of\s+figures'),
        }

        # Petrophysical terminology patterns
        self.technical_patterns = {
            'well_info': re.compile(r'(?i)well\s+(?:name|number|id)|wellbore'),
            'formations': re.compile(r'(?i)formation|zone|interval|layer'),
            'measurements': re.compile(r'(?i)porosity|permeability|saturation|resistivity'),
            'depth': re.compile(r'(?i)depth|measured\s+depth|true\s+vertical\s+depth'),
            'lithology': re.compile(r'(?i)lithology|rock\s+type|mineralogy'),
            'fluids': re.compile(r'(?i)oil|water|gas|hydrocarbon|fluid'),
        }

    def chunk_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkingResult:
        """
        Intelligently chunk a document into optimal pieces.

        Args:
            text: Full document text
            metadata: Optional document metadata

        Returns:
            ChunkingResult with chunks and metadata
        """
        if not text.strip():
            return ChunkingResult([], text, {'error': 'Empty text'})

        # Preprocess text
        cleaned_text = self._preprocess_text(text)

        # Detect document sections
        sections = self._detect_sections(cleaned_text) if self.preserve_sections else []

        # Choose chunking strategy based on document type
        if sections and len(sections) > 3:
            chunks = self._chunk_by_sections(cleaned_text, sections)
        else:
            chunks = self._chunk_by_semantic_boundaries(cleaned_text)

        # Post-process chunks
        processed_chunks = self._post_process_chunks(chunks, cleaned_text)

        # Create result metadata
        result_metadata = {
            'total_chunks': len(processed_chunks),
            'avg_chunk_size': np.mean([c.token_count for c in processed_chunks]) if processed_chunks else 0,
            'total_tokens': sum(c.token_count for c in processed_chunks),
            'sections_detected': len(sections),
            'chunking_strategy': 'sections' if sections else 'semantic',
            'original_length': len(text),
            'processed_length': len(cleaned_text)
        }

        return ChunkingResult(processed_chunks, text, result_metadata)

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for chunking."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single

        # Normalize quotes and dashes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[-–—]', '-', text)

        # Remove page headers/footers (common in reports)
        text = self._remove_headers_footers(text)

        return text.strip()

    def _remove_headers_footers(self, text: str) -> str:
        """Remove repetitive headers/footers from reports."""
        lines = text.split('\n')
        if len(lines) < 10:
            return text

        # Check for repetitive patterns at start/end of pages
        # This is a simplified version - production would use more sophisticated detection
        first_lines = lines[:5]
        last_lines = lines[-5:]

        # Remove if first/last lines are very similar across "pages"
        # For now, just clean up obvious page breaks
        cleaned_lines = []
        for line in lines:
            # Skip lines that look like page numbers or headers
            if re.match(r'^\s*\d+\s*$', line.strip()):  # Just numbers
                continue
            if re.match(r'^\s*(page\s*\d+|confidential|draft)', line.lower()):
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect document sections using pattern matching."""
        sections = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Skip empty lines or very short lines
            if len(line_lower) < 10:
                continue

            # Check for section headers
            for section_type, pattern in self.section_patterns.items():
                if pattern.search(line_lower):
                    # Look for capitalization patterns typical of headers
                    if (line.istitle() or  # Title Case
                        line.isupper() or  # ALL CAPS
                        re.match(r'^[A-Z][^a-z]*[A-Z]', line)):  # Mixed case headers

                        sections.append({
                            'type': section_type,
                            'header': line.strip(),
                            'line_number': i,
                            'start_pos': sum(len(lines[j]) + 1 for j in range(i))
                        })
                        break

        return sections

    def _chunk_by_sections(self, text: str, sections: List[Dict[str, Any]]) -> List[TextChunk]:
        """Chunk document by preserving section boundaries."""
        chunks = []
        char_pos = 0

        # Sort sections by position
        sections.sort(key=lambda x: x['line_number'])

        for i, section in enumerate(sections):
            start_pos = section['start_pos']
            end_pos = sections[i + 1]['start_pos'] if i < len(sections) - 1 else len(text)

            section_text = text[start_pos:end_pos]

            # Further chunk large sections if needed
            if len(section_text.split()) > self.chunk_size * 1.5:
                sub_chunks = self._chunk_section_content(section_text, start_pos, section['header'])
                chunks.extend(sub_chunks)
            else:
                chunk = TextChunk(
                    text=section_text.strip(),
                    chunk_id=len(chunks),
                    start_char=start_pos,
                    end_char=end_pos,
                    token_count=len(section_text.split()),
                    sentence_count=len(sent_tokenize(section_text)),
                    section_header=section['header']
                )
                chunks.append(chunk)

        return chunks

    def _chunk_section_content(self, section_text: str, base_pos: int, section_header: str) -> List[TextChunk]:
        """Chunk large section content intelligently."""
        sentences = sent_tokenize(section_text)
        chunks = []

        current_chunk = ""
        current_tokens = 0
        chunk_sentences = []
        start_pos = base_pos

        for sentence in sentences:
            sentence_tokens = len(word_tokenize(sentence))

            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(chunk_sentences)
                chunk = TextChunk(
                    text=chunk_text.strip(),
                    chunk_id=len(chunks),
                    start_char=start_pos,
                    end_char=start_pos + len(chunk_text),
                    token_count=current_tokens,
                    sentence_count=len(chunk_sentences),
                    section_header=section_header
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_sentences = chunk_sentences[-self.overlap // 10:] if len(chunk_sentences) > 2 else []
                current_chunk = ' '.join(overlap_sentences) + ' ' + sentence
                chunk_sentences = overlap_sentences + [sentence]
                current_tokens = len(word_tokenize(current_chunk))
                start_pos += len(chunk_text) - len(' '.join(overlap_sentences))
            else:
                current_chunk += ' ' + sentence
                chunk_sentences.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk.strip():
            chunk = TextChunk(
                text=current_chunk.strip(),
                chunk_id=len(chunks),
                start_char=start_pos,
                end_char=base_pos + len(section_text),
                token_count=current_tokens,
                sentence_count=len(chunk_sentences),
                section_header=section_header
            )
            chunks.append(chunk)

        return chunks

    def _chunk_by_semantic_boundaries(self, text: str) -> List[TextChunk]:
        """Chunk text using semantic boundaries when sections aren't clear."""
        sentences = sent_tokenize(text)
        chunks = []

        current_chunk = ""
        current_tokens = 0
        chunk_sentences = []
        start_pos = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = len(word_tokenize(sentence))

            # Check for semantic boundaries (technical terms, numbers, etc.)
            is_boundary = self._is_semantic_boundary(sentence, sentences[i-1] if i > 0 else None)

            # Create chunk if we'd exceed size or hit semantic boundary
            if (current_tokens + sentence_tokens > self.chunk_size and current_chunk) or \
               (is_boundary and current_chunk and len(chunk_sentences) > 3):

                chunk_text = ' '.join(chunk_sentences)
                chunk = TextChunk(
                    text=chunk_text.strip(),
                    chunk_id=len(chunks),
                    start_char=start_pos,
                    end_char=start_pos + len(chunk_text),
                    token_count=current_tokens,
                    sentence_count=len(chunk_sentences)
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_count = min(len(chunk_sentences), self.overlap // 10)
                overlap_sentences = chunk_sentences[-overlap_count:] if overlap_count > 0 else []
                current_chunk = ' '.join(overlap_sentences) + ' ' + sentence
                chunk_sentences = overlap_sentences + [sentence]
                current_tokens = len(word_tokenize(current_chunk))
                start_pos += len(chunk_text) - len(' '.join(overlap_sentences))
            else:
                current_chunk += ' ' + sentence
                chunk_sentences.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk.strip():
            chunk = TextChunk(
                text=current_chunk.strip(),
                chunk_id=len(chunks),
                start_char=start_pos,
                end_char=len(text),
                token_count=current_tokens,
                sentence_count=len(chunk_sentences)
            )
            chunks.append(chunk)

        return chunks

    def _is_semantic_boundary(self, current_sentence: str, prev_sentence: str) -> bool:
        """Check if sentence represents a semantic boundary."""
        # Technical term transitions
        current_tech = any(pattern.search(current_sentence) for pattern in self.technical_patterns.values())
        prev_tech = prev_sentence and any(pattern.search(prev_sentence) for pattern in self.technical_patterns.values())

        if current_tech != prev_tech:
            return True

        # Number/measurement transitions
        current_has_numbers = bool(re.search(r'\d', current_sentence))
        prev_has_numbers = prev_sentence and bool(re.search(r'\d', prev_sentence))

        if current_has_numbers != prev_has_numbers:
            return True

        # Length-based boundaries (very long sentences might be boundaries)
        if len(word_tokenize(current_sentence)) > 50:
            return True

        return False

    def _post_process_chunks(self, chunks: List[TextChunk], original_text: str) -> List[TextChunk]:
        """Post-process chunks for quality and coherence."""
        processed_chunks = []

        for chunk in chunks:
            # Skip very small chunks (likely noise)
            if chunk.token_count < 20:
                continue

            # Ensure chunk doesn't end mid-sentence
            if not chunk.text.endswith(('.', '!', '?', ':')):
                # Try to find sentence boundary
                sentences = sent_tokenize(chunk.text)
                if len(sentences) > 1:
                    # Keep only complete sentences
                    chunk.text = ' '.join(sentences[:-1])
                    chunk.token_count = len(word_tokenize(chunk.text))
                    chunk.sentence_count = len(sentences) - 1

            # Calculate confidence score
            chunk.confidence_score = self._calculate_chunk_confidence(chunk, original_text)

            processed_chunks.append(chunk)

        return processed_chunks

    def _calculate_chunk_confidence(self, chunk: TextChunk, original_text: str) -> float:
        """Calculate confidence score for chunk quality."""
        score = 1.0

        # Penalize very small chunks
        if chunk.token_count < 50:
            score -= 0.2

        # Penalize chunks that don't end with punctuation
        if not chunk.text.endswith(('.', '!', '?', ':')):
            score -= 0.1

        # Reward chunks with technical content
        tech_matches = sum(1 for pattern in self.technical_patterns.values()
                          if pattern.search(chunk.text))
        if tech_matches > 0:
            score += 0.1

        # Check for good sentence structure
        sentences = sent_tokenize(chunk.text)
        avg_sentence_length = np.mean([len(word_tokenize(s)) for s in sentences])
        if 10 <= avg_sentence_length <= 30:
            score += 0.1

        return max(0.1, min(1.0, score))

#!/usr/bin/env python3
import os
import sys
import glob
import re
import json
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from chemical_formula_processor import ChemicalFormulaProcessor

import logging
import sys


logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    handlers=[
        logging.FileHandler("logs/step1.out", mode="w", encoding="utf-8"), 
        logging.StreamHandler(sys.stdout)  
    ]
)

err_handler = logging.FileHandler("logs/step1.err", mode="w", encoding="utf-8")
err_handler.setLevel(logging.ERROR)

logger = logging.getLogger()
logger.addHandler(err_handler)

# Hardcoded configuration
MARKDOWN_DIR = "./files_mmd"
FAISS_DIR = "./embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('document_indexing.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_mmd_metadata(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse MultiMarkdown-style metadata header and return (metadata, body).
    Stops at the first empty line or first non key:value line.
    """
    lines = content.split('\n')
    header: Dict[str, Any] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            break
        if ':' in line:
            key, value = line.split(':', 1)
            header[key.strip()] = value.strip()
            i += 1
        else:
            break
    body = '\n'.join(lines[i:])
    # Normalize some common fields
    norm = {k.lower(): v for k, v in header.items()}
    parsed = {
        'title': norm.get('title', ''),
        'authors': [a.strip() for a in norm.get('author', '').split(',')] if 'author' in norm else [],
        'year': norm.get('date', ''),
        'language': norm.get('language', ''),
        'format': norm.get('format', ''),
        'math': norm.get('math', ''),
        'raw': header,
    }
    return parsed, body

def extract_metadata(content: str, file_name: str) -> Dict[str, Any]:
    """Extract metadata such as title, authors, year, and abstract from markdown content."""
    metadata = {
        'title': os.path.basename(file_name).replace('.mmd', '').replace('.pdf', ''),
        'authors': [],
        'year': '',
        'abstract': '',
        'id': file_name
    }
    
    # Try to extract title from content (usually at the beginning)
    title_match = re.search(r'^#\s+(.+?)(?=\n\n|\n#)', content, re.MULTILINE)
    if title_match:
        metadata['title'] = title_match.group(1).strip()
    
    # Try to extract authors (look for patterns like "Author1, Author2, and Author3")
    author_patterns = [
        r'(?:^|\n)by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:,\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)?)',
        r'(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:,\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)?)\n(?:University|Institute|Department|School)',
        r'^(?:Authors?|AUTHOR[S]?)[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:,\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)?)'
    ]
    
    for pattern in author_patterns:
        author_match = re.search(pattern, content, re.MULTILINE)
        if author_match:
            author_text = author_match.group(1)
            # Split by commas and 'and'
            authors = re.split(r',\s+|\s+and\s+', author_text)
            metadata['authors'] = [author.strip() for author in authors if author.strip()]
            break
    
    # Try to extract year (look for 4-digit numbers that could be years)
    year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', content[:2000])
    if year_matches:
        metadata['year'] = year_matches[0]
    
    # Try to extract abstract
    abstract_patterns = [
        r'(?:^|\n)Abstract[:\s]+(.*?)(?=\n\n|\n#)',
        r'(?:^|\n)ABSTRACT[:\s]+(.*?)(?=\n\n|\n#)',
        r'(?:^|\n)Summary[:\s]+(.*?)(?=\n\n|\n#)',
    ]
    
    for pattern in abstract_patterns:
        abstract_match = re.search(pattern, content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
            # Clean up the abstract
            abstract_text = re.sub(r'\s+', ' ', abstract_text)
            metadata['abstract'] = abstract_text
            break
    
    return metadata

def chunk_document(content: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split document content into smaller overlapping chunks with formula protection."""
    # Treat page separators (---) as hard boundaries by inserting blank lines
    preprocessed = re.sub(r'^\s*---\s*$', '\n\n', content, flags=re.MULTILINE)
    # Split by paragraphs
    paragraphs = re.split(r'\n\s*\n', preprocessed)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        # Check if adding this paragraph would exceed chunk size
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            # Before chunking, check if we're about to split a formula
            if _contains_incomplete_formula(current_chunk + "\n\n" + paragraph, len(current_chunk)):
                # If the current chunk would split a formula, try to include the complete formula
                extended_chunk = current_chunk + "\n\n" + paragraph
                if len(extended_chunk) <= chunk_size * 1.5:  # Allow 50% overflow for formula integrity
                    current_chunk = extended_chunk
                    continue
            
            chunks.append(current_chunk.strip())
            # Keep some overlap between chunks, but ensure we don't split formulas in overlap
            overlap_text = _safe_overlap(current_chunk, overlap)
            current_chunk = overlap_text
            
        current_chunk += "\n\n" + paragraph
    
    # Add the last chunk if not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def _contains_incomplete_formula(text: str, split_point: int) -> bool:
    """Check if splitting at split_point would break a LaTeX formula."""
    # Check for incomplete $$ blocks
    before_split = text[:split_point]
    after_split = text[split_point:]
    
    # Count $$ pairs before split point
    double_dollar_count = before_split.count('$$')
    if double_dollar_count % 2 == 1:  # Odd count means incomplete formula
        return True
    
    # Check for incomplete $ pairs (inline math)
    # Remove $$ blocks first to avoid false positives
    temp_text = re.sub(r'\$\$.*?\$\$', '', before_split, flags=re.DOTALL)
    single_dollar_count = temp_text.count('$')
    if single_dollar_count % 2 == 1:  # Odd count means incomplete inline formula
        return True
    
    return False

def _safe_overlap(text: str, overlap_size: int) -> str:
    """Create overlap text that doesn't split formulas."""
    if overlap_size <= 0:
        return ""
    
    # Get the desired overlap region
    overlap_text = text[-overlap_size:] if len(text) > overlap_size else text
    
    # Check if this overlap starts in the middle of a formula
    # If so, extend backwards to include the complete formula
    double_dollar_count = overlap_text.count('$$')
    if double_dollar_count % 2 == 1:  # Incomplete $$ formula
        # Find the start of the incomplete formula
        remaining_text = text[:-overlap_size] if len(text) > overlap_size else ""
        last_double_dollar = remaining_text.rfind('$$')
        if last_double_dollar != -1:
            # Include from the start of the formula
            return text[last_double_dollar:]
    
    # Check for incomplete single $ formulas
    temp_overlap = re.sub(r'\$\$.*?\$\$', '', overlap_text, flags=re.DOTALL)
    single_dollar_count = temp_overlap.count('$')
    if single_dollar_count % 2 == 1:  # Incomplete $ formula
        remaining_text = text[:-overlap_size] if len(text) > overlap_size else ""
        temp_remaining = re.sub(r'\$\$.*?\$\$', '', remaining_text, flags=re.DOTALL)
        # Find the last single $ in remaining text
        last_single_dollar = temp_remaining.rfind('$')
        if last_single_dollar != -1:
            # Calculate position in original text
            actual_pos = last_single_dollar
            # Account for removed $$ blocks
            for match in re.finditer(r'\$\$.*?\$\$', remaining_text, flags=re.DOTALL):
                if match.start() < last_single_dollar:
                    actual_pos += len(match.group()) - 0  # We removed the $$...$$ content
            return text[actual_pos:]
    
    return overlap_text

def read_markdown_files(markdown_dir: str) -> List[Dict[str, Any]]:
    """Read all markdown files, extract metadata, and chunk content for indexing."""
    markdown_files = glob.glob(os.path.join(markdown_dir, "*.mmd"))
    
    chunked_documents = []
    processor = ChemicalFormulaProcessor()
    
    for file_path in markdown_files:
        file_name = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                original_content = file.read()
            # Parse MMD header (Title/Author/Date/Language/Format/Math) and body
            mmd_meta, content = parse_mmd_metadata(original_content)
            # Preserve chemical formulas/symbols before chunking to avoid splitting inside formulas
            protected_content = processor.preserve_chemical_content(content)

            # Extract metadata
            metadata = extract_metadata(content, file_name)
            # Merge MMD header fields (header overrides heuristics when present)
            if mmd_meta.get('title'):
                metadata['title'] = mmd_meta['title']
            if mmd_meta.get('authors'):
                metadata['authors'] = mmd_meta['authors']
            if mmd_meta.get('year'):
                metadata['year'] = mmd_meta['year']
            # Add protection flags to metadata-like fields we will store alongside chunks
            latex_formulas = processor.extract_latex_formulas(content)
            metadata_protection = {
                'protected': True,
                'latex_formula_count': len(latex_formulas),
            }
            
            # Chunk the document content
            chunks = chunk_document(protected_content)
            
            # Create a document entry for each chunk
            for i, chunk in enumerate(chunks):
                chunked_documents.append({
                    'file_name': file_name,
                    'content': chunk,
                    'path': file_path,
                    'title': metadata['title'],
                    'authors': metadata['authors'],
                    'year': metadata['year'],
                    'abstract': metadata['abstract'],
                    'language': mmd_meta.get('language', ''),
                    'format': mmd_meta.get('format', ''),
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    # Protection-related hints for downstream consumers
                    **metadata_protection,
                })
                
            logger.info(f"Processed {file_name}: {len(chunks)} chunks, authors: {metadata['authors']}")
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            
    return chunked_documents

def create_faiss_database(documents: List[Dict[str, Any]], output_dir: str):
    """Create a FAISS database from the chunked markdown documents."""
    if not documents:
        logger.error("No documents to index.")
        return False
        
    # Create embeddings directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the sentence transformer model
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    # Create embeddings
    logger.info("Creating embeddings for document chunks...")
    embeddings = []
    metadata = []
    
    for doc in tqdm(documents):
        # Create embedding for the document chunk
        embedding = model.encode(doc['content'])
        embeddings.append(embedding)
        
        # Store rich metadata
        metadata.append({
            'file_name': doc['file_name'],
            'path': doc['path'],
            'title': doc['title'],
            'authors': doc['authors'],
            'year': doc['year'],
            'abstract': doc['abstract'],
            'chunk_id': doc['chunk_id'],
            'total_chunks': doc['total_chunks'],
            'excerpt': doc['content'][:500],  # Store first 500 chars as excerpt
            'id': f"{doc['file_name']}_{doc['chunk_id']}",
            # Propagate protection flags for downstream awareness
            'protected': bool(doc.get('protected', False)),
            'latex_formula_count': int(doc.get('latex_formula_count', 0)),
        })
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Add debug logging
    logger.info(f"Number of embeddings created: {len(embeddings)}")
    logger.info(f"Number of metadata entries: {len(metadata)}")
    logger.info(f"Embeddings array shape: {embeddings_array.shape}")
    
    # Create FAISS index
    logger.info("Creating FAISS index...")
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    # Add more debug logging
    logger.info(f"FAISS index total entries: {index.ntotal}")
    
    # Save the index
    faiss_path = os.path.join(output_dir, "faiss.index")
    faiss.write_index(index, faiss_path)
    logger.info(f"FAISS index saved to {faiss_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.npy")
    np.save(metadata_path, metadata)
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Also save metadata as JSON for easier inspection
    json_path = os.path.join(output_dir, "metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata also saved as JSON to {json_path}")
    
    # Save statistics
    stats = {
        "total_documents": len(set(doc['file_name'] for doc in documents)),
        "total_chunks": len(documents),
        "average_chunks_per_document": len(documents) / len(set(doc['file_name'] for doc in documents)) if documents else 0,
        "model_name": MODEL_NAME,
        "embedding_dimension": dimension,
        "index_size": index.ntotal,
        "metadata_size": len(metadata)
    }
    
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Index statistics saved to {stats_path}")
    
    return True

def main():
    # Step 1: Read markdown files, extract metadata, and chunk content
    logger.info(f"Reading Markdown files from {MARKDOWN_DIR}")
    chunked_documents = read_markdown_files(MARKDOWN_DIR)
    logger.info(f"Created {len(chunked_documents)} document chunks from {len(set(doc['file_name'] for doc in chunked_documents))} documents.")
    
    # Step 2: Create FAISS database with rich metadata
    logger.info(f"Creating FAISS database in {FAISS_DIR}")
    success = create_faiss_database(chunked_documents, FAISS_DIR)
    if not success:
        logger.error("FAISS database creation failed.")
        sys.exit(1)
    
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()



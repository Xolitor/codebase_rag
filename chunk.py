from config import (
    COLLECTION_NAME,
    VECTOR_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_PRICE_PER_1M_TOKENS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MODE
)

#CHUNK
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])
    return chunks

#SYNTAX AWARE CHUNKING
def chunk_code_syntax_aware():
    pass

#LANGUAGE SPECIFIC CHUNKING
import re
import ast
import os

def chunk_code_by_language(code, filename):
    ext = os.path.splitext(filename)[1]

    if ext == ".py":
        print(f"Chunking Python code in {filename} using syntax-aware chunking.")
        return chunk_python(code)
    elif ext in [".js", ".ts"]:
        print(f"Chunking JavaScript/TypeScript code in {filename} using syntax-aware chunking.")
        return chunk_js_ts(code)
    else:
        print(f"Chunking generic code in {filename} using text-based chunking.")
        return chunk_text(code)
    
def chunk_python(code):
    chunks = []

    try:
        tree = ast.parse(code)

        for node in tree.body:
            chunk = ast.get_source_segment(code, node)
            if chunk:
                if len(chunk) > CHUNK_SIZE:
                    chunks.extend(chunk_text(chunk))
                else:
                    chunks.append(chunk)

    except Exception:
        return chunk_text(code)

    return chunks

def chunk_js_ts(code):
    pattern = r"(function\s+\w+.*?\{.*?\})|(class\s+\w+.*?\{.*?\})"
    matches = re.finditer(pattern, code, re.DOTALL)

    chunks = []
    for match in matches:
        chunk = match.group()

        if len(chunk) > CHUNK_SIZE:
            chunks.extend(chunk_text(chunk))
        else:
            chunks.append(chunk)

    if not chunks:
        return chunk_text(code)

    return chunks
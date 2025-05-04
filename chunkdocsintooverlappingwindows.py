def chunk_text(tokenizer, text, chunk_size=500, overlap=50):
    tokens = tokenizer.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# Example usage
# tokenizer = SomeTokenizer()
# chunks = chunk_text(tokenizer, large_document_text)

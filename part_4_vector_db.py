import json
import hashlib
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

#config
CHUNKS_FILE = "chunks.json"
VECTOR_DB_PATH = "./vector_store"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
COLLECTION_NAME = "math_chunks"

# in case of the same id, make unique one
def ensure_unique_id(chunk, existing_ids):
    original_id = chunk.get("id", "")
    base_content = chunk["content"][:200]
    
    if original_id and original_id not in existing_ids:
        return original_id
    
    counter = 0
    while True:
        new_id = hashlib.md5(f"{base_content}_{counter}".encode()).hexdigest()[:12]
        if new_id not in existing_ids:
            return new_id
        counter += 1

# loading all data
def load_chunks():
    print(f"loading chunks from {CHUNKS_FILE}")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    existing_ids = set()
    unique_chunks = []
    
    for chunk in chunks:
        unique_id = ensure_unique_id(chunk, existing_ids)
        chunk["id"] = unique_id
        existing_ids.add(unique_id)
        unique_chunks.append(chunk)
    
    print(f"loaded {len(unique_chunks)} unique chunks")
    return unique_chunks

#creation of vector DB
def create_vector_store(chunks):
    print(f"loading embedding more: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL) # embedding model
    
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH) # chroma client 
    
    try:
        client.delete_collection(name=COLLECTION_NAME) # remove existing
    except:
        pass
    
    collection = client.create_collection( # creation of collection
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # uses HNSW index + cosine similarity
    )
    
    print("generating embeddings")
    batch_size = 100
    
    for i in tqdm(range(0, len(chunks), batch_size)): # for each batch put into DB with metadata
        batch = chunks[i:i + batch_size]
        
        ids = [c["id"] for c in batch]
        texts = [c["content"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        
        embeddings = model.encode(texts, show_progress_bar=False)
        
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
    
    print(f"db created: {collection.count()} chunks stored")
    return collection, model

def test_retrieval(collection, model): # for testing purposes
    print("testing")
    
    test_queries = [
        "інтеграли",
        "похідна функції",
        "теорема Піфагора",
        "квадратні рівняння"
    ]
    
    for query in test_queries:
        print(f"query: '{query}'")
        query_embedding = model.encode([query])[0].tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"similarity: {1-distance:.3f}")
            print(f"source: {metadata.get('source', 'unknown')}")
            print(f"strategy: {metadata.get('strategy', 'unknown')}")
            if 'section' in metadata:
                print(f"section: {metadata['section']}")
            print(f"preview: {doc[:150]}")

def main():
    chunks = load_chunks()
    collection, model = create_vector_store(chunks)
    test_retrieval(collection, model)
    
    print("db ready")
    print(f"location: {VECTOR_DB_PATH}")
    print(f"collection: {COLLECTION_NAME}")
    print(f"total amount of chunks: {collection.count()}")

if __name__ == "__main__":
    main()
# imports
import json
import hashlib
from FlagEmbedding import BGEM3FlagModel
import chromadb
from tqdm import tqdm

#basic config
CHUNKS_FILE = "chunks_v3.json"
VECTOR_DB_PATH = "./vector_store_v3"
COLLECTION_NAME = "math_chunks_v2"
EMBEDDING_MODEL = "BAAI/bge-m3"

#ensuring unique id in case needed
def ensure_unique_id(chunk, existing_ids):
    original_id = chunk.get("id", "")
    if original_id and original_id not in existing_ids:
        return original_id
    
    base_content = chunk["content"][:200]
    counter = 0
    while True:
        new_id = hashlib.md5(f"{base_content}_{counter}".encode()).hexdigest()[:12]
        if new_id not in existing_ids:
            return new_id
        counter += 1

#chroma crashes in case metadata is None or complex
def sanitize_metadata(metadata):
    sanitized = {}
    for k, v in metadata.items():
        if v is None:
            sanitized[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        else:
            sanitized[k] = str(v)
    return sanitized

#loading data
def load_chunks(chunks_file = CHUNKS_FILE):
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    existing_ids = set()
    unique_chunks = []
    
    for chunk in chunks:
        unique_id = ensure_unique_id(chunk, existing_ids)
        chunk["id"] = unique_id
        chunk["metadata"] = sanitize_metadata(chunk.get("metadata", {}))
        existing_ids.add(unique_id)
        unique_chunks.append(chunk)
    
    return unique_chunks

#initializing model and DB
def create_vector_store(chunks):
    model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)
    
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    
    for name in [COLLECTION_NAME, f"{COLLECTION_NAME}_sparse"]: # removing old one
        try:
            client.delete_collection(name=name)
        except:
            pass
    
    collection_dense = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    batch_size = 32
    
    sparse_index = {}
    
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        ids = [c["id"] for c in batch]
        texts = [c["content"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        
        output = model.encode(texts, return_dense=True, return_sparse=True)
        dense_embeddings = output['dense_vecs']
        sparse_weights = output['lexical_weights']
        #storing dense vectors in chroma
        collection_dense.add(
            ids=ids,
            embeddings=dense_embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        # sparse in memory, since chroma does not support BGE format
        for idx, chunk_id in enumerate(ids):
            sparse_index[chunk_id] = sparse_weights[idx]
    
    sparse_path = f"{VECTOR_DB_PATH}/sparse_index.json" # saving as json
    serializable_sparse = {k: {str(tok): float(w) for tok, w in v.items()} 
                          for k, v in sparse_index.items()}
    with open(sparse_path, 'w') as f:
        json.dump(serializable_sparse, f)
    
    print(f"db created: {collection_dense.count()} chunks stored")
    print(f"index saved to {sparse_path}")
    return collection_dense, model, client

#initializes search engine
#weights to control balance. math requires precision (sparse), but we still need semantic (dense)
class MathRetriever:
    
    def __init__(self, db_path = VECTOR_DB_PATH, 
                 collection_name = COLLECTION_NAME,
                 dense_weight = 0.6,
                 sparse_weight = 0.4):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)
        self.model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        sparse_path = f"{db_path}/sparse_index.json"
        try:
            with open(sparse_path, 'r') as f:
                self.sparse_index = json.load(f)
            print(f"loaded index: {len(self.sparse_index)} chunks")
        except FileNotFoundError:
            print("index not found, using dense only")
            self.sparse_index = {}
    
    #manual calculation how well keywords match, since it is json
    def compute_sparse_scores(self, query_sparse, candidate_ids):
        scores = {}
        for chunk_id in candidate_ids:
            if chunk_id not in self.sparse_index:
                scores[chunk_id] = 0.0
                continue
            
            chunk_sparse = self.sparse_index[chunk_id]
            score = sum(query_sparse.get(tok, 0) * chunk_sparse.get(tok, 0) 
                       for tok in set(query_sparse) | set(chunk_sparse))
            scores[chunk_id] = score
        return scores
    
    # retrieval
    def retrieve(self, query, n_results = 5,
                 chunk_type = None,
                 source = None,
                 use_hybrid = True):
        # step 1:
        # dense vector to find relevant context. fetch more to ensure good pool to rerank
        output = self.model.encode([query], return_dense=True, return_sparse=True)
        query_dense = output['dense_vecs'][0].tolist()
        query_sparse = {str(k): float(v) for k, v in output['lexical_weights'][0].items()}
        
        where_conditions = []
        if chunk_type:
            where_conditions.append({"type": chunk_type})
        if source:
            where_conditions.append({"source": source})
        
        where = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}
        
        n_candidates = n_results * 3 if use_hybrid else n_results
        
        results = self.collection.query(
            query_embeddings=[query_dense],
            n_results=n_candidates,
            where=where
        )
        
        if not results['ids'][0]:
            return []
        
        candidates = []
        for i in range(len(results['ids'][0])):
            chunk_id = results['ids'][0][i]
            dense_score = 1 - results['distances'][0][i]
            candidates.append({
                'id': chunk_id,
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'dense_score': dense_score
            })
        # step 2: reranking
        # rescoring candidates using sparse overlap. since sparse are not normalized, we do it to make [0, 1]
        if use_hybrid and self.sparse_index:
            sparse_scores = self.compute_sparse_scores(
                query_sparse, 
                [c['id'] for c in candidates]
            )
            
            max_sparse = max(sparse_scores.values()) if sparse_scores.values() else 1
            if max_sparse > 0:
                sparse_scores = {k: v / max_sparse for k, v in sparse_scores.items()}
            
            for c in candidates:
                c['sparse_score'] = sparse_scores.get(c['id'], 0)
                c['similarity'] = (self.dense_weight * c['dense_score'] + 
                                  self.sparse_weight * c['sparse_score'])
        else:
            for c in candidates:
                c['similarity'] = c['dense_score']
        
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return candidates[:n_results]
    
    

#1. read json, clean it
#2. generate dense vectors (context) + sparse (keywords)
#3. dense -> chroma, sparse -> json
#4. retrieve:
#  search chroma
#  check json
#  combine and return top scores

def main():
    chunks = load_chunks()
    
    collection, model, client = create_vector_store(chunks)
    
    retriever = MathRetriever()
    
    print('db ready')
    print(f"location: {VECTOR_DB_PATH}")
    print(f"oollection: {COLLECTION_NAME}")
    print(f"total amount of chunks: {collection.count()}")


if __name__ == "__main__":
    main()
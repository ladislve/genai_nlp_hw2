#imports and config
import json
import re
import time
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
import chromadb
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

VECTOR_DB_PATH = "./vector_store_v3"
COLLECTION_NAME = "math_chunks_v2"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DATASET_PATH = "math_dataset_ukr.jsonl"
MAX_WORKERS = 5 # for speed up
TIMEOUT_PER_TASK = 120
CODE_TIMEOUT = 10

# additional prompts and parsting scripts
from additional_scripts.part_6_prompts import *
from additional_scripts.part_6_parsing import *

#data class to store the outcome of a single problem
class SolverResult:
    def __init__(self, task_id, question, expected_answer, predicted_answer, solution, method, time_seconds, retrieved_chunks=None, search_queries=None, error=None, verification=None, code_output=None):
        self.task_id = task_id
        self.question = question
        self.expected_answer = expected_answer
        self.predicted_answer = predicted_answer
        self.solution = solution
        self.method = method
        self.time_seconds = time_seconds
        self.retrieved_chunks = retrieved_chunks if retrieved_chunks is not None else []
        self.search_queries = search_queries if search_queries is not None else []
        self.error = error
        self.verification = verification
        self.code_output = code_output

#cleans answer strings
def normalize_answer(answer):
    if not answer: return ""
    digits = re.sub(r'[^\d]', '', str(answer).strip())
    return str(int(digits)) if digits else ""

# writes ai-generated code and runs in separare process, since if in current main crushes 
def run_python_code(code, timeout=CODE_TIMEOUT):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ['python', f.name],
                capture_output=True, text=True, timeout=timeout
            )
            output = result.stdout.strip() or result.stderr.strip()
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, str(e)

# enchanced retrieval from part 4
class HybridRetriever:
    def __init__(self, db_path=VECTOR_DB_PATH, dense_weight=0.6, sparse_weight=0.4):
        self.embedding_model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        
        with open(f"{db_path}/sparse_index.json", 'r') as f:
            self.sparse_index = json.load(f)
        print(f"loaded index: {len(self.sparse_index)} chunks")
    
    def compute_sparse_scores(self, query_sparse, ids):
        scores = {}
        for chunk_id in ids:
            if chunk_id not in self.sparse_index:
                scores[chunk_id] = 0.0
                continue
            chunk_sparse = self.sparse_index[chunk_id]
            score = sum(query_sparse.get(t, 0) * chunk_sparse.get(t, 0) 
                       for t in set(query_sparse) | set(chunk_sparse))
            scores[chunk_id] = score
        return scores
    
    # retrieval
    def retrieve(self, queries, top_k=10, final_k=5):
        all_chunks = {}
        
        # dense and sparse search, grabbing top candidates
        for query in queries:
            output = self.embedding_model.encode([query], return_dense=True, return_sparse=True)
            query_dense = output['dense_vecs'][0].tolist()
            query_sparse = {str(k): float(v) for k, v in output['lexical_weights'][0].items()}
            
            for chunk_type in ['blueprint', 'theory']:
                try:
                    results = self.collection.query(
                        query_embeddings=[query_dense],
                        n_results=top_k,
                        where={"type": chunk_type}
                    )
                    
                    ids = results['ids'][0]
                    sparse_scores = self.compute_sparse_scores(query_sparse, ids)
                    max_sparse = max(sparse_scores.values()) if sparse_scores.values() else 1
                    if max_sparse > 0:
                        sparse_scores = {k: v/max_sparse for k, v in sparse_scores.items()}
                    
                    for i, chunk_id in enumerate(ids):
                        if chunk_id in all_chunks:
                            continue
                        dense_score = 1 - results['distances'][0][i]
                        sparse_score = sparse_scores.get(chunk_id, 0)
                        hybrid = self.dense_weight * dense_score + self.sparse_weight * sparse_score
                        
                        all_chunks[chunk_id] = {
                            'id': chunk_id,
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'hybrid_score': hybrid,
                            'query': query
                        }
                except: continue
        
        if not all_chunks:
            return []
        
        chunks = list(all_chunks.values())
        chunks.sort(key=lambda x: x['hybrid_score'], reverse=True)
        candidates = chunks[:top_k * 2]
        # reranking
        #cross encoder reads query and docs, rearranges them for better retrieval
        if candidates:
            main_query = queries[0] if queries else ""
            pairs = [(main_query, c['content'][:500]) for c in candidates]
            scores = self.reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c['rerank_score'] = float(s)
            candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return candidates[:final_k]

# simple agent. nothing special
class BaselineAgent:
    def __init__(self, model="gemini-2.0-flash", temperature=0.1):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", MATH_SOLVER_SYSTEM),
            ("human", """Розв'яжи олімпіадну задачу:

            {question}

            Міркуй покроково. Наприкінці ОБОВ'ЯЗКОВО надай відповідь:
            **Відповідь:** [ціле число]""")
        ])
    
    def solve(self, task_id, question, expected):
        start = time.time()
        raw = ""
        try:
            response = self.llm.invoke(self.prompt.format_messages(question=question))
            raw = response.content
            result = parse_json_response(raw, raw)
            return SolverResult(
                task_id=task_id, question=question, expected_answer=expected,
                predicted_answer=normalize_answer(result.get("answer", "")),
                solution=raw, method="baseline", time_seconds=time.time() - start
            )
        except Exception as e:
            return self.error(task_id, question, expected, e, start, raw)
    
    def error(self, task_id, question, expected, e, start, raw=""):
        msg = f"{type(e).__name__}: {str(e)[:80]}"
        if isinstance(e, ResourceExhausted): msg = "RATE_LIMIT_429"
        print(f"   {task_id}: {msg}")
        return SolverResult(
            task_id=task_id, question=question, expected_answer=expected,
            predicted_answer="", solution=raw, method="baseline",
            time_seconds=time.time() - start, error=msg
        )

# rag
class RAGAgent:
    
    def __init__(self, model="gemini-2.0-flash", temperature=0.1, top_k=12, final_k=4, verify=True):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.retriever = HybridRetriever()
        self.top_k = top_k
        self.final_k = final_k
        self.verify = verify
        
        self.analysis_prompt = ChatPromptTemplate.from_messages(ANALYSIS_PROMPT)
        
        self.solve_prompt = ChatPromptTemplate.from_messages(SOLVE_PROMPT)
        
        self.verify_prompt = ChatPromptTemplate.from_messages(VERIFY_PROMPT)
    
    # transform question into search query (ask llm to identify)
    def analyze(self, question):
        try:
            resp = self.llm.invoke(self.analysis_prompt.format_messages(question=question[:800]))
            content = resp.content.lower()
            
            prob_type = "unknown"
            for t in ['алгебра', 'геометрія', 'комбінаторика', 'теорія_чисел']:
                if t in content:
                    prob_type = t
                    break
            
            methods = []
            m = re.search(r'метод[иів]?[:\s]+([^\n]+)', content)
            if m:
                methods = [w.strip() for w in re.split(r'[,;]', m.group(1)) if w.strip()][:3]
            
            return {"type": prob_type, "methods": methods}
        except:
            return {"type": "unknown", "methods": []}
    # creates search queries based on analysis
    def generate_queries(self, question, analysis):
        queries = []
        
        if analysis.get('methods'):
            queries.append(' '.join(analysis['methods']))
        
        math_terms = re.findall(
            r'(рівняння|нерівність|многочлен|функція|послідовність|'
            r'трикутник|коло|площа|ймовірність|комбінаці|перестановк|'
            r'подільність|остача|степін|логарифм|границ|інтеграл)',
            question.lower()
        )
        if math_terms:
            queries.append(f"{analysis.get('type', '')} {' '.join(set(math_terms[:3]))}")
        
        queries.append(question[:200])
        return [q.strip() for q in queries if q.strip()]
    
    def format_context(self, chunks):
        if not chunks:
            return "[relevant context not found]"
        
        parts = []
        for i, c in enumerate(chunks, 1):
            meta = c['metadata']
            header = f"[{i}] {meta.get('source', '?')} | {meta.get('type', '?')}"
            content = c['content'][:1500] + "..." if len(c['content']) > 1500 else c['content']
            parts.append(f"{header}\n{content}")
        return "\n\n---\n\n".join(parts)
    # asks llm to write script to verify its own answer
    def verify_with_code(self, question, solution, answer, context):
        try:
            resp = self.llm.invoke(self.verify_prompt.format_messages(
                question=question[:600],
                context=context[:1500],
                solution=solution[:2000],
                answer=answer
            ))
            
            code_match = re.search(r'```python\s*([\s\S]*?)```', resp.content)
            if not code_match:
                return answer, None, "no code generated"
            
            code = code_match.group(1).strip()
            
            success, output = run_python_code(code)
            
            if not success:
                return answer, None, f"Code error: {output[:100]}"
            
            if "VERIFIED" in output:
                match = re.search(r'VERIFIED[:\s]*(\d+)', output)
                verified_ans = match.group(1) if match else answer
                return verified_ans, "verified", output
            elif "CORRECTED" in output:
                match = re.search(r'CORRECTED[:\s]*(\d+)', output)
                if match:
                    return match.group(1), "corrected", output
            
            nums = re.findall(r'\b(\d{1,4})\b', output)
            if nums:
                return nums[-1], "extracted", output
            
            return answer, None, output
            
        except Exception as e:
            return answer, None, f"verify error: {str(e)[:80]}"
    
    #workflow for single problem
    #1. analyze question
    #2. retrieve context
    #3. generate solution
    #4. run python verification 
    #5. return results
    def solve(self, task_id, question, expected):
        start = time.time()
        queries, chunks, raw, context = [], [], "", ""
        
        try:
            analysis = self.analyze(question)
            
            queries = self.generate_queries(question, analysis)
            
            chunks = self.retriever.retrieve(queries, self.top_k, self.final_k)
            context = self.format_context(chunks)
            
            resp = self.llm.invoke(self.solve_prompt.format_messages(
                context=context, question=question
            ))
            raw = resp.content
            result = parse_json_response(raw, raw)
            predicted = normalize_answer(result.get("answer", ""))
            
            verification = None
            code_output = None
            if self.verify and predicted:
                verified_ans, status, output = self.verify_with_code(
                    question, raw, predicted, context
                )
                code_output = output
                if status == "corrected":
                    verification = f"{predicted} -> {verified_ans}"
                    predicted = normalize_answer(verified_ans)
                elif status == "verified":
                    verification = "verified"
            
            return SolverResult(
                task_id=task_id, question=question, expected_answer=expected,
                predicted_answer=predicted, solution=raw,
                method="rag_v3" + ("_verify" if self.verify else ""),
                time_seconds=time.time() - start,
                retrieved_chunks=[{
                    'source': c['metadata'].get('source'),
                    'type': c['metadata'].get('type'),
                    'score': round(c.get('rerank_score', 0), 3)
                } for c in chunks],
                search_queries=queries,
                verification=verification,
                code_output=code_output
            )
        except Exception as e:
            return self.error(task_id, question, expected, e, start, queries, chunks, raw)
    
    def error(self, task_id, question, expected, e, start, queries=None, chunks=None, raw=""):
        msg = f"{type(e).__name__}: {str(e)[:80]}"
        if isinstance(e, ResourceExhausted): msg = "RATE_LIMIT_429"
        print(f"   {task_id}: {msg}")
        return SolverResult(
            task_id=task_id, question=question, expected_answer=expected,
            predicted_answer="", solution=raw, method="rag_v3",
            time_seconds=time.time() - start, error=msg,
            search_queries=queries or [],
            retrieved_chunks=[{'source': c['metadata'].get('source')} for c in (chunks or [])]
        )

def load_dataset(path=DATASET_PATH):
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            dataset.append({
                'id': item.get('ID', item.get('id', '')),
                'question': item.get('Problem', item.get('question', '')),
                'answer': str(item.get('Answer', item.get('answer', '')))
            })
    print(f"Loaded {len(dataset)} tasks")
    return dataset

# engine loop, speeding up using parralel processing
def run_evaluation(agent, dataset, max_workers=MAX_WORKERS):
    results = []
    stats = {'success': 0, 'correct': 0, 'error': 0}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(agent.solve, t['id'], t['question'], t['answer']): t
            for t in dataset
        }
        
        for future in as_completed(futures):
            task = futures[future]
            try:
                r = future.result(timeout=TIMEOUT_PER_TASK)
                results.append(r)
                
                if r.error:
                    stats['error'] += 1
                else:
                    stats['success'] += 1
                    correct = normalize_answer(r.predicted_answer) == normalize_answer(r.expected_answer)
                    if correct:
                        stats['correct'] += 1
                        v = f" [{r.verification}]" if r.verification else ""
                        print(f"   {r.task_id}: {r.predicted_answer}{v} [{r.time_seconds:.1f}s]")
                    else:
                        print(f"   {r.task_id}: {r.predicted_answer} (exp: {r.expected_answer}) [{r.time_seconds:.1f}s]")
            except TimeoutError:
                print(f"   {task['id']}: TIMEOUT")
                stats['error'] += 1
    
    print(f"\n   Stats: {stats['correct']}/{stats['success']} correct, {stats['error']} errors")
    return sorted(results, key=lambda r: r.task_id)

def save_results(results, path):
    with open(path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps({
                'task_id': r.task_id,
                'question': r.question[:300] + '...',
                'expected': r.expected_answer,
                'predicted': r.predicted_answer,
                'correct': normalize_answer(r.predicted_answer) == normalize_answer(r.expected_answer),
                'solution': r.solution,
                'method': r.method,
                'time': round(r.time_seconds, 2),
                'queries': r.search_queries,
                'retrieved': r.retrieved_chunks,
                'verification': r.verification,
                'code_output': r.code_output,
                'error': r.error
            }, ensure_ascii=False) + '\n')
    print(f"saved to {path}")

def compare_results(results_list):
    print(f"{'method':<25} {'Done':>8} {'Correct':>8} {'Acc':>8} {'Time':>8}")
    
    for name, results in results_list:
        ok = [r for r in results if not r.error]
        correct = sum(1 for r in ok if normalize_answer(r.predicted_answer) == normalize_answer(r.expected_answer))
        acc = correct / len(ok) * 100 if ok else 0
        avg_t = sum(r.time_seconds for r in ok) / len(ok) if ok else 0
        print(f"{name:<25} {len(ok):>8} {correct:>8} {acc:>7.1f}% {avg_t:>7.1f}s")

def main():
    dataset = load_dataset()
    all_results = []
    
    print("baseline")
    baseline = BaselineAgent()
    baseline_results = run_evaluation(baseline, dataset)
    save_results(baseline_results, "results_baseline.jsonl")
    all_results.append(("baseline", baseline_results))

    print("\n")

    print("RAG (no verify)")
    rag_fast = RAGAgent(verify=False)
    rag_fast_results = run_evaluation(rag_fast, dataset)
    save_results(rag_fast_results, "results_rag.jsonl")
    all_results.append(("RAG", rag_fast_results))
    
    print("\n")

    print("RAG + CODE VERIFY")
    rag_verify = RAGAgent(verify=True)
    rag_verify_results = run_evaluation(rag_verify, dataset)
    save_results(rag_verify_results, "results_rag_verify.jsonl")
    all_results.append(("RAG-verify", rag_verify_results))
    
    compare_results(all_results)

if __name__ == "__main__":
    main()
# imports
import json
import re
import os
import subprocess
import tempfile
import requests
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
import chromadb
from langgraph.graph import StateGraph, END
import operator
from additional_scripts.part_7_prompts import *
from additional_scripts.part_7_parsing import *
load_dotenv()

#config
VECTOR_DB_PATH = "./vector_store_v3"
COLLECTION_NAME = "math_chunks_v2"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CODE_TIMEOUT = 20

#shared state for all agents
class AgentState(TypedDict):
    question: str
    problem_type: str
    key_concepts: List[str]
    search_queries: List[str]
    math_expression: str
    retrieved_context: str
    retrieved_chunks: List[Dict]
    solution_draft: str
    answer_draft: str
    wolfram_result: str
    wolfram_verified: bool
    code_result: str
    code_verified: bool
    final_answer: str
    final_solution: str
    iteration: int
    max_iterations: int
    messages: Annotated[List[str], operator.add]
    error: Optional[str]

#helper function. normalizing answer and code runner
def normalize_answer(answer):
    if not answer:
        return ""
    digits = re.sub(r'[^\d]', '', str(answer).strip())
    return str(int(digits)) if digits else ""

def run_python_code(code, timeout = CODE_TIMEOUT):
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
        finally:
            os.unlink(f.name)

# same retriever as in 6
class HybridRetriever:
    def __init__(self, db_path = VECTOR_DB_PATH, dense_weight = 0.6, sparse_weight = 0.4):
        self.embedding_model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        
        sparse_path = f"{db_path}/sparse_index.json"
        with open(sparse_path, 'r') as f:
            self.sparse_index = json.load(f)
        print(f"loaded sparse index: {len(self.sparse_index)} chunks")
    
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
    
    def retrieve(self, queries, top_k = 12, final_k = 5, chunk_types = None):
        all_chunks = {}
        chunk_types = chunk_types or ['blueprint', 'theory']
        
        for query in queries:
            output = self.embedding_model.encode([query], return_dense=True, return_sparse=True)
            query_dense = output['dense_vecs'][0].tolist()
            query_sparse = {str(k): float(v) for k, v in output['lexical_weights'][0].items()}
            
            for chunk_type in chunk_types:
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
                except Exception as e:
                    continue
        
        if not all_chunks:
            return []
        
        chunks = sorted(all_chunks.values(), key=lambda x: x['hybrid_score'], reverse=True)
        candidates = chunks[:top_k * 2]
        
        if candidates:
            main_query = queries[0] if queries else ""
            pairs = [(main_query, c['content'][:500]) for c in candidates]
            scores = self.reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c['rerank_score'] = float(s)
            candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return candidates[:final_k]

# question analyzer. plans solution. extracts math expression for wolfram
class QuestionAnalyzer:
    def __init__(self, model = "gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(Question_Analyzer_PROMPT)
    
    def analyze(self, state):
        try:
            response = self.llm.invoke(self.prompt.format_messages(
                question=state["question"][:1500]
            ))
            content = clean_json_response(response.content)
            result = json.loads(content)
            
            return {
                "problem_type": result.get("problem_type", "математика"),
                "key_concepts": result.get("key_concepts", [])[:5],
                "search_queries": result.get("search_queries", [])[:3],
                "math_expression": result.get("math_expression", ""),
                "messages": [f"[Analyzer] type={result.get('problem_type')}, expr={result.get('math_expression', '')[:40]}"]
            }
        except Exception as e:
            return {
                "problem_type": "математика",
                "key_concepts": [],
                "search_queries": [state["question"][:200]],
                "math_expression": "",
                "messages": [f"[Analyzer] error: {str(e)[:80]}"]
            }

class ContextRetriever:
    def __init__(self, top_k = 12, final_k = 5):
        self.retriever = HybridRetriever()
        self.top_k = top_k
        self.final_k = final_k
    
    def retrieve(self, state):
        try:
            queries = state.get("search_queries", [])
            if not queries:
                queries = [state["question"][:200]]
            
            if state.get("key_concepts"):
                queries.append(" ".join(state["key_concepts"]))
            
            chunks = self.retriever.retrieve(queries, self.top_k, self.final_k)
            
            context_parts = []
            for i, c in enumerate(chunks, 1):
                meta = c['metadata']
                header = f"[{i}] {meta.get('source', '?')} | {meta.get('type', '?')}"
                content = c['content'][:1200]
                context_parts.append(f"{header}\n{content}")
            
            context = "\n\n---\n\n".join(context_parts) if context_parts else "[context not found]"
            
            return {
                "retrieved_context": context,
                "retrieved_chunks": [{
                    'source': c['metadata'].get('source'),
                    'type': c['metadata'].get('type'),
                    'score': round(c.get('rerank_score', 0), 3)
                } for c in chunks],
                "messages": [f"[retriever] found {len(chunks)} chunks"]
            }
        except Exception as e:
            return {
                "retrieved_context": "[error searching]",
                "retrieved_chunks": [],
                "messages": [f"[retriever] error: {str(e)[:80]}"]
            }
# core solver. takes context and generates answer - draft for review
class SolutionGenerator:
    def __init__(self, model = "gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.2)
        self.prompt = ChatPromptTemplate.from_messages(Solution_Generator_PROMPT)
    
    def generate(self, state, force = False):
        if state.get("answer_draft") and not force:
            return {}
        
        try:
            context = state.get("retrieved_context", "")[:6000]
            response = self.llm.invoke(self.prompt.format_messages(
                context=context,
                question=state["question"]
            ))
            
            content = clean_json_response(response.content)
            result = json.loads(content)
            
            answer = result.get("answer", "")
            if not answer.isdigit():
                answer = parse_answer(response.content)
            
            return {
                "solution_draft": result.get("solution", response.content),
                "answer_draft": normalize_answer(answer),
                "messages": [f"[Generator] answer={answer[:20]}"]
            }
        except Exception as e:
            answer = parse_answer(response.content if 'response' in dir() else "")
            return {
                "solution_draft": response.content if 'response' in dir() else "",
                "answer_draft": normalize_answer(answer),
                "messages": [f"[Generator] parse fallback: {str(e)[:60]}"]
            }
# symbolic judge, deterministic. if confirms, we can be sure that problem is solved correctly
class WolframVerifier:
    def __init__(self):
        self.app_id = os.getenv("WOLFRAM_ALPHA_APPID")
        self.url = "http://api.wolframalpha.com/v1/result"
    
    def extract_number(self, text):
        if not text:
            return None
        text = str(text).lower().strip()
        
        frac_match = re.search(r'(\d+)/(\d+)', text)
        if frac_match:
            num, denom = int(frac_match.group(1)), int(frac_match.group(2))
            if denom != 0:
                return num / denom
        
        numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', text)
        if numbers:
            try:
                return float(numbers[0])
            except:
                pass
        return None
    
    def verify(self, state):
        math_expr = state.get("math_expression", "").strip()
        
        if not math_expr or not self.app_id:
            return {
                "wolfram_result": "no expression" if not math_expr else "no API key",
                "wolfram_verified": False,
                "messages": ["[Wolfram] skipped - no expression"]
            }
        
        try:
            params = {"appid": self.app_id, "i": math_expr}
            response = requests.get(self.url, params=params, timeout=30)
            
            if response.status_code == 501:
                return {
                    "wolfram_result": "query not supported",
                    "wolfram_verified": False,
                    "messages": ["[Wolfram] 501 - query not supported"]
                }
            
            if response.status_code != 200:
                return {
                    "wolfram_result": f"error {response.status_code}",
                    "wolfram_verified": False,
                    "messages": [f"[Wolfram] API error {response.status_code}"]
                }
            
            result = response.text.strip()
            wolfram_num = self.extract_number(result)
            answer_num = self.extract_number(state.get("answer_draft", ""))
            
            if wolfram_num is not None and answer_num is not None:
                if abs(wolfram_num - round(wolfram_num)) < 0.001:
                    wolfram_int = int(round(wolfram_num))
                    answer_int = int(round(answer_num)) if abs(answer_num - round(answer_num)) < 0.001 else None
                    
                    if answer_int is not None and wolfram_int == answer_int:
                        return {
                            "wolfram_result": result[:200],
                            "wolfram_verified": True,
                            "messages": [f"[Wolfram] VERIFIED: {wolfram_int} == {answer_int}"]
                        }
                    else:
                        return {
                            "wolfram_result": result[:200],
                            "wolfram_verified": False,
                            "messages": [f"[Wolfram] MISMATCH: wolfram={wolfram_int}, answer={answer_int}"]
                        }
                
                if abs(wolfram_num - answer_num) / max(abs(wolfram_num), 1) < 0.01:
                    return {
                        "wolfram_result": result[:200],
                        "wolfram_verified": True,
                        "messages": [f"[Wolfram] VERIFIED: {wolfram_num:.4f} approx {answer_num:.4f}"]
                    }
                else:
                    return {
                        "wolfram_result": result[:200],
                        "wolfram_verified": False,
                        "messages": [f"[Wolfram] MISMATCH: wolfram={wolfram_num:.4f}, answer={answer_num:.4f}"]
                    }
            
            return {
                "wolfram_result": result[:200],
                "wolfram_verified": False,
                "messages": [f"[Wolfram] cannot compare: {result[:50]}"]
            }
            
        except requests.Timeout:
            return {"wolfram_result": "timeout", "wolfram_verified": False, 
                    "messages": ["[Wolfram] timeout"]}
        except Exception as e:
            return {"wolfram_result": f"error: {str(e)[:80]}", "wolfram_verified": False,
                    "messages": [f"[Wolfram] error: {str(e)[:60]}"]}

# if wolfram is unavailable or problem does not consists of only formulaes, agent makes code to increase confidence
class CodeVerifier:
    def __init__(self, model = "gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(Code_Verifier_PROMPT)
    
    def verify(self, state):
        if not state.get("answer_draft"):
            return {"code_result": "", "code_verified": False,
                    "messages": ["[CodeVerifier] no answer to verify"]}
        
        try:
            response = self.llm.invoke(self.prompt.format_messages(
                question=state["question"][:1000],
                answer=state.get("answer_draft", "")
            ))
            
            code_match = re.search(r'```python\s*([\s\S]*?)```', response.content)
            if not code_match:
                code_match = re.search(r'```\s*([\s\S]*?)```', response.content)
            
            if not code_match:
                return {"code_result": "no code generated", "code_verified": False,
                        "messages": ["[CodeVerifier] no code in response"]}
            
            code = code_match.group(1).strip()
            
            success, output = run_python_code(code)
            
            if not success:
                return {"code_result": output[:200], "code_verified": False,
                        "messages": [f"[CodeVerifier] execution failed: {output[:60]}"]}
            
            result_match = re.search(r'RESULT:\s*(\d+)', output)
            if result_match:
                code_answer = result_match.group(1)
                answer_draft = normalize_answer(state.get("answer_draft", ""))
                
                if code_answer == answer_draft:
                    return {"code_result": output[:200], "code_verified": True,
                            "messages": [f"[CodeVerifier] VERIFIED: {code_answer}"]}
                else:
                    return {"code_result": output[:200], "code_verified": False,
                            "answer_draft": code_answer,
                            "messages": [f"[CodeVerifier] CORRECTED: {answer_draft} -> {code_answer}"]}
            
            numbers = re.findall(r'\b(\d+)\b', output)
            if numbers:
                return {"code_result": output[:200], "code_verified": False,
                        "messages": [f"[CodeVerifier] found number: {numbers[-1]}"]}
            
            return {"code_result": output[:200], "code_verified": False,
                    "messages": ["[CodeVerifier] no result extracted"]}
            
        except Exception as e:
            return {"code_result": str(e)[:100], "code_verified": False,
                    "messages": [f"[CodeVerifier] error: {str(e)[:60]}"]}

# if verification fails, creates new answer
class Refiner:
    def __init__(self, model = "gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.1)
        self.prompt = ChatPromptTemplate.from_messages(Refiner_PROMPT)
    
    def refine(self, state):
        try:
            response = self.llm.invoke(self.prompt.format_messages(
                question=state["question"][:800],
                prev_answer=state.get("answer_draft", ""),
                wolfram=state.get("wolfram_result", "N/A"),
                code=state.get("code_result", "N/A")
            ))
            
            content = clean_json_response(response.content)
            result = json.loads(content)
            new_answer = normalize_answer(result.get("answer", ""))
            
            return {
                "answer_draft": new_answer,
                "solution_draft": result.get("solution", state.get("solution_draft", "")),
                "messages": [f"[Refiner] corrected to: {new_answer}"]
            }
        except Exception as e:
            return {"messages": [f"[Refiner] error: {str(e)[:60]}"]}

# making graph
class MultiAgentMathSolver:
    def __init__(self, model = "gemini-2.0-flash"):
        self.analyzer = QuestionAnalyzer(model)
        self.retriever = ContextRetriever()
        self.generator = SolutionGenerator(model)
        self.wolfram = WolframVerifier()
        self.code_verifier = CodeVerifier(model)
        self.refiner = Refiner(model)
        
        workflow = StateGraph(AgentState)
        
        workflow.add_node("analyze", self.analyze)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.add_node("verify_wolfram", self.verify_wolfram)
        workflow.add_node("verify_code", self.verify_code)
        workflow.add_node("refine", self.refine)
        workflow.add_node("finalize", self.finalize)
        
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "verify_wolfram")
        
        workflow.add_conditional_edges(
            "verify_wolfram",
            self.after_wolfram,
            {"verify_code": "verify_code", "finalize": "finalize"}
        )
        
        workflow.add_conditional_edges(
            "verify_code", 
            self.after_code,
            {"refine": "refine", "finalize": "finalize"}
        )
        
        workflow.add_edge("refine", "generate")
        workflow.add_edge("finalize", END)
        
        self.graph = workflow.compile()
    
    def analyze(self, state):
        return self.analyzer.analyze(state)
    
    def retrieve(self, state):
        return self.retriever.retrieve(state)
    
    def generate(self, state):
        force = state.get("iteration", 0) > 0
        return self.generator.generate(state, force=force)
    
    def verify_wolfram(self, state):
        result = self.wolfram.verify(state)
        return result
    
    def verify_code(self, state):
        result = self.code_verifier.verify(state)
        result["iteration"] = state.get("iteration", 0) + 1
        return result
    
    def refine(self, state):
        return self.refiner.refine(state)
    
    def finalize(self, state):
        return {
            "final_answer": state.get("answer_draft", ""),
            "final_solution": state.get("solution_draft", ""),
            "messages": ["[Finalized]"]
        }
    
    def after_wolfram(self, state):
        if state.get("wolfram_verified", False):
            return "finalize"
        return "verify_code"
    
    def after_code(self, state):
        if state.get("code_verified", False):
            return "finalize"
        if state.get("iteration", 0) >= state.get("max_iterations", 2):
            return "finalize"
        return "refine"
    
    def solve(self, question, max_iterations = 2):
        initial_state = {
            "question": question,
            "problem_type": "",
            "key_concepts": [],
            "search_queries": [],
            "math_expression": "",
            "retrieved_context": "",
            "retrieved_chunks": [],
            "solution_draft": "",
            "answer_draft": "",
            "wolfram_result": "",
            "wolfram_verified": False,
            "code_result": "",
            "code_verified": False,
            "final_answer": "",
            "final_solution": "",
            "iteration": 0,
            "max_iterations": max_iterations,
            "messages": [],
            "error": None
        }
        
        final = self.graph.invoke(initial_state)
        
        return {
            "question": question,
            "answer": final.get("final_answer", ""),
            "solution": final.get("final_solution", ""),
            "problem_type": final.get("problem_type", ""),
            "wolfram_verified": final.get("wolfram_verified", False),
            "code_verified": final.get("code_verified", False),
            "wolfram_result": final.get("wolfram_result", ""),
            "iterations": final.get("iteration", 0),
            "logs": final.get("messages", []),
            "retrieved_chunks": final.get("retrieved_chunks", [])
        }

# testing code
def load_dataset(path = "math_dataset_ukr.jsonl"):
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

def evaluate(solver, dataset, max_tasks = None):
    results = []
    correct = 0
    
    tasks = dataset[:max_tasks] if max_tasks else dataset
    
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['id']}")
        
        result = solver.solve(task['question'])
        
        pred = normalize_answer(result['answer'])
        exp = normalize_answer(task['answer'])
        is_correct = pred == exp
        
        if is_correct:
            correct += 1
            v = "Wrong" if result['wolfram_verified'] else ("Correct" if result['code_verified'] else "")
            print(f"  {pred} [{v}]")
        else:
            print(f"  {pred} (expected: {exp})")
        
        results.append({
            'task_id': task['id'],
            'question': task['question'][:200],
            'expected': exp,
            'predicted': pred,
            'correct': is_correct,
            'wolfram_verified': result['wolfram_verified'],
            'code_verified': result['code_verified'],
            'iterations': result['iterations'],
            'logs': result['logs']
        })
    
    print(f"results: {correct}/{len(tasks)} = {correct/len(tasks)*100:.1f}%")
    
    return results

def main():
    solver = MultiAgentMathSolver(model="gemini-2.0-flash")
    
    test_q = """Знайдіть остачу від ділення 7^100 на 13."""
    
    print(f"test: {test_q}")
    result = solver.solve(test_q)
    
    print(f"answer: {result['answer']}")
    print(f"wolfram verified: {result['wolfram_verified']}")
    print(f"code verified: {result['code_verified']}")
    print("logs:")
    for log in result['logs']:
        print(f"  {log}")
    
    dataset = load_dataset()
    results = evaluate(solver, dataset, max_tasks=50)
    with open("results_multiagent.jsonl", 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
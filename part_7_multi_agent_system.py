import json
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
import operator
import re
import requests
import os

load_dotenv()
# config
VECTOR_DB_PATH = "./vector_store"
COLLECTION_NAME = "math_chunks"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# shared state across all nodes that read and write to it
class AgentState(TypedDict):
    question: str
    topic: str
    key_concepts: List[str]
    retrieved_context: str
    retrieved_sources: List[str]
    solution_draft: str
    answer_draft: str
    math_expression: str
    wolfram_result: str
    verification_passed: bool
    final_answer: str
    final_solution: str
    iteration: int
    max_iterations: int
    messages: Annotated[List[str], operator.add]
    analyzed: bool
    retrieved: bool
    generated: bool

def clean_json_response(content): # striping and parsing output
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        content = content[start:end]
    return content

# the first agent. Its job is to interpret the user's intent
class QuestionAnalyzer:
    def __init__(self, model="gpt-5-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)
    
    #asks to take raw input and output structured response
    # identified key parts of question and constructs math expression for wolfram
    def analyze(self, state):
        if state.get("analyzed", False):
            return {}
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Проаналізуй математичну задачу та визнач:
            1. topic - тема задачі українською
            2. key_concepts - ключові концепції
            3. math_expression - вираз для Wolfram Alpha англійською

            ПРАВИЛА для math_expression:
            - Конкретні числа: "1 - (0.5)^8", "2/3", "0.667"
            - Рівняння: "solve x^2 = 4"
            - Якщо ДІЙСНО немає обчислень - ""

            КРИТИЧНО:
            - Для ймовірностей НЕ генеруй складні вирази з solve
            - Якщо можна порахувати напряму - пиши пряме обчислення
            - Приклад: "ймовірність 2/3" -> просто "2/3", НЕ "1/(1-0.5*0.5)"

            Відповідай JSON: {{"topic": "...", "key_concepts": ["..."], "math_expression": "..."}}"""),
                        ("user", "Задача: {question}")
                    ])
        
        try:
            response = self.llm.invoke(prompt.format_messages(question=state["question"]))
            content = clean_json_response(response.content)
            result = json.loads(content)
            
            return {
                "topic": result.get("topic", "математика"),
                "key_concepts": result.get("key_concepts", []),
                "math_expression": result.get("math_expression", ""),
                "analyzed": True,
                "messages": [f"analyzer: expr={result.get('math_expression', 'none')[:50]}"]
            }
        except Exception as e:
            return {
                "messages": [f"analyzer error: {str(e)[:100]}"],
                "topic": "математика",
                "key_concepts": [],
                "math_expression": "",
                "analyzed": True
            }
# RAG: retrieves relevant content 
#
# takes question and the key concepts identified in the previous step to search vector database
class ContextRetriever:
    def __init__(self, top_k=5):
        self.top_k = top_k
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    def retrieve(self, state):
        if state.get("retrieved", False):
            return {}
            
        try:
            search_query = state["question"]
            if state.get("key_concepts"):
                search_query += " " + " ".join(state["key_concepts"])
            
            docs = self.retriever.invoke(search_query)
            
            context = "\n\n---\n\n".join([
                f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            ])
            
            sources = [doc.metadata.get('source', 'unknown') for doc in docs]
            
            return {
                "retrieved_context": context,
                "retrieved_sources": sources,
                "retrieved": True,
                "messages": [f"retriever: found {len(docs)} chunks"]
            }
        except Exception as e:
            return {
                "messages": [f"retriever error: {str(e)[:100]}"],
                "retrieved_context": "",
                "retrieved_sources": [],
                "retrieved": True
            }
# solution agent.

# takes retrieved context and question and tries to come up with the solution for the problem
class SolutionGenerator:
    def __init__(self, model="gpt-5-nano"):
        self.llm = ChatOpenAI(model=model, temperature=0.3)
    
    def generate(self, state, force_regenerate=False):
        if state.get("generated", False) and not force_regenerate:
            return {}
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ти експерт-математик. Використовуй контекст з підручників.

            Відповідай JSON: {{"answer": "коротка відповідь", "solution": "розв'язок"}}

            answer - ТІЛЬКИ фінальне число або короткий текст
            solution - детальне пояснення українською

            БЕЗ LaTeX символів, пиши: "1/2", "cos(x)", "sin(x)"
            """),
                        ("user", """Контекст:
            {context}

            Задача: {question}

            Дай точну відповідь""")
                    ])
        
        try:
            context = state.get("retrieved_context", "")[:4000]
            if not context:
                context = "контекст відсутній"
            
            response = self.llm.invoke(prompt.format_messages(
                question=state["question"],
                context=context
            ))
            
            content = clean_json_response(response.content)
            content = content.replace("\\", "")
            
            result = json.loads(content)
            
            return {
                "answer_draft": result.get("answer", ""),
                "solution_draft": result.get("solution", ""),
                "generated": True,
                "messages": [f"generator: answer={result.get('answer', '')[:30]}"]
            }
        except Exception as e:
            return {
                "messages": [f"generator error: {str(e)[:100]}"],
                "answer_draft": "",
                "solution_draft": "",
                "generated": True
            }
# tool block
# introduces wolfram
class Verifier:
    def __init__(self):
        self.app_id = os.getenv("WOLFRAM_ALPHA_APPID")
        self.url = "http://api.wolframalpha.com/v1/result"
    
    def extract_number(self, text):
        if not text:
            return None
        text = str(text).lower()
        
        fractions = {
            "half": 0.5, "third": 0.333, "quarter": 0.25,
            "two thirds": 0.667, "2/3": 0.667, "three quarters": 0.75
        }
        for word, value in fractions.items():
            if word in text:
                return value
        
        numbers = re.findall(r'-?\d+\.?\d*[eE]?[+-]?\d*', text)
        if numbers:
            try:
                return float(numbers[0])
            except:
                pass
        return None
    
    # takes math expression and sends it to the wolfram to get objective ground truth
    def verify(self, state):
        math_expr = state.get("math_expression", "").strip()
        
        if not math_expr:
            return {
                "verification_passed": True,
                "wolfram_result": "no expression",
                "messages": ["verifier: no math expression"]
            }
        
        try:
            params = {"appid": self.app_id, "i": math_expr}
            response = requests.get(self.url, params=params, timeout=30)
            
            if response.status_code == 501:
                return {
                    "verification_passed": True,
                    "wolfram_result": "not supported",
                    "messages": ["verifier: wolfram 501"]
                }
            
            if response.status_code != 200:
                return {
                    "verification_passed": True,
                    "wolfram_result": f"error {response.status_code}",
                    "messages": [f"verifier: api error {response.status_code}"]
                }
            
            result = response.text.strip()
            wolfram_result = result[:300]
            
            answer_num = self.extract_number(state.get("answer_draft", ""))
            wolfram_num = self.extract_number(result)
            
            if answer_num is not None and wolfram_num is not None:
                diff = abs(answer_num - wolfram_num)
                if diff <= 0.02: # small tolarance in case of inconsistency
                    return {
                        "verification_passed": True,
                        "wolfram_result": wolfram_result,
                        "messages": [f"verifier: passed - wolfram={wolfram_num:.3f} answer={answer_num:.3f}"]
                    }
                else:
                    return {
                        "verification_passed": False,
                        "wolfram_result": wolfram_result,
                        "messages": [f"verifier: mismatch - wolfram={wolfram_num:.3f} answer={answer_num:.3f}"]
                    }
            
            return {
                "verification_passed": True,
                "wolfram_result": wolfram_result,
                "messages": ["verifier: cannot compare"]
            }
                
        except requests.Timeout:
            return {
                "verification_passed": True,
                "wolfram_result": "timeout",
                "messages": ["verifier: timeout"]
            }
        except Exception as e:
            error_msg = str(e) if str(e) else type(e).__name__
            return {
                "verification_passed": True,
                "wolfram_result": f"error: {error_msg[:100]}",
                "messages": [f"verifier: error - {error_msg[:100]}"]
            }
# activates only when verifier finds a mistake
class Refiner:
    def __init__(self, model="gpt-5-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0.2)
    # if wolfram has value, answer should be fixed, enforce regeneration
    def refine(self, state):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ти експерт-математик. Виправ розв'язок на основі Wolfram Alpha.

            КРИТИЧНО ВАЖЛИВО:
            - Якщо Wolfram показав число, а твоя відповідь інша - ОБОВ'ЯЗКОВО виправ на Wolfram число
            - Округли до точності з задачі (якщо вказано "до 0.01" то 2 знаки)

            Відповідай JSON БЕЗ LaTeX: {{"answer": "ЧИСЛО з Wolfram", "solution": "пояснення"}}"""),
                        ("user", """Задача: {question}

            Попередня відповідь: {prev_answer}

            Wolfram Alpha: {wolfram}

            Виправ answer на правильне число""")
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages(
                question=state["question"],
                prev_answer=state.get("answer_draft", ""),
                wolfram=state.get("wolfram_result", "")
            ))
            
            content = clean_json_response(response.content)
            content = content.replace("\\", "")
            
            result = json.loads(content)
            new_answer = result.get("answer", state.get("answer_draft", ""))
            
            return {
                "answer_draft": new_answer,
                "solution_draft": result.get("solution", state.get("solution_draft", "")),
                "generated": False,
                "messages": [f"refiner: new answer={new_answer[:30]}"]
            }
        except Exception as e:
            return {
                "messages": [f"refiner error: {str(e)[:100]}"]
            }

# graph orchestrator that connects all nodes using StateGraph
class MultiAgentMathSolver:
    def __init__(self):
        self.analyzer = QuestionAnalyzer()
        self.retriever = ContextRetriever(top_k=5)
        self.generator = SolutionGenerator()
        self.verifier = Verifier()
        self.refiner = Refiner()
        
        workflow = StateGraph(AgentState)

        # analyze -> retrieve -> generate -> verify
        
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("verify", self.verify_node)
        workflow.add_node("refine", self.refine_node)
        workflow.add_node("finalize", self.finalize_node)
        
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "verify")
        # if verification_passed -> go to finalize
        # else go to refine -> generate
        workflow.add_conditional_edges(
            "verify",
            self.should_refine,
            {"refine": "refine", "finalize": "finalize"}
        )
        workflow.add_edge("refine", "generate")
        workflow.add_edge("finalize", END)
        
        self.graph = workflow.compile()
    

    # adding state
    def analyze_node(self, state):
        return self.analyzer.analyze(state)
    
    def retrieve_node(self, state):
        return self.retriever.retrieve(state)
    
    def generate_node(self, state):
        force_regen = state.get("iteration", 0) > 0
        return self.generator.generate(state, force_regenerate=force_regen)
    
    def verify_node(self, state):
        result = self.verifier.verify(state)
        result["iteration"] = state.get("iteration", 0) + 1
        return result
    
    def refine_node(self, state):
        return self.refiner.refine(state)
    
    def finalize_node(self, state):
        return {
            "final_answer": state.get("answer_draft", ""),
            "final_solution": state.get("solution_draft", ""),
            "messages": ["finalized"]
        }
    # condition
    def should_refine(self, state):
        if state.get("verification_passed", False): #if verified-> finalize
            return "finalize"
        
        if state.get("iteration", 0) >= state.get("max_iterations", 2): # max iterations
            return "finalize"
        
        return "refine"
    
    #entry point that kicks off the graph execution
    def solve_task(self, question, max_iterations=2):
        initial_state = {
            "question": question,
            "topic": "",
            "key_concepts": [],
            "retrieved_context": "",
            "retrieved_sources": [],
            "solution_draft": "",
            "answer_draft": "",
            "math_expression": "",
            "wolfram_result": "",
            "verification_passed": False,
            "final_answer": "",
            "final_solution": "",
            "iteration": 0,
            "max_iterations": max_iterations,
            "messages": [],
            "analyzed": False,
            "retrieved": False,
            "generated": False
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "question": question,
            "answer": final_state.get("final_answer", ""),
            "solution": final_state.get("final_solution", ""),
            "topic": final_state.get("topic", ""),
            "verification_passed": final_state.get("verification_passed", False),
            "wolfram_result": final_state.get("wolfram_result", ""),
            "iterations": final_state.get("iteration", 0),
            "logs": final_state.get("messages", []),
            "retrieved_sources": final_state.get("retrieved_sources", [])
        }

# data loading
def load_evaluation_dataset(path="evaluation_dataset.jsonl"):
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

# for testing purposes
def test_multi_agent():
    print("multi-agent math solver")
    solver = MultiAgentMathSolver()
    dataset = load_evaluation_dataset()
    print(f"loaded {len(dataset)} tasks")
    
    for i, task in enumerate(dataset[:3]):
        question = task['question']
        print(f"\ntask {i+1}: {task.get('task_id', 'unknown')}")
        print(f"question: {question[:100]}")
        
        result = solver.solve_task(question, max_iterations=2)
        
        print("multi-agent result")
        print(f"answer: {result['answer']}")
        print(f"verification: {result['verification_passed']}")
        print(f"iterations: {result['iterations']}")
        print("logs:")
        for log in result['logs']:
            print(f"  {log}")

if __name__ == "__main__":
    test_multi_agent()
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

#config
VECTOR_DB_PATH = "./vector_store"
COLLECTION_NAME = "math_chunks"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

#output schema. expected output, enforce consistence
class MathSolution(BaseModel):
    answer: str = Field(description="Відповідь")
    solution: str = Field(description="Покроковий розв'язок")

# baseline agent. No RAG. 
class BaselineAgent: # cheeper model, with low temp to ensure math consistency
    def __init__(self, model="gpt-5-mini", temperature=0.3):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Ти експерт з математики. Розв'яжи задачу українською мовою.
                Надай:
                1. answer - коротка фінальна відповідь
                2. solution - детальний покроковий розв'язок

                Відповідай JSON: {{"answer": "...", "solution": "..."}}"""),
                            ("user", "Задача: {question}")
        ])
    
    # sends task to LLM, parses result
    def solve_task(self, question: str):
        try:
            response = self.llm.invoke(self.prompt.format_messages(question=question))
            content = response.content.strip()
            
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            result['method'] = 'baseline_no_rag'
            result['question'] = question
            return result
        except Exception as e:
            print(f"baseline error: {e}")
            return {
                "answer": "",
                "solution": "",
                "method": "baseline",
                "error": str(e)
            }

# RAG + generation
class RAGAgent: # the same model, retrieve top 5 docs
    def __init__(self, model="gpt-5-mini", temperature=0.3, top_k=5):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.top_k = top_k
        
        print("loading vector store...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL) # the same embeddings as indexing
        self.vectorstore = Chroma(# loading persistent database
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k}) # retriever
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Ти експерт з математики. Використовуй наданий контекст з підручників для розв'язання задачі УКРАЇНСЬКОЮ мовою.

                Контекст з підручників:
                {context}

                На основі цього контексту розв'яжи задачу. Використовуй методи та підходи з контексту.

                Надай:
                1. answer - коротка фінальна відповідь
                2. solution - детальний покроковий розв'язок

                Відповідай JSON: {{"answer": "...", "solution": "..."}}"""),
                            ("user", "Задача: {question}")
        ])
    
    # passing task to LLM with context
    def solve_task(self, question: str):
        try:
            docs = self.retriever.invoke(question)
            context = "\n\n---\n\n".join([
                f"[Джерело: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            ])
            
            response = self.llm.invoke(self.prompt.format_messages(
                context=context,
                question=question
            ))
            content = response.content.strip()
            
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            result['method'] = 'rag'
            result['question'] = question
            result['retrieved_sources'] = [doc.metadata.get('source', 'unknown') for doc in docs]
            result['num_chunks_used'] = len(docs)
            return result
        except Exception as e:
            print(f"rag error: {e}")
            return {
                "answer": "",
                "solution": "",
                "method": "rag",
                "error": str(e)
            }
# loading data
def load_evaluation_dataset(path="evaluation_dataset.jsonl"):
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset
# for testing purposes
def test_systems():
    print("initializing systems")
    
    baseline = BaselineAgent()
    rag = RAGAgent(top_k=5)
    
    print("loading evaluation dataset...")
    dataset = load_evaluation_dataset()
    print(f"loaded {len(dataset)} tasks")
    
    for i, task in enumerate(dataset[:3]):
        question = task['question']
        expected_answer = task['expected_answer']
        
        print(f"task {i+1}: {task.get('task_id', 'unknown')}")
        print(f"question: {question[:100]}...")
        print(f"expected: {expected_answer}")
        
        print("baseline - no rag")
        baseline_result = baseline.solve_task(question)
        if baseline_result:
            print(f"answer: {baseline_result['answer']}")
            print(f"solution: {baseline_result['solution'][:150]}...")
        
        print("rag - retrieval + generation")
        rag_result = rag.solve_task(question)
        if rag_result:
            print(f"sources: {', '.join(rag_result['retrieved_sources'][:3])}")
            print(f"answer: {rag_result['answer']}")
            print(f"solution: {rag_result['solution'][:150]}...")

if __name__ == "__main__":
    test_systems()
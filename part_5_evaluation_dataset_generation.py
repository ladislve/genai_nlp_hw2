import re
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from pydantic import BaseModel, Field

load_dotenv()

#config
BOOK_PATH = "parsed_output/kapinosov_am_bilousova_gi_ta_in_zno_2014_matematika_kompleks/kapinosov_am_bilousova_gi_ta_in_zno_2014_matematika_kompleks.md"
OUTPUT_FILE = "evaluation_dataset.jsonl"
TARGET_SAMPLES = 40
MIN_TASK_NUMBER = 15 # tasks with lower number are easier

#output schema to enforce structure
class TaskExtraction(BaseModel):
    question: str
    answer: str
    math_expression: str

#load the book
def load_markdown(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

#finding task blocks
def find_task_blocks(content):
    pattern = r'(\d+)\.(\d+)\.?\s+(.*?)(?=\d+\.\d+\.|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    tasks = []
    for section, number, text in matches:
        task_num = int(number)
        if task_num >= MIN_TASK_NUMBER and len(text.strip()) > 100:
            if '```markdown\n\\blacksquare\n```' not in text and '\\blacksquare' not in text: # exclude $blocksquare$ since it is solved
                tasks.append({
                    'section': section,
                    'number': number,
                    'raw_text': text.strip(),
                    'task_id': f"{section}.{number}"
                })
    
    tasks.sort(key=lambda x: int(x['number']), reverse=True)
    return tasks[:TARGET_SAMPLES * 2]

# infering topic from headers to use as context for llm
def extract_topic_from_section(content, section_id):
    header_pattern = rf'#+ .*?{section_id}[.\s]+(.*?)(?:\n|$)'
    match = re.search(header_pattern, content)
    if match:
        return match.group(1).strip()
    
    topics = {
        '1': 'Алгебра та числа',
        '2': 'Рівняння та нерівності',
        '3': 'Функції та графіки',
        '4': 'Геометрія',
        '5': 'Тригонометрія',
        '6': 'Похідні та інтеграли',
        '7': 'Комбінаторика та ймовірність'
    }
    return topics.get(section_id, 'Математика')

# single agent extraction task
# force the agent to extract and formalize the task
def agent_extract_task(raw_text, task_id, topic, agent_id):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, model_kwargs={"seed": agent_id})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "ти математичний експерт. витягни з тексту: 1. question - чисте формулювання задачі, 2. answer - відповідь, 3. math_expression - вираз для wolfram alpha. відповідай json."),
        ("user", "тема: {topic}\nтекст:\n{raw_text}")
    ])
    
    parser = JsonOutputParser(pydantic_object=TaskExtraction)
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "topic": topic,
            "raw_text": raw_text[:2000]
        })
        
        if isinstance(result, dict) and all(k in result for k in ['question', 'answer', 'math_expression']):
            return result
    except Exception as e:
        print(f"agent {agent_id} error: {e}")
    
    return None

# external symbolic check with wolfram
def verify_with_wolfram(math_expr):
    try:
        wolfram = WolframAlphaAPIWrapper()
        return wolfram.run(math_expr)
    except Exception as e:
        print(f"wolfram error: {e}")
        return None

# judge decision
def judge_consensus(agents_results):
    # judge 
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # sees all answers of agents
    agents_summary = "\n".join([
        f"agent {i+1}: answer: {r['answer']} math: {r['math_expression']}"
        for i, r in enumerate(agents_results)
    ])
    # and decideds whethere there is a consensus between them
    prompt = ChatPromptTemplate.from_messages([
        ("system", "ти суддя-експерт. проаналізуй відповіді та визнач консенсус. відповідай тільки валідним json."),
        ("user", f"відповіді агентів:\n{agents_summary}")
    ])
    
    try:
        response = llm.invoke(prompt.format_messages())
        content = response.content.strip()
        
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        return json.loads(content)
    except Exception as e:
        print(f"judge error: {e}")
        return None

# achieving consensus across agents
def consensus_extract(raw_text, task_id, topic): 
    print(f"running 3 agents for consensus")
    #running  3 independent agents
    agents_results = []
    for agent_id in range(3):
        result = agent_extract_task(raw_text, task_id, topic, agent_id)
        if result: # collect successful tasks
            agents_results.append(result)
            print(f"agent {agent_id+1}: {result['answer']}")
    
    if len(agents_results) < 2: # require at least 2 successful agents
        print(f"less than 2 agents succeeded")
        return None
    
    # send answers to judge
    print(f"judge analyzing consensus")
    judge_decision = judge_consensus(agents_results)
    
    # based on result accept into dataset or deny
    if not judge_decision or not judge_decision.get('consensus'):
        print(f"judge: no consensus")
        return None
    
    print(f"judge consensus: {judge_decision['final_answer']}")
    
    final_result = {
        'question': agents_results[0]['question'],
        'answer': judge_decision['final_answer'],
        'math_expression': judge_decision['final_math_expression'],
        'judge_reasoning': judge_decision.get('reasoning'),
        'agent_answers': [r['answer'] for r in agents_results]
    }
    
    print(f"verifying: {final_result['math_expression'][:50]}")
    wolfram_result = verify_with_wolfram(final_result['math_expression']) # verifying with wolfram
    
    if wolfram_result:
        final_result['wolfram_verification'] = wolfram_result[:200]
        print(f"wolfram: {wolfram_result[:80]}")
    else:
        final_result['wolfram_verification'] = "not verified"
    
    return final_result

# dataset creation
def create_dataset():
    print(f"loading {BOOK_PATH}")
    content = load_markdown(BOOK_PATH)
    
    print("finding complex tasks")
    task_blocks = find_task_blocks(content)
    print(f"found {len(task_blocks)} candidate tasks")
    
    dataset = []
    for i, task in enumerate(task_blocks):
        if len(dataset) >= TARGET_SAMPLES:
            break
            
        print(f"task {task['task_id']}")
        
        topic = extract_topic_from_section(content, task['section'])
        extracted = consensus_extract(task['raw_text'], task['task_id'], topic)
        
        if extracted:
            dataset.append({
                "input": topic,
                "question": extracted['question'],
                "expected_answer": extracted['answer'],
                "math_expression": extracted['math_expression'],
                "wolfram_verification": extracted['wolfram_verification'],
                "judge_reasoning": extracted['judge_reasoning'],
                "agent_answers": extracted['agent_answers'],
                "type": "task",
                "task_id": task['task_id'],
                "difficulty": "high"
            })
            print(f"added {len(dataset)} of {TARGET_SAMPLES}")
    
    print(f"saving {len(dataset)} tasks to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"dataset created: {len(dataset)} verified tasks")
    
    with open('evaluation_dataset_preview.json', 'w', encoding='utf-8') as f:
        json.dump(dataset[:3], f, ensure_ascii=False, indent=2)
    print("preview saved")

if __name__ == "__main__":
    create_dataset()
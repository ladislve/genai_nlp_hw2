import json
import time
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from part_6_base_systems import BaselineAgent, RAGAgent
from part_7_multi_agent_system import MultiAgentMathSolver

load_dotenv()

#evaluation data
DATASET_PATH = "evaluation_dataset.jsonl"
RESULTS_DIR = "evaluation_results"
def load_dataset(path=DATASET_PATH):
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

# llm judge, whose job is comparison
class AnswerJudge:
    def __init__(self, model="gpt-5-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Ти експерт-математик. Визнач, чи є дві відповіді математично еквівалентними.

            ПРАВИЛА:
            1. Числові відповіді: порівнюй значення (5 = 5.0 = 5.00)
            2. Допускай округлення в межах 1% або 0.01
            3. Ігноруй різний формат запису (x=5 vs 5 vs "п'ять")
            4. Ігноруй одиниці виміру якщо значення співпадають
            5. Для виразів: перевір математичну еквівалентність

            ВІДПОВІДАЙ ТІЛЬКИ JSON: {{"equivalent": true/false, "reasoning": "пояснення"}}"""),
                        ("user", """Питання: {question}

            Очікувана відповідь: {expected}
            Згенерована відповідь: {generated}

            Чи еквівалентні ці відповіді?""")
                    ])
    
    def judge(self, question, expected, generated):
        if not generated or generated == "ERROR":
            return False, "no answer generated"
        
        response = self.llm.invoke(self.prompt.format_messages(
            question=question[:200],
            expected=str(expected),
            generated=str(generated)
        ))
        
        content = response.content.strip()
        
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            content = content[start:end]
        
        result = json.loads(content)
        return result.get("equivalent", False), result.get("reasoning", "")
        
# experiment loop
# for each method
# run solver
# measure time
# judge correctness
# collect logs
def evaluate_system(system, dataset, system_name, judge, max_tasks=None, verbose=False):
    print(f"evaluating: {system_name.lower()}")
    
    results = []
    correct = 0
    total = 0
    total_time = 0
    wolfram_verified = 0
    
    tasks_to_eval = dataset[:max_tasks] if max_tasks else dataset
    # solving tasks
    for i, task in enumerate(tasks_to_eval):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(tasks_to_eval)}] task {task.get('task_id', 'unknown')}")
        print(f"question: {task['question'][:100]}")
        print(f"expected: {task['expected_answer']}")
        
        start_time = time.time()
        try:
            if system_name == "multi-agent":
                result = system.solve_task(task['question'], max_iterations=2)
            else:
                result = system.solve_task(task['question'])
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            if result is None:
                print(f"{system_name} error: solve_task returned None")
                results.append({
                    'task_id': task.get('task_id', 'unknown'),
                    'question': task['question'],
                    'expected_answer': task['expected_answer'],
                    'generated_answer': 'ERROR',
                    'error': 'solve_task returned None',
                    'correct': False,
                    'time_seconds': elapsed
                })
                total += 1
                continue
            
            generated = result.get('answer', '')
            
            if not generated:
                generated = 'ERROR'
            
            print(f"generated: {generated}")
            
            if verbose and 'logs' in result:
                print(f"\n--- {system_name} logs ---")
                for log in result['logs']:
                    print(f"  {log}")
                print(f"--- end logs ---\n")
            
            print("  judging answer", end=" ")
            
            correct_flag, reasoning = judge.judge(
                task['question'],
                task['expected_answer'],
                generated
            )
            
            if correct_flag:
                correct += 1
                print("✓ correct")
            else:
                print("✗ wrong")
            print(f"  reasoning: {reasoning}")
            
            if 'verification_passed' in result and result['verification_passed']:
                wolfram_verified += 1
            
            results.append({
                'task_id': task.get('task_id', 'unknown'),
                'question': task['question'],
                'expected_answer': task['expected_answer'],
                'generated_answer': generated,
                'solution': result.get('solution', result.get('logs', [])),
                'correct': correct_flag,
                'judge_reasoning': reasoning,
                'time_seconds': elapsed,
                'method': result.get('method', system_name),
                'wolfram_verified': result.get('verification_passed', None),
                'wolfram_result': result.get('wolfram_result', 'N/A'),
                'topic': result.get('topic', None),
                'iterations': result.get('iterations', None),
                'logs': result.get('logs', [])
            })
            total += 1
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"{system_name} error: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'task_id': task.get('task_id', 'unknown'),
                'question': task['question'],
                'expected_answer': task['expected_answer'],
                'generated_answer': 'ERROR',
                'error': str(e),
                'correct': False,
                'time_seconds': elapsed
            })
            total += 1
    # summary metrics
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    wolfram_rate = (wolfram_verified / total * 100) if total > 0 else 0
    
    summary = {
        'system': system_name,
        'total_tasks': total,
        'correct': correct,
        'accuracy': accuracy,
        'avg_time_seconds': avg_time,
        'total_time_seconds': total_time,
        'wolfram_verified': wolfram_verified,
        'wolfram_verification_rate': wolfram_rate,
        'results': results
    }
    
    print(f"summary: {system_name.lower()}")
    print(f"accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"avg time: {avg_time:.2f}s per task")
    print(f"total time: {total_time:.2f}s")
    if wolfram_verified > 0:
        print(f"wolfram verified: {wolfram_verified}/{total} = {wolfram_rate:.2f}%")
    
    return summary
# saving results
def save_results(all_results, output_dir=RESULTS_DIR):
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    full_path = Path(output_dir) / f"full_results_{timestamp}.json"
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"saved full results to: {full_path}")
    
    summary_path = Path(output_dir) / f"summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("evaluation summary\n\n")
        
        for result in all_results:
            f.write(f"\n{result['system'].lower()}:\n")
            f.write(f"  accuracy: {result['accuracy']:.2f}% ({result['correct']}/{result['total_tasks']})\n")
            f.write(f"  avg time: {result['avg_time_seconds']:.2f}s\n")
            if result.get('wolfram_verification_rate', 0) > 0:
                f.write(f"  wolfram verification: {result['wolfram_verification_rate']:.2f}%\n")
        
        f.write("\ncomparison\n\n")
        sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
        f.write("ranking by accuracy:\n")
        for i, result in enumerate(sorted_results, 1):
            f.write(f"  {i}. {result['system'].lower()}: {result['accuracy']:.2f}%\n")
    
    print(f"saved summary to: {summary_path}")
    
    comparison_path = Path(output_dir) / f"per_task_comparison_{timestamp}.jsonl"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        if all_results:
            num_tasks = len(all_results[0]['results'])
            for task_idx in range(num_tasks):
                task_comparison = {
                    'task_id': all_results[0]['results'][task_idx]['task_id'],
                    'question': all_results[0]['results'][task_idx]['question'],
                    'expected_answer': all_results[0]['results'][task_idx]['expected_answer'],
                    'systems': {}
                }
                
                for system_result in all_results:
                    if task_idx < len(system_result['results']):
                        task_result = system_result['results'][task_idx]
                        task_comparison['systems'][system_result['system']] = {
                            'answer': task_result['generated_answer'],
                            'correct': task_result['correct'],
                            'time': task_result.get('time_seconds', 0),
                            'judge_reasoning': task_result.get('judge_reasoning', '')
                        }
                
                f.write(json.dumps(task_comparison, ensure_ascii=False) + '\n')
    
    print(f"saved per-task comparison to: {comparison_path}")
# main evaluation loop
def main():
    print("math tutor system evaluation")
    print("loading evaluation dataset")
    dataset = load_dataset()
    print(f"loaded {len(dataset)} tasks")
    
    MAX_TASKS = 47 # num samples in dataset, needed it for testing on smaller amount of data
    
    print("initializing llm judge")
    judge = AnswerJudge(model="gpt-5-mini")
    
    print("initializing systems")
    baseline = BaselineAgent()
    rag = RAGAgent(top_k=5)
    multi_agent = MultiAgentMathSolver()
    
    all_results = []
    # baseline
    baseline_results = evaluate_system(baseline, dataset, "baseline (no rag)", judge, MAX_TASKS)
    all_results.append(baseline_results)
    #single agent + rag
    rag_results = evaluate_system(rag, dataset, "rag", judge, MAX_TASKS)
    all_results.append(rag_results)
    # multi-agent approach
    multi_agent_results = evaluate_system(multi_agent, dataset, "multi-agent", judge, MAX_TASKS)
    all_results.append(multi_agent_results)
    
    save_results(all_results)
    
    print("final comparison")
    for result in all_results:
        print(f"{result['system'].lower():20} | accuracy: {result['accuracy']:6.2f}% | avg time: {result['avg_time_seconds']:6.2f}s")
    
    print("evaluation complete")

if __name__ == "__main__":
    main()
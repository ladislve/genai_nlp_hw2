# imports
import json
import time
from pathlib import Path
from datetime import datetime
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from part_6_2_base_systems import BaselineAgent, RAGAgent, normalize_answer
from part_7_2_multi_agent_system import MultiAgentMathSolver
from additional_scripts.part_8_prompts import *

load_dotenv()
#config
DATASET_PATH = "math_dataset_ukr.jsonl"
RESULTS_DIR = "evaluation_results"

# for comparison
#singe grade 
@dataclass
class JudgeResult:
    is_correct: bool
    score: float
    reasoning: str

#complete report for single problem
@dataclass
class TaskResult:
    task_id: str
    question: str
    expected: str
    predicted: str
    solution: str
    time_seconds: float
    method: str
    value_correct: bool
    value_score: float = 0.0
    value_reasoning: str = ""
    approach_correct: bool = False
    approach_score: float = 0.0
    approach_reasoning: str = ""
    combined_score: float = 0.0
    error: Optional[str] = None
    wolfram_verified: bool = False
    code_verified: bool = False
    iterations: int = 0
    retrieved_chunks: List[Dict] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

# for metrics results
@dataclass
class EvaluationSummary:
    system: str
    total_tasks: int
    completed: int
    errors: int
    value_correct: int
    value_accuracy: float
    avg_value_score: float
    approach_correct: int
    approach_accuracy: float
    avg_approach_score: float
    avg_combined_score: float
    avg_time: float
    total_time: float
    wolfram_verified: int
    code_verified: int
    results: List[TaskResult] = field(default_factory=list)

# judge to determine whether answer numerically is correct
class ValueJudge:
    def __init__(self, model="gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(VALUE_JUDJE_PROMPT)
    
    def judge(self, expected, predicted):
        if not predicted or predicted.upper() == "ERROR":
            return JudgeResult(False, 0.0, "відповідь відсутня")
        
        exp_norm = normalize_answer(expected)
        pred_norm = normalize_answer(predicted)
        if exp_norm == pred_norm:
            return JudgeResult(True, 1.0, "точний збіг")
        
        try:
            response = self.llm.invoke(self.prompt.format_messages(
                expected=str(expected),
                predicted=str(predicted)
            ))
            
            content = self.extract_json(response.content)
            result = json.loads(content)
            
            if not isinstance(result, dict):
                raise ValueError("LLM returned valid JSON but not a dictionary")

            return JudgeResult(
                is_correct=result.get("is_correct", False),
                score=float(result.get("score", 0.0)),
                reasoning=result.get("reasoning", "")
            )
        except Exception as e:
            return JudgeResult(False, 0.0, f"Judge Error: {str(e)[:50]}")
    
    def extract_json(self, content):
        text = content.strip()
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return text[start:end+1]
        return text

# approach judge. evaluates whether correct steps solving task were taken
class ApproachJudge:
    def __init__(self, model="gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(APPROACH_JUDGE_PROMPT)
    
    def judge(self, question, expected_answer, expected_solution, predicted_answer, system_solution):
        if not system_solution or len(system_solution.strip()) < 10:
            return JudgeResult(False, 0.0, "розв'язок відсутній")
        
        try:
            response = self.llm.invoke(self.prompt.format_messages(
                question=question[:1500],
                expected_solution=expected_solution[:2500] if expected_solution else "Не надано",
                expected_answer=str(expected_answer),
                system_solution=system_solution[:3000],
                predicted_answer=str(predicted_answer)
            ))
            
            content = self.extract_json(response.content)
            result = json.loads(content)
            
            if not isinstance(result, dict):
                 return JudgeResult(False, 0.0, "judge error: output is not json")

            return JudgeResult(
                is_correct=result.get("is_correct", False),
                score=float(result.get("score", 0.0)),
                reasoning=result.get("reasoning", "")
            )
        except Exception as e:
            return JudgeResult(False, 0.0, f"judge error: {str(e)[:100]}")
    
    def extract_json(self, content):
        text = content.strip()
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return text[start:end+1]
        return text

# final grade. prioritize value, but include good math thinking
class CombinedJudge:
    def __init__(self, model="gemini-2.0-flash", value_weight=0.6, approach_weight=0.4):
        self.value_judge = ValueJudge(model)
        self.approach_judge = ApproachJudge(model)
        self.value_weight = value_weight
        self.approach_weight = approach_weight
    
    def judge(self, question, expected, expected_solution, predicted, solution):
        value_result = self.value_judge.judge(expected, predicted)
        
        approach_result = self.approach_judge.judge(
            question=question, 
            expected_answer=expected,
            expected_solution=expected_solution,
            predicted_answer=predicted, 
            system_solution=solution
        )
        
        combined = (
            self.value_weight * value_result.score +
            self.approach_weight * approach_result.score
        )
        
        return value_result, approach_result, combined

# a few wrappers to make standard interface
class SystemWrapper:
    def __init__(self, name):
        self.name = name
    
    def solve(self, task_id, question, expected):
        raise NotImplementedError

class BaselineWrapper(SystemWrapper):
    def __init__(self):
        super().__init__("Baseline")
        self.agent = BaselineAgent()
    
    def solve(self, task_id, question, expected):
        start = time.time()
        try:
            result = self.agent.solve(task_id, question, expected)
            return {
                'predicted': result.predicted_answer,
                'solution': result.solution,
                'time': result.time_seconds,
                'method': result.method,
                'error': result.error
            }
        except Exception as e:
            return {
                'predicted': '', 'solution': '', 'time': time.time()-start,
                'method': 'baseline', 'error': str(e)[:100]
            }

class RAGVerifyWrapper(SystemWrapper):
    def __init__(self):
        super().__init__("RAG+Verify")
        self.agent = RAGAgent(verify=True)
    
    def solve(self, task_id, question, expected):
        start = time.time()
        try:
            result = self.agent.solve(task_id, question, expected)
            return {
                'predicted': result.predicted_answer,
                'solution': result.solution,
                'time': result.time_seconds,
                'method': result.method,
                'error': result.error,
                'code_verified': result.verification == "verified",
                'retrieved_chunks': result.retrieved_chunks
            }
        except Exception as e:
            return {
                'predicted': '', 'solution': '', 'time': time.time()-start,
                'method': 'rag_verify', 'error': str(e)[:100]
            }

class MultiAgentWrapper(SystemWrapper):
    def __init__(self):
        super().__init__("MultiAgent")
        self.agent = MultiAgentMathSolver()
    
    def solve(self, task_id, question, expected):
        start = time.time()
        try:
            result = self.agent.solve(question, max_iterations=2)
            return {
                'predicted': result.get('answer', ''),
                'solution': result.get('solution', ''),
                'time': time.time() - start,
                'method': 'multi_agent',
                'wolfram_verified': result.get('wolfram_verified', False),
                'code_verified': result.get('code_verified', False),
                'iterations': result.get('iterations', 0),
                'retrieved_chunks': result.get('retrieved_chunks', []),
                'logs': result.get('logs', [])
            }
        except Exception as e:
            return {
                'predicted': '', 'solution': '', 'time': time.time()-start,
                'method': 'multi_agent', 'error': str(e)[:100]
            }

# data loading
def load_dataset(path=DATASET_PATH):
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            dataset.append({
                'id': item.get('ID', item.get('id', '')),
                'question': item.get('Problem', item.get('question', '')),
                'answer': str(item.get('Answer', item.get('answer', ''))),
                'solution_gt': str(item.get('Solution', item.get('solution', '')))
            })
    print(f"loaded {len(dataset)} tasks from {path}")
    return dataset
# evaluation loop
def evaluate_system(system, dataset, judge, max_tasks=None):
    print(f"EVALUATING: {system.name}")
    
    tasks = dataset[:max_tasks] if max_tasks else dataset
    results = []
    
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['id']}")
        # system solves problem
        solve_result = system.solve(task['id'], task['question'], task['answer'])
        
        #judges grade result
        value_result, approach_result, combined = judge.judge(
            question=task['question'],
            expected=task['answer'],
            expected_solution=task.get('solution_gt', ''), 
            predicted=solve_result['predicted'],
            solution=solve_result.get('solution', '')
        )
        # storing logs
        result = TaskResult(
            task_id=task['id'],
            question=task['question'],
            expected=task['answer'],
            predicted=solve_result['predicted'],
            solution=solve_result.get('solution', ''),
            time_seconds=solve_result['time'],
            method=solve_result['method'],
            value_correct=value_result.is_correct,
            value_score=value_result.score,
            value_reasoning=value_result.reasoning,
            approach_correct=approach_result.is_correct,
            approach_score=approach_result.score,
            approach_reasoning=approach_result.reasoning,
            combined_score=combined,
            error=solve_result.get('error'),
            wolfram_verified=solve_result.get('wolfram_verified', False),
            code_verified=solve_result.get('code_verified', False),
            iterations=solve_result.get('iterations', 0),
            retrieved_chunks=solve_result.get('retrieved_chunks', []),
            logs=solve_result.get('logs', [])
        )
        
        print_task_result(result)
        results.append(result)
    
    summary = compute_summary(system.name, results)
    print_summary(summary)
    
    return summary

def print_task_result(r):
    if r.error:
        print(f"error: {r.error[:50]}")
        return
    
    v_icon = "Y" if r.value_correct else "N"
    v_info = f"[Wrong]" if r.wolfram_verified else ("[Correct]" if r.code_verified else "")
    print(f"Value: {v_icon} {r.predicted} (exp: {r.expected}) score={r.value_score:.2f} {v_info}")
    
    a_icon = "Yes" if r.approach_correct else "No"
    print(f"Approach: {a_icon} score={r.approach_score:.2f} | {r.approach_reasoning[:60]}")
    
    print(f"Combined: {r.combined_score:.2f} | time={r.time_seconds:.1f}s")

def compute_summary(system_name, results):
    completed = [r for r in results if not r.error]
    
    return EvaluationSummary(
        system=system_name,
        total_tasks=len(results),
        completed=len(completed),
        errors=sum(1 for r in results if r.error),
        value_correct=sum(1 for r in results if r.value_correct),
        value_accuracy=sum(1 for r in results if r.value_correct) / len(results) * 100 if results else 0,
        avg_value_score=sum(r.value_score for r in results) / len(results) if results else 0,
        approach_correct=sum(1 for r in results if r.approach_correct),
        approach_accuracy=sum(1 for r in results if r.approach_correct) / len(results) * 100 if results else 0,
        avg_approach_score=sum(r.approach_score for r in results) / len(results) if results else 0,
        avg_combined_score=sum(r.combined_score for r in results) / len(results) if results else 0,
        avg_time=sum(r.time_seconds for r in completed) / len(completed) if completed else 0,
        total_time=sum(r.time_seconds for r in results),
        wolfram_verified=sum(1 for r in results if r.wolfram_verified),
        code_verified=sum(1 for r in results if r.code_verified),
        results=results
    )

def print_summary(s):
    print(f"{s.system} SUMMARY")
    print(f" Tasks: {s.total_tasks} | Completed: {s.completed} | Errors: {s.errors}")
    print(f"VALUE ACCURACY:    {s.value_correct}/{s.total_tasks} = {s.value_accuracy:.1f}%")
    print(f"Avg Value Score:   {s.avg_value_score:.3f}")
    print(f"APPROACH ACCURACY: {s.approach_correct}/{s.total_tasks} = {s.approach_accuracy:.1f}%")
    print(f"Avg Approach Score: {s.avg_approach_score:.3f}")
    print(f"COMBINED SCORE:    {s.avg_combined_score:.3f}")
    print(f"Avg Time: {s.avg_time:.2f}s | Total: {s.total_time:.1f}s")
    if s.wolfram_verified > 0:
        print(f"Wolfram verified: {s.wolfram_verified}")
    if s.code_verified > 0:
        print(f"Code verified: {s.code_verified}")



def print_comparison_table(summaries):
    print("FINAL COMPARISON")
    
    print(f"\n{'System':<15} {'Value':>8} {'Value%':>8} {'Approach':>10} {'Appr%':>8} {'Combined':>10} {'Time':>8}")
    
    for s in sorted(summaries, key=lambda x: x.avg_combined_score, reverse=True):
        print(f"{s.system:<15} {s.value_correct:>8} {s.value_accuracy:>7.1f}% "
              f"{s.approach_correct:>10} {s.approach_accuracy:>7.1f}% "
              f"{s.avg_combined_score:>10.3f} {s.avg_time:>7.2f}s")
    
    
    best_value = max(summaries, key=lambda x: x.value_accuracy)
    best_approach = max(summaries, key=lambda x: x.approach_accuracy)
    best_combined = max(summaries, key=lambda x: x.avg_combined_score)
    
    print(f"Best Value Accuracy:    {best_value.system} ({best_value.value_accuracy:.1f}%)")
    print(f"Best Approach Accuracy: {best_approach.system} ({best_approach.approach_accuracy:.1f}%)")
    print(f"Best Combined Score:    {best_combined.system} ({best_combined.avg_combined_score:.3f})")

def save_results(summaries, output_dir=RESULTS_DIR):
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    full_path = Path(output_dir) / f"full_results_{timestamp}.json"
    full_data = []
    for s in summaries:
        data = {
            'system': s.system,
            'total_tasks': s.total_tasks,
            'completed': s.completed,
            'errors': s.errors,
            'value_correct': s.value_correct,
            'value_accuracy': s.value_accuracy,
            'avg_value_score': s.avg_value_score,
            'approach_correct': s.approach_correct,
            'approach_accuracy': s.approach_accuracy,
            'avg_approach_score': s.avg_approach_score,
            'avg_combined_score': s.avg_combined_score,
            'avg_time': s.avg_time,
            'total_time': s.total_time,
            'wolfram_verified': s.wolfram_verified,
            'code_verified': s.code_verified,
            'results': [asdict(r) for r in s.results]
        }
        full_data.append(data)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full results: {full_path}")
    
    summary_path = Path(output_dir) / f"summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("EVALUATION SUMMARY\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        f.write(f"{'System':<15} {'Value Acc':>10} {'Approach Acc':>12} {'Combined':>10} {'Avg Time':>10}\n")
        
        for s in sorted(summaries, key=lambda x: x.avg_combined_score, reverse=True):
            f.write(f"{s.system:<15} {s.value_accuracy:>9.1f}% {s.approach_accuracy:>11.1f}% ")
            f.write(f"{s.avg_combined_score:>10.3f} {s.avg_time:>9.2f}s\n")
        
        f.write("DETAILED METRICS\n")
        
        for s in summaries:
            f.write(f"\n{s.system}:\n")
            f.write(f"Total tasks: {s.total_tasks}\n")
            f.write(f"Completed: {s.completed}, Errors: {s.errors}\n")
            f.write(f"\n")
            f.write(f"VALUE METRICS:\n")
            f.write(f"Correct: {s.value_correct}/{s.total_tasks} ({s.value_accuracy:.1f}%)\n")
            f.write(f"Avg Score: {s.avg_value_score:.3f}\n")
            f.write(f"\n")
            f.write(f"APPROACH METRICS:\n")
            f.write(f"Correct: {s.approach_correct}/{s.total_tasks} ({s.approach_accuracy:.1f}%)\n")
            f.write(f"Avg Score: {s.avg_approach_score:.3f}\n")
            f.write(f"\n")
            f.write(f"COMBINED: {s.avg_combined_score:.3f}\n")
            f.write(f"Time: {s.avg_time:.2f}s avg, {s.total_time:.1f}s total\n")
            if s.wolfram_verified > 0:
                f.write(f"Wolfram verified: {s.wolfram_verified}\n")
            if s.code_verified > 0:
                f.write(f"Code verified: {s.code_verified}\n")


def main():
    
    dataset = load_dataset()
    
    MAX_TASKS = None
    VALUE_WEIGHT = 0.6
    APPROACH_WEIGHT = 0.4
    
    judge = CombinedJudge(
        model="gemini-2.0-flash",
        value_weight=VALUE_WEIGHT,
        approach_weight=APPROACH_WEIGHT
    )
    
    systems = [
        BaselineWrapper(),
        RAGVerifyWrapper(),
        MultiAgentWrapper(),
    ]
    
    summaries = []
    for system in systems:
        try:
            summary = evaluate_system(
                system, dataset, judge,
                max_tasks=MAX_TASKS,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"failed to evaluate {system.name}: {e}")
            import traceback
            traceback.print_exc()
    
    if summaries:
        save_results(summaries)
        print_comparison_table(summaries)
    

if __name__ == "__main__":
    main()
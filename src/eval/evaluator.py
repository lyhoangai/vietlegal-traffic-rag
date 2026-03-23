"""
src/eval/evaluator.py — Ragas evaluation with correctness metric.

Usage:
  from src.eval.evaluator import run_evaluation
  scores = await run_evaluation(question, answer, contexts, ground_truth)
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ContextPrecision,
    AnswerRelevancy,
    Faithfulness,
    AnswerCorrectness,
)
from src.eval.db import save_result

METRICS = [
    ContextPrecision(),
    AnswerRelevancy(),
    Faithfulness(),
    AnswerCorrectness(),   # Needs ground_truth — the missing piece before
]


async def run_evaluation(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = "",
) -> dict:
    """Run Ragas evaluation and persist results to SQLite."""
    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth],
    })

    result = evaluate(dataset, metrics=METRICS)

    scores = {
        "context_precision": float(result["context_precision"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "faithfulness": float(result["faithfulness"]),
        "answer_correctness": float(result["answer_correctness"]) if ground_truth else 0.0,
    }
    save_result(question, answer, scores)
    return scores

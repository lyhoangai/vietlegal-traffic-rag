"""src/eval/db.py — SQLite persistence for Ragas eval results."""
import sqlite3
import time

DB_PATH = "eval_results.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         REAL,
                question          TEXT,
                answer            TEXT,
                context_precision REAL,
                answer_relevancy  REAL,
                faithfulness      REAL,
                answer_correctness REAL
            )
        """)


def save_result(question: str, answer: str, scores: dict):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO eval_results VALUES (NULL,?,?,?,?,?,?,?)",
            (
                time.time(),
                question,
                answer,
                scores.get("context_precision", 0),
                scores.get("answer_relevancy", 0),
                scores.get("faithfulness", 0),
                scores.get("answer_correctness", 0),
            ),
        )


def get_avg_metrics() -> dict:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT
                AVG(context_precision),
                AVG(answer_relevancy),
                AVG(faithfulness),
                AVG(answer_correctness),
                COUNT(*)
            FROM eval_results
        """).fetchone()
    return {
        "context_precision": round(row[0] or 0, 3),
        "answer_relevancy": round(row[1] or 0, 3),
        "faithfulness": round(row[2] or 0, 3),
        "answer_correctness": round(row[3] or 0, 3),
        "total_evaluations": row[4] or 0,
        "unit": "0.0 to 1.0",
    }

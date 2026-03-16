import json
from datetime import datetime
from pathlib import Path

from src.evaluator import EvaluationScores
from src.logger import get_logger

logger = get_logger("output_writer")

_OUTPUT_DIR = Path("output")


def save_output(query: str, result: dict, latency: float, scores: EvaluationScores | None) -> None:
    """Serializa la respuesta completa (pregunta, dominios, retrievers, scores, latencia, respuesta) en output/."""
    _OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    domains = result.get("domains", [])
    # agent_outputs se acumula entre queries (operator.add + MemorySaver).
    # Solo nos interesan los outputs de esta query: los últimos len(domains) items.
    n = len(domains)
    current_outputs = result.get("agent_outputs", [])[-n:] if n else []

    payload = {
        "timestamp": ts,
        "query": query,
        "domains": domains,
        "latency_seconds": round(latency, 3),
        "agent_outputs": [
            {
                "domain": o["domain"],
                "retrievers": o.get("retrievers", []),
            }
            for o in current_outputs
        ],
        "scores": {
            "relevance": scores.relevance,
            "completeness": scores.completeness,
            "accuracy": scores.accuracy,
            "reasoning": scores.reasoning,
        } if scores else None,
        "final_answer": result.get("final_answer", ""),
    }
    out_path = _OUTPUT_DIR / f"response_{ts}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Respuesta guardada en %s", out_path)

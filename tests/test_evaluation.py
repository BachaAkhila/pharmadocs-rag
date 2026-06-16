from app.agents.rag_chain import RAGResult
from app.evaluation.ragas_eval import (
    RegressionGate,
    answer_relevancy,
    context_recall,
    evaluate_result,
    faithfulness,
    summarize_scores,
)
from app.ingestion.chunking import Chunk
from app.retrieval.vectorstore import RetrievedChunk


def _make_result(answer: str, query: str, context_text: str) -> RAGResult:
    chunk = Chunk(chunk_id="c1", doc_id="d1", title="T", source="S", year=2024, text=context_text, chunk_index=0)
    retrieved = RetrievedChunk(chunk=chunk, score=1.0, retrieval_method="hybrid")
    return RAGResult(query=query, answer=answer, retrieved_chunks=[retrieved], confidence=0.9, model_name="mock", latency_ms=1.0)


def test_faithfulness_high_when_answer_grounded_in_context():
    result = _make_result(
        answer="The reduction was 5.1x from 125.75 to 24.86 MMACs.",
        query="What was the MAC reduction?",
        context_text="The framework achieves a 5.1x reduction from 125.75 to 24.86 MMACs on ResNet-56.",
    )

    score = faithfulness(result)
    assert score > 0.5


def test_faithfulness_low_when_answer_introduces_unsupported_terms():
    result = _make_result(
        answer="Quantum entanglement enables teleportation of biological samples.",
        query="What was the MAC reduction?",
        context_text="The framework achieves a 5.1x reduction from 125.75 to 24.86 MMACs on ResNet-56.",
    )

    score = faithfulness(result)
    assert score < 0.3


def test_answer_relevancy_with_reference():
    result = _make_result(
        answer="The reduction was 5.1x MAC.",
        query="What MAC reduction was achieved?",
        context_text="irrelevant context",
    )

    score = answer_relevancy(result, reference_answer="The MAC reduction achieved was 5.1x.")
    assert score > 0.3


def test_context_recall_full_coverage():
    result = _make_result(
        answer="anything",
        query="anything",
        context_text="The reduction was 5.1x from 125.75 to 24.86 MMACs on ResNet-56.",
    )

    score = context_recall(result, reference_answer="The reduction was 5.1x from 125.75 to 24.86 MMACs.")
    assert score == 1.0


def test_evaluate_result_returns_overall_average():
    result = _make_result(
        answer="The reduction was 5.1x MAC.",
        query="What MAC reduction?",
        context_text="The reduction was 5.1x MAC.",
    )

    scores = evaluate_result(result, reference_answer="The reduction was 5.1x MAC.")

    assert 0 <= scores.overall <= 1
    assert scores.overall == round((scores.faithfulness + scores.answer_relevancy + scores.context_recall) / 3, 4)


def test_summarize_scores_empty_list_returns_zeros():
    summary = summarize_scores([])
    assert summary == {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0, "overall": 0.0}


def test_regression_gate_passes_when_no_baseline(tmp_path):
    gate = RegressionGate(baseline_path=tmp_path / "baseline.json", tolerance=0.05)

    passed, failures = gate.check({"overall": 0.8})

    assert passed is True
    assert failures == []
    assert (tmp_path / "baseline.json").exists()


def test_regression_gate_fails_on_regression(tmp_path):
    gate = RegressionGate(baseline_path=tmp_path / "baseline.json", tolerance=0.05)
    gate.save_baseline({"overall": 0.85, "faithfulness": 0.9})

    passed, failures = gate.check({"overall": 0.70, "faithfulness": 0.9})

    assert passed is False
    assert len(failures) == 1
    assert "overall" in failures[0]


def test_regression_gate_passes_within_tolerance(tmp_path):
    gate = RegressionGate(baseline_path=tmp_path / "baseline.json", tolerance=0.05)
    gate.save_baseline({"overall": 0.85})

    passed, failures = gate.check({"overall": 0.83})

    assert passed is True
    assert failures == []

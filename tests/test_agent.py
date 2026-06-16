from app.agents.agent_graph import LangGraphAgent, _reformulate_query
from app.agents.llm_client import MockLLMClient
from app.agents.rag_chain import RAGChain
from app.ingestion.chunking import Chunk
from app.retrieval.embeddings import get_default_embedding_model
from app.retrieval.vectorstore import BM25Index, HybridRetriever, VectorStore


SAMPLE_CHUNKS = [
    Chunk(chunk_id="c1", doc_id="d1", title="Pruning Paper", source="S", year=2026,
          text="The pruning framework achieves a 5.1x MAC reduction from 125.75 to 24.86 MMACs on ResNet-56.",
          chunk_index=0),
    Chunk(chunk_id="c2", doc_id="d2", title="MLOps Paper", source="S", year=2024,
          text="Drift detection using PSI and the KS-test triggers automated Airflow retraining pipelines.",
          chunk_index=0),
]


def _build_chain(top_k=2):
    embedding_model = get_default_embedding_model(corpus_for_fit=[c.text for c in SAMPLE_CHUNKS])
    vs = VectorStore(embedding_model)
    vs.build(SAMPLE_CHUNKS)
    bm25 = BM25Index()
    bm25.build(SAMPLE_CHUNKS)
    retriever = HybridRetriever(vs, bm25)
    return RAGChain(retriever, llm_client=MockLLMClient(), top_k=top_k)


def test_rag_chain_returns_grounded_answer():
    chain = _build_chain()

    result = chain.run("What MAC reduction was achieved on ResNet-56?")

    assert "5.1x" in result.answer or "MAC" in result.answer
    assert result.confidence > 0
    assert len(result.retrieved_chunks) > 0
    assert result.model_name == "mock-extractive-v1"


def test_rag_chain_handles_query_with_no_relevant_context():
    chain = _build_chain()

    result = chain.run("zzzzz nonexistent gibberish term")

    assert result.confidence < 0.5


def test_reformulate_query_strips_stopwords():
    reformulated = _reformulate_query("What is the MAC reduction achieved by the pruning framework?")

    tokens = reformulated.lower().split()
    assert "what" not in tokens
    assert "is" not in tokens
    assert "mac" in reformulated.lower() or "reduction" in reformulated.lower()


def test_agent_returns_high_confidence_for_relevant_query():
    chain = _build_chain()
    agent = LangGraphAgent(chain, confidence_threshold=0.1, max_attempts=2)

    result = agent.run("What MAC reduction was achieved on ResNet-56?")

    assert result.attempts == 1
    assert result.low_confidence is False


def test_agent_retries_on_low_confidence_query():
    chain = _build_chain()
    agent = LangGraphAgent(chain, confidence_threshold=0.99, max_attempts=3)

    result = agent.run("Tell me about something completely unrelated")

    assert result.attempts == 3
    assert result.low_confidence is True

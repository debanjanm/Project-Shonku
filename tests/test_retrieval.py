from langchain_core.documents import Document

from backend.retrieval import _filter_by_period, _parse_query_period, _rrf_merge, tokenize


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("What was the Revenue in Q1?") == ["revenue", "q1"]


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_parse_query_period_requires_both_year_and_quarter():
    assert _parse_query_period("Apple revenue in Q1 2026") == (2026, 1)
    assert _parse_query_period("Apple revenue in 2026") is None
    assert _parse_query_period("Apple revenue in Q1") is None
    assert _parse_query_period("Apple revenue") is None


def test_filter_by_period_keeps_matching_and_untagged_docs():
    docs = [
        Document("a", metadata={"fiscal_year": 2026, "fiscal_quarter": 1}),
        Document("b", metadata={"fiscal_year": 2026, "fiscal_quarter": 2}),
        Document("c", metadata={}),  # non-quarterly KB, no fiscal metadata at all
    ]
    kept = _filter_by_period(docs, 2026, 1)
    assert [d.page_content for d in kept] == ["a", "c"]


def test_rrf_merge_ranks_docs_present_in_multiple_lists_higher():
    doc_a = Document("a", metadata={"source": "a.pdf", "chunk_index": 0})
    doc_b = Document("b", metadata={"source": "b.pdf", "chunk_index": 0})
    doc_c = Document("c", metadata={"source": "c.pdf", "chunk_index": 0})

    dense = [doc_a, doc_b]  # a ranks higher than b
    sparse = [doc_b, doc_c]  # b ranks higher than c

    merged = _rrf_merge([dense, sparse], k=3)

    # b appears (rank 1) in both lists, so it should out-score a and c, which
    # each appear in only one list.
    assert [d.page_content for d in merged] == ["b", "a", "c"]


def test_rrf_merge_dedups_same_doc_across_lists():
    doc_a = Document("a", metadata={"source": "a.pdf", "chunk_index": 0})
    merged = _rrf_merge([[doc_a], [doc_a]], k=5)
    assert len(merged) == 1


def test_rrf_merge_respects_k():
    docs = [Document(str(i), metadata={"source": "x.pdf", "chunk_index": i}) for i in range(5)]
    merged = _rrf_merge([docs], k=2)
    assert len(merged) == 2

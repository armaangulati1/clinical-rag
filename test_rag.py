from rag import reciprocal_rank_fusion, chunk_text, answer_cited

def test_rrf_ranks_items_in_both_lists_first():
    a = [(1,"d","s","x"), (2,"d","s","y")]
    b = [(2,"d","s","y"), (3,"d","s","z")]
    ids = [r[0] for r in reciprocal_rank_fusion([a, b])]
    assert ids[0] == 2                       # in both lists -> top
    assert set(ids) == {1, 2, 3}

def test_chunking_respects_size():
    chunks = chunk_text("a" * 2000, size=800, overlap=100)
    assert chunks and all(len(c) <= 800 for c in chunks)

def test_out_of_corpus_cites_nothing():
    _, cites = answer_cited("What is the dose of ibuprofen?")   # not in our 5 drugs
    assert cites == []

def test_in_corpus_cites_the_right_drug():
    ans, cites = answer_cited("What are adverse reactions to metoprolol?")
    assert ans.strip() and len(cites) > 0
    assert any(c["drug"].lower() == "metoprolol" for c in cites)

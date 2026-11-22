from benchmarks.squad_retrieval import answer_in_chunks, normalize, reciprocal_rank


def test_normalize():
    assert normalize("The U.S.A., 1970!") == "the u s a 1970"


def test_answer_in_chunks():
    hits = [{"chunk": "The code is L-7781 and it opens at 06:30."}]
    assert answer_in_chunks("L-7781", hits)
    assert not answer_in_chunks("W-9900", hits)


def test_reciprocal_rank():
    hits = [{"source": "a"}, {"source": "b"}]
    assert reciprocal_rank("a", hits) == 1.0
    assert reciprocal_rank("b", hits) == 0.5
    assert reciprocal_rank("c", hits) == 0.0

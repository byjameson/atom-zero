from benchmarks.answerability_eval import features


def test_features_extract_lexical_and_type_signals():
    hit = {
        "chunk": "The launch code is L-7781. It was set in 2024.",
        "title": "Launch",
        "section": "Codes",
        "rerank_score": 0.8,
        "embedding_score": 0.7,
    }
    result = features("what is the launch code?", hit)
    assert result["terms"] >= 2
    assert result["coverage_chunk"] > 0
    assert result["top_score"] == 0.8
    assert result["embedding"] == 0.7
    assert result["novel_numbers"] >= 1
    assert result["novel_years"] >= 1


def test_features_flag_answer_type_mismatch():
    hit = {
        "chunk": "The meeting happened in 2024.",
        "title": "Meeting",
        "section": "",
        "rerank_score": 0.4,
        "embedding_score": 0.2,
    }
    result = features("who attended the meeting?", hit)
    assert result["who"] is True
    assert result["type_answer"] is False

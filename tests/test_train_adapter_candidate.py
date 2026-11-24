import torch

from research.train_adapter_candidate import _token_weights


class _Tokenizer:
    def decode(self, ids):
        return {1: "hi", 2: "{", 3: "Ada", 4: "}"}[int(ids[0])]


def test_token_weights_only_scale_scored_content_tokens():
    y = torch.tensor([[1, 2, 3, 4, -100]])
    weights = _token_weights(_Tokenizer(), y, 5.0)
    assert weights.tolist() == [[5.0, 1.0, 5.0, 1.0, 0.0]]


def test_token_weights_default_is_plain_cross_entropy():
    y = torch.tensor([[1, 2, 3, -100]])
    weights = _token_weights(_Tokenizer(), y, 1.0)
    assert weights.tolist() == [[1.0, 1.0, 1.0, 0.0]]

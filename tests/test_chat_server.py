import sys

import pytest

from atomzero import __version__
from atomzero.chat_server import chat_prompt, main


def test_chat_prompt_uses_training_format():
    prompt = chat_prompt([{"role": "user", "content": "hello"}])
    assert prompt == "<|user|> hello <|assistant|>"


def test_chat_prompt_includes_the_last_turns():
    prompt = chat_prompt([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "how are you?"},
    ])
    assert "<|user|> hi" in prompt
    assert "<|assistant|> Hello!" in prompt
    assert prompt.endswith("<|assistant|>")


def test_version_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["atomzero", "--version"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out

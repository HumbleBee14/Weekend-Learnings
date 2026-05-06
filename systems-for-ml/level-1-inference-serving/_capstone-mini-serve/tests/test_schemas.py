"""Tests that don't need the model loaded — fast, run on every commit."""

import pytest

from mini_serve.schemas import GenerateRequest


def test_generate_request_validates_max_tokens():
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", max_tokens=0)
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", max_tokens=999999)


def test_generate_request_validates_temperature():
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", temperature=-0.1)
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", temperature=2.5)


def test_generate_request_rejects_empty_prompt():
    with pytest.raises(ValueError):
        GenerateRequest(prompt="")


def test_generate_request_defaults():
    r = GenerateRequest(prompt="hello")
    assert r.max_tokens == 128
    assert r.temperature == 0.7
    assert r.stream is False

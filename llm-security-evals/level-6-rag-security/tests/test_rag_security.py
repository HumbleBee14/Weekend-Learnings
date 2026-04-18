"""Tests for RAG security defenses.

Verifies:
- Poisoned docs are caught at ingestion
- Injection payloads are stripped from context
- Auth prevents cross-user data leakage
"""
import pytest

# TODO: implement

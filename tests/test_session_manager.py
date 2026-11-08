
from pathlib import Path
from types import SimpleNamespace

import pytest

from session_manager import get_session_vectorstore_dir


def test_different_sessions_get_different_directories(tmp_path):
    request_a = SimpleNamespace(session_hash="session-a")
    request_b = SimpleNamespace(session_hash="session-b")

    path_a = get_session_vectorstore_dir(request_a, tmp_path)
    path_b = get_session_vectorstore_dir(request_b, tmp_path)

    assert path_a != path_b
    assert path_a.exists()
    assert path_b.exists()


def test_same_session_gets_same_directory(tmp_path):
    request = SimpleNamespace(session_hash="session-a")

    first = get_session_vectorstore_dir(request, tmp_path)
    second = get_session_vectorstore_dir(request, tmp_path)

    assert first == second


def test_raw_session_hash_is_not_used_in_path(tmp_path):
    session_hash = "private-session-123"
    request = SimpleNamespace(session_hash=session_hash)

    path = get_session_vectorstore_dir(request, tmp_path)

    assert session_hash not in str(path)


def test_missing_session_hash_is_rejected(tmp_path):
    request = SimpleNamespace(session_hash=None)

    with pytest.raises(ValueError):
        get_session_vectorstore_dir(request, tmp_path)


def test_blank_session_hash_is_rejected(tmp_path):
    request = SimpleNamespace(session_hash="   ")

    with pytest.raises(ValueError):
        get_session_vectorstore_dir(request, tmp_path)

def test_path_traversal_session_hash_stays_inside_sessions_dir(tmp_path):
    request = SimpleNamespace(session_hash="../../outside/../../attacker")

    path = get_session_vectorstore_dir(request, tmp_path)

    assert path.is_relative_to(tmp_path.resolve())
    assert path.exists()
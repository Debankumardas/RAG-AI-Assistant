
import hashlib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SESSIONS_DIR = BASE_DIR / "sessions"


def get_session_vectorstore_dir(request, sessions_dir=None):
    """
    Return the vectorstore directory belonging to one Gradio session.

    The raw session hash is never used as a filesystem path.
    """

    session_hash = getattr(request, "session_hash", None)

    if not isinstance(session_hash, str) or not session_hash.strip():
        raise ValueError("A valid Gradio session hash is required.")

    session_hash = session_hash.strip()

    hashed_session_id = hashlib.sha256(
        session_hash.encode("utf-8")
    ).hexdigest()

    root = Path(sessions_dir or DEFAULT_SESSIONS_DIR).resolve()

    vectorstore_dir = (
        root / hashed_session_id / "vectorstore"
    ).resolve()

    if not vectorstore_dir.is_relative_to(root):
        raise ValueError("Invalid session storage path.")

    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    return vectorstore_dir

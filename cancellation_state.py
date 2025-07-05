"""
Shared cancellation state module to avoid circular imports between server.py and connector.py
"""

import threading
import logging

logger = logging.getLogger(__name__)

# Thread-safe cancellation flags storage
_cancellation_lock = threading.RLock()
_cancellation_flags = {}

def set_cancellation_flag(session_id: str) -> None:
    """Set cancellation flag for a session"""
    if not session_id:
        return
    
    with _cancellation_lock:
        _cancellation_flags[session_id] = True
        logger.info(f"Cancellation flag set for session: {session_id}")

def is_generation_cancelled(session_id: str) -> bool:
    """Check if generation has been cancelled for a session"""
    if not session_id:
        return False
    
    with _cancellation_lock:
        return _cancellation_flags.get(session_id, False)

def clear_cancellation_flag(session_id: str) -> None:
    """Clear cancellation flag for a session"""
    if not session_id:
        return
    
    with _cancellation_lock:
        if session_id in _cancellation_flags:
            del _cancellation_flags[session_id]
            logger.info(f"Cancellation flag cleared for session: {session_id}")

def get_all_cancelled_sessions() -> list:
    """Get list of all cancelled session IDs (for debugging)"""
    with _cancellation_lock:
        return list(_cancellation_flags.keys())
"""
Small helpers used across the codebase.
"""

import logging

from realtimertc import config


def generate_id(prefix="evt"):
    """Monotonically incrementing ID (e.g. 'evt_0000000000000001')."""
    config._id_counter += 1
    return f"{prefix}_{config._id_counter:016x}"


def channel_open(channel):
    """True if the DataChannel is connected and ready to send."""
    return channel is not None and channel.readyState == "open"


def trim_history(history, max_length=config.MAX_HISTORY_LENGTH):
    """Trim conversation history, always preserving the system message."""
    if len(history) <= max_length:
        return
    system_msgs = [m for m in history if m.get("role") == "system"]
    other_msgs  = [m for m in history if m.get("role") != "system"]
    overflow = len(other_msgs) - (max_length - len(system_msgs))
    if overflow > 0:
        del other_msgs[:overflow]
    history[:] = system_msgs + other_msgs


def cleanup_session(session_id, reason=""):
    """Cancel any in-flight response task and remove the session."""
    session_data = config.active_sessions.pop(session_id, None)
    if session_data:
        task = session_data.get("response_task")
        if task and not task.done():
            task.cancel()
        config.pcs.discard(session_data.get("pc"))
        logging.info(f"[{session_id}] Session cleaned up. {reason}")
    return session_data

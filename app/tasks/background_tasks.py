"""
Background task management.
"""
from typing import Dict, Any

processing_tasks: Dict[str, Dict[str, Any]] = {}


def create_task(task_id: str):
    """Create new task."""
    processing_tasks[task_id] = {
        "status": "pending",
        "message": "Task created",
        "result": None,
    }


def get_task(task_id: str) -> Dict[str, Any] | None:
    """Get task by ID."""
    return processing_tasks.get(task_id)
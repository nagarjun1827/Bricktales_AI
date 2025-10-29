from typing import Dict, Any

processing_tasks: Dict[str, Dict[str, Any]] = {}

def create_task(task_id: str):
    processing_tasks[task_id] = {
        "status": "pending",
        "message": "Task created",
        "result": None,
    }

def get_task(task_id: str) -> Dict[str, Any] | None:
    return processing_tasks.get(task_id)
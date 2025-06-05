import glob
import json
import shutil
import os
from datetime import datetime

from config.constants import TASKS_FILE, TEMP_DIR

# Global dictionary to store running task references
_running_tasks = {}


def clear_temp_files():
    """Clear all temp files including tasks"""
    try:
        # Clear tasks file
        if os.path.exists(TASKS_FILE):
            os.remove(TASKS_FILE)

        # Clear Temp Directory
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)

        # Create Temp Directory
        os.makedirs(TEMP_DIR, exist_ok=True)
    except:
        pass


def get_past_generated_files():
    """Get list of past generated audiobook files"""
    try:
        audiobooks_dir = "generated_audiobooks"
        if not os.path.exists(audiobooks_dir):
            return []

        # Get all files in the generated_audiobooks directory
        pattern = os.path.join(audiobooks_dir, "*")
        files = glob.glob(pattern)

        # Filter out directories, hidden files (starting with .), and get file info
        audiobook_files = []
        for file_path in files:
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                # Skip hidden files (starting with a period)
                if filename.startswith('.'):
                    continue
                # Get file size and modification time
                stat_info = os.stat(file_path)
                size_mb = round(stat_info.st_size / (1024 * 1024), 2)
                mod_time = datetime.fromtimestamp(stat_info.st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                )

                audiobook_files.append(
                    {
                        "path": file_path,
                        "filename": filename,
                        "size_mb": size_mb,
                        "modified": mod_time,
                    }
                )

        # Sort by modification time (newest first)
        audiobook_files.sort(key=lambda x: os.path.getmtime(x["path"]), reverse=True)
        return audiobook_files
    except Exception as e:
        print(f"Error getting past files: {e}")
        return []


def load_tasks():
    """Load current tasks from file"""
    try:
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, "r") as f:
                return json.load(f)
    except:
        pass
    return {}


def save_tasks(tasks):
    """Save tasks to file"""
    try:
        with open(TASKS_FILE, "w") as f:
            json.dump(tasks, f)
    except:
        pass


def update_task_status(task_id, status, progress="", error=None, params=None):
    """Update task status and optionally store parameters"""
    tasks = load_tasks()
    existing_task = tasks.get(task_id, {})
    
    # Preserve existing progress if no new progress is provided and we're marking as failed/cancelled
    if not progress and status in ["failed", "cancelled"] and existing_task.get("progress"):
        progress = existing_task.get("progress", "")
        print(f"[DEBUG] Preserving progress for {status} task: {progress}")
    
    if task_id in tasks:
        # Merge params if already present
        if params:
            old_params = tasks[task_id].get("params", {})
            params = {**old_params, **params}
    
    tasks[task_id] = {
        "status": status,
        "progress": progress,
        "timestamp": datetime.now().isoformat(),
        "error": error,
        "params": params or existing_task.get("params", {}),
    }
    save_tasks(tasks)


def register_running_task(task_id, task_ref):
    """Register a running task for potential cancellation"""
    global _running_tasks
    _running_tasks[task_id] = task_ref


def unregister_running_task(task_id):
    """Unregister a running task"""
    global _running_tasks
    if task_id in _running_tasks:
        del _running_tasks[task_id]


def cancel_task(task_id):
    """Cancel a running task"""
    global _running_tasks

    # Check if task is currently running
    if task_id in _running_tasks:
        task_ref = _running_tasks[task_id]
        try:
            # Cancel the asyncio task
            task_ref.cancel()
            # Update task status
            update_task_status(task_id, "cancelled", "Task cancelled by user")
            # Remove from running tasks
            unregister_running_task(task_id)
            return True, "Task cancelled successfully"
        except Exception as e:
            return False, f"Error cancelling task: {str(e)}"
    else:
        # Check if task exists in task file
        tasks = load_tasks()
        if task_id in tasks:
            # Mark as cancelled in the task file
            update_task_status(task_id, "cancelled", "Task cancelled by user")
            return True, "Task marked as cancelled"
        else:
            return False, "Task not found"


def is_task_cancelled(task_id):
    """Check if a task has been cancelled"""
    tasks = load_tasks()
    if task_id in tasks:
        return tasks[task_id].get("status") == "cancelled"
    return False


def get_active_tasks():
    """Get list of active generation tasks"""
    tasks = load_tasks()
    active_tasks = []
    current_time = datetime.now()

    for task_id, task_info in tasks.items():
        try:
            timestamp = datetime.fromisoformat(task_info["timestamp"])
            hours_old = (current_time - timestamp).total_seconds() / 3600

            if task_info.get("status") in ["running", "starting", "generating", "resuming", "failed"]:
                # Check if task is still actually running (simple heuristic)
                if hours_old < 5:  # 5 hours timeout (changed from minutes)
                    active_tasks.append(
                        {
                            "id": task_id,
                            "progress": task_info.get("progress", ""),
                            "timestamp": task_info["timestamp"],
                            "status": task_info.get("status", "running"),
                        }
                    )

        except:
            pass

    return active_tasks


def remove_task(task_id):
    """Remove a task from the tasks file"""
    tasks = load_tasks()
    if task_id in tasks:
        del tasks[task_id]
        save_tasks(tasks)


def get_task_progress_index(task_id):
    """Get the last completed line/chapter index for a task (if any)"""
    tasks = load_tasks()
    if task_id in tasks:
        progress = tasks[task_id].get("progress", "")
        import re

        m = re.search(r"Progress: (\d+)/(\d+)", progress)
        if m:
            return int(m.group(1)), int(m.group(2))
    return 0, 0


def set_task_progress_index(task_id, index, total):
    """Set the last completed line/chapter index for a task"""
    tasks = load_tasks()
    if task_id in tasks:
        tasks[task_id]["progress"] = f"Generating audiobook. Progress: {index}/{total}"
        save_tasks(tasks)

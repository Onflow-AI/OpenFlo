"""
Checklist utility functions for status computation and formatting.
Only contains functions actually used by ChecklistManager.
"""


def get_checklist_status(task_checklist):
    """
    Compute checklist status and progress metrics from a list of items.
    """
    if not task_checklist:
        return {
            "total": 0,
            "completed": 0,
            "in_progress": 0,
            "pending": 0,
            "failed": 0,
            "progress": 0.0,
            "completion_rate": 0.0,
            "success_rate": 0.0
        }

    status_counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}

    for item in task_checklist:
        status = (item.get('status', 'pending') or 'pending').strip().lower().replace(' ', '_')
        status_counts[status] = status_counts.get(status, 0) + 1

    total = len(task_checklist)
    completed = status_counts['completed']
    failed = status_counts['failed']

    progress = (completed / total * 100) if total > 0 else 0
    completion_rate = (completed / total) if total > 0 else 0
    success_rate = (completed / (completed + failed)) if (completed + failed) > 0 else 0.0

    return {
        "total": total,
        "completed": completed,
        "in_progress": status_counts['in_progress'],
        "pending": status_counts['pending'],
        "failed": failed,
        "progress": progress,
        "completion_rate": completion_rate,
        "success_rate": success_rate
    }


def format_checklist_for_prompt(task_checklist):
    """
    Format checklist for inclusion in LLM prompts.
    """
    if not task_checklist:
        return "No checklist available"

    status = get_checklist_status(task_checklist)

    checklist_str = "TASK CHECKLIST:\n"
    checklist_str += "=" * 50 + "\n"

    for i, item in enumerate(task_checklist, 1):
        item_status = item.get('status', 'pending')
        description = item.get('description', '')
        item_id = item.get('id', f'requirement_{i}')

        status_indicator = {
            'pending': '[PENDING]',
            'in_progress': '[IN_PROGRESS]',
            'completed': '[COMPLETED]',
            'failed': '[FAILED]'
        }.get(item_status, '[PENDING]')
        checklist_str += f"{i:2d}. {status_indicator} {item_id}: {description}\n"

    checklist_str += "=" * 50 + "\n"
    checklist_str += f"PROGRESS SUMMARY:\n"
    checklist_str += f"  Total Items: {status['total']}\n"
    checklist_str += f"  Completed: {status['completed']} ({status['progress']:.1f}%)\n"
    checklist_str += f"  In Progress: {status['in_progress']}\n"
    checklist_str += f"  Pending: {status['pending']}\n"
    checklist_str += f"  Failed: {status['failed']}\n"
    checklist_str += f"  Success Rate: {status['success_rate']:.1%}\n"
    checklist_str += "=" * 50

    return checklist_str

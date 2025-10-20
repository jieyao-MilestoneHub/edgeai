"""
Annotation Manager for Human-in-the-Loop (HITL) preference labeling.

This module manages the annotation workflow for creating preference pairs.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AnnotationTask:
    """Represents a single annotation task."""
    task_id: str
    prompt: str
    responses: List[str]
    metadata: Optional[Dict] = None
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class PreferencePair:
    """Represents an annotated preference pair."""
    task_id: str
    prompt: str
    chosen: str
    rejected: str
    annotator: str
    annotation_time: str
    confidence: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return asdict(self)


class AnnotationManager:
    """Manages annotation tasks and results.

    This class handles:
    - Loading/saving annotation tasks
    - Tracking annotation progress
    - Managing multiple annotators
    - Quality control and validation
    """

    def __init__(
        self,
        tasks_file: str,
        output_file: str,
        progress_file: Optional[str] = None
    ):
        """Initialize annotation manager.

        Args:
            tasks_file: Path to file containing annotation tasks (JSONL)
            output_file: Path to save annotated preference pairs (JSONL)
            progress_file: Path to save annotation progress
        """
        self.tasks_file = Path(tasks_file)
        self.output_file = Path(output_file)
        self.progress_file = Path(progress_file) if progress_file else Path(output_file).with_suffix('.progress.json')

        self.tasks: List[AnnotationTask] = []
        self.annotations: List[PreferencePair] = []
        self.progress: Dict = {}

        # Create directories
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def load_tasks(self) -> int:
        """Load annotation tasks from file.

        Returns:
            Number of tasks loaded
        """
        if not self.tasks_file.exists():
            logger.warning(f"Tasks file not found: {self.tasks_file}")
            return 0

        self.tasks = []
        with open(self.tasks_file, 'r', encoding='utf-8') as f:
            for line in f:
                task_dict = json.loads(line.strip())
                task = AnnotationTask(**task_dict)
                self.tasks.append(task)

        logger.info(f"✅ Loaded {len(self.tasks)} annotation tasks")
        return len(self.tasks)

    def load_progress(self) -> None:
        """Load annotation progress."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                self.progress = json.load(f)
            logger.info(f"📊 Loaded progress: {len(self.progress.get('completed', []))} completed")
        else:
            self.progress = {
                'completed': [],
                'skipped': [],
                'last_task_index': 0
            }

    def save_progress(self) -> None:
        """Save annotation progress."""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2)

    def save_annotation(self, annotation: PreferencePair) -> None:
        """Save a single annotation.

        Args:
            annotation: Annotated preference pair
        """
        # Append to output file
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(annotation.to_dict()) + '\n')

        # Update progress
        self.progress['completed'].append(annotation.task_id)
        self.save_progress()

        logger.info(f"✅ Saved annotation for task: {annotation.task_id}")

    def get_next_task(self) -> Optional[AnnotationTask]:
        """Get next unannotated task.

        Returns:
            Next task to annotate, or None if all done
        """
        completed = set(self.progress.get('completed', []))
        skipped = set(self.progress.get('skipped', []))

        for task in self.tasks:
            if task.task_id not in completed and task.task_id not in skipped:
                return task

        return None

    def skip_task(self, task_id: str) -> None:
        """Mark a task as skipped.

        Args:
            task_id: ID of task to skip
        """
        if task_id not in self.progress.get('skipped', []):
            self.progress['skipped'].append(task_id)
            self.save_progress()
            logger.info(f"⏭️  Skipped task: {task_id}")

    def get_statistics(self) -> Dict:
        """Get annotation statistics.

        Returns:
            Dictionary with statistics
        """
        total = len(self.tasks)
        completed = len(self.progress.get('completed', []))
        skipped = len(self.progress.get('skipped', []))
        remaining = total - completed - skipped

        return {
            'total_tasks': total,
            'completed': completed,
            'skipped': skipped,
            'remaining': remaining,
            'completion_rate': completed / total if total > 0 else 0,
            'progress_percentage': (completed / total * 100) if total > 0 else 0
        }

    def validate_annotations(self) -> Tuple[int, List[str]]:
        """Validate all saved annotations.

        Returns:
            Tuple of (valid_count, list_of_errors)
        """
        if not self.output_file.exists():
            return 0, ["Output file does not exist"]

        valid_count = 0
        errors = []

        with open(self.output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    annotation = json.loads(line.strip())

                    # Validate required fields
                    required = ['task_id', 'prompt', 'chosen', 'rejected', 'annotator']
                    missing = [field for field in required if field not in annotation]

                    if missing:
                        errors.append(f"Line {i}: Missing fields {missing}")
                    else:
                        valid_count += 1

                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: Invalid JSON - {e}")

        return valid_count, errors

    def export_for_training(self, output_path: str) -> int:
        """Export annotations in DPO training format.

        Args:
            output_path: Path to save training-ready dataset

        Returns:
            Number of pairs exported
        """
        if not self.output_file.exists():
            logger.error("No annotations to export")
            return 0

        export_path = Path(output_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(self.output_file, 'r', encoding='utf-8') as f_in, \
             open(export_path, 'w', encoding='utf-8') as f_out:

            for line in f_in:
                annotation = json.loads(line.strip())

                # Convert to DPO format
                dpo_pair = {
                    'prompt': annotation['prompt'],
                    'chosen': annotation['chosen'],
                    'rejected': annotation['rejected']
                }

                f_out.write(json.dumps(dpo_pair, ensure_ascii=False) + '\n')
                count += 1

        logger.info(f"✅ Exported {count} preference pairs to: {export_path}")
        return count


def create_annotation_tasks_from_prompts(
    prompts_file: str,
    output_file: str,
    num_responses: int = 3
) -> int:
    """Create annotation tasks from a list of prompts.

    This is a helper function to generate tasks that will be annotated.

    Args:
        prompts_file: File containing prompts (one per line)
        output_file: Output file for tasks (JSONL)
        num_responses: Number of responses to generate per prompt

    Returns:
        Number of tasks created
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(prompts_file, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            prompt = line.strip()
            if not prompt:
                continue

            task = AnnotationTask(
                task_id=str(uuid.uuid4()),
                prompt=prompt,
                responses=[f"Response {i+1} (to be generated)" for i in range(num_responses)],
                metadata={'source': prompts_file}
            )

            f_out.write(json.dumps(asdict(task)) + '\n')
            count += 1

    logger.info(f"✅ Created {count} annotation tasks in: {output_path}")
    return count

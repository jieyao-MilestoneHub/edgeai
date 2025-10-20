"""
Human-in-the-Loop (HITL) Preference Annotation Tool

This is an interactive CLI tool for annotating preference pairs for DPO training.

Features:
- Load tasks from file or generate from model
- Interactive annotation interface
- Progress tracking and resume support
- Quality control and validation
- Export to DPO training format

Usage:
    # Annotate from pre-generated tasks
    python annotate_preferences.py --tasks tasks.jsonl --output annotations.jsonl

    # Generate responses and annotate
    python annotate_preferences.py --model path/to/model --prompts prompts.txt --output annotations.jsonl

    # Resume previous session
    python annotate_preferences.py --tasks tasks.jsonl --output annotations.jsonl --resume
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preference.annotation_manager import (
    AnnotationManager,
    PreferencePair,
    AnnotationTask,
    create_annotation_tasks_from_prompts
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PreferenceAnnotator:
    """Interactive preference annotation tool."""

    def __init__(
        self,
        manager: AnnotationManager,
        annotator_name: str = "default"
    ):
        """Initialize annotator.

        Args:
            manager: AnnotationManager instance
            annotator_name: Name of current annotator
        """
        self.manager = manager
        self.annotator_name = annotator_name

    def display_task(self, task: AnnotationTask) -> None:
        """Display annotation task in a formatted way.

        Args:
            task: Task to display
        """
        print("\n" + "="*80)
        print(f"📝 Task ID: {task.task_id}")
        print("="*80)
        print(f"\n💬 Prompt:\n{task.prompt}\n")
        print("-"*80)

        for i, response in enumerate(task.responses, 1):
            print(f"\n[{i}] Response {i}:")
            print(f"{response}")
            print()

        print("-"*80)

    def get_user_choice(self, num_options: int) -> Optional[str]:
        """Get user's annotation choice.

        Args:
            num_options: Number of response options

        Returns:
            User's choice or None if skipped
        """
        print("\n📊 Actions:")
        print("  - Enter two numbers (e.g., '1 3') to select chosen and rejected")
        print("  - Enter 's' to skip this task")
        print("  - Enter 'q' to quit and save progress")
        print("  - Enter 'stats' to view statistics")

        while True:
            choice = input("\n> Your choice: ").strip().lower()

            if choice == 's':
                return 'skip'
            elif choice == 'q':
                return 'quit'
            elif choice == 'stats':
                self.show_statistics()
                continue

            # Parse two numbers
            parts = choice.split()
            if len(parts) == 2:
                try:
                    chosen_idx = int(parts[0])
                    rejected_idx = int(parts[1])

                    if 1 <= chosen_idx <= num_options and 1 <= rejected_idx <= num_options:
                        if chosen_idx == rejected_idx:
                            print("❌ Chosen and rejected must be different!")
                            continue
                        return f"{chosen_idx},{rejected_idx}"
                    else:
                        print(f"❌ Numbers must be between 1 and {num_options}")
                except ValueError:
                    print("❌ Invalid input! Please enter two numbers.")
            else:
                print("❌ Invalid input! Please enter two numbers (e.g., '1 3')")

    def annotate_task(self, task: AnnotationTask) -> Optional[PreferencePair]:
        """Annotate a single task.

        Args:
            task: Task to annotate

        Returns:
            PreferencePair if annotated, None if skipped/quit
        """
        self.display_task(task)

        choice = self.get_user_choice(len(task.responses))

        if choice == 'skip':
            self.manager.skip_task(task.task_id)
            return None

        if choice == 'quit':
            return 'quit'

        # Parse choice
        chosen_idx, rejected_idx = map(int, choice.split(','))

        # Optional: Get confidence and notes
        print("\n💭 Optional:")
        confidence_str = input("  Confidence (0-1, press Enter to skip): ").strip()
        confidence = float(confidence_str) if confidence_str else None

        notes = input("  Notes (press Enter to skip): ").strip()
        notes = notes if notes else None

        # Create preference pair
        from datetime import datetime
        pair = PreferencePair(
            task_id=task.task_id,
            prompt=task.prompt,
            chosen=task.responses[chosen_idx - 1],
            rejected=task.responses[rejected_idx - 1],
            annotator=self.annotator_name,
            annotation_time=datetime.now().isoformat(),
            confidence=confidence,
            notes=notes
        )

        return pair

    def show_statistics(self) -> None:
        """Display annotation statistics."""
        stats = self.manager.get_statistics()

        print("\n" + "="*60)
        print("📊 Annotation Statistics")
        print("="*60)
        print(f"Total Tasks:    {stats['total_tasks']}")
        print(f"Completed:      {stats['completed']} ✅")
        print(f"Skipped:        {stats['skipped']} ⏭️")
        print(f"Remaining:      {stats['remaining']} 📝")
        print(f"Progress:       {stats['progress_percentage']:.1f}%")
        print("="*60)

    def run(self) -> None:
        """Run the annotation session."""
        logger.info("\n" + "="*60)
        logger.info("🏷️  Human-in-the-Loop Preference Annotation Tool")
        logger.info("="*60)

        # Load tasks and progress
        task_count = self.manager.load_tasks()
        if task_count == 0:
            logger.error("No tasks to annotate!")
            return

        self.manager.load_progress()
        self.show_statistics()

        logger.info(f"\n👤 Annotator: {self.annotator_name}")
        logger.info("🚀 Starting annotation session...\n")

        # Main annotation loop
        annotated_count = 0
        while True:
            task = self.manager.get_next_task()

            if task is None:
                logger.info("\n🎉 All tasks completed!")
                break

            result = self.annotate_task(task)

            if result == 'quit':
                logger.info("\n💾 Saving progress and quitting...")
                break

            if result is not None:
                # Save annotation
                self.manager.save_annotation(result)
                annotated_count += 1
                print(f"\n✅ Annotation saved! ({annotated_count} in this session)")

        # Final statistics
        self.show_statistics()

        # Validation
        logger.info("\n🔍 Validating annotations...")
        valid_count, errors = self.manager.validate_annotations()

        if errors:
            logger.warning(f"⚠️  Found {len(errors)} validation errors:")
            for error in errors[:5]:  # Show first 5
                logger.warning(f"  - {error}")
        else:
            logger.info(f"✅ All {valid_count} annotations are valid!")

        logger.info(f"\n📁 Annotations saved to: {self.manager.output_file}")
        logger.info(f"📁 Progress saved to: {self.manager.progress_file}")


def generate_responses_with_model(
    model_path: str,
    prompts: List[str],
    num_responses: int = 3
) -> List[AnnotationTask]:
    """Generate responses using a language model.

    Args:
        model_path: Path to the model
        prompts: List of prompts
        num_responses: Number of responses per prompt

    Returns:
        List of annotation tasks with generated responses
    """
    logger.info(f"🤖 Loading model from: {model_path}")

    # TODO: Implement model loading and generation
    # This is a placeholder - actual implementation would use transformers

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import uuid

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    tasks = []
    for prompt in prompts:
        logger.info(f"Generating responses for: {prompt[:50]}...")

        responses = []
        for i in range(num_responses):
            # Generate with different sampling parameters
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7 + i * 0.2,  # Vary temperature
                    top_p=0.9,
                    do_sample=True
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()  # Remove prompt
            responses.append(response)

        task = AnnotationTask(
            task_id=str(uuid.uuid4()),
            prompt=prompt,
            responses=responses,
            metadata={'model': model_path, 'num_responses': num_responses}
        )
        tasks.append(task)

    logger.info(f"✅ Generated {len(tasks)} annotation tasks")
    return tasks


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Human-in-the-Loop Preference Annotation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate from pre-generated tasks
  python annotate_preferences.py --tasks tasks.jsonl --output annotations.jsonl

  # Generate responses and annotate
  python annotate_preferences.py --model experiments/sft/r16 --prompts prompts.txt --output annotations.jsonl

  # Resume previous session
  python annotate_preferences.py --tasks tasks.jsonl --output annotations.jsonl --resume
        """
    )

    parser.add_argument(
        "--tasks",
        type=str,
        help="Path to annotation tasks file (JSONL)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save annotations (JSONL)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model for generating responses"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        help="Path to prompts file (one per line)"
    )
    parser.add_argument(
        "--num-responses",
        type=int,
        default=3,
        help="Number of responses to generate per prompt (default: 3)"
    )
    parser.add_argument(
        "--annotator",
        type=str,
        default="default",
        help="Name of annotator (default: 'default')"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous annotation session"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export annotations to DPO training format"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.tasks and not (args.model and args.prompts):
        parser.error("Either --tasks or (--model and --prompts) must be provided")

    # Export mode
    if args.export:
        logger.info("📤 Export mode")
        manager = AnnotationManager(args.tasks, args.output)
        count = manager.export_for_training(args.export)
        logger.info(f"✅ Exported {count} preference pairs")
        return

    # Generate tasks if needed
    if args.model and args.prompts:
        logger.info("🤖 Generating responses with model...")

        # Load prompts
        with open(args.prompts, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        # Generate responses
        tasks = generate_responses_with_model(args.model, prompts, args.num_responses)

        # Save tasks
        tasks_file = Path(args.output).with_suffix('.tasks.jsonl')
        import json
        from dataclasses import asdict
        with open(tasks_file, 'w', encoding='utf-8') as f:
            for task in tasks:
                f.write(json.dumps(asdict(task)) + '\n')

        logger.info(f"✅ Saved tasks to: {tasks_file}")
        args.tasks = str(tasks_file)

    # Create manager and annotator
    manager = AnnotationManager(args.tasks, args.output)
    annotator = PreferenceAnnotator(manager, args.annotator)

    # Run annotation
    annotator.run()


if __name__ == "__main__":
    main()

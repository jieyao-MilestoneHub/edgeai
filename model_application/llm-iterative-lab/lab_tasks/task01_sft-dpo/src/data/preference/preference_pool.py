"""
Preference Pool Manager for accumulating high-quality preference pairs.

This module manages a buffer that triggers micro-DPO training when threshold is reached.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import deque
from dataclasses import dataclass, asdict

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreferencePair:
    """Single preference pair with quality metadata."""
    prompt: str
    chosen: str
    rejected: str
    confidence: Optional[float] = None
    source: Optional[str] = None


class PreferencePool:
    """Manages accumulation and filtering of preference pairs for micro-DPO.

    This class implements a FIFO buffer that:
    - Accumulates preference pairs from various sources
    - Filters by quality threshold
    - Triggers when batch size is reached
    - Provides batches for micro-DPO training

    Example:
        >>> pool = PreferencePool(batch_size=5, quality_threshold=0.7)
        >>> pool.add_pair(prompt="Q", chosen="A1", rejected="A2", confidence=0.9)
        >>> if pool.is_ready():
        ...     batch = pool.drain()
        ...     # Run micro-DPO with batch
    """

    def __init__(
        self,
        batch_size: int = 5,
        quality_threshold: float = 0.7,
        data_file: Optional[str] = None,
        max_pool_size: Optional[int] = None
    ):
        """Initialize preference pool.

        Args:
            batch_size: Number of pairs to trigger micro-DPO
            quality_threshold: Minimum confidence score to accept pair
            data_file: Optional file to load/save pairs (JSONL format)
            max_pool_size: Maximum pool size (None for unlimited)
        """
        self.batch_size = batch_size
        self.quality_threshold = quality_threshold
        self.data_file = Path(data_file) if data_file else None
        self.max_pool_size = max_pool_size

        # Use deque for efficient FIFO operations
        self.pool: deque = deque(maxlen=max_pool_size)

        # Statistics
        self.total_added = 0
        self.total_filtered = 0
        self.total_drained = 0

        # Load existing pairs if file provided
        if self.data_file and self.data_file.exists():
            self._load_from_file()

    def add_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        confidence: Optional[float] = None,
        source: Optional[str] = None
    ) -> bool:
        """Add a preference pair to the pool.

        Args:
            prompt: Input prompt
            chosen: Preferred response
            rejected: Rejected response
            confidence: Quality score (0-1), None means always accept
            source: Source identifier for tracking

        Returns:
            True if pair was added, False if filtered out
        """
        # Quality filtering
        if confidence is not None and confidence < self.quality_threshold:
            self.total_filtered += 1
            logger.debug(
                f"Filtered pair (confidence={confidence:.3f} < {self.quality_threshold})"
            )
            return False

        # Create pair
        pair = PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            confidence=confidence,
            source=source
        )

        # Add to pool
        self.pool.append(pair)
        self.total_added += 1

        logger.info(
            f"✅ Added preference pair (pool: {len(self.pool)}/{self.batch_size}, "
            f"confidence: {confidence if confidence else 'N/A'})"
        )

        # Save to file if configured
        if self.data_file:
            self._append_to_file(pair)

        return True

    def is_ready(self) -> bool:
        """Check if pool has enough pairs to trigger micro-DPO.

        Returns:
            True if pool size >= batch_size
        """
        return len(self.pool) >= self.batch_size

    def drain(self, count: Optional[int] = None) -> List[Dict]:
        """Remove and return pairs from pool for training.

        Args:
            count: Number of pairs to drain (None = batch_size)

        Returns:
            List of preference pair dictionaries
        """
        if count is None:
            count = self.batch_size

        drained = []
        for _ in range(min(count, len(self.pool))):
            pair = self.pool.popleft()
            drained.append({
                'prompt': pair.prompt,
                'chosen': pair.chosen,
                'rejected': pair.rejected
            })

        self.total_drained += len(drained)

        logger.info(
            f"🔄 Drained {len(drained)} pairs from pool "
            f"(remaining: {len(self.pool)})"
        )

        return drained

    def peek(self, count: int = 5) -> List[Dict]:
        """View pairs without removing them.

        Args:
            count: Number of pairs to view

        Returns:
            List of preference pair dictionaries
        """
        pairs = list(self.pool)[:count]
        return [
            {
                'prompt': p.prompt,
                'chosen': p.chosen,
                'rejected': p.rejected,
                'confidence': p.confidence
            }
            for p in pairs
        ]

    def clear(self) -> None:
        """Clear all pairs from pool."""
        cleared = len(self.pool)
        self.pool.clear()
        logger.info(f"🗑️  Cleared {cleared} pairs from pool")

    def get_statistics(self) -> Dict:
        """Get pool statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'current_size': len(self.pool),
            'batch_size': self.batch_size,
            'is_ready': self.is_ready(),
            'total_added': self.total_added,
            'total_filtered': self.total_filtered,
            'total_drained': self.total_drained,
            'acceptance_rate': (
                self.total_added / (self.total_added + self.total_filtered)
                if (self.total_added + self.total_filtered) > 0 else 0
            )
        }

    def _load_from_file(self) -> None:
        """Load preference pairs from JSONL file."""
        if not self.data_file.exists():
            return

        count = 0
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    pair = PreferencePair(**data)

                    # Apply quality filter
                    if pair.confidence is None or pair.confidence >= self.quality_threshold:
                        self.pool.append(pair)
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to load pair: {e}")

        logger.info(f"📂 Loaded {count} pairs from {self.data_file}")

    def _append_to_file(self, pair: PreferencePair) -> None:
        """Append a single pair to file.

        Args:
            pair: Preference pair to save
        """
        if not self.data_file:
            return

        # Create parent directory if needed
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + '\n')

    def load_batch_from_file(self, file_path: str, max_pairs: Optional[int] = None) -> int:
        """Load preference pairs from external JSONL file.

        Args:
            file_path: Path to JSONL file with preference pairs
            max_pairs: Maximum number of pairs to load (None for all)

        Returns:
            Number of pairs loaded
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return 0

        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_pairs and count >= max_pairs:
                    break

                try:
                    data = json.loads(line.strip())

                    # Extract fields (flexible format)
                    prompt = data.get('prompt', '')
                    chosen = data.get('chosen', '')
                    rejected = data.get('rejected', '')
                    confidence = data.get('confidence')
                    source = data.get('source', file_path.name)

                    if prompt and chosen and rejected:
                        if self.add_pair(prompt, chosen, rejected, confidence, source):
                            count += 1

                except Exception as e:
                    logger.warning(f"Failed to parse line: {e}")

        logger.info(f"📥 Loaded {count} pairs from {file_path}")
        return count


def create_pool_from_annotations(
    annotation_file: str,
    batch_size: int = 5,
    quality_threshold: float = 0.7
) -> PreferencePool:
    """Create a preference pool from annotation file.

    This is a helper function to quickly create and populate a pool.

    Args:
        annotation_file: Path to annotated preference pairs (JSONL)
        batch_size: Batch size for triggering micro-DPO
        quality_threshold: Quality threshold for filtering

    Returns:
        Initialized and populated PreferencePool
    """
    pool = PreferencePool(
        batch_size=batch_size,
        quality_threshold=quality_threshold
    )

    pool.load_batch_from_file(annotation_file)

    return pool

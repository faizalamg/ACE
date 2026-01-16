"""
LinUCB Contextual Bandit for Adaptive Retrieval Strategy Selection

Part of P7 ARIA (Adaptive Retrieval Intelligence Architecture).

Implements P7.3 LinUCB algorithm with 4-arm retrieval strategy:
- FAST: Low latency, minimal context
- BALANCED: Moderate speed/quality tradeoff (cold start default)
- DEEP: Maximum semantic depth
- DIVERSE: Multi-perspective retrieval

Formula: UCB(arm) = theta^T * x + alpha * sqrt(x^T * A^-1 * x)
where:
- theta = A^-1 * b (parameter estimate)
- A = sum of x*x^T + I (regularized covariance)
- b = sum of r*x (reward-weighted features)
- alpha = exploration parameter

Algorithm Reference:
    Li, Lihong, et al. "A Contextual-Bandit Approach to Personalized News
    Article Recommendation." WWW 2010. arXiv:1003.0146
    https://arxiv.org/abs/1003.0146

The adaptation for RAG retrieval optimization (query feature extraction,
preset mapping, and quality feedback integration) is an original contribution.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional


class LinUCBRetrievalBandit:
    """LinUCB contextual bandit for adaptive retrieval strategy selection.

    Arms: FAST, BALANCED, DEEP, DIVERSE
    Formula: UCB(arm) = theta^T * x + alpha * sqrt(x^T * A^-1 * x)
    """

    ARMS = ['FAST', 'BALANCED', 'DEEP', 'DIVERSE']

    def __init__(self, alpha: float = 1.0, d: int = 4):
        """Initialize bandit with exploration parameter alpha and context dimension d.

        Args:
            alpha: Exploration parameter (higher = more exploration)
            d: Context vector dimension (default 4 for query features)
        """
        self.alpha = alpha
        self.d = d
        self.arms = self.ARMS.copy()

        # Initialize A matrices (d x d identity) and b vectors (d x 1 zero) per arm
        self._A = {arm: np.eye(d) for arm in self.arms}
        self._b = {arm: np.zeros(d) for arm in self.arms}

    def select_arm(self, context: np.ndarray) -> str:
        """Select arm using UCB scores. Cold start returns BALANCED.

        Args:
            context: Feature vector (d-dimensional)

        Returns:
            Selected arm name (FAST/BALANCED/DEEP/DIVERSE)
        """
        # Cold start: if NO arms have been updated, return BALANCED
        # Only true cold start is when b vectors are all zero
        if self._is_cold_start():
            return 'BALANCED'

        # Compute UCB scores for all arms
        ucb_scores = self._compute_ucb_scores(context)

        # Select arm with highest UCB score
        return max(ucb_scores.items(), key=lambda x: x[1])[0]

    def update(self, arm: str, context: np.ndarray, reward: float) -> None:
        """Update arm's A matrix and b vector after observing reward.

        Args:
            arm: Arm that was selected
            context: Feature vector used for selection
            reward: Observed reward (0.0 to 1.0)
        """
        # Update A_a = A_a + x*x^T
        self._A[arm] += np.outer(context, context)

        # Update b_a = b_a + r*x
        self._b[arm] += reward * context

    def _compute_ucb_scores(self, context: np.ndarray) -> Dict[str, float]:
        """Compute UCB score for each arm.

        UCB(arm) = theta^T * x + alpha * sqrt(x^T * A^-1 * x)
        where theta = A^-1 * b

        Args:
            context: Feature vector (d-dimensional)

        Returns:
            Dictionary mapping arm name to UCB score
        """
        scores = {}

        for arm in self.arms:
            # Compute theta = A^-1 * b
            A_inv = np.linalg.inv(self._A[arm])
            theta = A_inv @ self._b[arm]

            # Exploitation term: theta^T * x
            exploitation = theta @ context

            # Exploration term: alpha * sqrt(x^T * A^-1 * x)
            exploration = self.alpha * np.sqrt(context @ A_inv @ context)

            # UCB score
            scores[arm] = exploitation + exploration

        return scores

    def _is_cold_start(self) -> bool:
        """Check if bandit is in cold start state (all b vectors are zero).

        Cold start means no arm has received any feedback yet.
        """
        for arm in self.arms:
            # If any b vector is non-zero, we have feedback data
            if not np.allclose(self._b[arm], np.zeros(self.d)):
                return False
        return True

    def get_state_snapshot(self) -> str:
        """Return hashable state for comparison.

        Returns:
            JSON string representation of bandit state
        """
        state = {
            'arms': self.arms,
            'alpha': self.alpha,
            'd': self.d,
            'A': {arm: self._A[arm].tolist() for arm in self.arms},
            'b': {arm: self._b[arm].tolist() for arm in self.arms}
        }
        return json.dumps(state, sort_keys=True)

    def save_state(self, path: Path) -> None:
        """Save bandit state to JSON file.

        Args:
            path: Path to save JSON file
        """
        state = {
            'arms': self.arms,
            'alpha': self.alpha,
            'd': self.d,
            'matrices': {
                arm: {
                    'A': self._A[arm].tolist(),
                    'b': self._b[arm].tolist()
                }
                for arm in self.arms
            }
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, path: Path) -> 'LinUCBRetrievalBandit':
        """Load bandit state from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Restored LinUCBRetrievalBandit instance
        """
        with open(path, 'r') as f:
            state = json.load(f)

        # Create new instance
        bandit = cls(alpha=state['alpha'], d=state['d'])

        # Restore matrices
        for arm in state['arms']:
            bandit._A[arm] = np.array(state['matrices'][arm]['A'])
            bandit._b[arm] = np.array(state['matrices'][arm]['b'])

        return bandit

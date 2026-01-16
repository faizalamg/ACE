"""
Test suite for P7.4 Quality Feedback Loop implementation.

Tests the QualityFeedbackHandler that processes 1-5 star ratings
and updates bullet counters (helpful/neutral/harmful) with timestamps.

TDD RED PHASE - All tests intentionally fail until implementation exists.
"""

import unittest
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock


class TestQualityFeedbackHandler(unittest.TestCase):
    """Test suite for quality feedback handler functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # This will fail until QualityFeedbackHandler is implemented
        try:
            from ace.quality_feedback import QualityFeedbackHandler
            self.handler = QualityFeedbackHandler()
        except ImportError:
            self.handler = None

    def test_quality_feedback_handler_class_exists(self):
        """Test that QualityFeedbackHandler class exists and can be instantiated."""
        from ace.quality_feedback import QualityFeedbackHandler

        handler = QualityFeedbackHandler()
        self.assertIsNotNone(handler)
        self.assertTrue(hasattr(handler, 'process_feedback'))

    def test_rating_5_increments_helpful(self):
        """Test that 5-star rating increments helpful counter."""
        bullet_id = "test-bullet-1"

        result = self.handler.process_feedback(bullet_id, rating=5)

        self.assertEqual(result['helpful_delta'], 1)
        self.assertEqual(result['harmful_delta'], 0)
        self.assertEqual(result['neutral_delta'], 0)

    def test_rating_4_increments_helpful(self):
        """Test that 4-star rating increments helpful counter."""
        bullet_id = "test-bullet-2"

        result = self.handler.process_feedback(bullet_id, rating=4)

        self.assertEqual(result['helpful_delta'], 1)
        self.assertEqual(result['harmful_delta'], 0)
        self.assertEqual(result['neutral_delta'], 0)

    def test_rating_3_increments_neutral(self):
        """Test that 3-star rating increments neutral counter."""
        bullet_id = "test-bullet-3"

        result = self.handler.process_feedback(bullet_id, rating=3)

        self.assertEqual(result['helpful_delta'], 0)
        self.assertEqual(result['harmful_delta'], 0)
        self.assertEqual(result['neutral_delta'], 1)

    def test_rating_2_increments_harmful(self):
        """Test that 2-star rating increments harmful counter."""
        bullet_id = "test-bullet-4"

        result = self.handler.process_feedback(bullet_id, rating=2)

        self.assertEqual(result['helpful_delta'], 0)
        self.assertEqual(result['harmful_delta'], 1)
        self.assertEqual(result['neutral_delta'], 0)

    def test_rating_1_increments_harmful(self):
        """Test that 1-star rating increments harmful counter."""
        bullet_id = "test-bullet-5"

        result = self.handler.process_feedback(bullet_id, rating=1)

        self.assertEqual(result['helpful_delta'], 0)
        self.assertEqual(result['harmful_delta'], 1)
        self.assertEqual(result['neutral_delta'], 0)

    def test_feedback_updates_last_validated_timestamp(self):
        """Test that feedback updates last_validated timestamp (CRITICAL for confidence decay)."""
        bullet_id = "test-bullet-6"
        before_time = datetime.utcnow()

        result = self.handler.process_feedback(bullet_id, rating=5)

        after_time = datetime.utcnow()

        self.assertIn('last_validated', result)
        self.assertIsInstance(result['last_validated'], datetime)
        self.assertGreaterEqual(result['last_validated'], before_time)
        self.assertLessEqual(result['last_validated'], after_time)

    def test_feedback_updates_updated_at_timestamp(self):
        """Test that feedback updates updated_at timestamp."""
        bullet_id = "test-bullet-7"
        before_time = datetime.utcnow()

        result = self.handler.process_feedback(bullet_id, rating=5)

        after_time = datetime.utcnow()

        self.assertIn('updated_at', result)
        self.assertIsInstance(result['updated_at'], datetime)
        self.assertGreaterEqual(result['updated_at'], before_time)
        self.assertLessEqual(result['updated_at'], after_time)

    def test_feedback_latency_under_10ms(self):
        """Test that feedback processing completes in under 10ms (performance requirement)."""
        bullet_id = "test-bullet-8"

        start_time = time.perf_counter()
        self.handler.process_feedback(bullet_id, rating=5)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        self.assertLess(latency_ms, 10.0,
                       f"Feedback processing took {latency_ms:.2f}ms, must be under 10ms")

    def test_invalid_rating_raises_error(self):
        """Test that invalid ratings (not 1-5) raise ValueError."""
        bullet_id = "test-bullet-9"

        with self.assertRaises(ValueError) as context:
            self.handler.process_feedback(bullet_id, rating=0)

        self.assertIn("rating must be between 1 and 5", str(context.exception).lower())

        with self.assertRaises(ValueError):
            self.handler.process_feedback(bullet_id, rating=6)

        with self.assertRaises(ValueError):
            self.handler.process_feedback(bullet_id, rating=-1)

    def test_feedback_integrates_with_confidence_decay(self):
        """Test that feedback interacts correctly with confidence decay system."""
        import time
        # Mock the confidence decay retrieval with a timestamp in the past
        old_timestamp = datetime(2025, 1, 1, 0, 0, 0)  # Fixed past time
        mock_bullet = {
            'id': 'test-bullet-10',
            'helpful': 5,
            'harmful': 2,
            'neutral': 1,
            'last_validated': old_timestamp
        }

        with patch.object(self.handler, 'get_bullet', return_value=mock_bullet):
            result = self.handler.process_feedback('test-bullet-10', rating=5)

            # Verify last_validated was updated (critical for decay calculation)
            self.assertIn('last_validated', result)
            self.assertGreater(result['last_validated'], mock_bullet['last_validated'])

    def test_feedback_separate_from_session_tracking(self):
        """Test that quality feedback does NOT update session tracking counters."""
        bullet_id = "test-bullet-11"

        result = self.handler.process_feedback(bullet_id, rating=5)

        # Verify session tracking fields are NOT in result
        self.assertNotIn('session_helpful', result)
        self.assertNotIn('session_harmful', result)
        self.assertNotIn('session_neutral', result)

        # Only global counters should be updated
        self.assertIn('helpful_delta', result)
        self.assertIn('harmful_delta', result)
        self.assertIn('neutral_delta', result)

    def test_batch_feedback_processing(self):
        """Test processing multiple feedback items in batch."""
        feedback_batch = [
            {'bullet_id': 'bullet-1', 'rating': 5},
            {'bullet_id': 'bullet-2', 'rating': 4},
            {'bullet_id': 'bullet-3', 'rating': 3},
            {'bullet_id': 'bullet-4', 'rating': 2},
            {'bullet_id': 'bullet-5', 'rating': 1},
        ]

        results = self.handler.process_feedback_batch(feedback_batch)

        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]['helpful_delta'], 1)
        self.assertEqual(results[1]['helpful_delta'], 1)
        self.assertEqual(results[2]['neutral_delta'], 1)
        self.assertEqual(results[3]['harmful_delta'], 1)
        self.assertEqual(results[4]['harmful_delta'], 1)

    def test_feedback_persists_to_qdrant(self):
        """Test that feedback updates are persisted to Qdrant vector store."""
        bullet_id = "test-bullet-12"

        # Mock Qdrant client
        mock_qdrant = MagicMock()

        with patch.object(self.handler, 'qdrant_client', mock_qdrant):
            result = self.handler.process_feedback(bullet_id, rating=5)

            # Verify Qdrant update was called
            self.assertTrue(mock_qdrant.update_payload.called or
                          mock_qdrant.upsert.called,
                          "Feedback must persist to Qdrant")


if __name__ == '__main__':
    unittest.main()

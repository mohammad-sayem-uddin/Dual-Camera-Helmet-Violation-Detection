"""Unit Tests for Temporal Confidence Aggregation Algorithms."""
import unittest
from src.temporal.confidence_aggregator import (
    ConfidenceWeightedAggregator,
    EMAAggregator,
    SimpleVotingAggregator,
    create_aggregator,
)


class TestConfidenceAggregation(unittest.TestCase):
    def test_simple_voting_aggregator(self):
        agg = SimpleVotingAggregator(window_size=5)
        self.assertEqual(agg.get_aggregated_confidence(), {"no_helmet": 0.0, "helmet": 0.0})

        # Add 4 helmet, 1 no_helmet
        for _ in range(4):
            agg.add_observation("helmet", 0.90)
        agg.add_observation("no_helmet", 0.80)

        conf = agg.get_aggregated_confidence()
        self.assertAlmostEqual(conf["helmet"], 0.80, places=2)
        self.assertAlmostEqual(conf["no_helmet"], 0.20, places=2)

    def test_confidence_weighted_aggregator(self):
        agg = ConfidenceWeightedAggregator(window_size=5)
        # 1 very high-confidence no_helmet (0.95), 2 low-confidence helmet (0.25 each)
        agg.add_observation("no_helmet", 0.95)
        agg.add_observation("helmet", 0.25)
        agg.add_observation("helmet", 0.25)

        # Total conf = 0.95 + 0.50 = 1.45
        # no_helmet weight = 0.95 / 1.45 = 0.6552
        # helmet weight = 0.50 / 1.45 = 0.3448
        conf = agg.get_aggregated_confidence()
        self.assertGreater(conf["no_helmet"], conf["helmet"])
        self.assertAlmostEqual(conf["no_helmet"], 0.6552, places=2)
        self.assertAlmostEqual(conf["helmet"], 0.3448, places=2)

    def test_ema_aggregator_smoothing(self):
        agg = EMAAggregator(window_size=10, alpha=0.40)
        # Sequence of 5 consistent helmet frames
        for _ in range(5):
            agg.add_observation("helmet", 0.90)

        conf1 = agg.get_aggregated_confidence()
        self.assertGreater(conf1["helmet"], 0.70)

        # Single spurious no_helmet noise frame
        agg.add_observation("no_helmet", 0.60)
        conf2 = agg.get_aggregated_confidence()
        # Helmet must remain dominant despite single noise frame
        self.assertGreater(conf2["helmet"], conf2["no_helmet"])

    def test_factory_function(self):
        v = create_aggregator("voting")
        self.assertIsInstance(v, SimpleVotingAggregator)
        w = create_aggregator("weighted")
        self.assertIsInstance(w, ConfidenceWeightedAggregator)
        e = create_aggregator("ema")
        self.assertIsInstance(e, EMAAggregator)


if __name__ == "__main__":
    unittest.main()

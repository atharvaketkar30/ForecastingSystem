import unittest

import numpy as np
import pandas as pd

from causal_analysis import (
    _make_unavailable_result,
    _solve_synth_weights,
    eligible_synth_donors,
)


class CausalAnalysisTests(unittest.TestCase):
    def test_solve_synth_weights_respects_constraints(self):
        donors = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.6, 0.4],
                [0.4, 0.6],
            ]
        )
        treated = donors @ np.array([0.75, 0.25])
        weights = _solve_synth_weights(treated, donors)
        self.assertTrue(np.all(weights >= -1e-9))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertTrue(np.allclose(weights, np.array([0.75, 0.25]), atol=1e-2))

    def test_eligible_synth_donors_excludes_overlapping_series(self):
        category_views = pd.DataFrame(
            {
                "series_id": [
                    "en.wikipedia/tech",
                    "en.wikipedia/finance",
                    "de.wikipedia/tech",
                    "de.wikipedia/finance",
                ]
            }
        )
        launch = {
            "id": "launch_event",
            "type": "step_change",
            "date": "2025-06-01",
            "affected_projects": ["en.wikipedia"],
            "affected_categories": ["tech"],
        }
        price = {
            "id": "price_change",
            "type": "elasticity_shift",
            "date": "2023-09-01",
            "affected_projects": ["en.wikipedia"],
            "affected_categories": ["finance"],
        }
        donors = eligible_synth_donors(category_views, launch, [launch, price])
        self.assertEqual(donors, ["de.wikipedia/finance", "de.wikipedia/tech"])

    def test_unavailable_result_keeps_required_shape(self):
        intervention = {
            "id": "price_change",
            "type": "elasticity_shift",
            "affected_projects": ["en.wikipedia"],
            "affected_categories": ["tech", "finance"],
        }
        result = _make_unavailable_result(
            intervention,
            method="did_double_ml",
            diagnostics={"reason": "no_pre_period"},
        )
        self.assertEqual(result.summary["status"], "unavailable")
        self.assertEqual(result.summary["method"], "did_double_ml")
        self.assertIn("affected", result.summary)
        self.assertIn("diagnostics", result.summary)
        self.assertIsNone(result.summary["p_value"])


if __name__ == "__main__":
    unittest.main()

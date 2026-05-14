import os
import unittest
from unittest.mock import patch

from src.matcher import find_best_match


class MatcherDistanceModeTests(unittest.TestCase):
    def setUp(self):
        self.prev_distance_mode = os.environ.get("VALIDATION_DISTANCE_MODE")
        self.prev_ors_api_key = os.environ.get("ORS_API_KEY")
        os.environ["VALIDATION_DISTANCE_MODE"] = "route"

    def tearDown(self):
        if self.prev_distance_mode is None:
            os.environ.pop("VALIDATION_DISTANCE_MODE", None)
        else:
            os.environ["VALIDATION_DISTANCE_MODE"] = self.prev_distance_mode

        if self.prev_ors_api_key is None:
            os.environ.pop("ORS_API_KEY", None)
        else:
            os.environ["ORS_API_KEY"] = self.prev_ors_api_key

    def test_route_mode_uses_openrouteservice_when_key_exists(self):
        os.environ["ORS_API_KEY"] = "test-key"
        pred_poi = {"name": "Central Mall", "type": "mall"}
        real_pois = [
            {"name": "Central Mall", "type": "retail", "lat": -7.22, "lon": -35.88},
        ]

        with patch("src.matcher.get_route_distance_km", return_value=1.234) as route_mock:
            match = find_best_match(pred_poi, real_pois, -7.21, -35.87)

        self.assertIsNotNone(match)
        self.assertEqual(match["distance_mode"], "route")
        self.assertEqual(match["real_distance_km"], 1.234)
        route_mock.assert_called_once()

    def test_route_mode_falls_back_without_key(self):
        os.environ.pop("ORS_API_KEY", None)
        pred_poi = {"name": "Central Mall", "type": "mall"}
        real_pois = [
            {"name": "Central Mall", "type": "retail", "lat": -7.22, "lon": -35.88},
        ]

        with patch("src.matcher.get_route_distance_km") as route_mock:
            match = find_best_match(pred_poi, real_pois, -7.21, -35.87)

        self.assertIsNotNone(match)
        self.assertEqual(match["distance_mode"], "straight_fallback")
        self.assertEqual(match["real_distance_km"], match["straight_distance_km"])
        route_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

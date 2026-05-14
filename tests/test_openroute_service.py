import os
import unittest
from unittest.mock import Mock, patch

from src.openroute_service import get_ors_config, get_route_distance_km, is_ors_configured


class OpenRouteServiceTests(unittest.TestCase):
    def setUp(self):
        self.prev_env = {
            "ORS_API_KEY": os.environ.get("ORS_API_KEY"),
            "ORS_PROFILE": os.environ.get("ORS_PROFILE"),
            "ORS_BASE_URL": os.environ.get("ORS_BASE_URL"),
            "ORS_TIMEOUT": os.environ.get("ORS_TIMEOUT"),
        }

    def tearDown(self):
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_is_ors_configured_reflects_api_key(self):
        os.environ.pop("ORS_API_KEY", None)
        self.assertFalse(is_ors_configured())

        os.environ["ORS_API_KEY"] = "test-key"
        self.assertTrue(is_ors_configured())

    def test_get_route_distance_km_returns_none_without_key(self):
        os.environ.pop("ORS_API_KEY", None)
        self.assertIsNone(get_route_distance_km(-7.21, -35.87, -7.22, -35.88))

    def test_get_route_distance_km_parses_response_distance(self):
        os.environ["ORS_API_KEY"] = "test-key"
        os.environ["ORS_PROFILE"] = "driving-car"
        os.environ["ORS_BASE_URL"] = "https://example.org"
        os.environ["ORS_TIMEOUT"] = "45"

        response = Mock()
        response.json.return_value = {
            "routes": [
                {
                    "summary": {
                        "distance": 2345.6,
                    }
                }
            ]
        }
        response.raise_for_status.return_value = None

        with patch("src.openroute_service.requests.post", return_value=response) as post_mock:
            distance = get_route_distance_km(-7.21, -35.87, -7.22, -35.88)

        self.assertEqual(distance, 2.346)
        config = get_ors_config()
        self.assertEqual(config.profile, "driving-car")
        self.assertEqual(config.base_url, "https://example.org")
        self.assertEqual(config.timeout, 45)
        post_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""Lightweight Flask route checks; model inference is intentionally not tested here."""
import unittest
from app import app


class AppRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Research limitation", response.data)

    def test_missing_image_is_handled(self):
        response = self.client.post("/", data={})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Choose a JPG or PNG image", response.data)


if __name__ == "__main__":
    unittest.main()

from django.test import TestCase
from rest_framework.test import APIClient

# Create your tests here.


class EndpointTests(TestCase):
    def test_predict_view(self):
        client = APIClient()
        input_data = {
            "age": "lansia",
            "sex": 0,
            "cp": 0,
            "trestbps": 0,
            "chol": "tinggi",
            "fbs": 1,
            "restecg": 2,
            "thalach": "kecil",
            "exang": 0,
            "slope": 1,
            "ca": 2,
            "thal": 2,
        }
        classifier_url = "/api/v1/income_classifier/predict"
        response = client.post(classifier_url, input_data, format="json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["label"], "Sakit Jantung Koroner")
        self.assertTrue("request_id" in response.data)
        self.assertTrue("status" in response.data)

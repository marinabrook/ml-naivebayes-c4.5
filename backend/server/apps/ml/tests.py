import inspect

from django.test import TestCase

from apps.ml.income_classifier.naive_bayes import NaiveBayesClassifier
from apps.ml.registry import MLRegistry


class MLTests(TestCase):
    def test_nb_algorithm(self):
        # ["dewasa", 1, 2, 3, "tinggi", 1, 3, "besar", 1, 2, 1, 2]
        # age   sex cp	trestbps	chol	fbs	restecg	thalach	exang	slope	ca	thal
        input_data = {
            "age": "dewasa",
            "sex": 1,
            "cp": 2,
            "trestbps": 3,
            "chol": "tinggi",
            "fbs": 1,
            "restecg": 3,
            "thalach": "besar",
            "exang": 1,
            "slope": 2,
            "ca": 1,
            "thal": 2,
        }
        my_alg = NaiveBayesClassifier()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual("OK", response["status"])
        self.assertTrue("label" in response)
        self.assertEqual("Sakit Jantung Koroner", response["label"])

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "income_classifier"
        algorithm_object = NaiveBayesClassifier()
        algorithm_name = "naive bayes"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Daffa Ammar"
        algorithm_description = "Naive Bayes with simple pre- and post-processing"
        algorithm_code = inspect.getsource(NaiveBayesClassifier)
        # add to registry
        registry.add_algorithm(
            endpoint_name,
            algorithm_object,
            algorithm_name,
            algorithm_status,
            algorithm_version,
            algorithm_owner,
            algorithm_description,
            algorithm_code,
        )
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)

import joblib
import pandas as pd


class NaiveBayesClassifier:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.model = joblib.load(
            path_to_artifacts + "model/naivebayes_heartdisease.joblib"
        )

    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        return input_data

    def predict(self, input_data):
        return self.model.predict_proba(input_data)

    def postprocessing(self, input_data):
        label = "Tidak Sakit Jantung Koroner"
        if input_data[1] > 0.5:
            label = "Sakit Jantung Koroner"
        return {"probability": input_data[1], "label": label, "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction

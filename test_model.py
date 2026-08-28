import os
import pickle


def test_model_file_exists():
    assert os.path.exists("placement_model.pkl")


def test_model_can_predict():
    with open("placement_model.pkl", "rb") as file:
        model = pickle.load(file)

    student = [[8.5, 92, 85, 3, 1]]

    prediction = model.predict(student)

    assert prediction[0] in [0, 1]
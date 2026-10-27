from src.train import train_model


def test_model_trains_successfully():
    model, accuracy = train_model()
    assert model is not None
    assert accuracy > 0.5


def test_model_predicts_attrition():
    model, _ = train_model()
    sample = [[30, 50000, 3, 1]]
    prediction = model.predict(sample)
    assert prediction[0] in (0, 1)

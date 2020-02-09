from predict.run import predict

def test_predict():
    text = 'ruby on rails: how to change BG color of options in select list, ruby-on-rails'
    prediction = predict([text], model_path="../train/train/tests/models/")

    assert prediction[0] == 'ruby-on-rails'


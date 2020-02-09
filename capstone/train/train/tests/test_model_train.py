from unittest.mock import patch
from train.run import train

import os


def read_texts_test(train=True):

    texts = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails"]
    return texts, tags


@patch("train.run.read_texts", side_effect=read_texts_test)
def test_train(read_texts_test):

    params = {"dense_dim": 64, "epochs": 100, "batch_size": 8}

    model_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_path, "models")
    accuracy = train(model_path, params)

    assert accuracy == 1.0
    return

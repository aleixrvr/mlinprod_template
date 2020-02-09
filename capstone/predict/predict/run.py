import json
import argparse
import os
from functools import lru_cache

from tensorflow.keras.models import load_model
from numpy import argmax

from preprocessing.embeddings import embed

import logging

logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("predict")

MODEL_PATH = os.environ.get("MLINPROD_MODEL_PATH", "./model")


@lru_cache(maxsize=1)
def get_model(model_path):
    model = load_model(os.path.join(model_path, "model.h5"))
    with open(os.path.join(model_path, "params.json"), "r") as f:
        params = json.load(f)
    with open(os.path.join(model_path, "labels_index.json"), "r") as f:
        labels_index = json.load(f)

    labels_index_inv = {ind: lab for lab, ind in labels_index.items()}

    return model, params, labels_index_inv


def predict(texts, model_path=MODEL_PATH):

    logger.info("Loading model...")
    model, params, labels_index_inv = get_model(model_path)

    embeddings = embed(texts)
    scores = model.predict(embeddings)
    inds = [argmax(score) for score in scores]
    predictions = [labels_index_inv[ind] for ind in inds]

    logger.info("Prediction done!")

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="text to predict")
    args = parser.parse_args()
    predictions = predict([args.text])
    print(predictions)


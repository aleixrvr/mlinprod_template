import os
import json
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical

from preprocessing.embeddings import embed
from preprocessing.utils import labels_to_indexes

# model_path = '/Users/aleixrvr/Projects/Data/ml-in-prod/'


def read_texts(train=True):  # TODO
    return [], []


def train(model_path, params):

    texts_train, labels_train = read_texts(train=True)
    texts_test, labels_test = read_texts(train=False)

    labels_set = list(set(labels_train + labels_test))
    labels_n = len(labels_set)

    x_train = embed(texts_train)
    x_test = embed(texts_test)

    y_train, labels_index = labels_to_indexes(labels_train, labels_set)
    y_test, _ = labels_to_indexes(labels_test, labels_set)
    y_train = to_categorical(y_train, num_classes=labels_n)
    y_test = to_categorical(y_test, num_classes=labels_n)

    model = Sequential()
    model.add(Dense(params["dense_dim"], activation="relu"))
    model.add(Dense(labels_n, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=2,
    )

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    model.save(os.path.join(model_path, "model.h5"))
    with open(os.path.join(model_path, "params.json"), "w") as f:
        json.dump(params, f)
    with open(os.path.join(model_path, "labels_index.json"), "w") as f:
        json.dump(labels_index, f)

    return scores[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="saving model path")
    parser.add_argument("--dense_dim", default=64, help="Dimension of the dense")
    args = parser.parse_args()

    train(args.model_path, args.dense_dim)

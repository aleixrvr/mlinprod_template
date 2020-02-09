import json
import sys

from flask import Flask, request

from predict.run import predict

app = Flask(__name__)


@app.route("/tag/predict/", methods=["POST"])
def predict_tag():

    text = request.form.get("text")
    print("Predict Stackoverflow tag", file=sys.stderr)
    print(text, file=sys.stderr)

    prediction = predict([text])
    # global graph
    # with graph.as_default():
    #     preds = model.predict(data)

    return json.dumps(prediction[0])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

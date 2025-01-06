from flask import request,jsonify
from config import app
from Model_Machine_Sentiment import Prediction

@app.route("/prediction_comments",methods=["POST"])
def predict_comment():
    text = request.json.get('Comments')
    predicted = Prediction(text)
    return jsonify({'sentiment':predicted[0]})

if __name__ == "__main__":
    app.run(debug=True)
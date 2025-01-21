from flask import Flask, request, jsonify
from controllers import makePrediction
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
@app.route('/getPrediction', methods=['POST'])

def predict():
    data = request.get_json()
    model = data.get('model')

    if model is None:
        return jsonify({'error': 'model name is required'})
    
    classificationReport = makePrediction(model)

    return jsonify(classificationReport)


if __name__ == "__main__":
    app.run(debug=True)



from flask import Flask, request, jsonify
from controllers import makePrediction, getRegressionModelInfo
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

@app.route('/getRegressionHeatMap', methods=['GET'])
def getStats():
    regressionHeatMap = getRegressionModelInfo()
    
    return jsonify({'heatMap': regressionHeatMap})
    

if __name__ == "__main__":
    app.run(debug=True)



from flask import Flask, request, jsonify
from controllers import makePrediction, getRegressionModelHeatMap,getRegressionModelHistogram
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
#http://127.0.0.1:5000/getPrediction
@app.route('/getPrediction', methods=['POST'])

def predict():
    data = request.get_json()
    model = data.get('model')

    if model is None:
        return jsonify({'error': 'model name is required'})
    
    classificationReport = makePrediction(model)
    
    return jsonify(classificationReport)
#http://127.0.0.1:5000/getModelStats
@app.route('/getModelStats', methods=['GET'])

def getStats():
    regressionHeatMap = getRegressionModelHeatMap()
    regressionHistogram = getRegressionModelHistogram()
    return jsonify({'heatMap': regressionHeatMap,
                    'histogram': regressionHistogram
                    })
    

if __name__ == "__main__":
    app.run(debug=True)



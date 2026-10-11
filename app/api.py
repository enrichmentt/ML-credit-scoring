import os
import json
import logging
from flask import Flask, request, jsonify
from app.predictor import ModelPredictor

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


log_format = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)


app = Flask(__name__)


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
try:
    predictor = ModelPredictor(MODEL_DIR)
    logger.info(json.dumps({"msg": "Models loaded successfully"}))
except Exception as e:
    logger.error(json.dumps({"msg": f"Error loading models: {str(e)}"}))
    predictor = None


@app.route('/health', methods=['GET'])
def health():
    if predictor is None:
        return jsonify({"status": "unhealthy", "error": "Models not loaded"}), 500
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        logger.error(json.dumps({"msg": "Prediction error: Models not loaded"}))
        return jsonify({"error": "Models not loaded"}), 500

    if not request.is_json:
        logger.error(json.dumps({"msg": "Request is not JSON"}))
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    client_id = str(data.get('client_id', 'unknown'))

    try:
        result = predictor.predict(data)

        response = {
            "client_id": client_id,
            **result
        }

        logger.info(json.dumps({
            "msg": "Prediction successful",
            "request_data": data,
            "response_data": response
        }))

        return jsonify(response), 200

    except Exception as e:
        logger.error(json.dumps({"msg": f"Prediction error: {str(e)}", "client_id": client_id}))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

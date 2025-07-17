import json
import os
import numpy as np
from joblib import load

os.environ["JOBLIB_MULTIPROCESSING"] = "0"

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = load(model_path)

def lambda_handler(event, context):
    # Handle CORS preflight
    if event["requestContext"]["http"]["method"] == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps("Preflight OK")
        }

    try:
        body = event.get("body")
        if isinstance(body, str):
            body = json.loads(body)

        features = [
            float(body["average_speed"]),
            float(body["brake_events"]),
            float(body["acceleration_events"]),
            int(body["night_trip"]),
            float(body["trip_duration_min"]),
            float(body["distance_km"])
        ]
        features_np = np.array(features).reshape(1, -1)
        prediction = model.predict(features_np)[0]
        proba = model.predict_proba(features_np)[0].max()
        risk_label = {0: "Low", 1: "Medium", 2: "High"}[prediction]

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "risk_category": risk_label,
                "score": round(float(proba), 2),
                "estimated_cost_usd": 0.0000008
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({"error": str(e)})
        }

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

print("🔥 Flask app starting...")

app = Flask(__name__)
CORS(app)

model = joblib.load("land_price_model.pkl")
location_map = joblib.load("location_map.pkl")
print("✅ Model and location map loaded")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    location = data.get("location")
    year = int(data.get("year"))
    investment_per_sqft = float(data.get("investment"))
    area_sqft = float(data.get("area"))

    if location not in location_map:
        return jsonify({"error": f"Unknown location: {location}"}), 400

    loc_code = location_map[location]
    predicted_price_per_sqft = model.predict([[loc_code, year]])[0]

    total_investment = investment_per_sqft * area_sqft
    total_predicted_value = predicted_price_per_sqft * area_sqft
    roi = ((total_predicted_value - total_investment) / total_investment) * 100

    return jsonify({
        "predicted_price": round(total_predicted_value, 2),
        "roi_percent": round(roi, 2)
    })

if __name__ == '__main__':
    # ✅ Use 0.0.0.0 for production deployment
    print("🚀 Running Flask on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)

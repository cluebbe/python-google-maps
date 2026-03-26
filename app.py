from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
import numpy as np

load_dotenv()

app = Flask(__name__)

# === YOUR CUSTOM DATA POINTS ===
# Format: list of dictionaries
data_points = [
    {
        "lat": 40.4168,
        "lng": -3.7038,
        "title": "Puerta del Sol",
        "description": "The heart of Madrid 🌞",
        "icon": "https://maps.google.com/mapfiles/ms/icons/red-dot.png"
    },
    {
        "lat": 40.4154,
        "lng": -3.7074,
        "title": "Plaza Mayor",
        "description": "Historic square with beautiful architecture",
        "icon": "https://maps.google.com/mapfiles/ms/icons/blue-dot.png"
    },
    {
        "lat": 40.4075,
        "lng": -3.6925,
        "title": "Retiro Park",
        "description": "Beautiful park with lake and monuments",
        "icon": "https://maps.google.com/mapfiles/ms/icons/green-dot.png"
    },
    {
        "lat": 40.4304,
        "lng": -3.7023,
        "title": "Santiago Bernabéu Stadium",
        "description": "Home of Real Madrid ⚽",
        "icon": "https://maps.google.com/mapfiles/ms/icons/yellow-dot.png"
    }
]

@app.route("/")
def index():
    return render_template(
        "index.html",
        google_maps_key=os.getenv("GOOGLE_MAPS_KEY"),
        points=data_points
    )

@app.route("/knn_search", methods=["POST"])
def knn_search():
    """
    KNN search endpoint that finds nearest neighbors to clicked position
    """
    data = request.json
    lat = data.get("lat")
    lng = data.get("lng")
    k = data.get("k", 3)  # Default to 3 neighbors
    
    if lat is None or lng is None:
        return jsonify({"error": "Missing lat/lng"}), 400
    
    # Extract coordinates from data points
    X = np.array([[point["lat"], point["lng"]] for point in data_points])
    
    # Create KNN model
    knn = NearestNeighbors(n_neighbors=min(k, len(data_points)))
    knn.fit(X)
    
    # Query point
    query_point = np.array([[lat, lng]])
    
    # Find nearest neighbors
    distances, indices = knn.kneighbors(query_point)
    
    # Prepare results
    neighbors = []
    for idx, distance in zip(indices[0], distances[0]):
        point = data_points[int(idx)]
        neighbors.append({
            "lat": point["lat"],
            "lng": point["lng"],
            "title": point["title"],
            "description": point["description"],
            "distance": float(distance)
        })
    
    return jsonify({
        "query": {"lat": lat, "lng": lng},
        "neighbors": neighbors
    })

if __name__ == "__main__":
    app.run(debug=True)
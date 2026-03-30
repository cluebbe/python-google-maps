from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np

load_dotenv()

app = Flask(__name__)

# City center for distance feature (Puerta del Sol, Madrid)
CITY_CENTER = (40.4168, -3.7038)

def haversine_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two points in kilometers"""
    R = 6371  # Earth's radius in km
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def get_feature_vector(lat, lng, use_city_center=False):
    """Create feature vector with location and optionally distance to city center"""
    if use_city_center:
        dist_to_center = haversine_distance(lat, lng, CITY_CENTER[0], CITY_CENTER[1])
        return [lat, lng, dist_to_center]
    else:
        return [lat, lng]

# Add cache control headers to prevent aggressive caching
@app.after_request
def add_cache_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, public, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# === YOUR CUSTOM DATA POINTS ===
# Format: list of dictionaries with categories for classification
data_points = [
    # Restaurants cluster (around Puerta del Sol area) - RED markers
    {
        "lat": 40.4168,
        "lng": -3.7038,
        "title": "Puerta del Sol",
        "description": "The heart of Madrid 🌞",
        "icon": "https://maps.google.com/mapfiles/ms/icons/red-dot.png",
        "category": "restaurant"
    },
    {
        "lat": 40.4154,
        "lng": -3.7074,
        "title": "Plaza Mayor",
        "description": "Historic square with beautiful architecture",
        "icon": "https://maps.google.com/mapfiles/ms/icons/red-dot.png",
        "category": "restaurant"
    },
    {
        "lat": 40.4140,
        "lng": -3.7010,
        "title": "Casa Botín",
        "description": "Oldest restaurant in the world 🍽️",
        "icon": "https://maps.google.com/mapfiles/ms/icons/red-dot.png",
        "category": "restaurant"
    },
    {
        "lat": 40.4175,
        "lng": -3.7050,
        "title": "La Latina District",
        "description": "Famous for tapas and restaurants",
        "icon": "https://maps.google.com/mapfiles/ms/icons/red-dot.png",
        "category": "restaurant"
    },
    # Car workshops cluster (different area) - GREEN markers
    {
        "lat": 40.4075,
        "lng": -3.6925,
        "title": "Retiro Park",
        "description": "Beautiful park with lake and monuments",
        "icon": "https://maps.google.com/mapfiles/ms/icons/green-dot.png",
        "category": "car_workshop"
    },
    {
        "lat": 40.4304,
        "lng": -3.7023,
        "title": "Santiago Bernabéu Stadium",
        "description": "Home of Real Madrid ⚽",
        "icon": "https://maps.google.com/mapfiles/ms/icons/green-dot.png",
        "category": "car_workshop"
    },
    {
        "lat": 40.4050,
        "lng": -3.6900,
        "title": "AutoZone Madrid",
        "description": "Car parts and service center 🔧",
        "icon": "https://maps.google.com/mapfiles/ms/icons/green-dot.png",
        "category": "car_workshop"
    },
    {
        "lat": 40.4320,
        "lng": -3.7000,
        "title": "Bernabéu Auto Service",
        "description": "Professional car repair shop",
        "icon": "https://maps.google.com/mapfiles/ms/icons/green-dot.png",
        "category": "car_workshop"
    },
    # Construction shops cluster (outskirts) - BLUE markers
    {
        "lat": 40.4440,
        "lng": -3.7240,
        "title": "Madrid Construction Depot",
        "description": "Building materials and contractor equipment",
        "icon": "https://maps.google.com/mapfiles/ms/icons/blue-dot.png",
        "category": "construction_shop"
    },
    {
        "lat": 40.4485,
        "lng": -3.7330,
        "title": "La Vaguada Construction Hub",
        "description": "Outdoor equipment and supplies",
        "icon": "https://maps.google.com/mapfiles/ms/icons/blue-dot.png",
        "category": "construction_shop"
    },
    {
        "lat": 40.4500,
        "lng": -3.7460,
        "title": "Alcobendas Construction Center",
        "description": "Heavy machinery and tool rentals",
        "icon": "https://maps.google.com/mapfiles/ms/icons/blue-dot.png",
        "category": "construction_shop"
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
    
    # Extract feature vectors from data points (city center distance not used for search)
    X = np.array([get_feature_vector(point["lat"], point["lng"], False) for point in data_points])
    
    # Find nearest neighbors using Euclidean distance
    knn = NearestNeighbors(n_neighbors=min(k, len(data_points)), metric='euclidean')
    knn.fit(X)
    
    # Query point (search always uses lat/lng only)
    query_point = np.array([get_feature_vector(lat, lng, False)]).reshape(1, -1)
    
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

@app.route("/knn_classification", methods=["POST"])
def knn_classification():
    """
    KNN classification endpoint that predicts category for clicked position
    """
    data = request.json
    lat = data.get("lat")
    lng = data.get("lng")
    k = data.get("k", 3)  # Default to 3 neighbors
    use_city_center = data.get("use_city_center", False)
    
    if lat is None or lng is None:
        return jsonify({"error": "Missing lat/lng"}), 400
    
    # Extract feature vectors and labels from data points
    X = np.array([get_feature_vector(point["lat"], point["lng"], use_city_center) for point in data_points])
    y = np.array([point["category"] for point in data_points])
    
    # Create KNN classifier with Euclidean distance
    knn_classifier = KNeighborsClassifier(n_neighbors=min(k, len(data_points)), metric='euclidean')
    knn_classifier.fit(X, y)
    
    # Query point
    query_point = np.array([get_feature_vector(lat, lng, use_city_center)]).reshape(1, -1)
    
    # Predict category
    predicted_category = knn_classifier.predict(query_point)[0]
    
    # Get prediction probabilities
    probabilities = knn_classifier.predict_proba(query_point)[0]
    class_names = knn_classifier.classes_
    
    # Find similar points of the predicted category
    category_points = [point for point in data_points if point["category"] == predicted_category]
    X_category = np.array([get_feature_vector(point["lat"], point["lng"], use_city_center) for point in category_points])
    
    # Find nearest neighbors within the predicted category
    if len(category_points) > 0:
        category_knn = NearestNeighbors(n_neighbors=min(k, len(category_points)), metric='euclidean')
        category_knn.fit(X_category)
        distances, indices = category_knn.kneighbors(query_point)
        
        similar_points = []
        for idx, distance in zip(indices[0], distances[0]):
            point = category_points[int(idx)]
            similar_points.append({
                "lat": point["lat"],
                "lng": point["lng"],
                "title": point["title"],
                "description": point["description"],
                "distance": float(distance)
            })
    else:
        similar_points = []
    
    return jsonify({
        "query": {"lat": lat, "lng": lng},
        "predicted_category": predicted_category,
        "probabilities": {class_names[i]: float(probabilities[i]) for i in range(len(class_names))},
        "similar_points": similar_points
    })

if __name__ == "__main__":
    app.run(debug=True)
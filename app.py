from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from pymongo import MongoClient
import base64
import io
from PIL import Image
import os
from dotenv import load_dotenv

# ✅ Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# ✅ MongoDB Atlas connection
mongo_uri = os.environ.get("MONGO_URI")
database_name = os.environ.get("DATABASE_NAME", "intelliface_db")
client = MongoClient(mongo_uri)
db = client[database_name]
users_collection = db['users']

# ✅ Test MongoDB connection
try:
    print("🔗 MongoDB Connected. Total user documents:", users_collection.count_documents({}))
except Exception as e:
    print("❌ MongoDB connection failed:", e)

# ✅ Decode base64 image and convert to NumPy array
def loadImageFromBase64(image_base64_str):
    try:
        if ',' in image_base64_str:
            _, encoded = image_base64_str.split(",", 1)
        else:
            encoded = image_base64_str
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

# ✅ Enroll face route
@app.route('/enroll_face', methods=['POST'])
def enroll_face():
    data = request.get_json()
    if not data or 'employee_id' not in data or 'image_base64' not in data:
        return jsonify({"error": "Missing employee_id or image_base64"}), 400

    employee_id = data['employee_id']
    image_array = loadImageFromBase64(data['image_base64'])
    if image_array is None:
        return jsonify({"error": "Invalid base64 image"}), 400

    face_locations = face_recognition.face_locations(image_array, model="hog")
    if not face_locations:
        return jsonify({"error": "No face detected"}), 400
    if len(face_locations) > 1:
        return jsonify({"error": "Multiple faces detected"}), 400

    try:
        encoding = face_recognition.face_encodings(image_array, known_face_locations=[face_locations[0]])[0]
    except Exception as e:
        return jsonify({"error": f"Face encoding failed: {e}"}), 500

    result = users_collection.update_one(
        {'_id': employee_id},
        {
            '$push': {'faceEncodings': encoding.tolist()},
            '$setOnInsert': {'employee_id_ref': employee_id}
        },
        upsert=True
    )

    updated_user = users_collection.find_one({'_id': employee_id})
    num_encodings = len(updated_user['faceEncodings']) if updated_user and 'faceEncodings' in updated_user else 0

    return jsonify({
        "message": "Face enrolled successfully",
        "employee_id": employee_id,
        "num_encodings_for_user": num_encodings,
        "mongo_result": {
            "matched": result.matched_count,
            "modified": result.modified_count,
            "upserted_id": str(result.upserted_id) if result.upserted_id else None
        }
    }), 201

# ✅ Verify face route
@app.route('/verify_face', methods=['POST'])
def verify_face():
    data = request.get_json()
    if not data or 'employee_id' not in data or 'image_base64_to_check' not in data:
        return jsonify({"error": "Missing employee_id or image_base64_to_check"}), 400

    employee_id = data['employee_id']
    image_array = loadImageFromBase64(data['image_base64_to_check'])
    if image_array is None:
        return jsonify({"match": False, "reason": "Invalid base64 image"}), 400

    user_doc = users_collection.find_one({'_id': employee_id})
    if not user_doc or not user_doc.get('faceEncodings'):
        return jsonify({"match": False, "reason": "No enrolled faces"}), 200

    known_encodings = [np.array(enc) for enc in user_doc['faceEncodings']]
    face_locations = face_recognition.face_locations(image_array, model="hog")
    if not face_locations:
        return jsonify({"match": False, "reason": "No face detected"}), 200
    if len(face_locations) > 1:
        return jsonify({"match": False, "reason": "Multiple faces detected"}), 200

    try:
        unknown_encoding = face_recognition.face_encodings(image_array, known_face_locations=face_locations)[0]
    except Exception as e:
        return jsonify({"match": False, "reason": f"Encoding error: {e}"}), 500

    matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.55)
    is_match = any(matches)

    if is_match:
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        best_distance = min(d for i, d in enumerate(face_distances) if matches[i])
        return jsonify({
            "match": True,
            "employee_id": employee_id,
            "distance": float(best_distance)
        }), 200
    else:
        min_distance = float(min(face_recognition.face_distance(known_encodings, unknown_encoding)))
        return jsonify({
            "match": False,
            "employee_id": employee_id,
            "reason": "No match found",
            "min_distance": min_distance
        }), 200

# ✅ Start the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True)

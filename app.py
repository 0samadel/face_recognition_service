# --- face_recognition_service/app.py ---
from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from pymongo import MongoClient
import base64
import io
from PIL import Image

app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017')
db = client['intelliface']
users_collection = db['users']

def loadImageFromBase64(image_base64_str):
    try:
        if ',' in image_base64_str:
            header, encoded = image_base64_str.split(",", 1)
        else:
            encoded = image_base64_str
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        print(f"Python Service Error: Loading image from base64 - {e}")
        return None

@app.route('/enroll_face', methods=['POST'])
def enroll_face():
    print("Python Service: Received request for /enroll_face")
    data = request.get_json()
    if not data or 'employee_id' not in data or 'image_base64' not in data:
        return jsonify({"error": "Missing employee_id or image_base64"}), 400

    employee_id = data['employee_id'] # This is the MongoDB _id from main users DB
    image_base64_str = data['image_base64']
    print(f"Python Service: Enrolling face for employee_id: {employee_id}")

    image_array = loadImageFromBase64(image_base64_str)
    if image_array is None:
        return jsonify({"error": "Could not load image from base64 string for enrollment."}), 400

    face_locations = face_recognition.face_locations(image_array, model="hog") # Use "cnn" for more accuracy if you have dlib with CUDA
    if not face_locations:
        print(f"Python Service: No face found in enrollment image for employee_id: {employee_id}")
        return jsonify({"error": "No face found in the provided image for enrollment."}), 400
    if len(face_locations) > 1:
        print(f"Python Service: Multiple faces found in enrollment image for employee_id: {employee_id}")
        return jsonify({"error": "Multiple faces detected. Please use an image with a single clear face."}), 400

    try:
        # Using the first detected face
        current_encoding = face_recognition.face_encodings(image_array, known_face_locations=[face_locations[0]])[0]
    except IndexError:
        print(f"Python Service Error: Could not get encoding for {employee_id}")
        return jsonify({"error": "Could not get face encoding from image."}), 500

    # Update MongoDB: Add encoding to the user's document
    # The employee_id here is the string representation of the MongoDB ObjectId from your main User collection
    result = users_collection.update_one(
        {'_id': employee_id}, # Match document by its _id
        {
            '$push': {'faceEncodings': current_encoding.tolist()},
            '$setOnInsert': {'employee_id_ref': employee_id} # Optional: store original ID if _id is an ObjectId object
        },
        upsert=True # Creates the document if it doesn't exist with this _id
    )
    
    print(f"Python Service: Face enrolled for {employee_id}. Matched: {result.matched_count}, Modified: {result.modified_count}, UpsertedId: {result.upserted_id}")
    
    # Fetch the updated document to count encodings reliably
    updated_user_doc = users_collection.find_one({'_id': employee_id})
    num_encodings = 0
    if updated_user_doc and 'faceEncodings' in updated_user_doc:
        num_encodings = len(updated_user_doc['faceEncodings'])
    
    return jsonify({
        "message": f"Face successfully enrolled for employee_id: {employee_id}.",
        "employee_id": employee_id,
        "num_encodings_for_user": num_encodings,
        "mongo_result": {"matched": result.matched_count, "modified": result.modified_count, "upserted_id": str(result.upserted_id) if result.upserted_id else None}
    }), 201

# verify_face can remain largely the same
@app.route('/verify_face', methods=['POST'])
def verify_face():
    print("Python Service: Received request for /verify_face")
    data = request.get_json()
    if not data or 'employee_id' not in data or 'image_base64_to_check' not in data:
        return jsonify({"error": "Missing employee_id or image_base64_to_check"}), 400

    employee_id = data['employee_id'] # MongoDB _id
    image_to_check_base64 = data['image_base64_to_check']
    print(f"Python Service: Verifying face for employee_id: {employee_id}")

    user_doc = users_collection.find_one({'_id': employee_id}) # Find by MongoDB _id
    if not user_doc or not user_doc.get('faceEncodings') or not user_doc['faceEncodings']: # Check if list is empty too
        print(f"Python Service: No enrolled faces found for employee_id: {employee_id}")
        return jsonify({"match": False, "employee_id": employee_id, "reason": "No enrolled faces for this employee."}), 200

    known_encodings = [np.array(enc) for enc in user_doc['faceEncodings']]
    loaded_image_to_check = loadImageFromBase64(image_to_check_base64)
    if loaded_image_to_check is None:
        return jsonify({"match": False, "employee_id": employee_id, "reason": "Could not load image_to_check from base64."}), 400 # Changed to 400

    unknown_face_locations = face_recognition.face_locations(loaded_image_to_check, model="hog") # Use "cnn" for more accuracy
    if not unknown_face_locations:
        print(f"Python Service: No face found in image to verify for employee_id: {employee_id}")
        return jsonify({"match": False, "employee_id": employee_id, "reason": "No face detected in the provided snapshot."}), 200
    if len(unknown_face_locations) > 1:
         print(f"Python Service: Multiple faces found in verification image for employee_id: {employee_id}")
         return jsonify({"match": False, "employee_id": employee_id, "reason": "Multiple faces detected in snapshot. Please try again."}), 200


    try:
        unknown_encodings = face_recognition.face_encodings(loaded_image_to_check, known_face_locations=unknown_face_locations)
        if not unknown_encodings:
            print(f"Python Service Error: Could not get encoding from verification image for {employee_id}")
            return jsonify({"match": False, "employee_id": employee_id, "reason": "Could not process face from snapshot."}), 200 # Or 500
        unknown_encoding = unknown_encodings[0] # Use first detected face
    except Exception as e:
        print(f"Python Service Error: Encoding verification image failed for {employee_id}: {e}")
        return jsonify({"match": False, "employee_id": employee_id, "reason": "Error processing snapshot features."}), 500

    # Tolerance: Lower value means stricter matching. 0.6 is standard, 0.5 to 0.55 is stricter.
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.55) 
    is_match_found = any(matches) # True if any known_encoding matches unknown_encoding

    if is_match_found:
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        # Get the distance for the best match (smallest distance) among all known encodings that matched
        best_match_distance = min(d for i, d in enumerate(face_distances) if matches[i])
        print(f"Python Service: Face MATCH for {employee_id}. Best Distance: {best_match_distance:.4f}")
        return jsonify({"match": True, "employee_id": employee_id, "distance": float(best_match_distance)}), 200
    else:
        # If no match, you could still return the minimum distance to any known face for debugging
        min_distance_if_no_match = min(face_recognition.face_distance(known_encodings, unknown_encoding)) if known_encodings else float('inf')
        print(f"Python Service: Face NO MATCH for {employee_id}. Min distance: {min_distance_if_no_match:.4f}")
        return jsonify({"match": False, "employee_id": employee_id, "reason": "Face does not match enrolled data.", "min_distance": float(min_distance_if_no_match)}), 200

if __name__ == '__main__':
    # For production, use a WSGI server like Gunicorn:
    # gunicorn -w 4 -b 0.0.0.0:5001 app:app
    app.run(host='0.0.0.0', port=5001, debug=True) # debug=False for production
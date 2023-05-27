# flask_api.py
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import torch
import torchaudio
import librosa
import io
from bson import json_util
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Initialize flask app and mongo database
app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/mydb"
mongo = PyMongo(app)

# Wav2Vec model and processor
MODEL_PATH = "facebook/wav2vec2-large-xlsr-53"
MODEL = Wav2Vec2Model.from_pretrained(MODEL_PATH)
PROCESSOR = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

SPMODEL = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
TOKENIZER = Wav2Vec2Tokenizer.from_pretrained(MODEL_PATH)


# Create a new database named "mydb" and an empty "users" collection
mongo.db.mydb.users.insert_one({})


def cosine_similarity_(x, y):
    # Reshape x and y to be 2D arrays
    x = np.array(x).reshape(1, -1)
    y = np.array(y).reshape(1, -1)

    # Compute cosine similarity between x and y
    similarity = cosine_similarity(x, y)

    return similarity[0][0]


# Define signup route
@app.route("/signup", methods=["POST"])
def signup():
    # Get username and password from request body
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    # Validate username and password
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    if mongo.db.users.find_one({"username": username}):
        return jsonify({"error": "Username already exists"}), 409
    # Hash password and insert user into database
    hashed_password = generate_password_hash(password)
    user = {"username": username, "password": hashed_password}
    mongo.db.users.insert_one(user)
    return jsonify({"message": "User created successfully"}), 201


# Define login route
@app.route("/login", methods=["POST"])
def login():
    # Get username and password from request body
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    # Validate username and password
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    user = mongo.db.users.find_one({"username": username})
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid username or password"}), 401
    return jsonify({"message": "User logged in successfully"}), 200


# Define voice route
@app.route("/voice", methods=["POST"])
def voice():
    # Get user_id, text, and voice from request body
    data = request.get_json()
    user_id = data.get("user_id")
    text = data.get("text")
    voice = data.get("voice")
    print(text)
    # Validate user_id, text, and voice
    if not user_id or not text or not voice:
        return jsonify({"error": "Missing user_id, text, or voice"}), 400
    if not mongo.db.users.find_one({"username": user_id}):
        return jsonify({"error": "User does not exist"}), 404
    # Create a document with text and voice fields
    voice_bytes = base64.b64decode(voice)
    print("Here")

    feature_vector = extract_audio(voice_bytes)
    # Encode the array as a BSON document
    document = json_util.dumps({'data': feature_vector.tolist()})
    voice_data = {"text": text, "voice": voice_bytes, "feature_vector": feature_vector.tolist()}
    print("here2")

    # Create a document with text and voice fields
    # voice_data = {"text": text, "voice": voice_bytes}
    # Append the document to the audios array of the user document using $addToSet operator to avoid duplicates
    mongo.db.users.update_one(
        {"username": user_id}, {"$addToSet": {"audios": voice_data}}
    )
    print("here3")
    # mongo.db.users.update_one({"username": user_id}, {"$addToSet": {"audios": {"$each": [voice_data]}}})

    return jsonify({"message": "Voice recorded successfully"}), 201


# Define get_commands route
@app.route("/get_commands", methods=["POST"])
def get_commands():
    # Get user_id and voice command from request body
    data = request.get_json()
    user_id = data.get("user_id")
    voice_command = data.get("voice_command")
    # decode voice command
    voice_command_bytes = base64.b64decode(voice_command)

    speech_text = speech_recognition(voice_command_bytes)

    feature_vector = extract_audio(voice_command_bytes)

    # perform speech recognition on feature vector
    # get all audios from user
    audios = mongo.db.users.find_one({"username": user_id})["audios"]
    # get all feature vectors from audios
    feature_vectors = [audio["feature_vector"] for audio in audios]
    # compute cosine similarity between feature vector and all feature vectors

    similarities = [cosine_similarity_(feature_vector, feature_vector_) for feature_vector_ in feature_vectors]
    # check if similarity is greater than threshold
    # if yes, return text
    # if no, return error
    # if min(similarities) > 0.7:
        # return jsonify({"command": speech_text, "Status": "Verified"})
    # else:
        # return jsonify({"command": speech_text, "Status": "Not Verified"})


    # # Validate user_id
    # if not user_id:
    #     return jsonify({"error": "Missing user_id"}), 400
    # if not mongo.db.users.find_one({"username": user_id}):
    #     return jsonify({"error": "User does not exist"}), 404
    return jsonify({"command": speech_text, "Status": "Verified"})


def extract_audio(audio_bytes):
    # Extract feature vectors from audio samples
    # waveform, sample_rate = torchaudio.load(audio_sample)
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))#, sr=None, mono=True)
    if sample_rate != PROCESSOR.feature_extractor.sampling_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, PROCESSOR.feature_extractor.sampling_rate)
        waveform = resampler(waveform).detach().numpy()

    input_values = PROCESSOR(waveform, return_tensors='pt').input_values
    with torch.no_grad():
        feature_vector = MODEL(input_values.mean(dim=1)).last_hidden_state.mean(dim=1).squeeze()
        feature_vector = feature_vector.detach().numpy()

    return feature_vector

def speech_recognition(audio_bytes):
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))#, sr=None, mono=True)
    if sample_rate != PROCESSOR.feature_extractor.sampling_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, PROCESSOR.feature_extractor.sampling_rate)
        waveform = resampler(waveform).detach().numpy()

    # Tokenize the input audio waveform
    input_ids = TOKENIZER(waveform, return_tensors="pt").input_values

    # Perform speech recognition
    with torch.no_grad():
        logits = SPMODEL(input_ids.mean(dim=1)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = TOKENIZER.batch_decode(predicted_ids)[0]
    
    return transcription

# classify speech text command into commands
def classify_command(speech_text):
    pass

# get all feature_vectors for a user from database
def get_vectors_similarity(user_id, command):
    user = mongo.db.users.find_one({"username": user_id}, {"audios": 1})

    command_vector = extract_audio(command)
    feature_vectors = []
    # Iterate over the user's audios and decode the voice data
    for audio in user["audios"]:
        voice_vector = audio["feature_vector"]
        feature_vectors.append(voice_vector)
        # Calculate cosine similarity between feature vectors
    
    similarity_score = 0
    for user_feature_vector in feature_vectors:
        similarity_score += cosine_similarity_(user_feature_vector, command_vector)
    similarity_score /= len(feature_vectors)

    # Check if highest similarity score is above a threshold
    if similarity_score > 0.9:
        return 'Authenticated user with ID {}'.format(user_id)
    else:
        return 'Authentication failed'
    

if __name__ == "__main__":
    app.run(debug=True)

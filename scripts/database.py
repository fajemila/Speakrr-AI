import pymongo
from pymongo import MongoClient
import torchaudio, torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# MongoDB connection details
MONGO_URL = 'mongodb://localhost:27017/'
DB_NAME = 'voice_auth'
COLLECTION_NAME = 'users'

# Wav2Vec model and processor
MODEL_PATH = 'facebook/wav2vec2-base-960h'
MODEL = Wav2Vec2Model.from_pretrained(MODEL_PATH)
PROCESSOR = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

# Number of audio samples to store for each user
NUM_AUDIO_SAMPLES = 5

USER_ID = 1

audio_files = ["data/Record-000.wav", "data/Record-001.wav", "data/Record-002.wav", "data/Record-003.wav", "data/Record-004.wav"]

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URL)

# Create database and collection
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Connect to MongoDB
client = MongoClient(MONGO_URL)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Get user ID from request
user_id = USER_ID

# Get audio samples from request
audio_samples = audio_files

# Extract feature vectors from audio samples
feature_vectors = []
for audio_sample in audio_samples:
    waveform, sample_rate = torchaudio.load(audio_sample)
    if sample_rate != PROCESSOR.feature_extractor.sampling_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, PROCESSOR.feature_extractor.sampling_rate)
        waveform = resampler(waveform)
        sample_rate = PROCESSOR.feature_extractor.sampling_rate

    input_values = PROCESSOR(waveform, return_tensors='pt').input_values
    with torch.no_grad():
        feature_vector = MODEL(input_values.mean(dim=1)).last_hidden_state.mean(dim=1).squeeze()
        feature_vectors.append(feature_vector.numpy().tolist())

# Store feature vectors in MongoDB
collection.update_one({'user_id': user_id}, {'$set': {'feature_vectors': feature_vectors}}, upsert=True)


def cosine_similarity_(x, y):
    # Reshape x and y to be 2D arrays
    x = np.array(x).reshape(1, -1)
    y = np.array(y).reshape(1, -1)
    
    # Compute cosine similarity between x and y
    similarity = cosine_similarity(x, y)
    
    return similarity[0][0]

def authenticate():
    # Connect to MongoDB
    client = MongoClient(MONGO_URL)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Get audio sample from request
    audio_sample = audio_files[0]

    # Extract feature vector from audio sample
    waveform, sample_rate = torchaudio.load(audio_sample)
    if sample_rate != PROCESSOR.feature_extractor.sampling_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, PROCESSOR.feature_extractor.sampling_rate)
        waveform = resampler(waveform)
        sample_rate = PROCESSOR.feature_extractor.sampling_rate
    
    input_values = PROCESSOR(waveform, return_tensors='pt').input_values
    with torch.no_grad():
        feature_vector = MODEL(input_values.mean(dim=1)).last_hidden_state.mean(dim=1).squeeze()
        feature_vector = feature_vector.numpy()

    # Get feature vectors for all users from MongoDB
    cursor = collection.find({"user_id":USER_ID}, {'_id': 0, 'user_id': 1, 'feature_vectors': 1})


    # Calculate cosine similarity between feature vectors
    similarity_scores = []
    for user in cursor:
        user_id = user['user_id']
        feature_vectors = user['feature_vectors']
        similarity_score = 0
        for user_feature_vector in feature_vectors:
            similarity_score += cosine_similarity_(user_feature_vector, feature_vector)
        similarity_score /= len(feature_vectors)
        similarity_scores.append({'user_id': user_id, 'similarity_score': similarity_score})

    # Sort similarity scores in descending order
    similarity_scores = sorted(similarity_scores, key=lambda x: x['similarity_score'], reverse=True)

    # Check if highest similarity score is above a threshold
    if similarity_scores[0]['similarity_score'] > 0.9:
        return 'Authenticated user with ID {}'.format(similarity_scores[0]['user_id'])
    else:
        return 'Authentication failed'
print(authenticate())
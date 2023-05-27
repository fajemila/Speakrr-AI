import requests

API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-large-960h"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("audio_data/librispeech_test-other/367/367-130732-0002.flac")
print(output)
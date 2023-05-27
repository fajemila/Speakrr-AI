## this scripts receives performs speaker identity for donald trump

import requests

from resemblyzer import preprocess_wav, VoiceEncoder
from utils import *
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

## Load and preprocess the audio
# Set the directory path where the audio data is stored
data_dir = Path("audio_data", "donald_trump")

# Retrieve the file paths of all the MP3 files in the specified directory
wav_fpaths = list(data_dir.glob("**/*.mp3"))

# Preprocess each WAV file, using the 'preprocess_wav' function, and store the preprocessed data in 'wavs'
wavs = [preprocess_wav(wav_fpath) for wav_fpath in tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit=" utterances")]

## Compute the embeddings
# Create an instance of the VoiceEncoder class to compute embeddings
encoder = VoiceEncoder()

# Compute the embeddings for each preprocessed WAV file and store them in 'embeds'
embeds = np.array([encoder.embed_utterance(wav) for wav in wavs])

# Extract the speaker names and file names from the WAV file paths
speakers = np.array([fpath.parent.name for fpath in wav_fpaths])
names = np.array([fpath.stem for fpath in wav_fpaths])

# Randomly select 6 indices from the embeddings where the speaker is "real" for training, and mark them as ground truth
gt_indices = np.random.choice(*np.where(speakers == "real"), 6, replace=False)

# Create a boolean mask to separate the ground truth embeddings from the remaining embeddings
mask = np.zeros(len(embeds), dtype=np.bool)
mask[gt_indices] = True

# Separate the ground truth embeddings, speaker names, and file names from the rest of the data
gt_embeds = embeds[mask]
gt_names = names[mask]
gt_speakers = speakers[mask]
embeds, speakers, names = embeds[~mask], speakers[~mask], names[~mask]

## Compare all embeddings against the ground truth embeddings, and compute the average similarities.
scores = (gt_embeds @ embeds.T).mean(axis=0)

# Order the scores by decreasing order
sort = np.argsort(scores)[::-1]
scores, names, speakers = scores[sort], names[sort], speakers[sort]

## Plot the scores
fig, _ = plt.subplots(figsize=(6, 6))
indices = np.arange(len(scores))
plt.axhline(0.84, ls="dashed", label="Prediction threshold", c="black")
plt.bar(indices[speakers == "real"], scores[speakers == "real"], color="green", label="Real")
plt.bar(indices[speakers == "fake"], scores[speakers == "fake"], color="red", label="Fake")
plt.legend()
plt.xticks(indices, names, rotation="vertical", fontsize=8)
plt.xlabel("Youtube video IDs")
plt.ylim(0.7, 1)
plt.ylabel("Similarity to ground truth")
fig.subplots_adjust(bottom=0.25)
plt.show()

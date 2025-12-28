import numpy as np
from scipy.io import wavfile

# 1. Config
SAMPLE_RATE = 44100
DURATION = 10.0  # Seconds
FREQ = 110.0     # Low A (Guitar string)

# 2. Generate Input (Clean)
# We create a sine wave that grows from Gain 0.0 to Gain 5.0
# This teaches the network how to handle "Quiet Clean" vs "Loud Distortion"
t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
envelope = np.linspace(0.0, 5.0, len(t)) 
input_audio = np.sin(2 * np.pi * FREQ * t) * envelope

# 3. The "Black Box" Function (Asymmetric Soft Clip)
# This is the "Secret Sauce" we want the Neural Network to steal.
# Equation: tanh(2x) + 0.15x^2
def black_box_pedal(x):
    # The clean signal enters...
    distortion = np.tanh(2.0 * x)
    
    # Add some "Tube Asymmetry" (Even Harmonics)
    asymmetry = 0.15 * (x**2)
    
    return distortion + asymmetry

target_audio = black_box_pedal(input_audio)

# 4. Normalize and Save (Standard WAV format)
# We shrink it slightly to fit in -1.0 to 1.0 range so it plays nicely
def save_wav(filename, data):
    # Clamp to avoid ugly digital clipping on save
    data = np.clip(data, -1.0, 1.0)
    wavfile.write(filename, SAMPLE_RATE, data.astype(np.float32))

print("Generating synthetic training data...")
save_wav("audioFiles/dataset_input.wav", input_audio * 0.2) # Save input at normal volume
save_wav("audioFiles/dataset_target.wav", target_audio * 0.5) # Save target slightly quieter
print("Done! Created 'dataset_input.wav' and 'dataset_target.wav'")
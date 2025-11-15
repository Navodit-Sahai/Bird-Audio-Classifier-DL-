# Bird Audio Classifier - Deep Learning 🐦🔊

A deep learning-based audio classification system that detects Capuchin bird calls from forest recordings using CNN architecture and spectrogram analysis.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4.1-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project implements a binary classifier to detect Capuchin bird calls from ambient forest recordings. The model processes audio files by converting them into spectrograms using Short-Time Fourier Transform (STFT) and uses a Convolutional Neural Network (CNN) to classify the presence of bird calls.

## 🚀 Features

- **Audio Processing Pipeline**: Automatic resampling from 44.1kHz to 16kHz
- **Spectrogram Generation**: STFT-based feature extraction (frame length: 320, stride: 32)
- **Deep Learning Model**: Custom CNN architecture with Conv2D layers
- **Batch Processing**: Efficient processing of multiple forest recordings
- **Real-time Detection**: Capable of processing and classifying audio streams
- **Post-processing**: Consecutive detection grouping using itertools
- **CSV Export**: Automated result export for analysis

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## 🔧 Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Navodit-Sahai/Bird-Audio-Classifier-DL-.git
cd Bird-Audio-Classifier-DL-
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Requirements:**
```
tensorflow==2.4.1
tensorflow-gpu==2.4.1
tensorflow-io
matplotlib
numpy
```

## 📁 Dataset Structure

```
data/
├── Parsed_Capuchinbird_Clips/     # Positive samples (bird calls)
│   └── *.wav
├── Parsed_Not_Capuchinbird_Clips/ # Negative samples (other sounds)
│   └── *.wav
└── Forest Recordings/              # Test recordings
    └── *.mp3
```

## 🏗️ Model Architecture

### CNN Architecture
```
Input Shape: (1491, 257, 1)  # Spectrogram dimensions
├── Conv2D(16, (3,3), activation='relu')
├── Conv2D(16, (3,3), activation='relu')
├── Flatten()
├── Dense(128, activation='relu')
└── Dense(1, activation='sigmoid')  # Binary classification
```

### Training Configuration
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Precision, Recall
- **Batch Size**: 16
- **Epochs**: 4

## 💻 Usage

### 1. Data Preprocessing

The preprocessing pipeline includes:
- Audio loading and resampling (44.1kHz → 16kHz)
- Zero-padding to 48,000 samples
- STFT conversion to spectrograms
- Dataset creation with labels

### 2. Training the Model

```python
# Run the Jupyter notebook
jupyter notebook AudioClassification.ipynb
```

Or run training programmatically:
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# Load and preprocess data
# ... (data loading code)

# Build model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257, 1)),
    Conv2D(16, (3,3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile('Adam', loss='BinaryCrossentropy', 
              metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
hist = model.fit(train, epochs=4, validation_data=test)
```

### 3. Making Predictions

```python
# Load a forest recording
wav = load_mp3_16k_mono('path/to/recording.mp3')

# Create audio slices and preprocess
audio_slices = tf.keras.utils.timeseries_dataset_from_array(
    wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1
)
audio_slices = audio_slices.map(preprocess_mp3).batch(64)

# Predict
predictions = model.predict(audio_slices)
```

### 4. Batch Processing

Process multiple forest recordings:
```python
results = {}
for file in os.listdir('data/Forest Recordings'):
    wav = load_mp3_16k_mono(file)
    # ... preprocessing
    predictions = model.predict(audio_slices)
    results[file] = predictions
```

## 📊 Results

The model processes 100 forest recordings and exports detection results to `results.csv`:

```csv
recording,capuchin_calls
recording_00.mp3,5
recording_01.mp3,0
recording_02.mp3,3
...
```

### Performance Metrics
- **Precision**: High precision in detecting bird calls
- **Recall**: Effective recall with minimal false negatives
- **Processing**: Batch size of 16 for efficient training

## 🔍 Key Technical Details

### Audio Preprocessing
- **Resampling**: 44.1kHz → 16kHz using TensorFlow-IO
- **Normalization**: Zero-padding to 48,000 samples
- **Feature Extraction**: STFT with frame_length=320, frame_step=32

### Model Features
- **Input**: Spectrogram of shape (1491, 257, 1)
- **Architecture**: 2 Conv2D layers + Dense layers
- **Output**: Binary classification (bird call present/absent)
- **Post-processing**: Consecutive detection grouping with itertools.groupby

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Navodit Sahai**

- GitHub: [@Navodit-Sahai](https://github.com/Navodit-Sahai)
- LinkedIn: [Navodit Sahai](https://www.linkedin.com/in/navodit-sahai-491aa032a)
- Email: navodit.2024ug1071@iiitranchi.ac.in

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- TensorFlow-IO for audio processing utilities
- IIIT Ranchi for academic support

## 📚 References

- [TensorFlow Audio Documentation](https://www.tensorflow.org/io/tutorials/audio)
- [STFT for Audio Processing](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
- [CNN for Audio Classification](https://arxiv.org/abs/1609.04243)

---

⭐ If you find this project useful, please consider giving it a star!

**Project Link**: [https://github.com/Navodit-Sahai/Bird-Audio-Classifier-DL-](https://github.com/Navodit-Sahai/Bird-Audio-Classifier-DL-)

# Audio CNN Visualizer

A web-based visualization tool for analyzing audio classifications using Convolutional Neural Networks (CNNs). This project provides an interactive interface to upload audio files, visualize CNN layer activations, and understand how the model processes audio data for classification.

---

## Overview

The Audio CNN Visualizer consists of two main components:

- **Backend (Python/Modal):** A trained CNN model that classifies environmental sounds from the ESC-50 dataset.
- **Frontend (Next.js):** A web interface that displays predictions, spectrograms, waveforms, and feature maps from each layer of the CNN.

---

## Features

- Upload and analyze WAV audio files.
- Real-time audio classification with confidence scores.
<img width="1901" height="645" alt="image" src="https://github.com/user-attachments/assets/221624a2-8e21-49f3-96fb-692ce8a646a4" />
- Interactive visualizations, including:
  - Input spectrograms
  - Audio waveforms
  - Convolutional layer feature maps
  - Internal layer activations
<img width="1898" height="575" alt="image" src="https://github.com/user-attachments/assets/b7749a39-a067-49d5-976c-d4bacd520183" />
- Color-coded feature map representations.
<img width="1899" height="801" alt="image" src="https://github.com/user-attachments/assets/4c16204b-17e3-471a-a21e-4076c7a018a6" />
- Responsive design built with Tailwind CSS.

---

## Technology Stack

### Backend
- Python 3.9+
- PyTorch: Deep learning framework for model training and inference.
- Torchaudio: Audio processing library.
- Modal: Serverless compute platform for model deployment.
- FastAPI: API framework for inference endpoints.
- Librosa: Audio analysis library.
- Soundfile: Audio file I/O.

### Frontend
- Next.js 15.2.3: React framework for production.
- React 19.0.0: UI library.
- TypeScript 5.8.2: Typed JavaScript.
- Tailwind CSS 4.0.15: Utility-first CSS framework.
- Radix UI: Headless UI components.
- Lucide React: Icon library.

---

## Installation

### Prerequisites
- Node.js 18 or higher
- Python 3.9 or higher
- Modal CLI (for backend deployment)

### Frontend Setup
```bash
# Clone the repository
git clone <repository-url>
cd audio-cnn-visualization

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Rock Paper Scissors Classifier and Real-Time Detection

## Description

This project implements a  model to classify images of rock, paper, and scissors gestures with high accuracy. The classifier is trained on the [Rock Paper Scissors Dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) and achieves a **97% accuracy** on the test set. Additionally, a real-time detection system is developed using MediaPipe to detect hands and classify gestures (Rock, Paper, Scissors, or Rest) using the trained model.

### Important Note:
The classifier performs best when the background of the images consists of a single color, similar to the dataset. This ensures accurate predictions for new (test) images.

---

## Features

- **Image Classification**: Classifies gestures (Rock, Paper, Scissors, Rest) with high accuracy.
- **Real-Time Detection**: Detects gestures in live video using MediaPipe for hand tracking.
- **Environment Management**: All dependencies managed using Poetry for easy setup.

---

## Installation

### Prerequisites

- Python 3.9 or later
- Poetry (for dependency management)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Almotasembelah/Rock-Paper-Scissor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Rock-Paper-Scissor
   ```
3. Install dependencies using Poetry:
   ```bash
   poetry install
   ```
---


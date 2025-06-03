# Digit Classification: Machine Learning & Deep Learning Examples

This repository contains simple examples for digit classification using:

- A **Machine Learning** model (Logistic Regression) trained on the **`load_digits`** dataset (8x8 pixel digits)
- A **Deep Learning** model (Neural Network) trained on the **MNIST** dataset (28x28 pixel digits)

---

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Notes](#notes)
- [License](#license)

---

## Overview

This project demonstrates two approaches to digit recognition:

1. **Machine Learning (ML)** — Logistic Regression on a smaller 8x8 pixel digits dataset (`sklearn.datasets.load_digits`).
2. **Deep Learning (DL)** — A simple neural network trained on the larger, more complex MNIST dataset (28x28 pixels).

---

## Datasets

- **`load_digits`** (scikit-learn):  
  8x8 grayscale images of handwritten digits (1797 samples). Smaller and simpler dataset.

- **MNIST** (TensorFlow/Keras):  
  28x28 grayscale images of handwritten digits (70,000 samples). Standard benchmark for deep learning.

---

## Models

### 1. Machine Learning — Logistic Regression

- Uses the 8x8 `load_digits` dataset.
- Fast to train, simple linear model.
- Accuracy around ~90%.

### 2. Deep Learning — Neural Network

- Uses the 28x28 MNIST dataset.
- Simple feedforward neural network with one hidden layer (128 neurons).
- Achieves ~98-99% accuracy.

---

## Installation

```bash
pip install scikit-learn tensorflow matplotlib

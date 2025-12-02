# CIFAR-10 Neural Network Project

## Overview
This project involves designing, implementing, evaluating, and comparing neural networks to classify images from the CIFAR-10 dataset. The project is divided into two main components:
1. A **Baseline Multi-Layer Perceptron (MLP)** model.
2. A **Deep Neural Network (DNN)** model with advanced techniques.

The goal is to demonstrate an understanding of deep learning principles, including optimization, regularization, and model evaluation, while ensuring reproducibility.

---

## Dataset
The CIFAR-10 dataset contains 60,000 color images (32x32 pixels) across 10 classes, such as airplane, car, bird, and cat. For this project:
- A subset of 5 classes was selected.
- The dataset was split into training (70%) and validation (30%) sets.
- The test set was used for final evaluation.

---

## Project Design
### 1. **Baseline MLP Model**
The baseline model is a Multi-Layer Perceptron (MLP) with the following design:
- **Architecture**:
  - 3 hidden layers.
  - Neurons per layer: `[250, 150, 85, 5]`.
- **Activation Functions**:
  - Hidden layers: ReLU.
  - Output layer: Softmax.
- **Loss Function**: Cross-Entropy Loss.
- **Optimizer**: Stochastic Gradient Descent (SGD).
- **Hyperparameters**:
  - Learning rate: `0.02`.
  - Batch size: `64`.
  - Number of iterations: `100`.
- **Validation Split**: 30% of the training data.

**Key Features**:
- The model was trained without batch normalization or dropout.
- Training and validation loss curves were generated to evaluate performance.

### 2. **Deep Neural Network (DNN)**
The deep model is a 12-layer neural network designed to improve performance over the baseline. It incorporates advanced techniques for optimization and regularization.

- **Architecture**:
  - 12 hidden layers.
  - Neurons per layer: `[300, 250, 205, 180, 100, 50, 70, 80, 50, 60, 90, 50, 5]`.
- **Techniques Used**:
  - **Dropout**: Applied with a rate of `0.20` to prevent overfitting.
  - **Batch Normalization**: Used to stabilize training and improve convergence.
  - **Weight Decay (L2 Regularization)**: Lambda value set to `0.001`.
- **Loss Function**: Cross-Entropy Loss.
- **Optimizer**: Adam Optimizer.
- **Hyperparameters**:
  - Learning rate: `0.001`.
  - Batch size: `128`.
  - Number of iterations: `100`.
  - Early stopping patience: `5` epochs.
- **Validation Split**: 30% of the training data.

**Key Features**:
- The model was trained with dropout and batch normalization to improve generalization.
- Training and validation loss curves were generated to evaluate performance.
- An ablation study was conducted to analyze the impact of dropout and batch normalization.

---

## Evaluation Metrics
Both models were evaluated using the following metrics:
1. **Accuracy**: Percentage of correctly classified samples.
2. **Loss**: Cross-entropy loss to measure prediction error.
3. **Confusion Matrix**: To analyze class-wise performance.
4. **Training and Validation Loss Curves**: To monitor convergence and overfitting.

---

## Results
### Baseline MLP Model:
- **Training Accuracy**: ~76%.
- **Validation Accuracy**: ~65%.
- Observed overfitting due to the absence of regularization techniques.

### Deep Neural Network:
- **Training Accuracy**: Improved over the baseline.
- **Validation Accuracy**: Improved over the baseline.
- Regularization techniques (dropout, weight decay) and batch normalization reduced overfitting and improved generalization.

---

## Reproducibility
- Random seeds were fixed to ensure reproducibility.
- All code was written in Python without using high-level libraries like PyTorch or TensorFlow.
- A `requirements.txt` file is included to specify dependencies.

---

## Important Notes
1. **Code Structure**:
   - `my_mlp.py`: Implements the baseline MLP model.
   - `my_deep_network.py`: Implements the deep neural network.
   - `network.py`: Contains the generic `network_model` class used by both models.
   - `my_utils.py`: Contains utility functions for data processing and evaluation.
2. **Techniques**:
   - The deep model incorporates dropout, batch normalization, and weight decay.
   - Early stopping was used to prevent overfitting.
3. **Graphs**:
   - Training and validation loss curves were generated for both models.
   - Confusion matrices were computed to analyze class-wise performance.

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the baseline MLP model:
   ```bash
   python3 my_mlp.py
   ```
3. Run the deep neural network:
   ```bash
   python3 my_deep_network.py
   ```

---

## Conclusion
This project demonstrates the design, implementation, and evaluation of neural networks for CIFAR-10 classification. The deep model outperformed the baseline MLP by incorporating advanced techniques like dropout, batch normalization, and weight decay. The results highlight the importance of regularization and optimization in improving model generalization.
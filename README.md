# Federated-Learning-Using-Pytorch-and-Flower-MNIST-dataset
## Description
This repository implements a Federated Learning framework using PyTorch and Flower, focusing on the MNIST handwritten digit classification dataset. It demonstrates how to train a Convolutional Neural Network (CNN) in a federated manner, allowing multiple clients to collaboratively train a model without sharing their raw data.
## Key Features

- **Data Handling**: Efficiently loads and preprocesses the MNIST dataset with normalization and transformation techniques to prepare it for training.
- **Model Architecture**: Implements a straightforward Convolutional Neural Network (CNN) suitable for image classification, providing a solid foundation for further experimentation.
- **Centralized Training**: Includes functions for training and evaluating the model in a traditional centralized setting, allowing users to establish baseline performance metrics.
- **Federated Learning Setup**: Supports partitioning the dataset into IID (Independent and Identically Distributed) clients, enabling users to simulate real-world federated training scenarios.
- **Custom Evaluation Metrics**: Tracks and visualizes model performance, offering insights into accuracy and loss across training rounds.

## Requirements

To run the project, you need the following:

- Python 3.8+
- PyTorch
- Flower
- Hugging Face Datasets
- Matplotlib

Install the required packages using:

```bash
pip install flwr[simulation] flwr-datasets matplotlib


# Lab Report: Deep Learning with PyTorch for Computer Vision

## Objective
The primary goal of this lab is to gain hands-on experience with the PyTorch library and develop various neural network architectures for computer vision tasks. The lab involves building and evaluating models such as CNN, Faster R-CNN, and Vision Transformer (ViT) on the MNIST dataset.

## Work Overview

### Part 1: CNN Classifier
#### Dataset
- The dataset used for this lab is the MNIST dataset, which contains handwritten digit images.
- Link: [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

#### Tasks & Solutions
1. **Building a CNN Classifier**:
   - Designed a Convolutional Neural Network (CNN) using PyTorch.
   - The CNN architecture included multiple convolutional layers with ReLU activation, followed by max-pooling layers and fully connected layers for classification.
   - Used cross-entropy loss and Adam optimizer for training.
   - Implemented GPU acceleration using CUDA for faster training.
   - Trained the model on MNIST, achieving high classification accuracy.

2. **Implementing Faster R-CNN**:
   - Adapted a Faster R-CNN model, which is primarily used for object detection.
   - Converted the classification task into an object detection problem where digits were treated as separate objects.
   - Used a pre-trained Faster R-CNN model and fine-tuned it on MNIST.
   - The model detected and classified digits in images, but required additional processing time compared to CNN.

3. **Comparison of CNN and Faster R-CNN**:
   - Evaluated both models using metrics like Accuracy, F1-score, Loss, and Training time.
   - CNN provided faster training and better accuracy for simple classification.
   - Faster R-CNN was computationally expensive but more robust for detecting and classifying digits in a flexible manner.

4. **Fine-tuning with Pretrained Models**:
   - Used VGG16 and AlexNet, both pre-trained on ImageNet, and fine-tuned them on MNIST.
   - Removed the last classification layers and replaced them with MNIST-specific layers.
   - Fine-tuned models achieved better accuracy than CNN due to transfer learning but required additional training time.
   - Compared the results and concluded that fine-tuned models performed better than CNN but were still less efficient than Faster R-CNN in terms of adaptability.

### Part 2: Vision Transformer (ViT)
#### Overview
- Vision Transformers (ViT) have gained prominence in image classification tasks since their introduction by Dosovitskiy et al. in 2020.

#### Tasks & Solutions
1. **Implementing ViT from Scratch**:
   - Followed the tutorial: [ViT Guide](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)
   - Built a ViT model using PyTorch, including:
     - Splitting images into patches.
     - Embedding patches and applying positional encodings.
     - Passing embeddings through multiple Transformer encoder layers.
     - Using a classification head for digit recognition.
   - Trained the ViT model on MNIST and optimized it using Adam.

2. **Result Interpretation & Comparison**:
   - Analyzed the results and observed that ViT outperformed CNN in terms of accuracy but required significantly more computational resources.
   - Compared the performance of ViT against CNN and Faster R-CNN:
     - ViT provided superior accuracy and handled image distortions better.
     - However, CNN trained faster and was easier to implement on lower-end hardware.
     - Faster R-CNN remained the best option for detection-based tasks rather than simple classification.
   - Concluded that ViT is a powerful model for classification but requires high computational power, making it suitable for large-scale image processing tasks.

## Conclusion
- This lab provided hands-on experience with different deep learning architectures for image classification.
- CNN and Faster R-CNN were implemented and compared, along with fine-tuning using VGG16 and AlexNet.
- Vision Transformer (ViT) was implemented from scratch and evaluated on the MNIST dataset.
- Performance comparisons highlighted the strengths and weaknesses of each model in terms of accuracy, training time, and computational efficiency.
- ViT proved to be a strong alternative to CNN but required more computational power, making it ideal for advanced classification tasks with sufficient hardware resources.


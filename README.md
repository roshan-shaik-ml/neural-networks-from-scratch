# Neural Networks from Scratch

A comprehensive study repository covering neural networks from basics (perceptron) to LLMs, including MLPs, backpropagation, CNNs, RNNs, transformers, attention mechanisms, pretraining, and fine-tuning.

## 📚 Study Roadmap

This repository provides a structured learning path to master neural networks from the ground up. Each topic builds upon previous concepts, creating a solid foundation for understanding modern AI systems.

### Phase 1: Foundations (Weeks 1-3)

#### 1.1 Mathematical Prerequisites
- [ ] **Linear Algebra Review**
  - Vectors, matrices, and tensor operations
  - Matrix multiplication and broadcasting
  - Eigenvalues and eigenvectors
  - Implementation: NumPy operations

- [ ] **Calculus & Optimization**
  - Partial derivatives and gradients
  - Chain rule (crucial for backpropagation)
  - Optimization basics: gradient descent variants
  - Implementation: Manual gradient calculations

- [ ] **Probability & Statistics**
  - Probability distributions
  - Bayes' theorem
  - Maximum likelihood estimation
  - Information theory basics

#### 1.2 The Perceptron
- [ ] **Theory**
  - Biological inspiration
  - Mathematical formulation
  - Activation functions (step, sigmoid, ReLU)
  - Perceptron learning algorithm

- [ ] **Implementation**
  - Single perceptron from scratch
  - Training on linearly separable data
  - Visualization of decision boundaries
  - Limitations and XOR problem

### Phase 2: Multi-Layer Perceptrons (Weeks 4-6)

#### 2.1 MLP Architecture
- [ ] **Forward Propagation**
  - Layer-wise computation
  - Activation functions comparison
  - Hidden layer representations
  - Implementation: Feed-forward network

- [ ] **Backpropagation Algorithm**
  - Derivation from chain rule
  - Error propagation through layers
  - Weight and bias updates
  - Implementation: Manual backprop

- [ ] **Training Dynamics**
  - Loss functions (MSE, cross-entropy)
  - Gradient descent variants (SGD, momentum)
  - Learning rate scheduling
  - Batch vs. mini-batch vs. online learning

#### 2.2 Regularization & Optimization
- [ ] **Overfitting Prevention**
  - L1/L2 regularization
  - Dropout technique
  - Early stopping
  - Data augmentation basics

- [ ] **Advanced Optimizers**
  - AdaGrad, RMSprop
  - Adam and variants
  - Learning rate decay strategies
  - Implementation comparisons

### Phase 3: Convolutional Neural Networks (Weeks 7-10)

#### 3.1 CNN Fundamentals
- [ ] **Convolution Operation**
  - Mathematical definition
  - Kernels/filters and feature maps
  - Stride, padding, and dilation
  - Implementation: 2D convolution

- [ ] **CNN Architecture**
  - Convolutional layers
  - Pooling layers (max, average)
  - Fully connected layers
  - Classic architectures: LeNet-5

#### 3.2 Advanced CNN Concepts
- [ ] **Modern Architectures**
  - AlexNet and deep learning revolution
  - VGGNet: depth matters
  - ResNet: skip connections
  - DenseNet, MobileNet, EfficientNet

- [ ] **Specialized Techniques**
  - Batch normalization
  - Transfer learning
  - Object detection basics
  - Semantic segmentation intro

### Phase 4: Recurrent Neural Networks (Weeks 11-14)

#### 4.1 RNN Basics
- [ ] **Sequential Data Processing**
  - Temporal dependencies
  - Hidden state concept
  - Vanilla RNN architecture
  - Implementation: Simple RNN

- [ ] **Training Challenges**
  - Backpropagation through time (BPTT)
  - Vanishing gradient problem
  - Exploding gradients
  - Gradient clipping

#### 4.2 Advanced RNN Variants
- [ ] **LSTM (Long Short-Term Memory)**
  - Cell state and gates mechanism
  - Forget, input, and output gates
  - Mathematical formulation
  - Implementation: LSTM cell

- [ ] **GRU (Gated Recurrent Unit)**
  - Simplified gating mechanism
  - Reset and update gates
  - LSTM vs. GRU comparison
  - Implementation: GRU cell

- [ ] **Bidirectional RNNs**
  - Forward and backward processing
  - Context from both directions
  - Applications in NLP
  - Implementation considerations

### Phase 5: Attention Mechanisms (Weeks 15-17)

#### 5.1 Attention Fundamentals
- [ ] **Motivation & Intuition**
  - Limitations of fixed-size representations
  - Selective focus concept
  - Alignment in sequence-to-sequence
  - Visual attention analogy

- [ ] **Basic Attention Mechanisms**
  - Additive (Bahdanau) attention
  - Multiplicative (Luong) attention
  - Attention score computation
  - Implementation: Basic attention

#### 5.2 Self-Attention & Transformers
- [ ] **Self-Attention Mechanism**
  - Query, Key, Value concept
  - Scaled dot-product attention
  - Multi-head attention
  - Implementation: Self-attention layer

- [ ] **Transformer Architecture**
  - Encoder-decoder structure
  - Positional encoding
  - Layer normalization
  - Feed-forward networks
  - Implementation: Mini-transformer

### Phase 6: Modern Language Models (Weeks 18-22)

#### 6.1 Pre-training Paradigms
- [ ] **Language Model Basics**
  - N-gram models to neural LMs
  - Autoregressive generation
  - Perplexity and evaluation metrics
  - Implementation: Simple language model

- [ ] **BERT & Masked Language Modeling**
  - Bidirectional encoding
  - Masked token prediction
  - Next sentence prediction
  - Pre-training vs. fine-tuning

- [ ] **GPT Series Evolution**
  - Generative pre-training
  - Transformer decoder architecture
  - Scaling laws and emergent abilities
  - From GPT-1 to modern variants

#### 6.2 Advanced Training Techniques
- [ ] **Transfer Learning in NLP**
  - Feature extraction vs. fine-tuning
  - Task-specific adaptations
  - Domain adaptation strategies
  - Implementation: Fine-tuning pipeline

- [ ] **Advanced Training Methods**
  - Gradient accumulation
  - Mixed precision training
  - Distributed training basics
  - Efficient fine-tuning (LoRA, adapters)

### Phase 7: Large Language Models & Modern AI (Weeks 23-26)

#### 7.1 LLM Architecture & Scaling
- [ ] **Scaling Laws**
  - Model size, data, compute relationships
  - Emergent abilities at scale
  - Efficiency considerations
  - Hardware requirements

- [ ] **Modern Architectures**
  - LLaMA, PaLM, GPT-4 insights
  - Mixture of Experts (MoE)
  - Retrieval-augmented generation (RAG)
  - Multi-modal models intro

#### 7.2 LLM Ecosystem & Applications
- [ ] **Training Infrastructure**
  - Data preprocessing pipelines
  - Distributed training systems
  - Model parallelism strategies
  - Gradient checkpointing

- [ ] **Inference & Deployment**
  - Model quantization techniques
  - KV-cache optimization
  - Speculative decoding
  - Edge deployment considerations

- [ ] **Safety & Alignment**
  - Constitutional AI principles
  - RLHF (Reinforcement Learning from Human Feedback)
  - Bias detection and mitigation
  - Responsible AI practices

## 🛠 Implementation Framework

Each phase includes:
- **Theory Notebooks**: Mathematical derivations and explanations
- **Implementation Code**: From-scratch implementations using NumPy/PyTorch
- **Experiments**: Hands-on projects and visualizations
- **Comparisons**: Different approaches and trade-offs
- **Applications**: Real-world use cases and demos

## 📁 Repository Structure

```
neural-networks-from-scratch/
├── 01-foundations/
│   ├── math-prerequisites/
│   ├── perceptron/
│   └── exercises/
├── 02-mlp/
│   ├── forward-propagation/
│   ├── backpropagation/
│   ├── optimizers/
│   └── regularization/
├── 03-cnn/
│   ├── convolution-basics/
│   ├── architectures/
│   └── applications/
├── 04-rnn/
│   ├── vanilla-rnn/
│   ├── lstm-gru/
│   └── applications/
├── 05-attention/
│   ├── basic-attention/
│   ├── transformers/
│   └── implementations/
├── 06-language-models/
│   ├── pre-training/
│   ├── fine-tuning/
│   └── evaluation/
├── 07-llm-ecosystem/
│   ├── scaling/
│   ├── deployment/
│   └── safety/
├── utils/
│   ├── data-loaders/
│   ├── visualization/
│   └── metrics/
└── resources/
    ├── papers/
    ├── datasets/
    └── references/
```

## 🎯 Learning Objectives

By completing this roadmap, you will:
- Understand neural networks from mathematical foundations to modern applications
- Implement key algorithms from scratch for deep understanding
- Gain practical experience with state-of-the-art models
- Develop skills in model training, evaluation, and deployment
- Build intuition for designing and debugging neural architectures

## 📖 Recommended Resources

### Books
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

### Papers (Key Milestones)
- Rosenblatt (1958): The Perceptron
- Rumelhart et al. (1986): Backpropagation
- LeCun et al. (1989): CNNs for handwritten digits
- Hochreiter & Schmidhuber (1997): LSTM
- Vaswani et al. (2017): Attention Is All You Need
- Devlin et al. (2018): BERT
- Radford et al. (2019): GPT-2
- Brown et al. (2020): GPT-3

### Online Courses
- CS231n: Convolutional Neural Networks for Visual Recognition
- CS224n: Natural Language Processing with Deep Learning
- Fast.ai Practical Deep Learning courses

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add implementations and examples
- Improve explanations and documentation
- Suggest additional topics or resources
- Report issues or bugs

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🌟 Acknowledgments

This roadmap is inspired by the collective wisdom of the deep learning community, seminal research papers, and educational resources that have shaped our understanding of neural networks.

---

*Happy Learning! 🚀*

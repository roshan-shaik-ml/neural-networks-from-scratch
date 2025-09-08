# 02-MLP (Multi-Layer Perceptron)

This directory contains Multi-Layer Perceptron implementations and training algorithms.

## Contents

### Forward Propagation
- **Layer-wise Computation**: Feed-forward network implementation
- **Activation Functions**: Sigmoid, ReLU, Tanh comparisons
- **Hidden Representations**: Understanding learned features

### Backpropagation Algorithm
- **Derivation**: From chain rule to error propagation
- **Implementation**: Manual backpropagation from scratch
- **Weight Updates**: Gradient computation and parameter updates

### Optimizers
- **Gradient Descent Variants**: SGD, momentum, Nesterov
- **Adaptive Methods**: AdaGrad, RMSprop, Adam
- **Learning Rate Scheduling**: Decay strategies and comparisons

### Regularization
- **L1/L2 Regularization**: Weight penalty methods
- **Dropout**: Random neuron deactivation
- **Early Stopping**: Preventing overfitting
- **Batch Normalization**: Input normalization techniques

## Structure
```
02-mlp/
├── forward-propagation/
│   ├── mlp_architecture.py
│   ├── activation_functions.py
│   └── layer_computation.py
├── backpropagation/
│   ├── backprop_algorithm.py
│   ├── gradient_computation.py
│   └── chain_rule_demo.py
├── optimizers/
│   ├── sgd_variants.py
│   ├── adaptive_optimizers.py
│   └── learning_rate_schedules.py
└── regularization/
    ├── weight_penalties.py
    ├── dropout.py
    └── early_stopping.py
```

## Learning Path
1. Implement forward propagation
2. Derive and code backpropagation
3. Compare different optimizers
4. Apply regularization techniques
5. Train on real datasets

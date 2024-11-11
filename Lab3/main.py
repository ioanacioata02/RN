import numpy as np
from torchvision.datasets import MNIST


def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten() / 255.0,
                    download=True,
                    train=is_train)

    mnist_data = np.array([np.array(image) for image, _ in dataset])
    mnist_labels = np.array([label for _, label in dataset])

    return mnist_data, mnist_labels


train_X, train_Y = download_mnist(is_train=True)
test_X, test_Y = download_mnist(is_train=False)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]


def accuracy(Y_pred, Y_true):
    predictions = np.argmax(Y_pred, axis=1)
    targets = np.argmax(Y_true, axis=1)
    return np.mean(predictions == targets)


def forward_propagation(X, weights, biases):
    activations = [X]
    zs = []

    output_layer_index = len(weights) - 1

    for i, (w, b) in enumerate(zip(weights, biases)):
        z = np.dot(activations[-1], w) + b
        zs.append(z)

        if i == output_layer_index:
            activation = softmax(z)
        else:
            activation = sigmoid(z)

        activations.append(activation)

    return activations, zs


def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights = [
        np.random.randn(input_size, hidden_size) * 0.01,
        np.random.randn(hidden_size, output_size) * 0.01
    ]
    biases = [
        np.zeros((1, hidden_size)),
        np.zeros((1, output_size))
    ]
    return weights, biases


def update_parameters(weights, biases, gradient_W, gradient_b, learning_rate):
    for l in range(len(weights)):
        weights[l] -= learning_rate * gradient_W[l]
        biases[l] -= learning_rate * gradient_b[l]
    return weights, biases


def backward_propagation(X, y, activations, zs, weights, lambda_reg=0.001):
    m = X.shape[0]

    gradient_W = [np.zeros_like(w) for w in weights]
    gradient_b = [np.zeros((1, b.shape[1])) for b in biases]

    error = activations[-1] - y

    gradient_b[-1] = np.sum(error, axis=0, keepdims=True) / m
    gradient_W[-1] = np.dot(activations[-2].T, error) / m + (lambda_reg / m) * weights[-1]

    for l in range(len(weights) - 2, -1, -1):
        z = zs[l]
        sp = sigmoid_prime(z)

        error = np.dot(error, weights[l + 1].T) * sp

        gradient_b[l] = np.sum(error, axis=0, keepdims=True) / m
        gradient_W[l] = np.dot(activations[l].T, error) / m + (lambda_reg / m) * weights[l]

    return gradient_W, gradient_b


def train(X, y, weights, biases, epochs=150, batch_size=128, learning_rate=0.1, lambda_reg=0.001):
    for epoch in range(epochs):
        permutation = np.random.permutation(X.shape[0])
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            activations, zs = forward_propagation(X_batch, weights, biases)
            gradient_W, gradient_b = backward_propagation(X_batch, y_batch, activations, zs, weights, lambda_reg)
            weights, biases = update_parameters(weights, biases, gradient_W, gradient_b, learning_rate)

        if (epoch + 1) % 5 == 0:
            activations_train, _ = forward_propagation(X, weights, biases)
            loss = -np.mean(np.sum(y * np.log(activations_train[-1]), axis=1)) + (lambda_reg / (2 * X.shape[0])) * sum(
                np.sum(w ** 2) for w in weights)

            train_accuracy = accuracy(activations_train[-1], y)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")

    return weights, biases


train_Y_one_hot = one_hot_encode(train_Y)
test_Y_one_hot = one_hot_encode(test_Y)

weights, biases = initialize_parameters(784, 100, 10)
weights, biases = train(train_X, train_Y_one_hot, weights, biases)

activations_test, _ = forward_propagation(test_X, weights, biases)
test_accuracy = accuracy(activations_test[-1], test_Y_one_hot)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

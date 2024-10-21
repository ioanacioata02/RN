import numpy as np
from torchvision.datasets import MNIST


def download_mnist(is_train: bool):
    dataset = MNIST(root='./ data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return mnist_data, mnist_labels


def normalize(X):
    return X / 255.0


def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X, W, b):
    z = np.dot(X, W) + b
    return softmax(z)


def cross_entropy_loss(Y_true, Y_pred):
    return -np.sum(Y_true * np.log(Y_pred))


def backward(X, Y_true, Y_hat, W, b, learning_rate):
    dZ = Y_true - Y_hat
    dW = np.dot(X.T, dZ)
    db = np.sum(dZ, axis=0)

    W += learning_rate * dW
    b += learning_rate * db

    return W, b


def train_perceptron(X_train, Y_train, epochs=50, batch_size=100,
                     learning_rate=0.01, decrease_rate=0.9, decrease_step=10):
    nr_samples, nr_features = X_train.shape
    nr_classes = Y_train.shape[1]

    W = np.zeros((nr_features, nr_classes))
    b = np.zeros(nr_classes)

    for epoch in range(epochs):
        if (epoch + 1) % decrease_step == 0:
            learning_rate *= decrease_rate

        indices = np.random.permutation(nr_samples)
        X_train = X_train[indices]
        Y_train = Y_train[indices]

        for i in range(0, nr_samples, batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            Y_pred = forward(X_batch, W, b)
            W, b = backward(X_batch, Y_batch, Y_pred, W, b, learning_rate)

        # if epoch % 10 == 0:
        #     Y_pred_train = forward(X_train, W, b)
        #     loss = cross_entropy_loss(Y_train, Y_pred_train)
        #     print(f'Epoch {epoch}, Cross Entropy: {loss}')

    return W, b


def accuracy(Y_pred, Y_true):
    predictions = np.argmax(Y_pred, axis=1)
    targets = np.argmax(Y_true, axis=1)
    return np.mean(predictions == targets)


train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

train_X = np.array(train_X)
test_X = np.array(test_X)

train_X = normalize(train_X)
test_X = normalize(test_X)

train_Y_one_hot = one_hot_encode(train_Y)
test_Y_one_hot = one_hot_encode(test_Y)

weigh, bias = train_perceptron(train_X, train_Y_one_hot, epochs=100)

Y_pred_test = forward(test_X, weigh, bias)
test_accuracy = accuracy(Y_pred_test, test_Y_one_hot)
print(f'Test Accuracy: {test_accuracy}')

import random
import numpy as np

class network(object):
    def __init__(self, sizes, weight_initialization="random"):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # use the random init
        if weight_initialization == "random":
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # use the improved init
        elif weight_initialization == "improved":
            self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

        # invalid weight init so raise error
        else:
            raise ValueError("weight_initialization must be 'random' or 'improved'")

    def feedforward(self, a):
        """ Return the output of the network if "a" is input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, cost_function="quadratic", image_shift=0, test_data=None, random_stat=None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, cost_function, image_shift)

            if test_data:
                correct = self.evaluate(test_data)
                accuracy = (correct / n_test) * 100

                print("Epoch {0}: {1} / {2} ({3:.2f}%)".format(j, correct, n_test, accuracy))
            else:
                print("epoch {0} complete".format(j))

        if test_data:
            final_correct = self.evaluate(test_data)
            final_accuracy = (final_correct / n_test) * 100

            print("\nTraining complete")
            print("Hidden layer neuron amount:", random_stat) #used to test running the model with different values
            print("Final accuracy: {0:.2f}%".format(final_accuracy))
            print("Correct predictions: {0} / {1}".format(final_correct, n_test))

            return final_accuracy, final_correct

    def update_mini_batch(self, mini_batch, eta, cost_function, image_shift):
        """Update the network’s weights and biases by applying gradient descent
        using backpropagation to a single mini batch. The "mini_batch" is a list
        of tuples "(x, y)", and "eta" is the learning rate"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            x = random_shift_image(x, max_shift=image_shift)

            delta_nabla_b, delta_nabla_w = self.backprop(x, y, cost_function)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y, cost_function="quadratic"):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        if cost_function == "quadratic":
            delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        elif cost_function == "cross_entropy":
            delta = self.cost_derivative(activations[-1], y)

        else:
            raise ValueError("cost_function must be 'quadratic' or 'cross_entropy'")

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activation, y):
        return (output_activation - y)

def random_shift_image(x, max_shift=1):
    image = x.reshape(28, 28)

    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)

    shifted = np.zeros((28, 28))

    old_x_start = max(0, -dx)
    old_x_end = min(28, 28 - dx)

    old_y_start = max(0, -dy)
    old_y_end = min(28, 28 - dy)

    new_x_start = max(0, dx)
    new_x_end = min(28, 28 + dx)

    new_y_start = max(0, dy)
    new_y_end = min(28, 28 + dy)

    shifted[new_y_start:new_y_end, new_x_start:new_x_end] = image[
        old_y_start:old_y_end,
        old_x_start:old_x_end
    ]

    return shifted.reshape(784, 1)

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
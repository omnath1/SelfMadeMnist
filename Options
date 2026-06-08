WEIGHT_INITIALIZATION

random:
    sets the weights randomly using a standard normal distribution
    can cause activations to become too large in deeper networks

improved:
    sets the weights randomly using a scaled normal distribution
    divides weights by sqrt(number_of_inputs)
    helps keep activations at a reasonable size
    usually trains faster and more reliably


COST_FUNCTION

quadratic:
    original Chapter 1 cost function
    output error is multiplied by sigmoid_prime
    can learn slowly when neurons are saturated

cross_entropy:
    Chapter 3 improved cost function
    output error is not multiplied by sigmoid_prime
    helps the network learn faster when predictions are very wrong
    usually works better for classification problems like MNIST


LMBDA

0.0:
    no L2 regularization
    weights are updated using only gradient descent

> 0.0:
    enables L2 regularization (weight decay)
    discourages very large weights
    helps reduce overfitting
    larger values apply stronger regularization


OUTPUT_ACTIVATION

sigmoid:
    original Chapter 1 output activation
    each output neuron is activated independently
    outputs values between 0 and 1
    works with quadratic cost and cross entropy cost
    does not guarantee that all outputs sum to 1

softmax:
    converts output activations into probabilities
    all output values sum to 1
    increases the probability of the most likely classes
    ! does not support quadratic cost in this implementation


DATASET_ARTIFICIAL_INCREASE

none:
    uses the original training images without changes

image_shift:
    randomly shifts each training image left, right, up, or down
    creates slightly different versions of the same image during training
    helps the network handle digits that are not perfectly centered
    acts like having more varied training data without permanently changing the dataset
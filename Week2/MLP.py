import numpy as np
import matplotlib.pyplot as plt
import random

x = [random.uniform(0,1) for _ in range(100)]
t = [i**3-i**2 for i in x]

print(x)
print(t)




plt.plot(x,t)
plt.xlabel("x")
plt.ylabel("Our function")

plt.show()


def relu(x):
    return np.maximum(0, x)


def reluDerivative(x):
    return x>0

def mean_squared_error(target, output):
    return (1/2) * (np.square(target - output))

def mseDerivative(target, output):
    return output - target

def transpose_Vector(inp):
    return np.transpose(inp)

class Layer:
    def __init__(self, input_units, n_units):
        self.input_units = input_units
        self.n_units = n_units
        self.weights = np.random.rand(n_units, input_units)
        self.bias = np.random.rand(n_units,1)
        self.learning = 0.01
        self.preactivation = 0
        self.inputs = 0
        self.target = None
        self.output = 0


    def forward_step(self, inputs):
        self.preactivation = self.weights @ inputs + self.bias
        return relu(self.preactivation)


    def compute_weight_gradient(self, output):
        gradient_weights = (reluDerivative(self.preactivation) * mseDerivative(self.target, output)) @ np.transpose(self.inputs)
        return gradient_weights


    def compute_input_gradient(self, output):
        gradient_input = np.matrix.transpose(self.weights) @ (reluDerivative(self.preactivation) * mseDerivative(self.target, output))
        return gradient_input

    def compute_bias_gradient(self, output):
        gradient_bias = reluDerivative(self.preactivation) * mseDerivative(self.target, output)
        return gradient_bias

    def backwards_step(self, inputs, target):
        self.inputs = inputs
        self.target = target
        output = self.forward_step(self.inputs)

        update_weights = self.compute_weight_gradient(output)
        update_bias = self.compute_bias_gradient(output)

        self.weights -= self.learning * update_weights
        self.bias -= self.learning * update_bias

        return self.compute_input_gradient(output)


ann1 = Layer(3, 2)


class MLP:
    def __init__(self):
        self.layer1 = Layer(1,10)
        self.layer2 = Layer(10,1)
        self.output = 0
        #self.layer1_activation = self.layer1.forward_step(inputs)

    def forward_prop(self, inputs):
        self.layer1_activation = self.layer1.forward_step(inputs)
        print(self.layer1_activation)
        layer2_activation = self.layer2.forward_step(self.layer1_activation)
        print(layer2_activation)
        self.output = layer2_activation
        return layer2_activation

    def backward_prop(self, inputs, target):
        #target = np.random.rand(1,1)
        update_layer2 = self.layer2.backwards_step(self.layer1_activation, target)
        print(update_layer2)
        update_layer1 = self.layer2.backwards_step(self.layer1_activation, target) * self.layer1.backwards_step(inputs, target)
        print(update_layer1)

        return update_layer1,update_layer2



""""# Training
mlp = MLP()
epochs = []
losses = []
accuracies = []

for epoch in range(500):
    epochs.append(epoch)




    for i in range(1):
        a = x[i]
        b = t[i]

        mlp.forward_prop([[a]])

        mlp.backward_prop([[a]], [[b]])"""





a = MLP()
a.forward_prop([[1]])
a.backward_prop([[1]],[[1]])

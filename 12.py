# # Without using keras

# import numpy as np

# # Activation function (sigmoid) and its derivative
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1 - x)

# # Define the training data for AND-NOT function
# X_and_not = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y_and_not = np.array([[0], [0], [1], [0]])

# # Define the training data for XOR function
# X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y_xor = np.array([[0], [1], [1], [0]])

# # Define the Multi-layer Perceptron class with one hidden layer
# class MLP:
#     def __init__(self, input_size, hidden_size, output_size):
#         # Initialize weights for input to hidden layer
#         self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        
#         # Initialize weights for hidden to output layer
#         self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        
#         # Initialize biases for hidden layer
#         self.bias_hidden = np.random.rand(1, hidden_size)
        
#         # Initialize biases for output layer
#         self.bias_output = np.random.rand(1, output_size)

#     def forward(self, X):
#         # Forward pass through the hidden layer
#         self.hidden = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        
#         # Forward pass through the output layer
#         self.output = sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
#         return self.output

#     def backward(self, X, y, output):
#         # Calculate the error for the output layer
#         output_error = y - output
#         output_delta = output_error * sigmoid_derivative(output)

#         # Calculate the error for the hidden layer
#         hidden_error = output_delta.dot(self.weights_hidden_output.T)
#         hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

#         # Update weights and biases
#         self.weights_hidden_output += self.hidden.T.dot(output_delta)
#         self.weights_input_hidden += X.T.dot(hidden_delta)
#         self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True)
#         self.bias_output += np.sum(output_delta, axis=0, keepdims=True)

#     def train(self, X, y, epochs):
#         for _ in range(epochs):
#             output = self.forward(X)
#             self.backward(X, y, output)

#     def predict(self, X):
#         return (self.forward(X) > 0.5).astype(int)

# # Training the MLP for AND-NOT function
# mlp_and_not = MLP(input_size=2, hidden_size=4, output_size=1)
# mlp_and_not.train(X_and_not, y_and_not, epochs=5000)

# # Training the MLP for XOR function
# mlp_xor = MLP(input_size=2, hidden_size=4, output_size=1)
# mlp_xor.train(X_xor, y_xor, epochs=5000)

# # Print training results
# print("AND-NOT Function Predictions:")
# print(mlp_and_not.predict(X_and_not))

# print("\nXOR Function Predictions:")
# print(mlp_xor.predict(X_xor))

# # Manually test specific input values
# and_not_test_input = np.array([[0, 1]])
# xor_test_input = np.array([[1, 0]])

# print("\nAND-NOT Function Prediction for input [0, 1]:")
# print(mlp_and_not.predict(and_not_test_input))

# print("\nXOR Function Prediction for input [1, 0]:")
# print(mlp_xor.predict(xor_test_input))


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Multi-Layer Perceptron class
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(output_size)

    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, x, y, output):
        error = y - output
        delta_output = error * sigmoid_derivative(output)
        
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * sigmoid_derivative(self.hidden_output)
        
        self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * self.learning_rate
        self.bias_output += np.sum(delta_output, axis=0) * self.learning_rate
        self.weights_input_hidden += x.T.dot(delta_hidden) * self.learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0) * self.learning_rate

    def train(self, X, y):
        for _ in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, x):
        return int(self.forward(x)>0.5)


X_and_not = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and_not = np.array([[0], [0], [1], [0]])

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

mlp_and_not = MLP(input_size=2, hidden_size=2, output_size=1)
mlp_and_not.train(X_and_not, y_and_not)

print("AND-NOT Function:")
for xi in X_and_not:
    print(f"{xi} -> {mlp_and_not.predict(xi)}")

mlp_xor = MLP(input_size=2, hidden_size=2, output_size=1)
mlp_xor.train(X_xor, y_xor)

print("\nXOR Function:")
for xi in X_xor:
    print(f"{xi} -> {mlp_xor.predict(xi)}")
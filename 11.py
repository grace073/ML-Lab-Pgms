# # Without using keras

import numpy as np

# Activation function (step function)
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Define the training data for the AND function
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[0], [0], [0], [1]])

# Define the training data for the OR function
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([[0], [1], [1], [1]])

# Define the Single-layer Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        # Initialize weights to zero
        self.weights = np.zeros((input_size, 1))
        # Initialize bias to zero
        self.bias = 0
        # Set the learning rate and the number of training epochs
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, y):
        # Training process
        for _ in range(self.epochs):
            for inputs, label in zip(X, y):
                # Reshape inputs to column vector
                inputs = inputs.reshape(-1, 1)
                # Calculate linear output
                linear_output = np.dot(inputs.T, self.weights) + self.bias
                # Apply step function to get the prediction
                prediction = step_function(linear_output)
                # Calculate the error
                error = label - prediction
                # Update weights and bias based on the error
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

    def predict(self, X):
        # Prediction process
        linear_output = np.dot(X, self.weights) + self.bias
        return step_function(linear_output)

# Training the Perceptron for AND function
perceptron_and = Perceptron(input_size=2)
perceptron_and.train(X_and, y_and)

# Training the Perceptron for OR function
perceptron_or = Perceptron(input_size=2)
perceptron_or.train(X_or, y_or)

# Print training results for the AND function
print("AND Function Predictions:")
print(perceptron_and.predict(X_and))

# Print training results for the OR function
print("\nOR Function Predictions:")
print(perceptron_or.predict(X_or))

# Manually test specific input values for the AND function
and_test_input = np.array([[1, 1]])
print("\nAND Function Prediction for input [1, 1]:")
print(perceptron_and.predict(and_test_input))

# Manually test specific input values for the OR function
or_test_input = np.array([[0, 1]])
print("\nOR Function Prediction for input [0, 1]:")
print(perceptron_or.predict(or_test_input))

import numpy as np

def step_function(x):
    return 1 if x>0 else 0

class Perceptron:
    def __init__(self,input_size,learning_rate=0.1,epochs = 10):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs =epochs

    def predict(self,x):
        output = np.dot(self.weights,x)+self.bias
        return step_function(output)
    
    def train(self,x,y):
        for _ in range(self.epochs):
            for xi,yi in zip(x,y):
                y_pred = self.predict(xi)
                cost = yi-y_pred #cost is similar as loss/error
                self.weights += self.learning_rate*cost*xi
                self.bias += self.learning_rate*cost


x = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])
y_or = np.array([0,1,1,1])

#training for AND OR Boolean functions

perceptron_and = Perceptron(input_size=2)
perceptron_or = Perceptron(input_size=2)

perceptron_and.train(x,y_and)
print(f"AND Boolean Function")
for xi in x:
    print(f"{xi[0]} and {xi[1]} is {perceptron_and.predict(xi)}")

print()

perceptron_or.train(x,y_or)
print(f"OR Boolean Function")
for xi in x:
    print(f"{xi[0]} or {xi[1]} is {perceptron_or.predict(xi)}")
import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.1):
        """
        Neural network with two hidden layers optimized for XOR problem
        
        Parameters:
        input_size (int): Number of neurons in input layer
        hidden1_size (int): Number of neurons in first hidden layer
        hidden2_size (int): Number of neurons in second hidden layer
        output_size (int): Number of neurons in output layer
        learning_rate (float): Learning rate for weight updates
        """
        # Initialize weights with larger values to break symmetry
        self.W1 = np.random.randn(input_size, hidden1_size) * 1.0
        self.b1 = np.zeros((1, hidden1_size)) + 0.1
        
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 1.0
        self.b2 = np.zeros((1, hidden2_size)) + 0.1
        
        self.W3 = np.random.randn(hidden2_size, output_size) * 1.0
        self.b3 = np.zeros((1, output_size)) + 0.1
        
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1.0/(1.0 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return np.multiply(x, 1.0-x)
    def tanh(self, x):
        return np.tanh(x)  # NumPy has a built-in implementation
    def tanh_derivative(self, x):
        return 1 - np.square(x)
    def relu(self, x):
        return np.maximum(0, x)
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    def forward(self, X):
        # Input to first hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # First hidden to second hidden layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        # Second hidden to output layer
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        
        return self.a3
    
    def backward(self, X, y, output):
        # Calculate error at output                                     # We need to find ∂L/∂z (how the loss changes with respect to the neuron input).
        output_error = y - output          # derivative of MSE          # ∂L/∂z = (∂L/∂a) × (∂a/∂z)
        output_delta = output_error * self.sigmoid_derivative(output)   # ∂L/∂a = derivative of the loss with respect to the output = y - output (for mean squared error)
                                                                        # ∂a/∂z = derivative of the sigmoid function = output * (1 - output)
        # Error at second hidden layer
                                                                        # output_delta shape (100,1) W3 shape (4, 1) result shape (100,4)
                                                                        # why multiply by weights ? cuz it must distribute the error to each weight from previous layer respectively
        hidden2_error = output_delta.dot(self.W3.T)                     # XXX_delta means how the neuron's input should change to reduce error
        hidden2_delta = hidden2_error * self.sigmoid_derivative(self.a2)# matrix multiplication with W3.T distributes the error based on how much each hidden neuron contributed
        
        # Error at first hidden layer
        hidden1_error = hidden2_delta.dot(self.W2.T)                    # (100, 4)
        hidden1_delta = hidden1_error * self.sigmoid_derivative(self.a1)# XXX_delta is computed by element-wise multiplication.
        
        # Update weights - from output to second hidden layer
        self.W3 += self.a2.T.dot(output_delta) * self.learning_rate
        self.b3 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        
        # Update weights - from second hidden to first hidden layer
        self.W2 += self.a1.T.dot(hidden2_delta) * self.learning_rate
        self.b2 += np.sum(hidden2_delta, axis=0, keepdims=True) * self.learning_rate
        
        # Update weights - from first hidden to input layer
        self.W1 += X.T.dot(hidden1_delta) * self.learning_rate
        self.b1 += np.sum(hidden1_delta, axis=0, keepdims=True) * self.learning_rate
    
    def train(self, X, y, epochs, verbose=False):
        errors = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate error
            error = np.mean(np.square(y - output))
            errors.append(error)
            
            # Backward pass
            self.backward(X, y, output)
            
            # Print progress
            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch}, Error: {error:.6f}")
                
                # Also print some sample predictions every 1000 epochs to monitor progress
                if epoch % 5000 == 0:
                    sample_indices = [0, 1, 10, 11]  # Check a few examples
                    print("Sample predictions:")
                    for idx in sample_indices:
                        if idx < len(X):
                            pred = self.forward(X[idx:idx+1])[0][0]
                            print(f"  Input: {X[idx]}, Target: {y[idx][0]}, Prediction: {pred:.4f}")
        
        return errors
    
    def predict(self, X):
        return self.forward(X)

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)
def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()
def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate XOR data
    train_x, train_y = generate_linear()
    test_x, test_y = generate_linear()
    
    # Try multiple training attempts with different initializations
    best_accuracy = 0
    best_model = None
    best_errors = None
    
    for attempt in range(5):
        print(f"\nTraining attempt {attempt+1}/5")
        
        # Create a neural network with two hidden layers
        # Vary sizes to avoid getting stuck in the same local minimum
        hidden1_size = 4 # Random size between 6-9
        hidden2_size = 4   # Random size between 4-7
        
        print(f"Architecture: 2-{hidden1_size}-{hidden2_size}-1")
        
        model = Network(
            input_size=2,
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size,
            output_size=1,
            learning_rate=0.3  # Slightly reduced learning rate for stability
        )
        
        # Train for more epochs
        errors = model.train(train_x, train_y, epochs=15000, verbose=True)
        
        # Evaluate
        predictions = model.predict(test_x)
        rounded_preds = np.round(predictions)
        accuracy = np.mean(rounded_preds == test_y) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_errors = errors
    
    # Print detailed results for the best model
    print("\n=== Final Results with Best Model ===")
    final_predictions = best_model.predict(test_x)
    final_rounded = np.round(final_predictions)
    
    # Print detailed predictions
    print("\nDetailed Predictions:")
    for i in range(len(test_x)):
        correct = "✓" if final_rounded[i][0] == test_y[i][0] else "✗"
        print(f"Input: [{test_x[i][0]:.1f}, {test_x[i][1]:.1f}], Target: {int(test_y[i][0])}, " +
              f"Predicted: {final_predictions[i][0]:.4f}, Rounded: {int(final_rounded[i][0])} {correct}")
    
    # Calculate final accuracy
    final_accuracy = np.mean(final_rounded == test_y) * 100
    print(f"\nFinal Accuracy: {final_accuracy:.2f}%")
    
    # Plot training error
    plt.figure(figsize=(10, 6))
    plt.plot(best_errors)
    plt.title('Training Error Over Time', fontsize=18)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=14)
    plt.grid(True)
    plt.yscale('log')  # Log scale to better visualize error reduction
    plt.savefig("training_error.png")
    plt.show()
    
    # Plot decision boundary
    # plot_decision_boundary(best_model, test_x, test_y)
    show_result(test_x, test_y, final_rounded)
    
    # Show model weights to understand what it learned
    print("\nModel Weights:")
    print(f"W1 shape: {best_model.W1.shape}")
    print(f"W2 shape: {best_model.W2.shape}")
    print(f"W3 shape: {best_model.W3.shape}")
    
    # Just show a few sample weights to avoid cluttering the output
    print("\nSample weights (W1 first row):", best_model.W1[0][:3])

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Neural network with two hidden layers optimized for XOR problem
        
        Parameters:
        input_size (int): Number of neurons in input layer
        hidden1_size (int): Number of neurons in first hidden layer
        hidden2_size (int): Number of neurons in second hidden layer
        output_size (int): Number of neurons in output layer
        learning_rate (float): Learning rate for weight updates
        beta1 (float): Exponential decay rate for first moment estimates (Adam optimizer)
        beta2 (float): Exponential decay rate for second moment estimates (Adam optimizer)
        epsilon (float): Small constant to prevent division by zero (Adam optimizer)
        """
        # Initialize weights with larger values to break symmetry
        self.W1 = np.random.randn(input_size, hidden1_size) * 0.1
        self.b1 = np.zeros((1, hidden1_size)) + 0.1
        
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.1
        self.b2 = np.zeros((1, hidden2_size)) + 0.1
        
        self.W3 = np.random.randn(hidden2_size, output_size) * 0.1
        self.b3 = np.zeros((1, output_size)) + 0.1
        
        # Adam optimizer parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize first and second moment vectors
        # W1 moments
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        # b1 moments
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        
        # W2 moments
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        # b2 moments
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        
        # W3 moments
        self.m_W3 = np.zeros_like(self.W3)
        self.v_W3 = np.zeros_like(self.W3)
        # b3 moments
        self.m_b3 = np.zeros_like(self.b3)
        self.v_b3 = np.zeros_like(self.b3)
        
        # Timestep counter for bias correction
        self.t = 0
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1.0/(1.0 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return np.multiply(x, 1.0-x)
    
    def forward(self, X):
        """Forward pass through the network"""
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
        """Backward pass with Adam optimizer updates"""
        # Increment timestep
        self.t += 1
        
        # Calculate output layer error
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        # Calculate hidden layer errors
        hidden2_error = output_delta.dot(self.W3.T)
        hidden2_delta = hidden2_error * self.sigmoid_derivative(self.a2)
        
        hidden1_error = hidden2_delta.dot(self.W2.T)
        hidden1_delta = hidden1_error * self.sigmoid_derivative(self.a1)
        
        # Calculate gradients
        dW3 = self.a2.T.dot(output_delta)
        db3 = np.sum(output_delta, axis=0, keepdims=True)
        
        dW2 = self.a1.T.dot(hidden2_delta)
        db2 = np.sum(hidden2_delta, axis=0, keepdims=True)
        
        dW1 = X.T.dot(hidden1_delta)
        db1 = np.sum(hidden1_delta, axis=0, keepdims=True)
        
        # Adam update for W3
        self.m_W3 = self.beta1 * self.m_W3 + (1 - self.beta1) * dW3
        self.v_W3 = self.beta2 * self.v_W3 + (1 - self.beta2) * (dW3 ** 2)
        m_W3_corrected = self.m_W3 / (1 - self.beta1 ** self.t)
        v_W3_corrected = self.v_W3 / (1 - self.beta2 ** self.t)
        self.W3 += self.learning_rate * m_W3_corrected / (np.sqrt(v_W3_corrected) + self.epsilon)
        
        # Adam update for b3
        self.m_b3 = self.beta1 * self.m_b3 + (1 - self.beta1) * db3
        self.v_b3 = self.beta2 * self.v_b3 + (1 - self.beta2) * (db3 ** 2)
        m_b3_corrected = self.m_b3 / (1 - self.beta1 ** self.t)
        v_b3_corrected = self.v_b3 / (1 - self.beta2 ** self.t)
        self.b3 += self.learning_rate * m_b3_corrected / (np.sqrt(v_b3_corrected) + self.epsilon)
        
        # Adam update for W2
        self.m_W2 = self.beta1 * self.m_W2 + (1 - self.beta1) * dW2
        self.v_W2 = self.beta2 * self.v_W2 + (1 - self.beta2) * (dW2 ** 2)
        m_W2_corrected = self.m_W2 / (1 - self.beta1 ** self.t)
        v_W2_corrected = self.v_W2 / (1 - self.beta2 ** self.t)
        self.W2 += self.learning_rate * m_W2_corrected / (np.sqrt(v_W2_corrected) + self.epsilon)
        
        # Adam update for b2
        self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * db2
        self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (db2 ** 2)
        m_b2_corrected = self.m_b2 / (1 - self.beta1 ** self.t)
        v_b2_corrected = self.v_b2 / (1 - self.beta2 ** self.t)
        self.b2 += self.learning_rate * m_b2_corrected / (np.sqrt(v_b2_corrected) + self.epsilon)
        
        # Adam update for W1
        self.m_W1 = self.beta1 * self.m_W1 + (1 - self.beta1) * dW1
        self.v_W1 = self.beta2 * self.v_W1 + (1 - self.beta2) * (dW1 ** 2)
        m_W1_corrected = self.m_W1 / (1 - self.beta1 ** self.t)
        v_W1_corrected = self.v_W1 / (1 - self.beta2 ** self.t)
        self.W1 += self.learning_rate * m_W1_corrected / (np.sqrt(v_W1_corrected) + self.epsilon)
        
        # Adam update for b1
        self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * db1
        self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (db1 ** 2)
        m_b1_corrected = self.m_b1 / (1 - self.beta1 ** self.t)
        v_b1_corrected = self.v_b1 / (1 - self.beta2 ** self.t)
        self.b1 += self.learning_rate * m_b1_corrected / (np.sqrt(v_b1_corrected) + self.epsilon)
    
    def train(self, X, y, epochs, verbose=False):
        """Train the neural network"""
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
                
                # Also print some sample predictions every 5000 epochs to monitor progress
                if epoch % 5000 == 0:
                    sample_indices = [0, 1, 10, 11]  # Check a few examples
                    print("Sample predictions:")
                    for idx in sample_indices:
                        if idx < len(X):
                            pred = self.forward(X[idx:idx+1])[0][0]
                            print(f"  Input: {X[idx]}, Target: {y[idx][0]}, Prediction: {pred:.4f}")
        
        return errors
    
    def predict(self, X):
        """Generate predictions from the network"""
        return self.forward(X)

def generate_linear(n=100):
    """Generate linearly separable data"""
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    """Generate XOR data"""
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
    
    # Generate data
    # Uncomment the line for the dataset you want to use:
    train_x, train_y = generate_linear()  # Linear data
    # train_x, train_y = generate_XOR_easy()  # XOR data
    
    test_x, test_y = generate_linear()
    # test_x, test_y = generate_XOR_easy()
    
    # Try multiple training attempts with different initializations
    best_accuracy = 0
    best_model = None
    best_errors = None
    
    for attempt in range(5):
        print(f"\nTraining attempt {attempt+1}/5")
        
        # Create a neural network with two hidden layers
        hidden1_size = 4
        hidden2_size = 4
        
        print(f"Architecture: 2-{hidden1_size}-{hidden2_size}-1")
        
        model = Network(
            input_size=2,
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size,
            output_size=1,
            learning_rate=0.01,  # Lower learning rate for Adam
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        )
        
        # Train for fewer epochs - Adam typically converges faster
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
    for i in range(min(10, len(test_x))):  # Show first 10 examples
        correct = "✓" if final_rounded[i][0] == test_y[i][0] else "✗"
        print(f"Input: [{test_x[i][0]:.1f}, {test_x[i][1]:.1f}], Target: {int(test_y[i][0])}, " +
              f"Predicted: {final_predictions[i][0]:.4f}, Rounded: {int(final_rounded[i][0])} {correct}")
    
    # Calculate final accuracy
    final_accuracy = np.mean(final_rounded == test_y) * 100
    print(f"\nFinal Accuracy: {final_accuracy:.2f}%")
    
    # Plot training error
    plt.figure(figsize=(10, 6))
    plt.plot(best_errors)
    plt.title('Training Error Over Time (Adam Optimizer)', fontsize=18)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=14)
    plt.grid(True)
    plt.yscale('log')  # Log scale to better visualize error reduction
    plt.savefig("training_error_adam.png")
    plt.show()
    
    # Plot decision boundary
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
import numpy as np
import matplotlib.pyplot as plt

class ConvLayer:
    def __init__(self, input_dim, filter_size, num_filters, stride=1, padding=0):
        """
        Simple convolutional layer implementation
        
        Parameters:
        input_dim (tuple): Dimensions of input (height, width)
        filter_size (int): Size of the filter
        num_filters (int): Number of filters
        stride (int): Stride size
        padding (int): Padding size
        """
        self.input_dim = input_dim
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding
        
        # Initialize filters with random values
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1
        self.biases = np.zeros(num_filters)
        
        # Calculate output dimensions
        h_in, w_in = input_dim
        h_out = int(((h_in + 2 * padding - filter_size) / stride) + 1)
        w_out = int(((w_in + 2 * padding - filter_size) / stride) + 1)
        self.output_dim = (h_out, w_out, num_filters)
        
    def forward(self, X):
        """
        Forward pass for convolutional layer
        
        Parameters:
        X (ndarray): Input tensor of shape (batch_size, features)
        
        Returns:
        ndarray: Output tensor after convolution
        """
        batch_size = X.shape[0]
        h_out, w_out, num_filters = self.output_dim
        
        # For 2D input data like (batch_size, 2), reshape to (batch_size, h, w)
        # where h=2 and w=1 for our specific case with 2 features
        if len(X.shape) == 2:
            h, w = self.input_dim
            X = X.reshape(batch_size, h, w)
        
        # Apply padding if needed
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        else:
            X_padded = X
        
        # Initialize output
        self.output = np.zeros((batch_size, h_out, w_out, num_filters))
        self.X_padded = X_padded  # Store for backpropagation
        
        # Perform convolution
        for b in range(batch_size):
            for h in range(h_out):
                for w in range(w_out):
                    for f in range(num_filters):
                        h_start = h * self.stride
                        h_end = h_start + self.filter_size
                        w_start = w * self.stride
                        w_end = w_start + self.filter_size
                        
                        # Extract the region to convolve with the filter
                        region = X_padded[b, h_start:h_end, w_start:w_end]
                        
                        # Convolution operation
                        self.output[b, h, w, f] = np.sum(region * self.filters[f]) + self.biases[f]
        
        # Apply ReLU activation
        self.output = np.maximum(0, self.output)
        
        # Flatten output for the next layer
        self.flattened_output = self.output.reshape(batch_size, -1)
        
        return self.flattened_output
    
    def backward(self, dout, learning_rate):
        """
        Backward pass for convolutional layer
        
        Parameters:
        dout (ndarray): Gradient of the loss with respect to the output
        learning_rate (float): Learning rate for parameter updates
        
        Returns:
        ndarray: Gradient of the loss with respect to the input
        """
        batch_size = dout.shape[0]
        h_out, w_out, num_filters = self.output_dim
        
        # Reshape dout to match output dimensions
        dout = dout.reshape(batch_size, h_out, w_out, num_filters)
        
        # Initialize gradients
        dX_padded = np.zeros_like(self.X_padded)
        dfilters = np.zeros_like(self.filters)
        dbiases = np.zeros_like(self.biases)
        
        # Compute gradients
        for b in range(batch_size):
            for h in range(h_out):
                for w in range(w_out):
                    for f in range(num_filters):
                        # Only backpropagate through ReLU if output was positive
                        if self.output[b, h, w, f] > 0:
                            h_start = h * self.stride
                            h_end = h_start + self.filter_size
                            w_start = w * self.stride
                            w_end = w_start + self.filter_size
                            
                            # Update filter gradients
                            dfilters[f] += self.X_padded[b, h_start:h_end, w_start:w_end] * dout[b, h, w, f]
                            
                            # Update bias gradients
                            dbiases[f] += dout[b, h, w, f]
                            
                            # Update input gradients
                            dX_padded[b, h_start:h_end, w_start:w_end] += self.filters[f] * dout[b, h, w, f]
        
        # Update parameters
        self.filters -= learning_rate * dfilters
        self.biases -= learning_rate * dbiases
        
        # Remove padding from dX_padded if necessary
        if self.padding > 0:
            dX = dX_padded[:, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
            
        # Reshape dX to match input dimensions if needed
        if len(dX.shape) > 2:
            dX = dX.reshape(batch_size, -1)
            
        return dX

class Network:
    def __init__(self, input_size, conv_input_dim, filter_size, num_filters, hidden1_size, hidden2_size, output_size, learning_rate=0.1):
        """
        Neural network with a convolutional layer followed by two hidden layers
        
        Parameters:
        input_size (int): Number of input features
        conv_input_dim (tuple): Dimensions of input for convolution (height, width)
        filter_size (int): Size of convolutional filters
        num_filters (int): Number of convolutional filters
        hidden1_size (int): Number of neurons in first hidden layer
        hidden2_size (int): Number of neurons in second hidden layer
        output_size (int): Number of neurons in output layer
        learning_rate (float): Learning rate for weight updates
        """
        # For simplicity with our small input, adjust filter size if needed
        filter_size = min(filter_size, min(conv_input_dim))
        
        self.conv_layer = ConvLayer(
            input_dim=conv_input_dim, 
            filter_size=filter_size, 
            num_filters=num_filters
        )
        
        # Calculate the size of the flattened output from conv layer
        conv_output_size = np.prod(self.conv_layer.output_dim)
        
        # Initialize weights for fully connected layers
        self.W1 = np.random.randn(conv_output_size, hidden1_size) * 0.1
        self.b1 = np.zeros((1, hidden1_size)) + 0.1
        
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.1
        self.b2 = np.zeros((1, hidden2_size)) + 0.1
        
        self.W3 = np.random.randn(hidden2_size, output_size) * 0.1
        self.b3 = np.zeros((1, output_size)) + 0.1
        
        # Adam optimizer parameters
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
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
        # Reshape input for convolution
        batch_size = X.shape[0]
        
        # Convolutional layer
        self.conv_output = self.conv_layer.forward(X)
        
        # First hidden layer
        self.z1 = np.dot(self.conv_output, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Second hidden layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        # Output layer
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        
        return self.a3
    
    def backward(self, X, y, output):
        """Backward pass with parameter updates"""
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
        
        # Calculate gradients for dense layers
        dW3 = self.a2.T.dot(output_delta)
        db3 = np.sum(output_delta, axis=0, keepdims=True)
        
        dW2 = self.a1.T.dot(hidden2_delta)
        db2 = np.sum(hidden2_delta, axis=0, keepdims=True)
        
        dW1 = self.conv_output.T.dot(hidden1_delta)
        db1 = np.sum(hidden1_delta, axis=0, keepdims=True)
        
        # Backpropagate to convolutional layer
        dconv = hidden1_delta.dot(self.W1.T)
        dX = self.conv_layer.backward(dconv, self.learning_rate)
        
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
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Error: {error:.6f}")
                
                # Also print some sample predictions
                if epoch % 1000 == 0:
                    sample_indices = [0, 1, 5]  # Check a few examples
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
    train_x, train_y = generate_linear(n=200)
    test_x, test_y = generate_linear(n=50)
    
    # Try multiple training attempts with different filter configurations
    best_accuracy = 0
    best_model = None
    best_errors = None
    
    for attempt in range(3):
        print(f"\nTraining attempt {attempt+1}/3")
        
        # Create neural network with convolutional layer
        # For our 2D inputs, we'll reshape them into tiny "images"
        filter_size = 1  # Use 1x1 filter since our "image" is just 2x1
        num_filters = 3 + attempt  # Vary number of filters
        
        # Define input dimensions for conv layer (reshape our 2D input)
        # Since our input has 2 features, we'll treat as 2x1 "image"
        conv_input_dim = (2, 1)  # Height=2, Width=1
        
        # Configure hidden layers
        hidden1_size = 4
        hidden2_size = 4
        
        print(f"Architecture: Conv({filter_size}x{filter_size}x{num_filters}) -> " +
              f"{hidden1_size} -> {hidden2_size} -> 1")
        
        model = Network(
            input_size=2,
            conv_input_dim=conv_input_dim,
            filter_size=filter_size,
            num_filters=num_filters,
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size,
            output_size=1,
            learning_rate=0.05
        )
        
        # Train model
        errors = model.train(train_x, train_y, epochs=3000, verbose=True)
        
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
    
    # Print detailed predictions for a sample
    print("\nSample Predictions:")
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
    plt.title('Training Error Over Time (Convolutional Network)', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=14)
    plt.grid(True)
    plt.yscale('log')  # Log scale to better visualize error reduction
    plt.savefig("conv_training_error.png")
    plt.show()
    
    # Plot decision boundary
    show_result(test_x, test_y, final_rounded)

if __name__ == "__main__":
    main()
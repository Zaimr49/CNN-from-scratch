import numpy as np


class ReLU:
    def __init__(self):
        self.inputs = None
        self.has_weights = False

    def forward(self, inputs):
        """Forward pass for the ReLU activation function

        Args:
            inputs (np.ndarray):
                input array, can have any shape

        Returns: (np.ndarray):
            array of the same shape as the input
        """

        self.inputs = inputs

        # ================ Insert Code Here ================
        return np.maximum(0, inputs)
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass for the ReLU activation function

        Args:
            d_outputs (np.ndarray): array of any shape

        Returns (dict):
            Dictionary containing the derivative of the loss with
            respect to the output of the layer. The key of the dictionary
            should be "d_out"
        """
        # ================ Insert Code Here ================
        d_inputs = d_outputs * (self.inputs > 0)
        return {"d_out": d_inputs}
        # ==================================================


class Sigmoid:
    def __init__(self):
        self.inputs = None
        self.has_weights = False

    def forward(self, inputs):
        """Forward pass for the Sigmoid activation function

        Args:
            inputs (np.ndarray):
                input array, can have any shape

        Returns: (np.ndarray):
            array of the same shape as the input
        """

        self.inputs = inputs

        # ================ Insert Code Here ================
        self.inputs = inputs
        return 1 / (1 + np.exp(-inputs))
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass for the Sigmoid activation function

        Args:
            d_outputs (np.ndarray): array of any shape

        Returns (dict):
            Dictionary containing the derivative of the loss with
            respect to the output of the layer. The key of the dictionary
            should be "d_out"
        """

        # ================ Insert Code Here ================
        sigmoid_outputs = 1 / (1 + np.exp(-self.inputs))
        d_inputs = d_outputs * sigmoid_outputs * (1 - sigmoid_outputs)
        return {"d_out": d_inputs}
        # ==================================================


class Softmax:
    def __init__(self):
        self.inputs = None
        self.outputs=None
        self.has_weights = False

    def forward(self, inputs):
        """Forward pass for the ReLU activation function

        Args:
            inputs (np.ndarray):
                input array, can have any shape

        Returns: (np.ndarray):
            array of the same shape as the input
        """
        self.inputs = inputs

        # ================ Insert Code Here ================
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.outputs
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass for the Softmax activation function

        Args:
            d_outputs (np.ndarray): array of any shape

        Returns (dict):
            Dictionary containing the derivative of the loss with
            respect to the output of the layer. The key of the dictionary
            should be "d_out"
        """

        # ================ Insert Code Here ================
        # Initialize the derivative of the inputs matrix.
        d_inputs = np.empty_like(d_outputs)

        # Iterate through each sample in the batch.
        for index, (single_output, single_d_output) in enumerate(zip(self.outputs, d_outputs)):
            # Reshape single_output to a column vector
            single_output = single_output.reshape(-1, 1)
            # Compute the Jacobian matrix for the Softmax function
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # Apply the chain rule to get the derivative of the loss with respect to the inputs of the softmax.
            d_inputs[index] = np.dot(jacobian_matrix, single_d_output)
        
        return {"d_out": d_inputs}
        # ==================================================

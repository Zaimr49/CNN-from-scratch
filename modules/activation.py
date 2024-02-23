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
        d_inputs = np.array(d_outputs, copy=True) 
        d_inputs[self.inputs <= 0] = 0 
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
        """Forward pass for the Softmax activation function."""
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
        return {"d_out": d_outputs}
        # ==================================================
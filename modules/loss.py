import numpy as np


class CrossEntropy:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.inputs = None
        self.targets = None

    def forward(self, inputs, targets):
        """Forward pass for the cross entropy loss function

        Args:
            inputs (np.ndarray): predictions from the model
            targets (_type_): ground truth labels (one-hot encoded)

        Returns: (float):
            loss value
        """

        self.inputs = inputs
        self.targets = targets

        # ================ Insert Code Here ================
        # Prevent division by zero and stabilize the log function
        inputs_clipped = np.clip(inputs, self.eps, 1 - self.eps)

        # Calculate cross-entropy loss
        loss = -np.sum(targets * np.log(inputs_clipped)) / inputs.shape[0]
        return loss

        # raise NotImplementedError
        # ==================================================

    def backward(self):
        """Backward pass for the cross entropy loss function

        Args:
            None

        Returns: (dict):
            Dictionary containing the derivative of the loss
            with respect to the inputs to the loss function.
            The key of the dictionary should be "d_out"
        """
        # ================ Insert Code Here ================
        # Number of samples
        num_samples = self.targets.shape[0]

        # The gradient of cross-entropy loss with respect to the inputs is
        # the difference between the predictions and the true values.
        # For softmax activation + cross-entropy loss, this simplifies to:
        # dL/dy = y_pred - y_true
        d_out = (self.inputs - self.targets) 

        # Wrap the gradient in a dictionary and return
        grad_output = {"d_out": d_out}
        return grad_output        
    # ==================================================

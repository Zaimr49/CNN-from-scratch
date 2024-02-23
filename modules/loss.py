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

        # Removing Zeros to Prevent Log(0)
        clipped_inputs = np.clip(inputs, self.eps, 1 - self.eps)

        step1 = np.log(clipped_inputs)
        step2 = targets * step1
        step3 = np.sum(step2)
        step4 = step3/inputs.shape[0]
        loss = -step4
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
        d_out = (self.inputs - self.targets) 
        grad_output = {"d_out": d_out}
        return grad_output        
    # ==================================================

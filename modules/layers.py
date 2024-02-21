import numpy as np


class ConvolutionLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # print("here",in_channels,out_channels,kernel_size,stride)

        self.weights = np.random.rand(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float32)
        self.bias = np.random.rand(out_channels).astype(np.float32)

        self.inputs = None
        self.has_weights = True

    def forward(self, inputs):
        """Forward pass for a convolution layer

        Args:
            inputs (np.ndarray):
                array of shape
                (batch_size, in_channels, height, width)

        Returns: (np.ndarray):
            array of shape
            (batch_size, out_channels, new_height, new_width)
        """

        self.inputs = inputs
        # ================ Insert Code Here ================
        batch_size, in_channels, in_height, in_width = inputs.shape
        # print("batch_size",batch_size)
        # print("in_channels",in_channels)
        # print("in_height",in_height)
        # print("in_width",in_width)
        
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1
        
        # print("out_height",out_height)
        # print("out_width",out_width)
        
        outputs = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Extracting the receptive field
                        # Calculate start and end indices for height
                        h_start = oh * self.stride
                        h_end = h_start + self.kernel_size

                        # Calculate start and end indices for width
                        w_start = ow * self.stride
                        w_end = w_start + self.kernel_size

                        # Extract the receptive field using calculated indices
                        receptive_field = inputs[b, :, h_start:h_end, w_start:w_end]

                        # Performing convolution
                        convolution_result = np.sum(receptive_field * self.weights[oc])
                        convolution_result += self.bias[oc]
                        outputs[b, oc, oh, ow] = convolution_result


        return outputs
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass of convolution layer

        Args:
            d_outputs (np.ndarray):
                derivative of loss with respect to the output
                of the layer. Will have shape
                (batch_size, out_channels, new_height, new_width)

        Returns: (dict):
            Dictionary containing the derivatives of loss with respect to
            the weights and bias and input of the layer. The keys of
            the dictionary should be "d_weights", "d_bias", and "d_input"

        """

        if self.inputs is None:
            raise NotImplementedError(
                "Need to call forward function before backward function"
            )

        # ================ Insert Code Here ================
        batch_size, _, out_height, out_width = d_outputs.shape

        # Initialize gradients
        d_weights = np.zeros(self.weights.shape)
        d_bias = np.zeros(self.bias.shape)
        d_output = np.zeros(self.inputs.shape)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = ow * self.stride
                        w_end = w_start + self.kernel_size

                        # Gradient with respect to the output (d_outputs) is used to calculate gradients
                        # with respect to weights, bias, and inputs
                        receptive_field = self.inputs[b, :, h_start:h_end, w_start:w_end]
                        d_weights[oc] += d_outputs[b, oc, oh, ow] * receptive_field
                        d_bias[oc] += d_outputs[b, oc, oh, ow]

                        # Distribute the gradient back to the inputs
                        d_output[b, :, h_start:h_end, w_start:w_end] += self.weights[oc] * d_outputs[b, oc, oh, ow]  
                        
        gradients = {
        "d_weights": d_weights,
        "d_bias": d_bias,
        "d_out": d_output
        }

        return gradients      
        # ==================================================

    def update(self, d_weights, d_bias, learning_rate):

        # ================ Insert Code Here ================
        # Update weights and biases with gradients
        self.weights = self.weights - learning_rate * d_weights
        self.bias = self.bias - learning_rate * d_bias
        # ==================================================


class Flatten:
    def __init__(self):
        self.inputs_shape = None
        self.has_weights = False

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, d_outputs):
        return {"d_out": d_outputs.reshape(self.inputs_shape)}


class LinearLayer:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.weights = np.random.rand(out_features, in_features).astype(np.float32)
        self.bias = np.random.rand(out_features).astype(np.float32)

        self.inputs = None
        self.has_weights = True

    def forward(self, inputs):
        """Forward pass for a linear layer

        Args:
            inputs (np.ndarray):
                array of shape (batch_size, in_features)

        Returns: (np.ndarray):
            array of shape (batch_size, out_features)
        """

        # ================ Insert Code Here ================
        self.inputs = inputs
        return np.dot(inputs, self.weights.T) + self.bias    
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass of Linear layer

        Args:
            d_outputs (np.ndarray):
                derivative of loss with respect to the output
                of the layer. Will have shape
                (batch_size, out_features)

        Returns: (dict):
            Dictionary containing the derivatives of loss with respect to
            the weights and bias and input of the layer. The keys of
            the dictionary should be "d_weights", "d_bias", and "d_output"
        """
        if self.inputs is None:
            raise NotImplementedError(
                "Need to call forward function before backward function"
            )
        # ================ Insert Code Here ================
        d_weights = np.dot(d_outputs.T, self.inputs)
        d_bias = np.sum(d_outputs, axis=0)
        d_outputs = np.dot(d_outputs, self.weights)

        # Prepare derivatives for update
        derivatives = {
            "d_weights": d_weights,
            "d_bias": d_bias,
            "d_out": d_outputs
        }
        
        return derivatives

        # ==================================================

    def update(self, d_weights, d_bias, learning_rate):

        # ================ Insert Code Here ================
        self.weights = self.weights - learning_rate * d_weights
        self.bias = self.bias - learning_rate * d_bias
        # ==================================================

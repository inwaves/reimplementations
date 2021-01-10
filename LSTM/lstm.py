from util import sigmoid
import numpy as np
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))


class LSTM:

    def __init__(self):
        pass
    
    # TODO: initialise parameters!
    def initialise_parameters():
        # what are my parameters?
        pass

    def forget_gate(self, concat_tensor, Wf, bf):
        return sigmoid(np.dot(Wf, concat_tensor) + bf)

    def input_gate(self, concat_tensor, Wi, bi):
        return sigmoid(np.dot(Wi, concat_tensor) + bi)

    def update_gate(self, concat_tensor, Wi, bi):
        return sigmoid(np.dot(Wi, concat_tensor) + bi)

    def output_gate(self, concat_tensor, Wo, bo):
        return sigmoid(np.dot(Wo, concat_tensor) + bo)

    def candidate_value(self, concat_tensor, Wc, bc):
        return np.nah(np.dot(Wc, concat_tensor) + bc)

    def LSTM_forward_pass(self, a_prev, c_prev, x, parameters):
        """Does one forward pass of the LSTM cell

        Args:
            a_prev (np.ndarray): The previous activation matrix.
            c_prev (np.ndarray): The previous memory cell matrix.
            x (np.ndarray): The current input.
            parameters (Dict(np.ndarray)): The weights and biases.
        """
        # grab the parameters
        Wf, bf = parameters["Wf"], parameters["bf"]
        Wi, bi = parameters["Wi"], parameters["bi"]
        Wc, bc = parameters["Wc"], parameters["bc"]
        Wo, bo = parameters["Wo"], parameters["bo"]

        # stack the input and previous activation
        concat_tensor = np.stack([a_prev, x])

        c = np.dot(self.forget_gate(concat_tensor, Wf, bf), c_prev) + \
            np.dot(self.update_gate(concat_tensor, Wi, bi),
                   self.candidate_value(concat_tensor, Wc, bc))

        a = np.dot(self.output_gate(concat_tensor, Wo, bo), np.tanh(c))

        return a, c

    # TODO: write the main optimisation function
    def optimise():
        
        # prepare data
        # initialise parameters
        # for a given number of iterations
            # run a forward pass
            # calculate loss of the output
            # run backward propagation
            
        pass


if __name__ == "__main__":
    print(sys.path)
    print(sigmoid(0))

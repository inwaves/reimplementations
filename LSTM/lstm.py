import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
from util import sigmoid


class LSTM:

    def __init__(self, path_to_dataset):
        with open(path_to_dataset, "r") as text_in:
            text = text_in.read()
    
        # get unique words, sort alphabetically
        self.vocab = sorted(set(text)) 
        
        # for each character in the text, associate an integer value
        # FIXME: this uses integers to encode, not one-hot! why one, why the other?
        self.char2idx = {u:i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

        # represent the text as an array of integers
        self.text_as_int = np.array([self.char2idx[c] for c in text])
        pass
    
    # TODO: initialise parameters!
    def initialise_parameters():

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
    model = LSTM("data/infinite_jest_text.txt")
    print(model.text_as_int.shape)
    print(model.vocab)

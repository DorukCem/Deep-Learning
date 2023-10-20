# https://sausheong.github.io/posts/how-to-build-a-simple-artificial-neural-network-with-go/
import numpy as np

# The fields inputs, hiddens and output define the number of neurons in each of the input, hidden and output layers
class Network:
   def __init__(self, input : int, hidden : int, output: int, rate : float):
      self.inputs = input
      self.hiddens = hidden 
      self.outputs = output
      """
      ----->  Each neurons weights is 1 row
         |w11 w21| | |w1| 
         |w12 w22| | |w2| Each w vector has the same dimensions as the imput vector (in this case 2)
         |w13 w23| V |w3|
                
      """
      self.hidden_weights : np.random.rand(self.hiddens, self.inputs) 
      self.output_weights : np.random.rand(self.outputs, self.hiddens)
      self.learning_rate = rate
   
   def train(self, input_data, target_data):
      # Forawrd propagation 
      hidden_outputs = sigmoid( np.dot(self.hidden_weights, input_data) )
      final_outputs = sigmoid( np.dot(self.output_weights, hidden_outputs) )
      # errors
      targets = target_data.copy()
      output_errors = targets - final_outputs
      hidden_errors = np.dot(self.output_weights.T, output_errors)
      #backpropagation
      self.output_weights += self.learning_rate * np.dot( (output_errors * sigmoid_prime(final_outputs) ), hidden_outputs)
      self.hidden_weights += self.learning_rate * np.dot( (hidden_errors * sigmoid_prime(hidden_outputs) ), input_data.T)


def sigmoid(z):
   """The sigmoid function."""
   return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
   """Derivative of the sigmoid function."""
   return sigmoid(z)*(1-sigmoid(z))
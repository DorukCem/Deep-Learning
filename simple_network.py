import numpy as np

def neural_network(input, weights):
   hid = input.dot(weights[0])
   pred = hid.dot(weights[1])
   return pred

# toes % win # fans
first_layer_weights = np.array([
    [0.1, 0.2, -0.1],   # hid[0] -> weight for output which will be fed to hidden neuron 0
    [-0.1, 0.1, 0.9],   # hid[1] -> ... 1
    [0.1, 0.4, 0.1]]).T # hid[2] -> ... 2
 
# hid[0] hid[1] hid[2]
second_layer_weights = np.array([
    [0.3, 1.1, -0.3],    # is anyone hurt?   <---- Outputs
    [0.1, 0.2, 0.0],     # did we win?
    [0.0, 1.3, 0.1]]).T  # is the team sad?

weights = [first_layer_weights, second_layer_weights]

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input = np.array([toes[0], wlrec[0], nfans[0]])
pred = neural_network(input, weights)
print(pred)
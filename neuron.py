weights = [0.1, 0.2, 0]

def w_sum(a,b):
   # this is the same as taking the dot product
   assert(len(a) == len(b))
   output = 0
   for i in range(len(a)):
      output += (a[i] * b[i])
   return output

def neuron(input, weights):
   pred = w_sum(input,weights)
   return pred

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
input = [toes[0],wlrec[0],nfans[0]]

prediction = neuron(input,weights)
print(prediction)
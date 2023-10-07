weight = 0.5
expected = 0.8
input = 2
alpha = 0.1

for _ in range(20):
   prediction = input * weight
   derivative = input * (prediction - expected)
   weight -= (alpha * derivative)
   
   print(f"Error: {(prediction - expected) ** 2} prediction: {prediction}")

print(f"weight: {weight}")

# --------------------------------------------------

def gradient_descent(start, gradient, learn_rate, max_iter, tol=0.01):
   w = start

   for _ in range(max_iter):
      diff = learn_rate * gradient(w)
      if abs(diff)<tol:
         break    
      w = w - diff 

   return w
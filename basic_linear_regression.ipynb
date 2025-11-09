import random
import math
import matplotlib.pyplot as plt


#Creating a small dataset
X=[i for i in range(100)]
y = [3*i + 4 + random.uniform(-2,2) for i in X]

def predict(x, w, b):
    return w*x + b

def mse(X, y ,w, b):
    sum = 0
    for i in range(len(X)):
        y_pred = predict(X[i], w, b)
        sum+= (y[i] - y_pred)**2
    
    return sum/len(X)

def compute_gradients(X, y, w, b):
    """
    returns dW, db
    """
    dW = 0
    db = 0
    for i in range(len(X)):
        y_pred = predict(X[i], w, b)
        dW += (-2/len(X))*(X[i]*(y[i]- y_pred))
        db += (-2/len(X))*(y[i]- y_pred)
    
    return dW, db

def train(X, y, learning_rate = 0.000001, epochs = 1000):
    w, b = 0, 0
    for i in range(epochs):
        dW, db = compute_gradients(X, y, w, b)
        w -= learning_rate*dW
        b -= learning_rate*db
        
        if i % 100 == 0:
            print(f"Epoch {i}, Loss = {mse(X, y, w, b)}")
    
    return w, b

w, b = train(X, y, learning_rate = 0.0000001, epochs=30000)
print(f"Result: {w}, {b}")

#Predicting using calculated weight and bias
y_pred = [w*x + b for x in X]


#Visualization
plt.scatter(X, y, color = "pink", label ="Original Data")
plt.plot(X, y_pred, color = "red", label = "Model Prediction")
plt.xlabel("x")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()
plt.show()

import random
import math
import matplotlib.pyplot as plt


#Creating a small dataset
X=[i for i in range(100)]
y = [1 if x > 65 else 0 for x in X]

def sigmoid(z):
    return 1/(1 + math.exp(-z))

def predict(x, w, b):
    z = w*x + b
    return sigmoid(z)

def cost_function(y, y_pred):
    loss = 0
    for i in range(len(y)):
        loss += - (y[i]*math.log(y_pred[i]) + (1-y[i])*math.log(1-y_pred[i]))
    return loss / len(y)

def compute_gradients(X, y, w, b):
    """
    returns dW, db
    """
    dW = 0
    db = 0
    for i in range(len(X)):
        y_pred = predict(X[i], w, b)
        dW += (-1/len(X))*(X[i]*(y[i]- y_pred))
        db += (-1/len(X))*(y[i]- y_pred)
    
    return dW, db

def train(X, y, learning_rate = 0.001, epochs = 1000):
    w, b = 0, 0
    for i in range(epochs):
        dW, db = compute_gradients(X, y, w, b)
        w -= learning_rate*dW
        b -= learning_rate*db
        
        if i % 100 == 0:
            y_preds = [predict(x, w, b) for x in X]
            loss = cost_function(y, y_preds)
            print(f"Epoch {i}, Loss = {loss}")
    return w, b


def accuracy(X, y, w, b, threshold = 0.5):
    correct = 0
    for i in range(len(X)):
        y_pred = predict(X[i], w, b)
        y_label = 1 if y_pred >= threshold else 0
        if y_label == y[i]:
            correct += 1
    return correct / len(X)


w, b = train(X, y, learning_rate = 0.001, epochs=15000)
print(f"Result: {w}, {b}")

print("Accuracy: ", accuracy(X, y, w, b))

#Predicting using calculated weight and bias
y_pred = [predict(x, w, b) for x in X]


#Visualization
plt.scatter(X, y, color = "pink", label ="Original Data")
plt.plot(X, y_pred, color = "red", label = "Model Prediction")
plt.xlabel("x")
plt.ylabel("Y")
plt.title("Logitic Regression")
plt.legend()
plt.show()

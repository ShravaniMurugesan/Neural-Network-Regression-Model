# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

<img width="1184" height="783" alt="image" src="https://github.com/user-attachments/assets/ec9a9a13-bb46-4b2b-8cb4-c51a60753465" />



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:SHRAVANI M
### Register Number:212224230263
```
class Neuralnet(nn.Module):
   def __init__(self):
        super().__init__()
        self.n1=nn.Linear(1,10)
        self.n2=nn.Linear(10,20)
        self.n3=nn.Linear(20,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}
   def forward(self,x):
        x=self.relu(self.n1(x))
        x=self.relu(self.n2(x))
        x=self.n3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
nithi=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(nithi.parameters(),lr=0.001)

def train_model(nithi, X_train, y_train, criterion, optimizer, epochs=1000):
    # initialize history before loop
    nithi.history = {'loss': []}

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = nithi(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # record loss
        nithi.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')




```
## Dataset Information

<img width="239" height="658" alt="image" src="https://github.com/user-attachments/assets/7075a09a-6c95-4da5-ab4e-f0385dc19249" />


## OUTPUT

<img width="598" height="151" alt="image" src="https://github.com/user-attachments/assets/d3788ec8-5d22-45b4-8d1b-270b8a9cd806" />

### Training Loss Vs Iteration Plot

<img width="1264" height="767" alt="image" src="https://github.com/user-attachments/assets/ebdd38b9-f64b-44c7-a323-480e4e403a33" />


### New Sample Data Prediction

```
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = nithi(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```
<img width="1132" height="59" alt="image" src="https://github.com/user-attachments/assets/5c15f379-eb14-4121-a73f-3326d454ffdc" />

## RESULT

Successfully executed the code to develop a neural network regression model.

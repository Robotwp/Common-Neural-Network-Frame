# AlexNet 
## ImageNet Classification with Deep Convolutional Neural Networks
#### Dataset:ImageNet LSVRC-2010  
1. million high-resolution images 1000 different classes
2. top-1 37.5%
3. top-5 17.0%

#### Network
1. 60 million parameters and 650,000 neurons, 
2. five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax.
3. To reduce overfitting in the fully-connected layers we employed a recently-developed regularization method called “dropout”that proved to be very effective.

#### Tips
1. this depth seems to be important: we found that removing any convolutional layer (each of which contains no more than 1% of the model’s parameters) resulted in inferior performance.


#### The Architecture
1. It contains eight learned layers—five convolutional and three fully-connected.
2. ReLU Nonlinearity   f(x) = max(0; x).


### learning
1. stochastic gradient descent with a batch size of 128 examples  
momentum of 0.9, and weight decay of 0.0005.
2. $$ v_{i+1}:=0.9*v_{i}-0.0005* \epsilon *w_{i}- \epsilon *(\frac{\partial L}{\partial w}|w_i)_{D_{i}}\\w_{i+1}:=w_i+v_{i+1}$ $
3. We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01.
4. We initialized the neuron biases in the second, fourth, and fifth convolutional layers,as well as in the fully-connected hidden layers, with the constant 1.
5. We initialized the neuron biases in the remaining layers with the constant 0.
6. The heuristic which we followed was to divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate. The learning rate was initialized at 0.01 and reduced three times prior to termination

![image.png](attachment:image.png)

```
from keras.layers import Conv2D, MaxPooling2D, Input, Dense,Dropout,Flatten,BatchNormalization
from keras.models import Model
from keras.initializers import Ones,Zeros,RandomNormal
import keras
# First, define the vision modules
inputs = Input(shape=(224, 224, 3))
Conv2D_1 = Conv2D(96, (11, 11),strides=(4, 4),activation='relu', 
                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),bias_initializer=Zeros())(inputs)
MaxPooling2D_1 = MaxPooling2D((3, 3), strides=2)(Conv2D_1)
Conv2D_2 = Conv2D(256, (5, 5),activation='relu',padding= 'same',
                 kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),bias_initializer=Ones())(MaxPooling2D_1)
MaxPooling2D_2 = MaxPooling2D((3, 3), strides=2)(Conv2D_2)
Conv2D_3 = Conv2D(384, (3, 3),activation='relu',padding= 'same',
                 kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),bias_initializer=Zeros())(MaxPooling2D_2)
Conv2D_4 = Conv2D(384, (5, 5),activation='relu',padding= 'same',
                 kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),bias_initializer=Ones())(Conv2D_3)
Conv2D_5 = Conv2D(256, (5, 5),activation='relu',padding= 'same',
                 kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),bias_initializer=Ones())(Conv2D_4)
MaxPooling2D_3 = MaxPooling2D((3, 3), strides=2)(Conv2D_5)
Flatten_1 = Flatten()(MaxPooling2D_3)
# a layer instance is callable on a tensor, and returns a tensor
Dense_1= Dense(4096, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),bias_initializer=Zeros())(Flatten_1)
Dropout_1=Dropout(0.5)(Dense_1)
Dense_2 = Dense(4096, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),bias_initializer=Zeros())(Dropout_1)
Dropout_2=Dropout(0.5)(Dense_2)
predictions = Dense(1000, activation='softmax',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),bias_initializer=Zeros())(Dropout_2)

model.summary()
```

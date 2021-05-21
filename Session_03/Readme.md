# Modified MNIST Network for Classification and Addition

The objective is not only predict the number from the MNIST dataset , but also predict the sum of the random number (0-9) added to it.

## Input Data Pre-processing

Build a combined dataset using

*  torchvision MNIST
*  Add random numbers between 0-9 as the second input
*  2 outputs : predicted number and sum of predicted number

### Data Representation
> The input data is created using a list. The first element of the list is the batch of images and the second element is the batch of random numbers to be added.

### Data Generation Strategy
> The data is created using the class Combined_Dataset. This class creates train or test data based on the input paramter train (True or False). It has the function __getitem__ which appends one batch of image to one batch of random numbers and returns this list. Upon calling next(iter()) on the object of this class, one batch ([images, random_numbers]) is yielded.

# Network Design and Combination of Input

* First, convolution blocks use the images as input
* **After the convolution part, we concatenate the 2nd input (random number) with the argmax of convolution output**
* Stack the tensors to get pairs of numbers and pass them through the linear layers
* There is no activation function required during addition of two numbers as it is a linear function

  ```
  The model layers are: 
  Network(
    (input1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1))
    (conv1): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (oneconv1): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
    (conv3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
    (conv4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (conv5): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (conv6): Conv2d(64, 10, kernel_size=(5, 5), stride=(1, 1))
    (input2): Linear(in_features=2, out_features=5, bias=True)
    (layer1): Linear(in_features=5, out_features=5, bias=True)
    (out2): Linear(in_features=5, out_features=1, bias=True)
  )

  Shape of parameters in each layer is: 
  input1.weight 		 torch.Size([16, 1, 3, 3])
  input1.bias 		 torch.Size([16])
  conv1.weight 		 torch.Size([32, 16, 3, 3])
  conv1.bias 		 torch.Size([32])
  conv2.weight 		 torch.Size([64, 32, 3, 3])
  conv2.bias 		 torch.Size([64])
  oneconv1.weight 		 torch.Size([16, 64, 1, 1])
  oneconv1.bias 		 torch.Size([16])
  conv3.weight 		 torch.Size([32, 16, 3, 3])
  conv3.bias 		 torch.Size([32])
  conv4.weight 		 torch.Size([64, 32, 3, 3])
  conv4.bias 		 torch.Size([64])
  conv5.weight 		 torch.Size([64, 64, 3, 3])
  conv5.bias 		 torch.Size([64])
  conv6.weight 		 torch.Size([10, 64, 5, 5])
  conv6.bias 		 torch.Size([10])
  input2.weight 		 torch.Size([5, 2])
  input2.bias 		 torch.Size([5])
  layer1.weight 		 torch.Size([5, 5])
  layer1.bias 		 torch.Size([5])
  out2.weight 		 torch.Size([1, 5])
  out2.bias 		 torch.Size([1])
  ```
 
## Training and Loss

* Number of epochs : 10
* Loss - as there are 2 components - image detection and sum - 2 loss functions are used 

### Loss function
> The loss functions picked are -
> 1. Crossentropy loss for the image classification part of the network.
> 2. Mean squared error loss for the addition part of the network. As adding two numbers is a regression problem, we use MSE loss to find the relationship between input and output. Here, CE loss is not effective as changing the range of inputs (from 0-9 to say 0-99) will lead the network to not work. However, MSE loss will still work although it might yield poor results. For this case, the network can be easily retrained.


  ```
  Epoch: 1, loss: 8588.646249156445, Classification Acc: 72.58666666666667, Addition Acc: 52.27
  Epoch: 2, loss: 1329.7578395307064, Classification Acc: 96.93, Addition Acc: 91.17333333333333
  Epoch: 3, loss: 968.9839583390858, Classification Acc: 97.88, Addition Acc: 95.56333333333333
  Epoch: 4, loss: 720.8626747094095, Classification Acc: 98.36, Addition Acc: 96.21666666666667
  Epoch: 5, loss: 597.3828105715802, Classification Acc: 98.67833333333334, Addition Acc: 97.28666666666666
  Epoch: 6, loss: 506.67549971531844, Classification Acc: 98.86666666666667, Addition Acc: 98.33666666666666
  Epoch: 7, loss: 450.0150337165105, Classification Acc: 98.98833333333333, Addition Acc: 98.30499999999999
  Epoch: 8, loss: 370.68281011172803, Classification Acc: 99.17, Addition Acc: 98.87
  Epoch: 9, loss: 342.45800989316194, Classification Acc: 99.21833333333333, Addition Acc: 98.965
  Epoch: 10, loss: 302.226245637954, Classification Acc: 99.325, Addition Acc: 99.08500000000001
  Finished Training

  ```

## Evaluation
> The network is evaluated by calculting the accuracy for the classification problem and the addition problem. For the addition problem, we have rounded off the final prediction to the nearest integer and then compared it with the target variables. It can be noticed that both the accuracies are same which means the addition part of the network is working 100% and gives the result as good as the classification.
```
Accuracy of the network on the 10,000 test images:  98.80191693290735
Accuracy of the network on the 10,000 test images:  98.80191693290735
```


## Observations/ Learning

* The model is able to predict the number and the sum very well
* Equal weightages were given to both the CE loss and MSE loss
* No activation function was used during addition of two numbers as it was a linear function

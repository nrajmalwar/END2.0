## Assignment

1. What is a neural network neuron?

> A neural network neuron is a unit in the layer of a network. It takes the weights from previous layer as the input and computes the activation function and the bias and
gives output to the next layer. Generally, a neuron will have a single activation function like sigmoid, softmax or ReLU. Depending on the application and the problem we're
trying to solve, the activation can change. It acts as a memory storage that provides non-linearity to our network through the use of activation functions.

---

2. What is the use of the learning rate?

> Learning rate is used to change the value of weights in such a way that the error is minimised. When we are trying to minimise the error function numerically, we need to use
the learning rate to iterate over our data continuously until error is minimised. The value chosen for learning rate should be optimal. Very small learning rate would result
in slow convergence, which means it will take many iterations to reach the minimum point in the loss function. A very high value can make the value of weights overshoot and
miss the minima and diverge. An optimal value of learning will reach the minima quickly.

---

3. How are weights initialized?

> The weights are initialized randomly with a normal distribution. We specify the mean and the standard deviation to control the spread of the randomly initialized weights.
The random initialization helps the weights to adjust their values for the network to perform well rather than incrementing them all from one value. The normal distribution
helps to control the spread of the weights so that we don't have to adjust their values drastically and it easier and faster for our model to train them.

---

4. What is "loss" in a neural network?

> When we train a neural network, we calculate an output and compare it with our known or expected output. The deviation of the calculated output from the original output is
called as loss. This loss can be in the form of difference, squared difference and so on. The loss helps the network train and adjust the value of its weights so as to minimise
the loss.

---

5. What is the "chain rule" in gradient flow?

> In order to minimize the loss, we adjust the weights in all the layers of our network. When we want to adjust the weights in the very first layer, we calculate the partial
derivative of the error function with respect to these weights. However, since the loss function is dependent of other weights, we need to calcute its partial derivative.
In order to do this, we use chain rule where we start differentiating from the last layer all the way to the first layer. This is called chain rule where the gradients are
flowing from the output layer all the way to the first hidden layer.

---


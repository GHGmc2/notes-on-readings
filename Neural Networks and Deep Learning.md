# Neural Networks and Deep Learning
> [官网](http://neuralnetworksanddeeplearning.com/), [Github](https://github.com/mnielsen/neural-networks-and-deep-learning)
> [douban](https://book.douban.com/subject/26727997/)
> [神经网络与深度学习（中文）](https://legacy.gitbook.com/book/hit-scir/neural-networks-and-deep-learning-zh_cn/details)

## Notes

### What this book is about

Neural networks are one of the most beautiful **programming paradigms** ever invented.

One conviction underlying the book is that it's better to obtain a solid understanding of **the core principles** of neural networks and deep learning, rather than a hazy understanding of a long laundry list of ideas.

You need to understand the durable, lasting insights underlying how neural networks work. Technologies come and technologies go, but **insight** is forever.

Be **principle-oriented and hands-on**.

### On the exercises and problems

My advice is that you really should attempt most of the exercises, and you should aim not to do most of the problems. Struggling with a project you care about.

**Emotional commitment is a key to achieving mastery.**

## Using neural nets to recognize handwritten digits

### Perceptrons
> 可自动调整的逻辑门

The NAND gate is universal for computation, that is, we can build any computation up out of NAND gates.

It turns out that we can devise **learning algorithms** which can **automatically tune the weights and biases** of a network of artificial neurons. These learning algorithms enable us to use artificial neurons in a way which is radically different to conventional logic gates(explicitly laying out a circuit of NAND and other gates).

### Sigmoid neurons
> 使平滑

A small change in the weights or bias of any single perceptron in the network can sometimes cause the output of that perceptron to completely flip. That flip may then cause the behavior of the rest of the network to completely change in some very complicated way.

Sigmoid neurons are similar to perceptrons, but modified so that **small changes in their weights and bias cause only a small change in their output**.

The sigmoid neuron has weights for each input $w_1,w_2,$…, and **an overall bias** $b$, $z = \sum_{i=1}^{n}{w_i}{x_i} + b$

The shape of the sigmoid function is a smoothed out version of a step function. It turns out that when we compute those partial derivatives later, using $\sigma=\frac{1}{1+e^{-z}}$ will simplify the algebra, simply because exponentials have lovely properties when differentiated.

### The architecture of neural networks
> 多层抽象，结点表示局部特征（输入间的相互影响？）

Recurrent neural networks(RNN) are much closer in spirit to how our brains work than feedforward networks.

### Learning with gradient descent

The cost function $C$ 's change:
$$
\Delta{C} \approx \sum_{i=1}^n \frac{\partial{C}}{\partial{v_i}} \Delta{v_i} = \nabla{C}\cdot\Delta{v}
$$

$\nabla{C}$ is called the gradient vector. The gradient of a function is the vector of partial derivatives with respect to all of the independent variables.

**Stochastic gradient descent** can be used to speed up learning. The idea is to estimate the gradient $\nabla{C}$ by computing $\nabla{C_x}$ for **a small sample** of randomly chosen training inputs.

The stochastic gradient descent works by picking out a randomly chosen mini-batch of training inputs, and training. Then we pick out another randomly chosen mini-batch and train. And so on, **until we've exhausted the training inputs**, which is said to complete an **epoch** of training.

Online or incremental learning: use a mini-batch size of just 1.

### Implementing our network to classify digits
> [network.py](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py)

moral: **sophisticated algorithm ≤ simple learning algorithm + good training data**
```python
import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # The biases and weights are stored as lists of NumPy matrices.
        # numpy.random.randn()与rand()的区别：https://zhuanlan.zhihu.com/p/34122964
        # Python zip()函数：http://www.runoob.com/python/python-func-zip.html
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network."""
        for b, w in  zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.
        The ``training_data`` is a list of tuples ``(x, y)`` representing the
        training inputs and the desired outputs."""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        # Python xrange() 函数：http://www.runoob.com/python/python-func-xrange.html
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent
        using backpropagation to a single mini batch.
        The ``eta`` is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta =  self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Here, l = 1 means the last layer of neurons, l = 2 is the second-last layer, and so on.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network outputs the correct result."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return  sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x / \partial a
        for the output activations."""
        return (output_activations-y)

"""vectorizing the function: when the input z is a vector or Numpy array,
Numpy automatically applies the function sigmoid elementwise, that is, in vectorized form."""
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
```

## How the backpropagation algorithm works
> If you're not crazy about mathematics you may be tempted to skip the chapter, and to treat backpropagation as a black box whose details you're willing to ignore. But at those points you should still be able to **understand the main conclusions**, even if you don't follow all the reasoning.

### Matrix-based approach to computing the output from a neural network

The activation $a_j^l$ of the $j^{th}$ neuron in the $l^{th}$ layer:
$$
a_j^l = \sigma(\sum_{k}w_{jk}^{l}a_k^{l-1} + b_j^l)
$$
where $w_{jk}^{l}$ denotes the weight for the connection **from the $k^{th}$** neuron in the $(l−1)^{th}$ layer **to the $j^{th}$** neuron in the $l^{th}$ layer, and the sum is over all neurons $k$ in the $(l−1)^{th}$ layer.

Equation can also be rewritten in vectorized form:
$$
a^l = \sigma({w^{l}a^{l-1}+b^l})
$$
where $w^l$ is a **weight matrix** for each layer $l$, and $b^l$ is a **bias vector** for each layer $l$. We call $a^l$ the **weighted input** to the neurons in layer $l$.

### The two assumptions about the cost function

 - the cost function can be written as an average $C=\frac{1}{n}\sum_{x}C_x$ over cost functions $C_x$ for individual training examples, $x$
 - the cost can be written as a function of the outputs from the neural network: $C=C(a^L)$. $L$ denotes the number of layers in the network, and $a^L=a^L(x)$ is the vector of activations output from the network

### The Hadamard product(Schur product)

Suppose $s$ and $t$ are two vectors of the same dimension. Then we use $s\odot t$ to denote **the elementwise product** of the two vectors: $(s\odot t)_j=s_{j}t_j$.

### The four fundamental equations behind backpropagation(and their proof)

 - the error in the output layer: $\delta_j^L=\frac{\partial C}{\partial{a_j^L}}\sigma'(z_j^L)$, or $\delta^L=\nabla_aC\odot\sigma'(z^L)$
 - the error $δ^l$ in terms of the error in the next layer $δ^{l+1}$: $\delta^l=((w^{l+1})^T \delta^{l+1})\odot\delta'(z^l)$
 - the rate of change of the cost with respect to any **bias** in the network: $\frac{\partial C}{\partial b_j^l}=\delta_j^l$
 - the rate of change of the cost with respect to any **weight** in the network: $\frac{\partial C}{\partial w_{jk}^l}=a_k^{l-1}\delta_j^l$

The four fundamental equations turn out to hold for any activation function(not just the standard sigmoid function). And so we can use these equations to **design** activation functions which have particular desired learning properties.

### The backpropagation algorithm

 1. **Input** x: Set the corresponding activation $a^1$ for the input layer.
 2. **Feedforward**: For each $l=2,3,…,L$ compute $z^l=w^la^{l−1}+b^l$ and $a^l=\sigma(z^l)$.
 3. **Output error** $δ^L$: Compute the vector $δ^L=\nabla_aC\odot \sigma'(z^L)$.
 4. **Backpropagate the error**: For each $l=L−1,L−2,…,2$ compute $\delta^l=((w^{l+1})^T\delta^{l+1})\odot\sigma'(z^l)$.
 5. **Output**: The gradient of the cost function is given by $\frac{\partial C}{\partial w_{jk}^l}=a_k^{l-1}\delta_j^l$ and $\frac{\partial C}{\partial b_j^l}=\delta_j^l$.

The backpropagation algorithm computes the gradient of the cost function for a single training example $C=C_x$. In practice, it's common to combine backpropagation with a learning algorithm such as stochastic gradient descent, in which we compute the gradient for many training examples.

### In what sense is backpropagation a fast algorithm?

What's clever about backpropagation is that it enables us to **simultaneously compute all the partial derivatives** ${\partial C}/{\partial w_j}$ using just one forward pass through the network, followed by one backward pass through the network. The total cost of backpropagation is roughly the same as making just two forward passes through the network.

### Backpropagation: the big picture

Every **edge** between two neurons in the network is associated with a rate factor which is just **the partial derivative** of one neuron's activation with respect to the other neuron's activation.
The rate factor for a **path** is just the **product** of the rate factors along the path. And the total rate of change $\partial C/\partial w^l_{jk}$ is just **the sum of the rate factors of all paths** from the initial weight to the final cost.

## Improving the way neural networks learn
> **Understanding the backpropagation algorithm** is the foundation for learning in most work on neural networks.
> The philosophy is that the best entree to the plethora of available techniques is **in-depth study of a few of the most important**.

### The cross-entropy cost function
The cross-entropy cost function for a neuron:
$$
C=-\frac{1}{n}\sum_x[y\ln a+(1-y)\ln(1-a)].
$$

Two properties in particular make it reasonable to interpret the cross-entropy as a cost function.

 - it's non-negative, that is, $C>0$.
 - if the neuron's actual output is close to the desired output, then the cross-entropy will be close to zero.

The cross-entropy cost function has the benefit that, unlike the quadratic cost, it **avoids the problem of learning slowing down**:
$$
\frac{\partial C}{\partial w_j}=\frac{1}{n}\sum_xx_j(\sigma(z)-y),
\frac{\partial C}{\partial b}=\frac{1}{n}\sum_x(\sigma(z)-y).
$$
This is a beautiful expression. It tells us that **the rate at which the weight learns is controlled by the error in the output**.

**Many-layer multi-neuron networks**:
$$
C=-\frac{1}{n}\sum_x\sum_j[y_j\ln a_j^L+(1-y_j)\ln(1-a_j^L)]
$$
where $\sum_j$ means summing over all the output neurons.

The partial derivative with respect to the weights and biases in the output layer:
$$
\frac{\partial C}{\partial w^L_{jk}}=\frac{1}{n}\sum_xa_k^{L-1}(a^L_j-y_j), \newline
\frac{\partial C}{\partial b_j^L}=\frac{1}{n}\sum_x(a_j^L-y_j).
$$

Using the quadratic cost **when we have linear neurons in the output layer** (then the quadratic cost will not give rise to any problems with a learning slowdown).

The cross-entropy is a measure of surprise.
[Cross entropy: Motivation](https://en.wikipedia.org/wiki/Cross_entropy#Motivation)
 the Kraft inequality in chapter 5 of [Elements of Information Theory](https://book.douban.com/subject/1822197/)

#### Softmax

The idea of softmax is to **define a new type of output layer** for our neural networks.

In a softmax layer, we apply the so-called **softmax function**(not the sigmoid function) to the the weighted inputs $z_j^L$, the activation $a^L_j$ of the $j^{th}$ **output neuron** is:
$$
a_j^L=\frac{e^{z_j^L}}{\sum_k e^{z_k^L}}
$$
where $\sum_k$ means summing over all the $k$ output neurons.

**Monotonicity** of softmax: increasing $z^L_j$ is guaranteed to increase the corresponding output activation, $a^L_j$, and will decrease all the other output activations.
**Non-locality** of softmax: any particular output activation $a^L_j$ depends on all the weighted inputs.

Equation implies that:

 - the output activations are all positive,
 - the output activations are guaranteed to always sum up to 1.

the output from the softmax layer can be thought of as **a probability distribution**.

Define **the log-likelihood cost function**: $C=-\ln a^L_y$, where $y$ denotes the desired output. Then
$$
\frac{\partial C}{\partial w^L_{jk}}=a^{L-1}_k(a^L_j-y_j),\newline
 \frac{\partial C}{\partial b^L_j=a^L_j-y_j}
$$
these expressions ensure that we will not encounter a learning slowdown.

It's useful to think of **a softmax output layer with log-likelihood cost** as being quite similar to **a sigmoid output layer with cross-entropy cost**. As a more general point of principle, softmax plus log-likelihood is worth using whenever you want to interpret the output activations as probabilities.

**Backpropagation** with softmax and the log-likelihood cost: $\sigma^L_j=a^L_j-y_j$.

### Overfitting and regularization

From a practical point of view, what we really care about is improving classification accuracy on the test data, while **the cost on the test data is no more than a proxy for classification accuracy**.

**early stopping**
Using the basic **hold out** method, based on the training_data, validation_data, and test_data.
We'll compute the classification **accuracy on the validation_data** at the end of each epoch. Once the classification accuracy on the validation_data has saturated, we stop training. In practice, we continue training until we're confident that the accuracy has saturated.

In general, one of the best ways of reducing overfitting is to increase the size of the training data.

#### Regularization

**L~2~ regularization**: add an extra term to the cost function, a term called the regularization term.
$$
C=C_0+\frac{\lambda}{2n}\sum_ww^2
$$
where $C_0$ is the original, unregularized cost function.

The regularization term doesn't include the biases.
Having a large bias doesn't make a neuron sensitive to its inputs in the same way as having large weights. At the same time, allowing large biases gives our networks more flexibility in behaviour - in particular, large biases make it easier for neurons to saturate.

This rescaling is sometimes referred to as **weight decay**, since it makes the weights smaller. It looks as though this means the weights are being driven unstoppably toward zero, but that's not right, since the other term may lead the weights to increase.

Empirically, when training with different (random) weight initializations, the unregularized runs will occasionally get "stuck", apparently caught in local minima of the cost function. Why is this going on? Heuristically, if the cost function is unregularized, then the length of the weight vector is likely to grow, since changes due to gradient descent only make tiny changes to the direction, when the length is long.

#### Why does regularization help reduce overfitting?

The smallness of the weights means that **the behavior of the network won't change too much if we change a few random inputs**. That makes it difficult for a regularized network to learn the effects of local noise in the data. Instead, a regularized network learns to respond to types of evidence which are **seen often** across the training set.

The true test of a model is not simplicity, but rather how well it does in predicting new phenomena, in new regimes of behavior.

It has been **conjectured** that the dynamics of gradient descent learning in multilayer nets has a `self-regularization' effect".

#### Other techniques for regularization

**L~1~ regularization**
L~1~ regularization adding the sum of the absolute values of the weights.
$$
C=C_0+\frac{\lambda}{n}\sum_w|w|
$$
In L~1~ regularization, the weights shrink by a constant amount toward 0. In L~2~ regularization, the weights shrink by an amount which is proportional to $w$. The net result is that **L~1~ regularization tends to concentrate the weight of the network in a relatively small number of high-importance connections**, while the other weights are driven toward zero.

**Dropout**
Dropout randomly (and temporarily) deleting half the hidden neurons in the network. When we actually run the full network that means that twice as many hidden neurons will be active. To compensate for that, we halve the weights outgoing from the hidden neurons.

Heuristically, when we dropout different sets of neurons, **it's rather like we're training different neural networks**. And so the dropout procedure is like averaging the effects of a very large number of different networks.

The true measure of dropout is that it has been very successful in **improving the performance** of neural networks. Dropout has been **especially useful in training large, deep networks**, where the problem of overfitting is often acute.

**Artificially expanding the training data**:
Artificially expand the training data by applying operations that reflect real-world variation.

Big data and what it means to compare classification accuracies?
More training data can sometimes compensate for differences in the machine learning algorithm used.
What we want is **both** better algorithms and better training data.

### Weight initialization

A clever choice of **cost function** helps with saturated **output** neurons, it does nothing at all for the problem with saturated **hidden** neurons.

Initialize weights as Gaussian random variables with mean 0 and standard deviation $1/ \sqrt{n_{in}}$ is not only the speed of learning which is improved, it's sometimes also the final performance.

### Handwriting recognition revisited: the code
> [代码见：network2.py](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py)

### How to choose a neural network's hyper-parameters?
> use gradient descent to try to learn good values for hyper-parameters?

During the early stages you should make sure you can **get quick feedback** from experiments. You can further speed up experimentation by **stripping your network down to the simplest network** likely to do meaningful learning.

**Learning rate $\eta$**
Find a value for $\eta$ where the cost oscillates or increases during the first few epochs, it will give us an order of magnitude estimate for the threshold value of $\eta$.

The learning rate's primary purpose is really to control the step size in gradient descent, and monitoring the training cost is the best way to detect if the step size is too big.

Use variable learning schedule: hold the learning rate constant until the validation accuracy starts to get worse(the same basic idea as early stopping). Then decrease the learning rate by some amount.

**The regularization parameter $\lambda$**:
I suggest starting initially with no regularization, and determining a value for $\eta$. Using that choice of $\eta$, we can then use the validation data to select a good value for $\lambda$. That done, you should return and re-optimize $\eta$ again.


**Mini-batch size**
Online learning: using a mini-batch size of 1.

Choosing the best mini-batch size is a compromise. Too small, and you don't get to take full advantage of the benefits of good matrix libraries optimized for fast hardware. Too large and you're simply not updating your weights often enough.

What you need is to choose a compromise value which **maximizes the speed of learning**. Fortunately, the choice of mini-batch size is relatively **independent** of the other hyper-parameters (apart from the overall architecture).

**Automated techniques**
**Grid search**: systematically searches through a grid in hyper-parameter space.
[Random search for hyper-parameter optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
[Practical Bayesian optimization of machine learning algorithms](https://arxiv.org/pdf/1206.2944.pdf)

**Summing up**
>[Neural Networks: Tricks of the Trade](https://www.springer.com/cn/book/9783642352881)

[Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/pdf/1206.5533.pdf)
[Effective BackProp](https://www.springer.com/cn/book/9783642352881)

### Other techniques

#### Variations on stochastic gradient descent

**Hessian technique**
The cost function $C=C(w)$, which $w=w_1,w_2,$…
By Taylor's theorem, the cost function can be approximated near a point $w$ by:
$$
C(w+\Delta w)=C(w) + \sum_j\frac{\partial C}{\partial w_j}\Delta w_j+\frac{1}{2}\sum_{jk}\Delta w_j\frac{\partial^2 C}{\partial w_j\partial w_k}\Delta w_k + \cdots \newline
=C(w)+\nabla C\cdot\Delta w+\frac{1}{2}\Delta w^TH\Delta w + \cdots
$$
where $H$ is a matrix known as the **[Hessian matrix](https://zh.wikipedia.org/wiki/%E6%B5%B7%E6%A3%AE%E7%9F%A9%E9%98%B5)**, whose $jk^{th}$ entry is $\partial^2 C/\partial w_jw_k$.

Suppose we approximate $C$ by discarding the higher-order terms, the expression on the right-hand side can be minimized by choosing $\Delta w=-H^{-1}\nabla C$, then we'd expect that moving from the point $w$ to $w'=w+\Delta w=w−H^{-1}\nabla C$ should significantly decrease the cost function. Repeating this process suggests a possible algorithm for minimizing the cost.

Intuitively, the advantage Hessian optimization has is that it incorporates not just information about the gradient, but also information about how the gradient is changing.

There are theoretical and empirical results showing that Hessian methods converge on a minimum in fewer steps than standard gradient descent. In particular, by incorporating information about second-order changes in the cost function it's possible for the Hessian approach to avoid many pathologies that can occur in gradient descent.

Unfortunately, it's very difficult to apply in practice. Part of the problem is the sheer size of the Hessian matrix.

**Momentum-based gradient descent**
The momentum technique modifies gradient descent in two ways that make it more similar to the physical picture:

 - introduces a notion of "velocity" for the parameters we're trying to optimize. The gradient acts to change the velocity, not (directly) the "position",
 - introduces a kind of friction term, which tends to gradually reduce the velocity.

A nice thing about the momentum technique is that it takes almost no work to modify an implementation of gradient descent to incorporate momentum. In practice, the momentum technique is commonly used, and often speeds up learning.

**Other approaches to minimizing the cost function**
[Effective BackProp](https://www.springer.com/cn/book/9783642352881)
Nesterov's accelerated gradient technique, which improves on the momentum technique. [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf)

#### Other models of artificial neuron

**tanh neuron**: replaces the sigmoid function by the hyperbolic tangent function($tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}$). the output from tanh neurons ranges from -1 to 1(not 0 to 1).

**ReLU**(rectified linear unit): $output=max(0, w\cdot x+b)$.
Increasing the weighted input to a ReLU will never cause it to saturate, and so there is no corresponding learning slowdown. On the other hand, when the weighted input to a ReLU is negative, the gradient vanishes, and so the neuron stops learning entirely.

We do not yet have a solid theory of how activation functions should be chosen.

#### On stories in neural networks

In neural networks there are large numbers of parameters and hyper-parameters, and extremely complex interactions between them. In such extraordinarily complex systems it's exceedingly difficult to establish reliable general statements.
Understanding neural networks in their full generality is a problem that, like quantum foundations, tests the limits of the human mind.

The great age of exploration!

## [A visual proof that neural nets can compute any function](http://neuralnetworksanddeeplearning.com/chap4.html)
> You do not need to have read earlier chapters in this book. Instead, the chapter is structured to be enjoyable as a self-contained essay.

The **universality** theorem: neural networks with a single hidden layer can be used to approximate any continuous function to any desired precision. The underlying reasons for universality are **simple and beautiful**.

Deep networks have a **hierarchical structure** which makes them particularly well adapted to learn the hierarchies of knowledge that seem to be useful in solving real-world problems.

## Why are deep neural networks hard to train?

Deep networks are intrinsically more powerful than shallow networks.
[Learning Deep Architectures for AI](https://book.douban.com/subject/6346890/)

### The vanishing gradient problem

The gradient in deep neural networks is unstable

 - the vanishing gradient problem: early hidden layers learn much more slowly than later hidden layers.
 - the exploding gradient problem

### What's causing the vanishing gradient problem? Unstable gradients in deep neural nets

The fundamental problem here it's that the gradient in early layers is the **product** of terms from all the later layers.

The maximum of derivative of sigmoid function $< 1$ (reaches a maximum at $\sigma'(0)=1/4$). And if we choose the weights using a Gaussian with mean 0 and standard deviation 1, the weights will usually satisfy $|w_j|<1$. When we take a product of many such terms, that is $w\sigma'(z)$, the product will tend to exponentially decrease.

In fact, when using sigmoid neurons **the gradient will usually vanish**. The reason is that the $\sigma'(z)$ term also depends on $w, \sigma'(z)=\sigma'(wa+b)$. When we make $w$ large we tend to make $wa+b$ very large, where $\sigma'$ takes very small values.

**Identity neuron**: a neuron whose output is the same (up to rescaling by a weight factor) as its input.

For more complex networks, the gradient form has lots of pairs of the form $(w_j)^T\sum'(z^j)$, each additional term $(w_j)^T\sum'(z^j)$ tends to make the gradient vector smaller, leading to a vanishing gradient.

### Other obstacles to deep learning

 - the choice of activation function
 - the way weights are initialized
 - implementation of learning by gradient descent
 - choice of network architecture and other hyper-parameters

## Deep learning

### CNN
[Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

Convolutional neural networks use three basic ideas:

 - local receptive fields
 - shared weights
 - pooling

**Local receptive fields**
Each neuron in the first hidden layer will be connected to its local receptive fields. **Each connection learns a weight**. And the hidden neuron learns **an overall bias** as well.

You can think of a particular hidden neuron as learning to analyze its particular local receptive field. For each local receptive field, there is a different hidden neuron in the first hidden layer. We then slide (a stride length) the local receptive field across the entire input image. And so on, building up the first hidden layer.

**Shared weights and biases**
**Feature map**: the map from the input layer to the hidden layer.
All the neurons in a feature map share the same weights and bias. The shared weights and bias are often said to define a **kernel** or **filter**. A complete convolutional layer consists of several different feature maps.

For the $j,k$th hidden neuron, the output is:
$$
\sigma(b+\sum_{l=0}^{l'}\sum_{m=0}^{m'} w_{l,m}a_{j+l,k+m})
$$
Here, $\sigma$ is the neural activation function. $b$ is the shared value for the bias. $w_{l,m}$ is a two-dimension **array** of shared weights. $a_{x,y}$ denotes the input activation at position $x,y$.

This means that all the neurons in the first hidden layer detect exactly the same feature, just at different locations in the input image.

It is useful to apply the same feature detector everywhere in the image. Convolutional networks are well adapted to **the translation invariance** of images.

[Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf)

A big advantage of sharing weights and biases is that it greatly reduces the number of parameters involved in a convolutional network.

**Pooling layers**
Pooling layers are usually used immediately after convolutional layers. It takes each feature map output from the convolutional layer and prepares a **condensed** feature map.

**Max-pooling**: in max-pooling, a pooling unit simply outputs **the maximum activation** in the $2×2$ input region. We can think of it as a way for the network to ask **whether a given feature is found anywhere in a region** of the image. It then throws away the exact positional information(once a feature has been found, its exact location isn't as important as its rough location relative to other features).
A big benefit is that there are many fewer pooled features, and so this helps reduce the number of parameters needed in later layers.

**L~2~ pooling**: take **the square root of the sum** of the squares of the activations in the $2×2$ region.
L~2~ pooling is a way of condensing information from the convolutional layer.

The final layer of connections in the convolutional neural network is a fully-connected layer.

### CNN in practice

[theano](http://deeplearning.net/software/theano/)

A common pattern in CNN
![](http://neuralnetworksanddeeplearning.com/images/simple_conv.png)
 - we can think of the convolutional and pooling layers as learning about local spatial structure in the input training image, while the later, fully-connected layer learns at a more abstract level, integrating global information from across the entire image.

In an ideal world we'd have a theory telling us which activation function to pick for which application. But at present we're a long way from such a world.

**Expanding the training data**
like rotating, translating, and skewing the training images.

**Using an ensemble of networks**

Why we only applied dropout to the fully-connected layers?
**The convolutional layers have considerable inbuilt resistance to overfitting**. The reason is that the shared weights mean that convolutional filters are forced to learn from across the entire image. This makes them less likely to pick up on local idiosyncracies in the training data. And so there is less need to apply other regularizers, such as dropout.

[Classification datasets results](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)

### The code for our convolutional networks
> [network3.py](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network3.py)

### [Recent progress in image recognition](http://neuralnetworksanddeeplearning.com/chap6.html#recent_progress_in_image_recognition)

### Other approaches to deep neural nets

**Recurrent neural networks (RNNs)**
RNNs are neural networks in which there is some notion of **dynamic change over time**. And, not surprisingly, they're particularly useful in **analysing data or processes that change over time**. Such data and processes arise naturally in problems such as **speech or natural language**, for example.

Long short-term memory units (**LSTM**s)

**Deep belief nets(DBN), generative models, and Boltzmann machines**

 - DBNs are an example of what's called a generative model: not only can it read digits, it can also write them.
 - DBNs can do unsupervised and semi-supervised learning.

DBNs and other generative models likely deserve more attention than they are currently receiving.
[Deep belief networks](http://www.scholarpedia.org/article/Deep_belief_networks)
[A Practical Guide to Training Restricted Boltzmann Machines](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)

Active areas of research:

 - NPL: [A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning](http://machinelearning.org/archive/icml2008/papers/391.pdf), [Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398)
 - Machine translation: [Sequence to Sequence Learning with Neural Networks](http://neuralnetworksanddeeplearning.com/assets/MachineTranslation.pdf)
 - DQN(Deep Q-Network): [Playing Atari with Deep Reinforcement Learning](http://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), [Human-level control through deep reinforcement learning](http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)

### On the future of neural networks

Intention-driven user interfaces

Machine learning, data science, and the virtuous circle of innovation

The role of neural networks and deep learning

**Conway's law**: Any organization that designs a system... will inevitably produce a design whose structure is a copy of the organization's communication structure.
The structure of our knowledge shapes the social organization of science. But that social shape in turn constrains and helps determine what we can discover. This is the scientific analogue of Conway's law.
Is there some way of measuring how powerful and promising a set of ideas is? Conway's law suggests that as a rough and heuristic proxy metric we can evaluate the complexity of the social structure associated to those ideas.
 1. how powerful a set of ideas are associated to deep learning, according to this metric of social complexity?
deep learning is still a rather shallow field. It's still possible for one person to master most of the deepest ideas in the field.
 2. how powerful a theory will we need, in order to be able to build a general artificial intelligence?
to get to such a point we will necessarily see the emergence of many **interrelating disciplines**, with a complex and surprising structure mirroring the structure in our deepest insights. We don't yet see this rich social structure in the use of neural networks and deep learning.

That's an exciting creative opportunity.

## Appendix: Is there a simple algorithm for intelligence?

Whether there is a simple set of principles which can be used to explain intelligence? In particular, and more concretely, is there a simple algorithm for intelligence?

It suggests that there are **common principles underlying** how different parts of the brain learn to respond to sensory data. That commonality provides at least some support for the idea that there is a set of simple principles underlying intelligence.

"Well, some of these developments may lie one hundred Nobel prizes away" --- Jack Schwartz


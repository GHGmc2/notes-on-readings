# Hands-On Machine Learning with Scikit-Learn and TensorFlow
> [主页](http://shop.oreilly.com/product/0636920052289.do), [勘误](https://www.oreilly.com/catalog/errata.csp?isbn=0636920052289)
> [Github](https://github.com/ageron/handson-ml), [Jupyter Viewer](https://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/index.ipynb), [中文注释](https://github.com/DeqianBai/Hands-on-Machine-Learning)
> [douban](https://book.douban.com/subject/30317874/)
> [中文翻译](https://github.com/apachecn/hands_on_Ml_with_Sklearn_and_TF)

## Preface

This book assumes that you know close to nothing about Machine Learning.

### Prerequisites
This book assumes that you have some Python programming experience and that you are familiar with Python’s main scientific libraries, in particular NumPy, Pandas, and Matplotlib.

### Roadmap
Don’t jump into deep waters too hastily, you should master the fundamentals first. Moreover, most problems can be solved quite well using simpler techniques.

### Other Resources
[What are the best, regularly updated machine learning blogs or resources available?](https://www.quora.com/What-are-the-best-regularly-updated-machine-learning-blogs-or-resources-available)
[deeplearning.net](http://deeplearning.net/)

[Dataquest](https://www.dataquest.io/)
[Kaggle](https://www.kaggle.com/)

# The Fundamentals of Machine Learning

## The Machine Learning Landscape

Machine Learning is about making machines get better at some task by **learning from data, instead of having to explicitly code rules**.

### Types of ML Systems

based on:

 - Whether or not they are trained with human supervision (supervised, unsupervised, semisupervised, and Reinforcement Learning)
 - Whether or not they can learn incrementally on the fly (**online vs. batch learning**)
 - Whether they work by simply comparing new data points to known data points, or instead detect patterns in the training data and build a predictive model, much like scientists do (instance-based versus model-based learning)

#### Batch and Online Learning

**Batch learning**: it must be trained using **all** the available data.
First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called **offline learning**.

**Online learning (incremental learning)**: train the system incrementally by feeding it data instances sequentially, either **individually or by mini-batches**.
A big **challenge** with online learning is that if bad data is fed to the system, the system’s performance will gradually decline. To reduce this risk, you need to **monitor** your system closely.

### Main Challenges

Bad data:

 - **Insufficient Quantity** of Training Data
 - **Nonrepresentative** Training Data: sample bias
 - **Poor-Quality** Data: errors, outliers, and noise
 - **Irrelevant Features**: the training data contains too many irrelevant features.
Feature engineering:
	 - Feature selection: selecting the most useful features to train on among existing features.
	 - Feature extraction: combining existing features to produce a more useful one.
	 - Creating new features by gathering new data.

Bad algorithm: 

 - **Overfitting** the Training Data
A **hyperparameter** is a parameter of a learning algorithm (not of the model).
Possible solutions: 
	 - **Simplify the model** by selecting one with fewer parameters (e.g., a linear model rather than a high-degree polynomial model), by **reducing the number of attributes** in the training data or by **constraining the model** (regularization)
	 - Gather **more training data**
	 - Reduce the noise in the training data (e.g., fix data errors and remove outliers)
 - **Underfitting** the Training Data
Main options: 
	 - Selecting a more powerful model, with more parameters
	 - Feeding better features to the learning algorithm
	 - Reducing the constraints on the model

### Testing and Validating

Generalization error (out-of- sample error): the error rate on new cases.

The training set, the validation set, and the test set.

**Cross-validation**: the training set is split into **complementary** subsets, and each model is trained against a **different combination** of these subsets and validated against the **remaining** parts.

Once the model type and hyperparameters have been selected, a final model is trained using these hyperparameters on the **full** training set, and the generalized error is measured on the **test** set.

No Free Lunch (NFL) theorem: if you make absolutely no assumption about the data, then there is no reason to prefer one model over any other.

## End-to-End Machine Learning Project
 
The main steps: [Machine Learning Project Checklist](http://www.ic.unicamp.br/~sandra/pdf/Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow-427-432.pdf)

 1. Look at the big picture.
 2. Get the data.
 3. Discover and visualize the data to gain insights.
 4. Prepare the data for Machine Learning algorithms.
 5. Select a model and train it.
 6. Fine-tune your model.
 7. Present your solution.
 8. Launch, monitor, and maintain your system.

### Working with Real Data

Open datasets

### Look at the big picture

#### Frame the Problem

two questions: 
 - what exactly is the business objective?
Getting this right is critical, as it will determine how you frame the problem, what algorithms you will select, what performance measure you will use to evaluate your model, and how much effort you should spend tweaking it.
 - what the current solution looks like (if any)?
It will often give you a reference performance, as well as insights on how to solve the problem.

A sequence of data processing components is called a data **pipeline**.

#### Select a Performance Measure

**The higher the norm index, the more it focuses on large values and neglects small ones.** (The RMSE is more sensitive to outliers than the MAE)

 - $\ell_1$ norm(Manhattan norm): MAE(Mean Absolute Error, also called the Average Absolute Deviation)
 - $\ell_2$ norm(Euclidian norm): RMSE(Root Mean Square Error)
 - $\ell_k$ norm

#### Check the Assumptions

### Get the data

#### Create the Workspace

Create a workspace directory:
```
$ export ML_PATH="$HOME/ml"
$ mkdir -p $ML_PATH
```

Creating an isolated environment: **virtualenv**
It is strongly recommended so you can work on different projects without having conflicting library versions. 
```
$ pip3 install --user --upgrade virtualenv
$ cd $ML_PATH
$ virtualenv env

Now every time you want to activate this environment, just open a terminal and type:
$ cd $ML_PATH
$ source env/bin/activate
```
While the environment is active, any package you install using pip will be installed in this isolated environment.

#### Download the Data

Automating the process of fetching(downloading and loading) the data with small functions.

#### Take a Quick Look at the Data Structure

**pandas** methods:

 - head(): look at the top five rows
 - info(): get a quick description
 - **describe()**: shows a summary of the numerical attributes
 - hist(): plot a histogram for a numerical attribute

#### Create a Test Set

 - random sampling: pick some instances(typically 20% of the dataset) randomly, and set them aside
	 - A common solution is to use each instance’s identifier(use the most stable features to build a unique identifier) to decide whether or not it should go in the test set.
 - stratified sampling: maintain the ratio in the sample.
	 - Scikit-Learn’s StratifiedShuffleSplit class

Ensure that the test set will remain consistent across multiple runs, even if you refresh the dataset.

### Discover and visualize the data to gain insights

### Prepare the data for Machine Learning algorithms

#### Data Cleaning

[API design for machine learning software: experiences from the scikit-learn project](https://arxiv.org/pdf/1309.0238v1.pdf)

 - Consistency
	 - Estimators
	 - Transformers
	 - Predictors
 - Inspection
 - Nonproliferation
 - Composition
 - Sensible defaults

#### Feature Scaling

 - Min-max scaling (normalization): values are shifted and rescaled so that they end up ranging from 0 to 1.
 - Standardization: first it subtracts the mean value, and then it divides by the variance so that the resulting distribution has unit variance. Standardization is much less affected by outliers.

Fit the scalers to the training data only, not to the full dataset (including the test set).

### Select a model and train it

### Fine-tune your model

#### Grid Search

Scikit-Learn’s GridSearchCV will evaluate all the possible combinations of hyperparameter values, using cross-validation.
The grid search approach is fine when you are exploring **relatively few combinations**.

#### Randomized Search

When the hyperparameter **search space is large**, it is often preferable to use RandomizedSearchCV. It evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration.

#### Ensemble Methods

Combine the models that perform best.

### Launch, monitor, and maintain your system

Write monitoring code to check your **system’s live performance** at regular intervals and trigger alerts when it drops.
Models tend to “rot” as data evolves over time, unless the models are regularly trained on fresh data.

Plug the human evaluation pipeline into your system. Evaluating your system’s performance will require sampling the system’s predictions and evaluating them.

Make sure you evaluate the system’s input data quality. Monitoring the inputs is particularly important for online learning systems.

Train your models on a regular basis using fresh data. You should automate this process as much as possible. If your system is an online learning system, you should make sure you save snapshots of its state at regular intervals so you can easily roll back to a previously working state.

### Try it out

It is probably preferable to be comfortable with the overall process and know three or four algorithms well rather than to spend all your time exploring advanced algorithms and not enough time on the overall process.

## Classification

### Performance Measures

#### Cross-Validation

#### Confusion Matrix

$$
precision=\frac{TP}{TP+FP}, recall=\frac{TP}{TP+FN}
$$

#### Precision and Recall

$$
F_1 = \frac{2}{\frac{1}{precision} + \frac{1}{recall}} = \frac{TP}{TP + \frac{FN + FP}{2}}
$$

#### Precision/Recall Tradeoff

decision threshold

#### The ROC Curve

Receiver operating characteristic (ROC) curve plots the true positive rate (another name for recall) against the false positive rate. (sensitivity (recall) vs. 1 – specificity)

Area under the curve (AUC)

### Multiclass Classification

strategies to perform multiclass classification using multiple binary classifiers:

 - one-versus-all (OvA), also called one-versus-the-rest: 
 - one-versus-one (OvO): 

Some algorithms scale poorly with the size of the training set, so for these algorithms OvO is preferred since it is faster to train many classifiers on small training sets than training few classifiers on large training sets. For most binary classification algorithms, however, OvA is preferred.

### Error Analysis

### Multilabel Classification

Multilabel classification system: outputs multiple binary labels.

### Multioutput Classification

Multioutput-multiclass classification (or simply multioutput classification): it is simply a generalization of multilabel classification where each label can be multiclass.

## Training Models
> [safari book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/ch04.html)

### Linear Regression

#### The normal equation

$$
\hat \theta=(X^T\cdot X)^{-1}\cdot X^T\cdot \bm y
$$
where $\hat \theta$ is the value of $\theta$ that minimizes the cost function, $\bm y$ is the vector of target values containing $y^{(1)}$ to $y^{(m)}$.

#### Computational Complexity

### Gradient Descent

#### Batch Gradient Descent

Computing the gradients based on the full training set.

#### Stochastic Gradient Descent

SGD just picks a random instance in the training set at every step and computes the gradients based only on that single instance.

Randomness is **good to escape from local optima**, but bad because it means that the algorithm can never settle at the minimum. One solution to this dilemma is to **gradually reduce the learning rate** (**Simulated annealing**). The function that determines the learning rate at each iteration is called the Learning schedule.

#### Mini-batch Gradient Descent

Computes the gradients on small random sets of instances called mini- batches.
The main advantage of Mini-batch GD over Stochastic GD is that you can **get a performance boost from hardware optimization of matrix operations**.

[Comparison of algorithms for Linear Regression](https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/ch04.html#linear_regression_algorithm_comparison)

### Polynomial Regression

Add powers of each feature as new features, then train a linear model on this extended set of features.

### Learning Curves

Learning curves are plots of the **model’s performance on the training set and the validation set** as a function of the training set size.

**The Bias/Variance Tradeoff**
A model’s generalization error can be expressed as the sum of three very different errors:

 - Bias: due to **wrong assumptions**. A high-bias model is most likely to **underfit** the training data.
 - Variance: due to the model’s **excessive sensitivity to small variations** in the training data. A model with many degrees of freedom is likely to have high variance, and thus to **overfit** the training data.
 - Irreducible error: due to the **noisiness of the data** itself. The only way to reduce this part of the error is to clean up the data.

Increasing a model’s complexity will typically increase its variance and reduce its bias. Conversely, reducing a model’s complexity increases its bias and reduces its variance.

### Regularized Linear Models

#### Ridge Regression

#### Lasso Regression

#### Elastic Net

#### Early Stopping

Stop training as soon as the validation error reaches a minimum. (stop only after the validation error has been above the minimum for some time, then roll back the model parameters to the point where the validation error was at a minimum)

### Logistic Regression

Logistic Regression is commonly used to estimate the **probability** that an instance belongs to a particular class.

Decision boundaries

#### Softmax Regression

When given an instance $x$, the Softmax Regression model first computes a score $s_k(x)$ for each class $k$, then estimates the probability of each class by applying the softmax function (also called the normalized exponential) to the scores.

Softmax function:
$$
\hat p_k = \frac{exp(s_k(x))}{\sum_{j=1}^{K}exp(s_j(x))}
$$
where $K$ is the number of classes.

The Softmax Regression classifier predicts only one class at a time.

Cross entropy: 

## SVM

## Decision Trees

## Ensemble Learning and Random Forests

## Dimensionality Reduction

Reducing dimensionality does lose some information, so even though it will speed up training, it may also make your system perform slightly worse. It also makes your pipelines a bit more complex and thus harder to maintain.

### The Curse of Dimensionality

The more dimensions the training set has, the greater the risk of **overfitting** it.

High-dimensional datasets are at risk of being very sparse: most training instances are likely to be far away from each other.

### Main Approaches for Dimensionality Reduction

#### Projection

In most real-world problems, training instances are not spread out uniformly across all dimensions. As a result, all training instances actually lie within (or close to) a much lower-dimensional subspace of the high-dimensional space.

#### Manifold Learning

A $d$-dimensional **manifold** is a part of an $n$-dimensional space (where $d < n$) that locally resembles a $d$-dimensional hyperplane.

Many dimensionality reduction algorithms work by **modeling the manifold on which the training instances lie**; this is called Manifold Learning.
It relies on the manifold **assumption** (also called the manifold hypothesis), which holds that **most real-world high-dimensional datasets lie close to a much lower-dimensional manifold**.

### PCA (Principal Component Analysis)

First it identifies the hyperplane that lies closest to the data, and then it projects the data onto it.

TODO.

### Kernel PCA

### LLE (Locally Linear Embedding)

First measuring how each training instance linearly relates to its closest neighbors, and then looking for a low-dimensional representation of the training set where these local relationships are best preserved.

### Other Dimensionality Reduction Techniques

 - Multidimensional Scaling (MDS)
 - Isomap
 - t-Distributed Stochastic Neighbor Embedding (t-SNE)
 - Linear Discriminant Analysis (LDA)

# Neural Networks and Deep Learning

## Up and Running with TensorFlow

## Introduction to Artificial Neural Networks

### From Biological to Artificial Neurons

#### Multi-Layer Perceptron and Backpropagation

activation functions:

 - logistic function: $\sigma(z) = 1 / (1 + exp(–z))$
 - hyperbolic tangent function: $tanh(z) = 2\sigma(2z)–1$
 - ReLU function: $ReLU(z) = max(0, z)$

## Training Deep Neural Nets

### Vanishing/Exploding Gradients Problems

[Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
We need the **variance** of the outputs of each layer to be equal to the variance of its inputs, and we also need the **gradients** to have equal variance before and after flowing through a layer in the reverse direction.

#### Xavier and He Initialization

Normal distribution with mean 0 and standard deviation $\sigma = \sqrt \frac{2}{n_{inputs} + n_{outputs}}$
Or a uniform distribution between $‐r$ and $+r$, with $r = \sqrt \frac{6}{n_{inputs} + n_{outputs}}$

#### Nonsaturating Activation Functions

leaky ReLU

 - randomized leaky ReLU (RReLU)
 - parametric leaky ReLU (PReLU)

ELU(exponential linear unit):

Which activation function should you use for the hidden layers of your deep neural networks?
In general, ELU > leaky ReLU (and its variants) > ReLU > tanh > logistic.

 - If you care a lot about runtime performance, then you may prefer leaky ReLUs over ELUs.
 - RReLU if your network is overfitting
 - PReLU if you have a huge training set

#### Batch Normalization

Let the model learn the optimal scale and mean of the **inputs** for each layer: adding an operation in the model just before the activation function of each layer, simply **zero-centering and normalizing the inputs, then scaling and shifting the result** using two new parameters per layer (one for scaling, the other for shifting).

#### Gradient Clipping

A popular technique to lessen the **exploding** gradients problem is to simply clip the gradients during backpropagation so that they never exceed some threshold (this is mostly useful for **RNN**).

### Reusing Pretrained Layers

#### Model Zoos

#### Unsupervised Pretraining

**Train the layers one by one**, starting with the lowest layer and then going up, using an **unsupervised feature detector algorithm** such as Restricted Boltzmann Machines (RBMs) or autoencoders. Each layer is trained on the output of the previously trained layers.
Once all layers have been trained this way, you can **fine-tune** the network using **supervised** learning.


### Faster Optimizers

You should almost always use Adam optimization.

#### Momentum optimization

Momentum optimization cares a great deal about what previous gradients were.

#### Adam(adaptive moment estimation) Optimization

It keeps track of an **exponentially decaying average of past gradients** and an **exponentially decaying average of past squared gradients**.

#### Learning Rate Scheduling

### Avoiding Overfitting Through Regularization

#### Early Stopping

#### l~1~ and l~2~ Regularization

#### Dropout

#### Max-Norm Regularization

#### Data Augmentation

### Practical Guidelines

## Distributing TensorFlow Across Devices and Servers

### Multiple Devices on a Single Machine

#### Installation

TensorFlow uses CUDA and cuDNN to control the GPU cards and accelerate computations.

 - Compute Unified Device Architecture library (CUDA)
 - CUDA Deep Neural Network library (cuDNN): **a GPU-accelerated library of primitives for DNNs**.

#### Managing the GPU RAM

By default TensorFlow automatically grabs all the RAM in all available GPUs the first time you run a graph.

Three solutions:

 - run each process on different GPU cards: set the CUDA_VISIBLE_DEVICES environment variable
 - tell TensorFlow to grab only a fraction of the memory: create a ConfigProto object, set its gpu_options.per_process_gpu_memory_fraction option
 - tell TensorFlow to grab memory only when it needs it: set config.gpu_options.allow_growth to True

#### Placing Operations on Devices

 - dynamic placer
 - simple placer

**Simple placement**
The simple placer respects the following rules:

 - If a node was already placed on a device in a previous run of the graph, it is left on that device.
 - Else, if the user pinned a node to a device, the placer places it on that device.
To pin nodes onto a device, you must create a device block using the device() function.
 - Else, it defaults to GPU #0, or the CPU if there is no GPU.
**The CPU is shared by all tasks located on the same machine.** The "/cpu:0" device aggregates all CPUs on a multi-CPU system. There is currently no way to pin nodes on specific CPUs.

**Logging placements**
Set the log_device_placement option to True; this tells the placer to log a message whenever it places a node.

**Dynamic placement function**
Specify a function (instead of a device name) when create a device block. TensorFlow will call this function for each operation it needs to place in the device block, and the function must return the name of the device to pin the operation on.

**Operations and kernels**
**kernel**: an implementation for a device.

**Soft placement**: fall back to the CPU when the operation has no kernel for GPU.
Set the allow_soft_placement configuration option to True.

#### Parallel Execution

graph

 - node (op)
 - edge (dependency)

When TensorFlow runs a graph, it starts by finding out the list of nodes that need to be evaluated, and it counts how many dependencies each of them has. Then it starts evaluating the nodes with zero dependencies.

TensorFlow manages a thread pool on each device to parallelize operations. These are called the **inter-op thread pools**.
Some operations have multi‐ threaded kernels: they can use other thread pools (one per device) called the **intra-op thread pools**.

#### Control Dependencies

To postpone evaluation of some nodes.

### Multiple Devices Across Multiple Servers

TensorFlow cluster:
|physical| logical |
|--|--|
| machine |  |
| CPU/GPU | device |
|  | client |
|  | server/cluster |
|  | task/job |
![](http://images2.imagebam.com/16/36/b1/efc4fa1024807754.png)

To run a graph across multiple servers, you first need to define a cluster.
A **cluster** is composed of one or more TensorFlow servers, called **tasks**, typically spread across several machines.

A **job** is just a named group of tasks that typically have a common role.

 - parameter server
 - worker

To start a TensorFlow server, you must create a Server object.

If you have several servers on one machine, you will need to ensure that they don’t all try to grab all the RAM of every GPU.
If you want the process to do nothing other than run the TensorFlow server, you can block the main thread by telling it to wait for the server to finish using the join() method.

#### Opening a Session

You can open a session on any of the servers, from a client located in any process on any machine.

#### The Master and Worker Services

The client uses the gRPC (Google Remote Procedure Call) protocol to communicate with the server.
Data is transmitted in the form of protocol buffers.

Every TensorFlow server provides two services:

 - the master service: **coordinates** the computations across tasks
 - the worker service: actually **execute** computations on tasks and get their results

#### Pinning Operations Across Tasks

You can use **device blocks** to pin operations on any device managed by any task.

Priority: device > task (or job) > session

#### Sharding Variables Across Multiple Parameter Servers

To reduce the risk of saturating a single parameter server’s network card.

TensorFlow provides the replica_device_setter() function to distribute variables across all the "ps" tasks in a round-robin fashion.

An inner device block can override the job, task, or device defined in an outer block.

#### Sharing State Across Sessions Using Resource Containers

 - local session
 - **distributed sessions**: variable state is managed by **resource containers** located on the cluster itself (not by the sessions).

If you want to run completely independent computations on the same cluster, use a **container block**. Advantages are variable names remain nice and short, and you can easily reset a named container.

Resource containers also take care of preserving the state of queues and readers. 

#### Asynchronous Communication Using TensorFlow Queues

To exchange data between multiple sessions.

 - FIFO queue
 - RandomShuffleQueue
 - PaddingFifoQueue

#### Loading Data Directly from the Graph

**Preload the data into a variable**
For datasets that can fit in memory, load the training data once and assign it to a variable, then just use that variable in your graph.

**Reading the training data directly from the graph**
If the training set does not fit in memory, a good solution is to use **reader operations**.

**Multithreaded readers using a Coordinator and a QueueRunner**
the Coordinator class and the QueueRunner class

**Other convenience functions**

### Parallelizing Neural Networks on a TensorFlow Cluster

#### One Neural Network per Device

**The speedup is almost linear.**

This solution is **perfect for hyperparameter tuning**: each device in the cluster will train a different model with its own set of hyperparameters.

It also works perfectly if you host a web service that receives a large number of queries per second (QPS) and you need your neural network to **make a prediction for each query**.
[TensorFlow Serving: for model deployment in production](https://www.tensorflow.org/serving/)

#### In-Graph Versus Between-Graph Replication

 - **in-graph replication**: create one big graph, containing every neural network
	 - simpler to implement since you don’t have to manage multiple clients and multiple queues
 - **between-graph replication**: create one **separate graph for each neural network** and handle synchronization between these graphs yourself
	 - easier to organize into well-bounded and easy-to-test modules. Moreover, it gives you more flexibility.
	 - one typical implementation is to **coordinate the execution of these graphs using queues**

#### Model Parallelism

**Run a single neural network across multiple devices.** This requires chopping your model into separate chunks and running each chunk on a different device.

Model parallelism can speed up running or training some types of neural networks(CNN and RNN), but not all(such as fully connected networks), **it really depends on the architecture of your neural network**. And it requires special care and tuning.

 - vertical split for CNN: contain layers that are only **partially** connected to the lower layers
 - horizontal split for RNN: **placing each layer on a different device**, active one device for each step, and by the time the signal propagates to the output layer all devices will be active simultaneously. The benefit of running multiple cells in parallel often outweighs the communication penalty.

#### Data Parallelism

Replicate the neural network on each device, run a training step simultaneously on all replicas **using a different mini-batch for each, and then aggregate the gradients** to update the model parameters.

**Synchronous updates**
The aggregator waits for **all** gradients to be available before computing the average and applying the result.

The **downside** is that some devices may be slower than others. To reduce the waiting time at each step, you could **ignore the gradients from the slowest few replicas** (typically ~10% spare replicas).

**Asynchronous updates**
Whenever a replica has finished computing the gradients, it **immediately** uses them to update the model parameters. There is no aggregation and synchronization.

**Stale gradients** can slow down convergence, introducing noise and wobble effects, or they can even make the training algorithm diverge.
Ways to reduce the effect of stale gradients:

 - Reduce the learning rate
 - Drop stale gradients or scale them down
 - Adjust the mini-batch size.
 - Start the first few epochs using just one replica (this is called the warmup phase), since stale gradients tend to be more damaging at the beginning of training.

[REVISITING DISTRIBUTED SYNCHRONOUS SGD](https://arxiv.org/pdf/1604.00981v2.pdf) found that **data parallelism with synchronous updates using a few spare replicas was the most efficient**.

**Bandwidth saturation**

## CNN

### The Architecture of the Visual Cortex

### Convolutional Layer

#### Filters(convolution kernels)

#### Stacking Multiple Feature Maps

Within one feature map, all neurons share the same parameters (weights and bias term), but different feature maps may have different parameters.

### Pooling Layer

Their goal is to **subsample** the input image in order to reduce the computational load, the memory usage, and the number of parameters(thereby limiting the risk of overfitting). It also makes the neural network tolerate a little bit of image shift (location invariance).

A pooling neuron has no weights, all it does is aggregate the inputs using an aggregation function such as the max or mean.

Only the max input value in each pooling kernel makes it to the next layer. The other inputs are dropped.

A pooling layer typically works on every input channel independently, so the output depth is the same as the input depth.

### CNN Architectures

 - LeNet-5 (1998)
 - AlexNet (2012)
 - GoogLeNet (2014)
 - ResNet (2015)

## RNN

### Recurrent Neurons

### Training RNNs

### Deep RNNs

### LSTM Cell

### GRU Cell

### NPL

## Autoencoders

## RL

# Appendix

## Machine Learning Project Checklist
> [pdf](http://www.ic.unicamp.br/~sandra/pdf/Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow-427-432.pdf)

## Other Popular ANN Architectures

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

pandas methods:

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

#### Feature Scaling

### Select a model and train it

#### Better Evaluation Using Cross-Validation

### Fine-tune your model

#### Grid Search

#### Randomized Search

#### Ensemble Methods

### Launch, monitor, and maintain your system

## Classification

### Training a Binary Classifier

### Performance Measures

#### Measuring Accuracy Using Cross-Validation

#### Confusion Matrix

#### Precision and Recall

#### Precision/Recall Tradeoff

#### The ROC Curve

### Multiclass Classification

### Error Analysis

### Multilabel Classification

### Multioutput Classification

## Training Models
> [safari book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/ch04.html)

### Linear Regression

### Gradient Descent

#### Batch Gradient Descent

#### Stochastic Gradient Descent

#### Mini-batch Gradient Descent

### Polynomial Regression

### Learning Curves

### Regularized Linear Models

### Logistic Regression

## SVM

## Decision Trees

## Ensemble Learning and Random Forests

## Dimensionality Reduction

### Main Approaches for Dimensionality Reduction

### PCA (Principal Component Analysis)

### Kernel PCA

### LLE (Locally Linear Embedding)

### Other Dimensionality Reduction Techniques

# Neural Networks and Deep Learning

## Up and Running with TensorFlow

## Introduction to Artificial Neural Networks

## Training Deep Neural Nets

### Vanishing/Exploding Gradients Problems

### Reusing Pretrained Layers

### Faster Optimizers

### Avoiding Overfitting Through Regularization

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

### Pooling Layer

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

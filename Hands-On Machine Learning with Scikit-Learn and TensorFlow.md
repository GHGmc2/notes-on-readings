# Hands-On Machine Learning with Scikit-Learn and TensorFlow
> [主页](http://shop.oreilly.com/product/0636920052289.do), [勘误](https://www.oreilly.com/catalog/errata.csp?isbn=0636920052289)
> [Github](https://github.com/ageron/handson-ml), [Jupyter Viewer](https://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/index.ipynb), [中文注释](https://github.com/DeqianBai/Hands-on-Machine-Learning)
> [douban](https://book.douban.com/subject/26840215/)
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
> [jupyter notebook](https://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/04_training_linear_models.ipynb)

### Linear Regression

Linear Regression model prediction (vectorized form):
$$
\hat y=h_{\theta}(\bm x)=\theta^T \cdot \bm x
$$
where $\theta$ is the model’s parameter vector, $\bm x$ is the instance’s feature vector, and $h_θ$ is the hypothesis function, using the model parameters $θ$.

**MSE cost function** for a Linear Regression model:
$$
MSE(\bm X, h_{\theta})=\frac{1}{m}\sum_{i=1}^m(\theta^T \cdot \bm x^{(i)} - y^{(i)})^2
$$


#### The normal equation

A closed-form solution:
$$
\hat \theta=(X^T\cdot X)^{-1}\cdot X^T\cdot \bm y
$$
where $\hat \theta$ is the value of $\theta$ that minimizes the cost function, $\bm y$ is the vector of target values containing $y^{(1)}$ to $y^{(m)}$.

#### Computational Complexity

The Normal Equation gets very **slow** when the number of **features** grows large.

The equation is **linear** with regards to the number of **instances** in the training set, so it handles large training sets efficiently, provided they can fit in memory.

### Gradient Descent

#### Batch Gradient Descent

Computing the gradients based on the full training set.

Gradient vector of the cost function:
$$
\nabla_{\theta}MSE(\theta)=\frac{2}{m}\bm X^T \cdot(\bm X \cdot \theta- \bm y)
$$

Gradient Descent step:
$$
\theta^{(next\space step)}=\theta -\eta \nabla_{\theta}MSE(\theta)
$$

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

For a linear model, regularization is typically achieved by constraining the weights ($\theta$) of the model.

The regularization term should only be added to the cost function **during training**. Once the model is trained, you want to evaluate the model’s performance using the unregularized performance measure.

#### Ridge Regression（岭回归）

regularization term: $\alpha\frac{1}{2} \sum_{i=1}^{n}\theta_i^2$

#### Lasso Regression（套索回归）

regularization term: $\alpha \sum_{i=1}^n|\theta_i|$

#### Elastic Net

Elastic Net is a middle ground between Ridge Regression and Lasso Regression(you can control the **mix ratio $r$**).
$$
J(\theta)=MSE(\theta) + r\alpha \sum_{i=1}^n |\theta_i| + \frac{1-r}{2}\alpha
\sum_{i=1}^n\theta_i^2$$

#### Early Stopping

Stop training as soon as the validation error reaches a minimum. (stop only after the validation error has been above the minimum for some time, then roll back the model parameters to the point where the validation error was at a minimum)

### Logistic Regression

Logistic Regression is commonly used to estimate the **probability** that an instance belongs to a particular class.

#### Estimating Probabilities

Logistic Regression model estimated probability: $\hat p = \sigma(\theta^T \cdot x)$, where $\sigma(t) = \frac{1}{1+exp(-t)}$.

#### Training and Cost Function

Logistic Regression **cost function** (log loss):
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^m[y^{(i)}\log (\hat p^{(i)}) + (1-y^{(i)})\log (1-\hat p^{(i)})]
$$

Logistic cost function **partial derivatives**:
$$
\frac{\partial}{\partial \theta_j}J(\theta)=\frac{1}{m_i}\sum_{i=1}^m(\sigma(\theta^T \cdot \bm x^{(i)})-y^{(i)})x_j^{(i)}
$$

#### Decision boundaries

#### Softmax Regression

The Logistic Regression model can be generalized to **support multiple classes directly**, without having to train and combine multiple binary classifiers. This is called Softmax Regression, or Multinomial Logistic Regression.

When given an instance $x$, the Softmax Regression model first computes a score $s_k(x)$ for each class $k$, then estimates the probability of each class by applying the softmax function (also called the normalized exponential) to the scores.

Softmax score for class $k$: 
$$
s_k(\bm x)=\theta_k^T \cdot \bm x
$$

Softmax **function**:
$$
\hat p_k = \frac{exp(s_k(x))}{\sum_{j=1}^{K}exp(s_j(x))}
$$
where $K$ is the number of classes.

The Softmax Regression classifier **predicts only one class at a time**.

**Cross entropy** is frequently used to measure how well a set of estimated class probabilities match the target classes.

Cross entropy **cost function** ($\Theta$ is the parameter matrix): 
$$
J(\Theta) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} \log (\hat p_k^{(i)})
$$

Cross entropy **gradient vector** for class k:
$$
\nabla_{\theta_k}J(\Theta)=\frac{1}{m}\sum_{i=1}^m(\hat p_k^{(i)}-y_k^{(i)})\bm x^{(i)}
$$

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
> the most popular dimensionality reduction algorithm.

First it identifies the hyperplane that lies closest to the data, and then it projects the data onto it.

#### Preserving the Variance

 - select the axis that preserves the maximum amount of variance, as it will most likely lose less information than the other projections.
 - select the axis that minimizes the mean squared distance between the original dataset and its projection onto that axis.

#### Principal Components

PCA identifies the axis that accounts for the largest amount of variance in the train‐ ing set.

The **unit vector that defines the i^th^ axis** is called the i^th^ principal component (PC).

The direction of the principal components is not stable. However, PCs will generally still lie on the same axes.

Singular Value Decomposition (SVD) can decompose the training set matrix $X$ into the dot product of three matrices $U \cdot \sum \cdot V^T$, where $V^T$ contains all the principal components that we are looking for.

#### Projecting Down to d Dimensions

Projecting the training set down to d dimensions: $X_{d-proj}=X \cdot W_d$, where  $X_d$ is the matrix containing the first $d$ principal components (i.e., the matrix composed of the first $d$ columns of $V^T$).

#### PCA for Compression

PCA inverse transformation, back to the original number of dimensions: $X_{recovered} = X_{d-proj} \cdot W_d^T$

#### Incremental PCA

The preceding implementation of PCA requires the whole training set to fit in memory in order for the SVD algorithm to run.

**Incremental PCA (IPCA)** algorithms have been developed: you can split the training set into mini-batches and feed an IPCA algorithm one **mini-batch** at a time. This is useful for large training sets, and also to apply PCA online.

#### Randomized PCA

It is a stochastic algorithm that quickly finds an approximation of the first $d$ principal components. It is dramatically faster than the previous algorithms when $d$ is much smaller than $n$.

### Kernel PCA (kPCA)

It is often good at preserving clusters of instances after projection, or sometimes even unrolling datasets that lie close to a twisted manifold.

#### Selecting a Kernel and Tuning Hyperparameters

kPCA is an unsupervised learning algorithm, there is no obvious performance measure.

Since Dimensionality reduction is often a preparation step for a **supervised** learning task, so you can simply use **grid search** to select the kernel and hyper‐parameters that lead to the best performance on that task.
Another approach (entirely **unsupervised**), is to select the kernel and hyper‐parameters that yield the **lowest reconstruction error**.


### LLE (Locally Linear Embedding)

First measuring how each training instance linearly relates to its closest neighbors, and then **looking for a low-dimensional representation of the training set where these local relationships are best preserved**.

### Other Dimensionality Reduction Techniques

 - Multidimensional Scaling (MDS)
 - Isomap
 - t-Distributed Stochastic Neighbor Embedding (t-SNE)
 - Linear Discriminant Analysis (LDA): during training it learns the most discriminative axes between the classes, and these axes can then be used to define a hyperplane onto which to project the data. The **benefit** is that the projection will keep classes as far apart as possible, so LDA is a good technique to **reduce dimensionality before running another classification algorithm** such as an SVM classifier.

# Neural Networks and Deep Learning

## Up and Running with TensorFlow
> [jupyter notebook](https://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/09_up_and_running_with_tensorflow.ipynb)

### Lifecycle of a Node Value

All node values are dropped between graph runs, except variable values, which are maintained by the session across graph runs.

In single-process TensorFlow, multiple sessions do not share any state, even if they reuse the same graph (each session would have its own copy of every variable).
In distributed TensorFlow, variable state is stored on the servers, not in the sessions, so multiple sessions can share the same variables.

### Implementing Gradient Descent

TensorFlow uses **reverse-mode autodiff**, which is perfect (efficient and accurate) when there are many inputs and few outputs, as is often the case in neural networks.

## Introduction to Artificial Neural Networks

### From Biological to Artificial Neurons

#### Multi-Layer Perceptron and Backpropagation

For each training instance the **BP algorithm** first makes a prediction (forward pass), measures the error, then goes through each layer in reverse to measure the error contribution from each connection (reverse pass), and finally slightly tweaks the connection weights to reduce the error (Gradient Descent step).

activation functions:

 - logistic function: $\sigma(z) = 1 / (1 + exp(–z))$
 - hyperbolic tangent function: $tanh(z) = 2\sigma(2z)–1$
 - ReLU function: $ReLU(z) = max(0, z)$

## Training Deep Neural Nets

### Vanishing/Exploding Gradients Problems

[Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
We need the **variance** of the **outputs** of each layer to be equal to the variance of its **inputs**, and we also need the **gradients** to have **equal variance** before and after flowing through a layer in the reverse direction.

#### Xavier and He Initialization

Normal distribution with mean 0 and standard deviation $\sigma = \sqrt \frac{2}{n_{inputs} + n_{outputs}}$
Or a uniform distribution between $‐r$ and $+r$, with $r = \sqrt \frac{6}{n_{inputs} + n_{outputs}}$

#### Nonsaturating Activation Functions

Leaky ReLU = $max(\alpha z, z)$ ($\alpha$ typically set to 0.01)

 - randomized leaky ReLU (RReLU)
 - parametric leaky ReLU (PReLU)

**ELU(exponential linear unit)** activation function:
$$
ELU_{ \alpha}(z)=
\begin{cases}
 \alpha(e^z-1), &\text{z<0} \\
 z, &\text{z>=0}
\end{cases}
$$

 - it takes on negative values when $z < 0$, which allows the unit to have an average output closer to 0. This helps alleviate the vanishing gradients problem. The hyperparameter $\alpha$ is usually set to 1.
 - it has a nonzero gradient for $z < 0$, which avoids the dying units issue.
 - the function is smooth everywhere, which helps speed up Gradient Descent

The main **drawback** of the ELU activation function is that it is slower to compute than the ReLU and its variants (due to the use of the exponential function), but during training this is compensated by **the faster convergence rate**.

Which activation function should you use for the hidden layers of your deep neural networks?
In general, ELU > leaky ReLU (and its variants) > ReLU > tanh > logistic.

 - If you care a lot about runtime performance, then you may prefer leaky ReLUs over ELUs.
 - RReLU if your network is overfitting
 - PReLU if you have a huge training set

#### Batch Normalization
> [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

The **Internal Covariate Shift** problem: the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change.

BN lets the model **learn the optimal scale and mean of the inputs for each layer**:

 - adding an operation in the model just **before the activation function of each layer**, simply **zero-centering and normalizing the inputs**
 - **scaling and shifting the result** using two new parameters per layer (one for scaling, the other for shifting).

Batch Normalization algorithm: TODO.

**Benefits**:

 - strongly **reducing the vanishing gradients problem**
 - the networks were also **much less sensitive to the weight initialization**. They were able to use much larger learning rates, significantly speeding up the learning process.
 - acts like a regularizer, **reducing the need for other regularization techniques** such as dropout.

Limits:

 - add some complexity to the model.
 - the neural network makes slower **predictions** due to the extra computations required at each layer.

#### Gradient Clipping

A popular technique to **lessen the exploding gradients problem** is to simply clip the gradients during backpropagation so that they **never exceed some threshold** (this is mostly useful for **RNN**).

### Reusing Pretrained Layers

**Transfer learning** will work only well if the inputs have similar low-level features.

#### Freezing the Lower Layers

#### Caching the Frozen Layers

#### Tweaking, Dropping, or Replacing the Upper Layers

#### Model Zoos

#### Unsupervised Pretraining

**Train the layers one by one**, starting with the lowest layer and then going up, using an **unsupervised feature detector algorithm** such as Restricted Boltzmann Machines (RBMs) or autoencoders. Each layer is trained on the output of the previously trained layers.
Once all layers have been trained this way, you can **fine-tune** the network using **supervised** learning.

#### Pretraining on an Auxiliary Task

max margin learning

### Faster Optimizers

You should almost always use Adam optimization.

#### Momentum optimization

Momentum optimization cares a great deal about what previous gradients were.

#### Adam(adaptive moment estimation) Optimization

It keeps track of an **exponentially decaying average of past gradients** and an **exponentially decaying average of past squared gradients**.

#### Learning Rate Scheduling

 - Performance scheduling
 - Exponential scheduling

### Avoiding Overfitting Through Regularization

#### Early Stopping

#### l~1~ and l~2~ Regularization

#### Dropout

At every training step, every neuron (including the input neurons but excluding the output neurons) has a probability $p$ (typically set to 50%) of being **temporarily** “dropped out”, but it may be active during the next step.

Neurons end up being less sensitive to slight changes in the inputs. In the end you get a more robust network that generalizes better.
A unique neural network is generated at each training step. The resulting neural network can be seen as an averaging **ensemble** of all these smaller neural networks.

We need to multiply each input connection weight by the keep probability $(1 – p)$ after training.

Dropout does tend to significantly slow down convergence, but it usually results in a much better model when tuned properly.

#### Max-Norm Regularization

For each neuron, it constrains the weights $w$ of the incoming connections.

#### Data Augmentation

### Practical Guidelines

## Distributing TensorFlow Across Devices and Servers
> [jupyter nootbook](https://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/12_distributed_tensorflow.ipynb)

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

When TensorFlow runs a graph, it starts by finding out the list of nodes that need to be evaluated, and it counts how many dependencies each of them has. Then it **starts evaluating the nodes with zero dependencies**.

TensorFlow **manages a thread pool on each device** to parallelize operations. These are called the **inter-op thread pools**.
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

The **speedup is almost linear.**

This solution is **perfect for hyperparameter tuning**: each device in the cluster will train a different model with its own set of hyperparameters.

It also works **perfectly if you host a web service** that receives a large number of queries per second (QPS) and you need your neural network to **make a prediction for each query**.
[TensorFlow Serving: for model deployment in production](https://www.tensorflow.org/serving/)

#### In-Graph Versus Between-Graph Replication

Two major approaches to handling a neural network **ensemble** (produce the ensemble’s prediction):

 - **in-graph replication**: create one big graph, containing every neural network
	 - just create one session
	 - simpler to implement since you don’t have to manage multiple clients and multiple queues
 - **between-graph replication**: create one **separate graph for each neural network** and handle synchronization between these graphs yourself
	 - easier to organize into well-bounded and easy-to-test modules. Moreover, it gives you more flexibility.
	 - one typical implementation is to **coordinate the execution of these graphs using queues**
	 - a set of clients handles one neural network each + one last client is in charge of reading one prediction from each prediction queue and aggregating them to produce the ensemble’s prediction

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

For models relatively small and trained on a very large training set, you are often better off training the model on a single machine with a single GPU.

Saturation is more severe for large dense models.

A few simple steps to reduce the saturation problem:

 - **Group your GPUs** on a few servers rather than scattering them across many servers.
 - **Shard the parameters** across multiple parameter servers
 - **Drop the model parameters’ float precision**. This will cut the amount of data to transfer, without much impact on the convergence rate or the model’s performance.

## CNN

### The Architecture of the Visual Cortex

### Convolutional Layer

#### Filters (convolution kernels)

#### Stacking Multiple Feature Maps

**Within one feature map, all neurons share the same parameters** (weights and bias term), but different feature maps may have different parameters.

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

time step(frame)

unrolling the network through time

Many researchers prefer to use the hyperbolic tangent (tanh) activation function in RNNs rather than the ReLU activation function.

#### Memory Cells

#### Input and Output Sequences

### Deep RNNs

encoder: a sequence-to-vector network
decoder: a vector-to-sequence network

#### The Difficulty of Training over Many Time Steps

### LSTM Cell

### GRU Cell

### NPL

## Autoencoders

## RL

# Appendix

## Machine Learning Project Checklist
> [pdf](http://www.ic.unicamp.br/~sandra/pdf/Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow-427-432.pdf)

## Other Popular ANN Architectures

# Libs

## NumPy
> The best way to learn is to experiment with NumPy, and go through the excellent [reference documentation](https://docs.scipy.org/doc/numpy/reference/index.html).
> [practice](https://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/tools_numpy.ipynb)

Some vocabulary

 - Each dimension is called an **axis**.
 - The number of axes is called the **rank**.
 - An array's list of axis lengths is called the **shape** of the array.
 - The **size** of an array is the total number of elements, which is the product of all axis lengths.

NumPy arrays have the type ndarrays.

### Creating arrays

`np.zeros` and `np.ones`:
```py
np.zeros((3,4))

np.ones((3,4))
```

`np.full` and `np.empty`:
```py
np.full((3,4), np.pi)

np.empty((2,3)) # its content is not predictable, as it is whatever is in memory at that point
```

`np.array`:
```py
np.array([[1,2,3,4], [10, 20, 30, 40]])
```

`np.arange`:
```py
np.arange(1, 5, 0.5)
```

**`np.linspace`**:
It is generally preferable to use the *linspace* function instead of arange when working with floats (the maximum value is included, contrary to arange).
```py
np.linspace(0, 5/3, 6)
```

**`np.rand`** and **`np.randn`**:
```py
# random floats between 0 and 1 (uniform distribution)
np.random.rand(3,4)
# random floats sampled from a univariate normal distribution of mean 0 and variance 1
np.random.randn(3,4)
```

`np.fromfunction`:  the function my_function is only called once, instead of once per element.
```py
def my_function(z, y, x):
    return x * y + z

np.fromfunction(my_function, (3, 2, 10))
```

### Array data

**`dtype`**
> Available data types include int8, int16, int32, int64, uint8|16|32|64, float16|32|64 and complex64|128. Check out [the documentation](http://docs.scipy.org/doc/numpy-1.10.1/user/basics.types.html) for the full list.

```py
c = np.arange(1, 5)
print(c.dtype, c)
```

**`itemsize`**: returns the size (in bytes) of each item.
```py
e = np.arange(1, 5, dtype=np.complex64)
e.itemsize
```

`data` buffer
An array's data is actually stored in memory as a flat (one dimensional) byte buffer. It is available via the data attribute.
```py
f = np.array([[1,2],[1000, 2000]], dtype=np.int32)
f.data
```

### Reshaping an array

In place: just setting its `shape` attribute. **The array's size must remain the same**.
```py
g = np.arange(24)
g.shape = (6, 4)
```

`reshape`: returns a new ndarray object pointing at the same data. This means that **modifying one array will also modify the other**.
```py
g2 = g.reshape(4,6)
```

`ravel`: returns a new one-dimensional ndarray that **also points to the same data**.
```py
g.ravel()
```

### Arithmetic operations

All the usual arithmetic operators (+, -, *, /, //, **, etc.) can be used with ndarrays. They apply **elementwise**.
The multiplication(`*`) is not a matrix multiplication.

The arrays must have the same shape. If they do not, NumPy will apply the broadcasting rules.

### Broadcasting

**First rule**
If the arrays do not have the same **rank**, then a **1 will be prepended to the smaller ranking** arrays until their ranks match.
```py
h = np.arange(5).reshape(1, 1, 5)
h + [10, 20, 30, 40, 50]  # same as: h + [[[10, 20, 30, 40, 50]]]
```

**Second rule**
Arrays with a 1 along a particular dimension act as if they had the size of the array with the largest shape along that dimension. The value of the array element is repeated along that dimension.
```py
k = np.arange(6).reshape(2, 3)
k + [[100], [200]]  # same as: k + [[100, 100, 100], [200, 200, 200]]

k + [100, 200, 300]  # after rule 1: [[100, 200, 300]], and after rule 2: [[100, 200, 300], [100, 200, 300]]
k + 1000  # same as: k + [[1000, 1000, 1000], [1000, 1000, 1000]]
```

**Third rule**
After rules 1 & 2, the sizes of all arrays must match.

**Upcasting**
When trying to combine arrays with different dtypes, NumPy will upcast to a type capable of handling all possible values (regardless of what the actual values are).

### Conditional operators

The conditional operators also apply **elementwise**:
```py
m = np.array([20, -5, 30, 40])
m < [15, 16, 35, 36]
# This is most useful in conjunction with boolean indexing
m[m < 25] # array([20, -5])
```

### Mathematical and statistical functions

**`ndarray` methods**
```py
for func in (a.mean, a.min, a.max, a.sum, a.prod, a.std, a.var):
    print(func.__name__, "=", func())
```
These functions accept an optional argument axis:
```py
c=np.arange(24).reshape(2,3,4)
c.sum(axis=(0,2))  # sum across matrices and columns
```

**Universal functions**
NumPy provides fast **elementwise functions** called universal functions, or **ufunc**.
```py
a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
for func in (np.square, np.abs, np.sqrt, np.exp, np.log, np.sign, np.ceil, np.modf, np.isnan, np.cos):
    print(func(a))
```

**Binary ufuncs**
Apply elementwise on two ndarrays. Broadcasting rules are applied if the arrays do not have the same shape.
```py
a = np.array([1, -2, 3, 4])
b = np.array([2, 8, -1, 7])

np.add(a, b)  # equivalent to a + b
np.greater(a, b)  # equivalent to a > b, array([False, False,  True, False], dtype=bool)
np.maximum(a, b) # array([2, 8, 3, 7])
np.copysign(a, b) # array([ 1.,  2., -3.,  4.])
```

### Array indexing

One-dimensional NumPy arrays can be accessed more or less like regular python arrays.

**Differences with regular python arrays**
If you assign a single value to an ndarray slice, it is **copied** across the whole slice, thanks to broadcasting rules.
```py
a = np.array([1, 5, 3, 19, 13, 7, 3])
a[2:5] = -1 # array([ 1,  5, -1, -1, -1,  7,  3])
```

`ndarray` slices are actually **views** on the same data buffer. This means that if you create a slice and modify it, you are **actually going to modify the original ndarray** as well!
```py
a_slice = a[2:6]
a_slice[1] = 1000 # array([   1,    5,   -1, 1000,   -1,    7,    3])
```
If you want a copy of the data, you need to use the copy method:
```py
another_slice = a[2:6].copy()
another_slice[1] = 3000
```

**Multi-dimensional arrays**
```py
b = np.arange(48).reshape(4, 12)
"""
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
       [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]])
"""

b[1, :] # returns row 1 as a 1D array of shape (12,): array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
b[1:2, :] # returns that same row as a 2D array of shape (1, 12): array([[12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])
```

**Fancy indexing**
If you provide multiple index arrays, you get a 1D ndarray containing the values of the elements at the specified coordinates.
```py
b[(-1, 2, -1, 2), (5, 9, 1, 9)]  # returns a 1D array with b[-1, 5], b[2, 9], b[-1, 1] and b[2, 9]: array([41, 33, 37, 33])
```

**Higher dimensions**
```py
c = b.reshape(4,2,6)
"""
array([[[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]],

       [[12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23]],

       [[24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35]],

       [[36, 37, 38, 39, 40, 41],
        [42, 43, 44, 45, 46, 47]]])
"""
c[2, :, 3]  # matrix 2, all rows, col 3: array([27, 33])
```

**Ellipsis (...)**
You may also write an ellipsis (...) to ask that all non-specified axes be entirely included.
```py
c[2, 1, ...]  # matrix 2, row 1, all columns.  This is equivalent to c[2, 1, :], array([30, 31, 32, 33, 34, 35])
```

**Boolean indexing**
Provide an ndarray of boolean values on **one axis** to specify the indices that you want to access.
```py
b = np.arange(48).reshape(4, 12)]

rows_on = np.array([True, False, True, False])
b[rows_on, :]  # Rows 0 and 2, all columns. Equivalent to b[(0, 2), :]

cols_on = np.array([False, True, False] * 4)
b[:, cols_on]  # All rows, columns 1, 4, 7 and 10 ?
```

**np.ix_**
You cannot use boolean indexing this way on **multiple axes**, but you can work around this by using the ix_ function:
```py
b[np.ix_(rows_on, cols_on)]

"""
array([[ 1,  4,  7, 10],
       [25, 28, 31, 34]])
"""
```

### Iterating

Iterating over multidimensional arrays is done with respect to the first axis.
```py
c = np.arange(24).reshape(2, 3, 4)  # A 3D array (composed of two 3x4 matrices)
for m in c:
    print("Item:")
    print(m)
"""
Item:
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
Item:
[[12 13 14 15]
 [16 17 18 19]
 [20 21 22 23]]
"""
```

If you want to iterate on **all elements** in the ndarray, simply iterate over the flat attribute:
```py
for i in c.flat:
    print("Item:", i)
```

### Stacking arrays

**`vstack`**: stack vertically
```py
q1 = np.full((3,4), 1.0)
q2 = np.full((4,4), 2.0)
q3 = np.full((3,4), 3.0)

q4 = np.vstack((q1, q2, q3))
```

**`hstack`**: stack horizontally
```py
q5 = np.hstack((q1, q3))
```

**`concatenate`**: stacks arrays along any given existing axis
```py
q7 = np.concatenate((q1, q2, q3), axis=0)  # Equivalent to vstack
```

**`stack`**: stacks arrays along a new axis. All arrays have to have the same shape.
```py
q8 = np.stack((q1, q3))
q8.shape # (2, 3, 4)
```

### Splitting arrays

Splitting is the opposite of stacking. There is also a split function which splits an array along any given axis.

```py
r = np.arange(24).reshape(6,4)
r1, r2, r3 = np.vsplit(r, 3)
r4, r5 = np.hsplit(r, 2)
```

### Transposing arrays

The `transpose` method creates a new view on an ndarray's data, with axes permuted in the given order.
```py
t = np.arange(24).reshape(4,2,3)
"""
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]],

       [[12, 13, 14],
        [15, 16, 17]],

       [[18, 19, 20],
        [21, 22, 23]]])
"""
t1 = t.transpose((1,2,0)) # the axes 0, 1, 2 (depth, height, width) are re-ordered to 1, 2, 0 (depth→width, height→depth, width→height)
"""
array([[[ 0,  6, 12, 18],
        [ 1,  7, 13, 19],
        [ 2,  8, 14, 20]],

       [[ 3,  9, 15, 21],
        [ 4, 10, 16, 22],
        [ 5, 11, 17, 23]]])
"""

# By default, `transpose` reverses the order of the dimensions.
t2 = t.transpose()  # equivalent to t.transpose((2, 1, 0))
t2.shape # (3, 2, 4)

# NumPy provides a convenience function `swapaxes` to swap two axes.
t3 = t.swapaxes(0,1)  # equivalent to t.transpose((1, 0, 2))
```

### Linear algebra

**Matrix transpose**
```py
m1 = np.arange(10).reshape(2,5)
m1.T
```
The T attribute has no effect on rank 0 (empty) or rank 1 arrays, we can get the desired transposition by first reshaping the 1D array to a single-row matrix (2D):
```py
m2 = np.arange(5)
m2r = m2.reshape(1,5)
m2r.T
```

**Matrix dot product**
```py
n1 = np.arange(10).reshape(2, 5)
n2 = np.arange(15).reshape(5, 3)
n1.dot(n2) # n1*n2 is not a dot product, it is an elementwise product.
```

**Matrix inverse**
Many of the linear algebra functions are available in the **`numpy.linalg` module**.
```py
import numpy.linalg as linalg

m3 = np.array([[1,2,3],[5,7,11],[21,29,31]])
linalg.inv(m3)
```

**Identity matrix**
You can create an identity matrix of size NxN by calling eye:
```py
np.eye(3)
"""
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
"""
```

**[QR decomposition](https://zh.wikipedia.org/wiki/QR%E5%88%86%E8%A7%A3)**
```py
q, r = linalg.qr(m3)
q.dot(r)  # q.r equals m3
```

**Determinant**
```py
linalg.det(m3)  # Computes the det(m3) or |m3|
```

**[Eigenvalues and eigenvectors](https://zh.wikipedia.org/wiki/%E7%89%B9%E5%BE%81%E5%80%BC%E5%92%8C%E7%89%B9%E5%BE%81%E5%90%91%E9%87%8F)**
```py
eigenvalues, eigenvectors = linalg.eig(m3)
m3.dot(eigenvectors) - eigenvalues * eigenvectors  # m3.v = λ*v
```

**[Singular Value Decomposition](https://zh.wikipedia.org/wiki/%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3)**
```py
m4 = np.array([[1,0,0,0,2], [0,0,3,0,0], [0,0,0,0,0], [0,2,0,0,0]])
U, S_diag, V = linalg.svd(m4)
S = np.zeros((4, 5))
S[np.diag_indices(4)] = S_diag
U.dot(S).dot(V) # U.Σ.V == m4
```

**Diagonal and trace**
```py
np.diag(m3) # top left to bottom right
np.trace(m3)  # equivalent to np.diag(m3).sum()
```

**Solving a system of linear scalar equations**
The  `solve`  function solves a system of linear scalar equations, such as:

-   2𝑥+6𝑦=6
-   5𝑥+3𝑦=−9

```py
coeffs  = np.array([[2, 6], [5, 3]])
depvars = np.array([6, -9])
solution = linalg.solve(coeffs, depvars)
```

### Vectorization

Your code is much more efficient if you try to stick to array operations. This is called vectorization. This way, you can benefit from NumPy's many optimizations.

NumPy's `meshgrid` function which generates coordinate matrices from coordinate vectors.
```py
x_coords = np.arange(0, 1024)  # [0, 1, 2, ..., 1023]
y_coords = np.arange(0, 768)   # [0, 1, 2, ..., 767]
X, Y = np.meshgrid(x_coords, y_coords) # both X and Y are 768x1024 arrays, and all values in X correspond to the horizontal coordinate, while all values in Y correspond to the the vertical coordinate.
data = np.sin(X*Y/40.5)
```

### Saving and loading

Save and load ndarrays in binary or text format.

**Binary `.npy` format**
```PY
a = np.random.rand(2,3)
# Since the file name contains no file extension was provided, NumPy automatically added .npy.
np.save("my_array", a)
a_loaded = np.load("my_array.npy")
```

**Text format**
```py
np.savetxt("my_array.csv", a, delimiter=",")
a_loaded = np.loadtxt("my_array.csv", delimiter=",")
```

**Zipped `.npz` format**
```py
b = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
np.savez("my_arrays", my_a=a, my_b=b)
my_arrays = np.load("my_arrays.npz")
```
This is a dict-like object which loads the arrays lazily:
```py
my_arrays.keys()
my_arrays["my_a"]
```

## Pandas
> Probably the best way to learn more is to **get your hands dirty** with some real-life data. It is also a good idea to go through pandas' excellent [documentation](http://pandas.pydata.org/pandas-docs/stable/index.html), in particular the [Cookbook](http://pandas.pydata.org/pandas-docs/stable/cookbook.html).
> [practice](https://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/tools_pandas.ipynb)

The pandas library provides high-performance, easy-to-use data structures and data analysis tools.
The main data structure is the `DataFrame`, which you can think of as an in-memory 2D table (like a spreadsheet, with column names and row labels).

The  `pandas`  library contains these useful data structures:

-   `Series`  objects. A  `Series`  object is 1D array, similar to a column in a spreadsheet (with a column name and row labels).
-   `DataFrame`  objects. This is a 2D table, similar to a spreadsheet (with column names and row labels).
-   `Panel`  objects. You can see a  `Panel`  as a dictionary of  `DataFrame`s. These are less used.


### Series objects

**Similar to a 1D ndarray**
Arithmetic operations on Series are also possible, and they apply **elementwise**:
```py
import pandas as pd

s = pd.Series([2,-1,3,5])
s + [1000,2000,3000,4000]
"""
0    1002
1    1999
2    3003
3    4005
dtype: int64
"""
```
If you add a single number to a Series, that number is added to **all items** in the Series(the same is true for all binary operations such as `*` or `/`, and even `conditional operations`). This is called **broadcasting**:
```py
s + 1000
"""
0    1002
1     999
2    1003
3    1005
dtype: int64
"""
```

**Index labels**
Each item in a Series object has a unique identifier called the index label.
```py
s2 = pd.Series([68, 83, 112, 68], index=["alice", "bob", "charles", "darwin"])
s2["bob"] # 83
s2[1] # 83
# It is recommended to always use the `loc` attribute when accessing by label, and the `iloc` attribute when accessing by integer location:
s2.loc["bob"]
s2.iloc[1]
```

Slicing a Series also slices the index labels:
```py
s2.iloc[1:3]
```

**Init from `dict`**
The keys will be used as index labels:
```py
weights = {"alice": 68, "bob": 83, "colin": 86, "darwin": 68}
s3 = pd.Series(weights)
s4 = pd.Series(weights, index = ["colin", "alice"])
```

**Automatic alignment**
When an operation involves multiple Series objects, pandas automatically aligns items by matching index labels.
Automatic alignment is very handy when working with data that may come from various sources with varying structure and missing items.
```py
s2 + s3
"""
Index(['alice', 'bob', 'charles', 'darwin'], dtype='object')
Index(['alice', 'bob', 'colin', 'darwin'], dtype='object')
alice      136.0
bob        166.0
charles      NaN
colin        NaN
darwin     136.0
dtype: float64
"""
```

**Init with a scalar**
All items will be set to the scalar:
```py
meaning = pd.Series(42, ["life", "universe", "everything"])
```

**`Series` name**
```py
s6 = pd.Series([83, 68], index=["bob", "alice"], name="weights")
```

**Plotting a `Series`**
Pandas makes it easy to plot Series data using matplotlib. Just import matplotlib and call the plot() method:
```py
%matplotlib inline
import matplotlib.pyplot as plt

temperatures = [4.4,5.1,6.1,6.2,6.1,6.1,5.7,5.2,4.7,4.1,3.9,3.5]
s7 = pd.Series(temperatures, name="Temperature")
s7.plot()
plt.show()
```
There are many options for plotting your data. ([Visualization](https://pandas.pydata.org/pandas-docs/stable/visualization.html))

### Handling time

**Time range**
```py
dates = pd.date_range('2016/10/29 5:30pm', periods=12, freq='H') 
# pd.date_range() returns a DatetimeIndex that may be used as an index in a Series:
temp_series = pd.Series(temperatures, dates)
```

**Resampling**
Just call the resample() method and specify a new frequency:
```py
temp_series_freq_2H = temp_series.resample("2H")
```
The resampling operation is actually **a deferred operation**, which is why we did not get a `Series` object, but a `DatetimeIndexResampler` object instead.
To actually perform the resampling operation, we can simply call the mean() method:
```py
temp_series_freq_2H = temp_series_freq_2H.mean()
```
any other aggregation function:
```py
temp_series_freq_2H = temp_series.resample("2H").min() # or temp_series_freq_2H = temp_series.resample("2H").apply(np.min)
```

**Upsampling and interpolation**
Call the `interpolate()` method to fill the gaps by interpolating. The default is to use linear interpolation, but we can also select another method, such as `cubic` interpolation:
```py
temp_series_freq_15min = temp_series.resample("15Min").interpolate(method="cubic")
temp_series_freq_15min.head(n=10) # `head` displays the top n values
```

**Timezones**
Make datetimes timezone aware by calling the tz_localize() method:
```py
temp_series_ny = temp_series.tz_localize("America/New_York")
```
We can convert these datetimes to Paris time:
```py
temp_series_paris = temp_series_ny.tz_convert("Europe/Paris")
```
Using the `ambiguous` argument we can tell pandas to infer the right DST (Daylight Saving Time) based on the order of the ambiguous timestamps:
```py
temp_series_paris_naive.tz_localize("Europe/Paris", ambiguous="infer")
```

**Periods**
The `pd.period_range()` function returns a `PeriodIndex` instead of a `DatetimeIndex`.
```py
quarters = pd.period_range('2016Q1', periods=8, freq='Q')
```
The asfreq() method lets us change the frequency of the PeriodIndex. All periods are lengthened or shortened accordingly.
```py
quarters.asfreq("M", how="start")
```
We can create a Series with a PeriodIndex:
```py
quarterly_revenue = pd.Series([300, 320, 290, 390, 320, 360, 310, 410], index = quarters)
```
We can convert periods to timestamps by calling `to_timestamp`:
```py
last_hours = quarterly_revenue.to_timestamp(how="end", freq="H")
```
And back to periods by calling to_period:
```py
last_hours.to_period()
```

### `DataFrame` objects

A `DataFrame` object represents a spreadsheet, with cell values, column names and row index labels. You can see `DataFrame`s as dictionaries of `Series`.

**Creating a `DataFrame`**
You can create a DataFrame by passing a dictionary of Series objects:
```py
people_dict = {
    "weight": pd.Series([68, 83, 112], index=["alice", "bob", "charles"]),
    "birthyear": pd.Series([1984, 1985, 1992], index=["bob", "alice", "charles"], name="year"),
    "children": pd.Series([0, 3], index=["charles", "bob"]),
    "hobby": pd.Series(["Biking", "Dancing"], index=["alice", "bob"]),
}
people = pd.DataFrame(people_dict)
```

You can access columns pretty much as you would expect. They are returned as `Series` objects:
```py
people[["birthyear", "hobby"]]
```

**Multi-indexing**
If all columns are tuples of the same size, then they are understood as a multi-index.

**Dropping a level**
We can drop a column or indices level by calling droplevel():
```py
d5.columns = d5.columns.droplevel(level = 0)
```

**Transposing**
Swap columns and indices using the T attribute:
```py
d6 = d5.T
```

**Stacking and unstacking levels**
Calling the stack() method will push the lowest column level after the lowest index:
```py
d7 = d6.stack()
d8 = d7.unstack() # unstack() does the reverse
```

The stack() and unstack() methods let you select the level to stack/unstack. You can even stack/unstack multiple levels at once:
```py
d10 = d9.unstack(level = (0,1))
```

**Most methods return modified copies**
The stack() and unstack() methods do not modify the object they apply to. Instead, they work on a copy and return that copy. This is true of **most** methods in pandas.

**Accessing rows**
```py
people[people["birthyear"] < 1990]
```

**Adding and removing columns**
You can generally treat DataFrame objects like dictionaries of Series:
```py
people

people["age"] = 2018 - people["birthyear"]  # adds a new column "age"
people["over 30"] = people["age"] > 30      # adds another column "over 30"
birthyears = people.pop("birthyear")
del people["children"]
```

When you add a new column (added at the end by default), it must have the same number of rows.

You can also insert a column anywhere else using the insert() method:
```py
people.insert(1, "height", [172, 181, 185])
```

**Assigning new columns**
The assign() method returns a new DataFrame object, the original is not modified:
```py
people.assign(
    body_mass_index = people["weight"] / (people["height"] / 100) ** 2,
    has_pets = people["pets"] > 0
)
```
You cannot access columns created within the same assignment, the solution is to split this assignment in two consecutive assignments:
```py
d6 = people.assign(body_mass_index = people["weight"] / (people["height"] / 100) ** 2)
d6.assign(overweight = d6["body_mass_index"] > 25)
```

You may want to just chain the assignment calls, pass a function to the assign() method (typically **a lambda function**), and this function will be called with the DataFrame as a parameter:
```py
(people
     .assign(body_mass_index = lambda df: df["weight"] / (df["height"] / 100) ** 2)
     .assign(overweight = lambda df: df["body_mass_index"] > 25)
)
```

**Evaluating an expression**
This relies on the `numexpr` library which must be installed.
```py
people.eval("body_mass_index = weight / (height/100) ** 2", inplace=True) # set inplace=True to directly modify the DataFrame rather than getting a modified copy
```
You can use a local or global variable in an expression by prefixing it with '@':
```py
overweight_threshold = 30
people.eval("overweight = body_mass_index > @overweight_threshold", inplace=True)
```

**Querying a `DataFrame`**
```py
people.query("age > 30 and pets == 0")
```

**Sorting a `DataFrame`**
`sort_index` returned a sorted copy of the `DataFrame`.
```py
people.sort_index(ascending=False, axis=1, inplace=True) # By default it sorts the rows by their index label, in ascending order.
```
To sort the DataFrame by the values instead of the labels, we can use sort_values:
```py
people.sort_values(by="age", inplace=True)
```

**Plotting a DataFrame**
The best option is to scroll through the [Visualization](https://pandas.pydata.org/pandas-docs/stable/visualization.html) page, find the plot you are interested in and look at the example code.

**Operations on DataFrames**
You can apply NumPy mathematical functions on a DataFrame: **the function is applied to all values**:
```py
grades_array = np.array([[8,8,9],[10,9,9],[4, 8, 2], [9, 10, 10]])
grades = pd.DataFrame(grades_array, columns=["sep", "oct", "nov"], index=["alice","bob","charles","darwin"])
np.sqrt(grades)
```
Aggregation operations of a DataFrame **apply to each column**, and you get back a Series object:
```py
grades.mean()
```
The `all` method is also an aggregation operation: it checks whether all values are True or not.
```py
(grades > 5).all(axis = 1)
```
The `any` method returns True if any value is True.
```py
(grades == 10).any(axis = 1)
```
If you add a `Series` object to a `DataFrame` (or execute any other binary operation), pandas attempts to **broadcast the operation to all rows** in the `DataFrame`. This only works if the `Series` has the same size as the `DataFrames` rows.
```py
grades - grades.mean()  # equivalent to: grades - [7.75, 8.75, 7.50]
```

**Automatic alignment**
When operating on multiple DataFrames, pandas automatically aligns them **by row index label, but also by column names.**
```py
bonus_array = np.array([[0,np.nan,2],[np.nan,1,0],[0, 1, 0], [3, 3, 0]])
bonus_points = pd.DataFrame(bonus_array, columns=["oct", "nov", "dec"], index=["bob","colin", "darwin", "charles"])
```

**Handling missing data**
```py
(grades + bonus_points).fillna(0)
```
Another way to handle missing data is to interpolate.
```py
bonus_points.interpolate(axis=1) # By default, it interpolates vertically (axis=0)
```
Call the dropna() method to get rid of rows that are full of NaNs:
```py
final_grades_clean = final_grades_clean.dropna(axis=1, how="all")
```

**Aggregating with `groupby`**
```py
grouped_grades = final_grades.groupby("hobby")
```

**Pivot tables**
Call the pd.pivot_table() function for this DataFrame, asking to group by the name column.
```py
pd.pivot_table(more_grades, index="name", values="grade", columns="month", margins=True) # By default, pivot_table() computes the mean of each numeric column:
```
We can specify multiple index or column names, and pandas will create multi-level indices:
```py
pd.pivot_table(more_grades, index=("name", "month"), margins=True)
```

**Overview functions**
```py
large_df.head() # The head() method returns the top 5 rows:
large_df.tail(n=2) # a tail() function to view the bottom 5 rows. You can pass the number of rows you want
large_df.info() # The info() method prints out a summary of each columns contents:
```
The describe() method gives a nice overview of the main aggregated values over each column:
-   `count`: number of non-null (not NaN) values
-   `mean`: mean of non-null values
-   `std`:  [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation)  of non-null values
-   `min`: minimum of non-null values
-   `25%`,  `50%`,  `75%`: 25th, 50th and 75th  [percentile](https://en.wikipedia.org/wiki/Percentile)  of non-null values
-   `max`: maximum of non-null values
```py
large_df.describe()
```

### Saving & loading

Pandas can save DataFrames to various backends, including file formats such as CSV, Excel, JSON, HTML and HDF5, or to a SQL database.

**Saving**
```py
my_df = pd.DataFrame(
    [["Biking", 68.5, 1985, np.nan], ["Dancing", 83.1, 1984, 3]], 
    columns=["hobby","weight","birthyear","children"],
    index=["alice", "bob"]
)
my_df.to_csv("my_df.csv")
my_df.to_html("my_df.html")
my_df.to_json("my_df.json")
```

**Loading**
```py
my_df_loaded = pd.read_csv("my_df.csv", index_col=0)
```
There are similar read_json, read_html, read_excel functions as well. We can also read data straight from the Internet.

### Combining DataFrames

**SQL-like joins**
```py
city_loc = pd.DataFrame(
    [
        ["CA", "San Francisco", 37.781334, -122.416728],
        ["NY", "New York", 40.705649, -74.008344],
        ["FL", "Miami", 25.791100, -80.320733],
        ["OH", "Cleveland", 41.473508, -81.739791],
        ["UT", "Salt Lake City", 40.755851, -111.896657]
    ], columns=["state", "city", "lat", "lng"])

city_pop = pd.DataFrame(
    [
        [808976, "San Francisco", "California"],
        [8363710, "New York", "New-York"],
        [413201, "Miami", "Florida"],
        [2242193, "Houston", "Texas"]
    ], index=[3,4,5,6], columns=["population", "city", "state"])

pd.merge(left=city_loc, right=city_pop, on="city") # INNER JOIN.
all_cities = pd.merge(left=city_loc, right=city_pop, on="city", how="outer") # FULL OUTER JOIN, and `how="left"` for LEFT OUTER JOIN, `how="right"` for RIGHT OUTER JOIN
```

If the key to join on is actually in one (or both) DataFrame's index, you must use left_index=True and/or right_index=True. If the key column names differ, you must use left_on and right_on.
```py
city_pop2 = city_pop.copy()
city_pop2.columns = ["population", "name", "state"]
pd.merge(left=city_loc, right=city_pop2, left_on="city", right_on="name")
```

**Concatenation**
```py
result_concat = pd.concat([city_loc, city_pop])
```

`concat()` aligns the data horizontally (by columns) but not vertically (by rows). We may end up with multiple rows having the same index. Or you can tell pandas to just ignore the index:
```py
pd.concat([city_loc, city_pop], ignore_index=True)
```

The `append()` method is a useful shorthand for concatenating DataFrames vertically, it works on a **copy** and returns the modified copy.
```py
city_loc.append(city_pop)
```

### Categories

```py
city_eco = city_pop.copy()
city_eco["eco_code"] = [17, 17, 34, 20]
city_eco["economy"] = city_eco["eco_code"].astype('category')
city_eco["economy"].cat.categories = ["Finance", "Energy", "Tourism"]
```

## Matplotlib
> The best way to learn more, is to visit the [gallery](http://matplotlib.org/gallery.html), look at the images, choose a plot that you are interested in, then just copy the code in a Jupyter notebook and play around with it.
> [practice](https://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/tools_matplotlib.ipynb)

Matplotlib can output graphs using various backend graphics libraries, such as Tk, wxPython, etc.

### Plotting your first graph

When running python using the command line, the graphs are typically shown in a separate window. In a Jupyter notebook, we can simply output the graphs within the notebook itself by running the `%matplotlib inline` magic command.
```py
import matplotlib

%matplotlib inline
```

### Line style and color

### Saving a figure

### Subplots

### Multiple figures

### Pyplot's state machine: implicit vs explicit

### Drawing text

### Legends

### Non linear scales

### Ticks and tickers

### Polar projection

### 3D projection

### Scatter plot

### Lines

### Histograms

### Images

### Animations

### Saving animations to video files

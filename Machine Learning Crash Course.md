# Machine Learning Crash Course
> [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/), [中文](https://developers.google.cn/machine-learning/crash-course/)
> [笔记](https://github.com/amusi/TensorFlow-From-Zero-To-One)

# 资源
> [Google AI Education](https://ai.google/education)
> [Pratica](https://developers.google.com/machine-learning/practica/)
> [Guides](https://developers.google.com/machine-learning/guides/)
> [面向普通开发者的机器学习应用方案](https://chinagdg.org/2016/03/machine-learning-recipes-for-new-developers/)

# Introduction

## Prework

### [pandas](https://pandas.pydata.org/)
> [Colaboratory](https://colab.research.google.com/)
> [Quick Introduction to pandas](https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=pandas-colab&hl=en#scrollTo=rHLcriKWLRe4)

The primary data structures in pandas are implemented as two classes:

 - **DataFrame**: a relational data table, with rows and named columns
 - **Series**: a single column. A DataFrame contains one or more Series and a name for each Series.

By default, at construction, pandas assigns index values(an identifier value to each Series item or DataFrame row) that reflect the ordering of the source data. Once created, the index values are stable; that is, they do not change when data is reordered.

**Feature column**s store only a description of the feature data; they do not contain the feature data itself.

### TensorFlow basics

**概念**

 - 张量：任意维度的数组。
 - 指令：创建、销毁和操控张量。
 - 图（也称为计算图或数据流图）：图的节点是指令；图的边是张量。
 - 会话：存储它所运行的图的状态。图必须在 TensorFlow 会话中运行，会话可以将图分发到多个机器上执行

TensorFlow 会实现**延迟执行**模型，意味着系统仅会根据相关节点的需求在需要时计算节点。

张量可以作为常量或变量存储在图中。常量和变量都只是图中的一种指令。常量是始终会返回同一张量值的指令。变量是会返回分配给它的任何张量的指令。

**流程**
TensorFlow 编程本质上是一个两步流程：

 - 将常量、变量和指令整合到一个图中。
 - 在一个会话中评估这些常量、变量和指令。

**组件**
TensorFlow consists of the following two **components**:

 - a  [graph protocol buffer](https://www.tensorflow.org/extend/tool_developers/#protocol_buffers)
 - a runtime that executes the (distributed) graph

These two components are analogous to Python code and the Python interpreter.

## Key Concepts and Tools

### Math

### Python

> [Tutorial](https://docs.python.org/3/tutorial/)

### Libraries

 - Matplotlib (for data visualization)
	 - [`pyplot`](http://matplotlib.org/api/pyplot_api.html)  module
	 - [`cm`](http://matplotlib.org/api/cm_api.html)  module
	 - [`gridspec`](http://matplotlib.org/api/gridspec_api.html)  module
 - pandas: for data manipulation
	 - [`DataFrame`](https://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe) class
 - NumPy: for low-level math operations
	 - [`linspace`](https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linspace.html)  function
	 - [`random`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.random.html#numpy.random.random)  function
	 - [`array`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html)  function
	 - [`arange`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html)  function
 - scikit-learn: for evaluation metrics
	 - [metrics](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)  module

### Command line

 - Bash
	 - [manual](https://tiswww.case.edu/php/chet/bash/bashref.html)
	 - [cheatsheet](https://github.com/LeCoupa/awesome-cheatsheets/blob/master/languages/bash.sh)
 - Shell

# ML Concepts
> 800 min

## Framing
### Models
A model defines the relationship between **features** and **label**. Two phases of a model's life:
 - **Training** means creating or learning the model. That is, you show the model labeled examples and enable the model to gradually learn the relationships between features and label.
 - **Inference** means applying the trained model to unlabeled examples. That is, you use the trained model to make useful predictions (y').

### Regression vs. classification
 - A **regression** model predicts continuous values.
 - A **classification** model predicts discrete values.

## Descending into ML

### Linear Regression
$$
y^{'} = b + \sum_{i\in D} \omega_{i} * x_{i}
$$
where:
-  $y^{'}$ is the predicted label(a desired output).
-   b  is the bias.
-   $\omega_i$  is the weight of feature i.
-   $x_i$  is a feature(a known input).
-   D  is a data set containing many labeled examples

### Training and Loss

In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called **empirical risk minimization(ERM)**.

**Loss** is a number indicating how bad the model's prediction was on a single example. The goal of training a model is to find a set of weights and biases that have low loss, on average, across all examples.

Loss function: **Squared loss**  (**$L_2 loss$**) & **Mean square error(MSE)**
MSE is the average squared loss per example over the whole dataset:
$$
MSE = \frac{1}{N}\sum_{x,y\in D} (y - prediction(x))^2
$$
where:

 - $(x,y)$  is an example in which
	 - $x$  is the set of features that the model uses to make predictions.
	 - $y$  is the example's label.
 - prediction(x)  is a function of the weights and bias in combination with the set of features  x.
 - D  is a data set containing many labeled examples, which are  (x,y)  pairs.
 - N  is the number of examples in  D.

Root Mean Square Error (RMSE): 
$$
RMSE = \sqrt{MSE}
$$

## Reducing Loss

### An Iterative Approach
![](http://flowtime-linear-regression.soft.today/images/c30/iterative-approach.png)

 - The "model" takes one or more features as input and returns one prediction (y') as output.
 - The "Compute Loss" part of the diagram is the loss function that the model will use.
 - The "Compute parameter updates" part examines the value of the loss function and generates new values for bias and weight.

And then the machine learning system re-evaluates all those features against all those labels, yielding a new value for the loss function, which yields new parameter values. Usually, you iterate until overall loss stops changing or at least changes extremely slowly. When that happens, we say that the model has **converged**.

### Gradient Descent

**Partial derivatives**
Intuitively, a partial derivative tells you how much the function changes when you perturb one variable a bit.

**Gradients**
The gradient of a function is **the vector of partial derivatives with respect to all of the independent variables**, the vector falls within the domain space of the function.

 - $\nabla f$	Points in the direction of greatest **increase** of the function.
 - $-\nabla f$	Points in the direction of greatest **decrease** of the function. We often try to minimize the loss function by following the negative of the gradient of the function.

### Learning Rate

Gradient descent algorithms **multiply the gradient by learning rate (step size)** to determine the next point. The **Goldilocks** learning rate is related to how flat the loss function is.

**[Hyperparameters](https://www.quora.com/What-are-hyperparameters-in-machine-learning)** are the knobs that programmers tweak in machine learning algorithms.

### Stochastic Gradient Descent

Stochastic gradient descent (SGD) uses only a single example (a batch size of 1) per iteration. Given enough iterations, SGD works but is very noisy. The term "stochastic" indicates that the one example is chosen at random.

Mini-batch stochastic gradient descent (mini-batch SGD) (typically between 10 and 1,000 examples) reduces the amount of noise in SGD but is still more efficient than full-batch.

## Generalization

 - We draw examples **independently and identically** (i.i.d) at random from the distribution. In other words, examples don't influence each other.
 - The distribution is **stationary**; that is the distribution doesn't change within the data set.
 - We draw examples from partitions from **the same distribution**.

When we know that any of the preceding three basic assumptions are violated, we must pay careful attention to metrics.

Overfitting
Ockham's razor in machine learning terms: The less complex an ML model, the more likely that a good empirical result is not just due to the peculiarities of the sample.

Good performance on the test set is a useful indicator of good performance on the new data in general, assuming that the test set is large enough.

## Training, Validation and Test Sets
You can greatly reduce your chances of overfitting by partitioning the data set into the three subsets: training set, validation set, test set.

Use the **validation set** to evaluate results from the **training set**. Then, use the **test set** to double-check your evaluation after the model has "passed" the validation set:
![](https://developers.google.com/machine-learning/crash-course/images/WorkflowWithValidationSet.svg)

**Never train on test data!**

Debugging in ML is often data debugging rather than code debugging.

## Representation

### Feature Engineering
Feature engineering means transforming raw data into a feature vector.
Since models cannot multiply strings by the learned weights, we use feature engineering to convert strings to numeric values.

OOV (out-of-vocabulary)

**Mapping categorical values**
Discrete features are usually converted into families of binary features before training a logistic regression model.
The representation is called **one-hot encoding** when a single value is 1, and a **multi-hot encoding** when multiple values are 1.

**Sparse Representation**: only nonzero values are stored.

### Qualities of Good Features

 - Avoid rarely used discrete feature values. Good feature values should appear more than 5 or so times in a data set.
 - Prefer clear and obvious meanings.
 - Don't mix "magic" values with actual data. Good floating-point features don't contain peculiar out-of-range discontinuities or "magic" values. 
 - Account for upstream instability. The definition of a feature shouldn't change over time.

### Cleaning Data
> Good ML relies on good data.
> [Rules of Machine Learning, ML Phase II: Feature Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml/#ml_phase_ii_feature_engineering)

**Scaling feature values**
Scaling means converting floating-point feature values from their natural range into a standard range

 - **linearly map** to a small scale
 - calculate the **Z score** of each value: ``` scaledvalue = (value - mean) / stddev ```

If a feature set consists of multiple features, then feature scaling provides the following benefits:

 - Helps gradient descent converge more quickly.
 - Helps avoid the "NaN trap"
 - Helps the model learn appropriate weights for each feature. Without feature scaling, the model will pay too much attention to the features having a wider range.

**Handling extreme outliers**

 - **take the log** of every value
 - **clip the maximum value**. All values that were greater than max value now become max value

**Binning**
With binning, our model can now learn completely different weights for each bin.

Another approach is to **bin by quantile**, which ensures that the number of examples in each bucket is equal. Binning by quantile completely removes the need to worry about outliers.


**Scrubbing**
Removing bad examples from the data set: 

 - Omitted values
 - Duplicate examples
 - Bad labels
 - Bad feature values

## Feature Crosses
A **feature cross** is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together. (The term cross comes from cross product.)

**Linear learners scale well to massive data**. Supplementing scaled linear models with feature crosses has traditionally been an efficient way to train on massive-scale data sets.

### Crossing One-Hot Vectors

Machine learning models do frequently cross one-hot feature vectors. Think of feature crosses of one-hot feature vectors as logical conjunctions. 

Using feature crosses on massive data sets is one efficient strategy for **learning highly complex models.** Neural networks provide another strategy.

[**FTRL Algorithm**](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf)

## Regularization for Simplicity

**Regularization** means penalizing the complexity of a model to reduce overfitting.

### L₂ Regularization

**Structural risk minimization**: $minimize(Loss(Data|Model) + \lambda\cdot complecity(Model))$

 - the **loss term**, which measures how well the model fits the data
 - the **regularization term**, which measures model complexity

Two common (and somewhat related) ways to think of model complexity:

 - Model complexity as a function of the **weights** of all the features in the model.
 - Model complexity as a function of the total **number of features** with nonzero weights.

L2 regularization formula:
$$
L_{2}\space Regularization\space term = \parallel\bm{w}\parallel_{2}^{2}
$$

### Lambda(regularization rate)

Performing L2 regularization has the following effect on a model:

 - Encourages weight values toward 0 (but not exactly 0)
 - Encourages the mean of the weights toward 0, with a normal (bell-shaped or Gaussian) distribution.

Increasing the lambda value strengthens the regularization effect.

When choosing a lambda value, the goal is to strike the right balance between simplicity and training-data fit:

 - lambda value is too high -> simple model -> underfitting
 - lambda value is too low -> complex model -> overfitting

**Early stopping** means ending training before the model fully reaches convergence.

The effects from changes to regularization parameters can be confounded with the effects from changes in learning rate or number of iterations. 
One useful practice (when training across a fixed batch of data) is to give yourself a high enough number of iterations that early stopping doesn't play into things.

 **Test loss** is the true measure of the model's ability to make good predictions on new data.

## Logistic Regression

Many problems require a probability estimate as output. Logistic regression is an extremely efficient mechanism for calculating probabilities.

**Loss function**
The loss function for linear regression is squared loss. The loss function for logistic regression is **Log Loss**:
$$
Log\space Loss = \sum_{x,y \in D} -y\log{y'} - (1-y)\log(1-y')
$$

 - every value of $y$ must either be 0 or 1.

Indeed, minimizing the loss function yields a maximum likelihood estimate.

**Regularization in Logistic Regression**

Without regularization, the asymptotic nature of logistic regression would keep driving loss towards 0 in high dimensions?
Strategies to dampen model complexity:

 - $L_2$ regularization
 - Early stopping
 - $L_1$ regularization

## Classification

### Thresholding

Logistic regression returns a probability.

In order to map a logistic regression value to a binary category, you must define a classification threshold (also called the decision threshold).

### True vs. False and Positive vs. Negative

[confusion matrix](https://developers.google.com/machine-learning/glossary#confusion_matrix)：row为预测值，column为真实值.

 - A **true positive(TP)** is an outcome where the model correctly predicts the positive class. 
 - A **true negative(TN)** is an outcome where the model correctly predicts the negative class.
 - A **false positive(FP)** is an outcome where the model incorrectly predicts the positive class. 
 - A **false negative(FN)** is an outcome where the model incorrectly predicts the negative class.

### Accuracy

Informally, accuracy is the fraction of predictions our model got right.
Accuracy alone doesn't tell the full story when you're working with a class-imbalanced data set.

### Precision and Recall

Precision: What proportion of positive identifications was actually correct?
$$
Precision = \frac{TP}{TP + FP}（以第1行为基准）
$$

Recall: What proportion of actual positives was identified correctly?
$$
Recall = \frac{TP}{TP + FN}（以第1列为基准）
$$

To fully evaluate the effectiveness of a model, you must examine **both** precision and recall.

[F1 score](https://wikipedia.org/wiki/F1_score)

### ROC and AUC

An ROC(receiver operating characteristic curve) curve plots two parameters:

 - True Positive Rate(TPR): a synonym for recall. The y-axis
 - False Positive Rate(FPR): $FPR = \frac{FP}{FP + TN}$（以第2列为基准）. The x-axis

One way of interpreting AUC(Area Under the ROC Curve) is as **the probability that the model ranks a random positive example more highly than a random negative example**.

AUC limits
 - Scale invariance is not always desirable.
 - Classification-threshold invariance is not always desirable. In cases where there are wide disparities in the cost of false negatives vs. false positives

### Prediction Bias

prediction bias = average of predictions - average of labels in data set

If possible, avoid calibration layers.

A good model will usually have near-zero bias. That said, a low prediction bias does not prove that your model is good. A really terrible model could have a zero prediction bias.

## Regularization for Sparsity

$L_0$ regularization: penalizes the count of non-zero coefficient values in a model. It would turn our convex optimization problem into a non-convex optimization problem that's NP-hard. It isn't something we can use effectively in practice.

### L₁ Regularization

L₁ vs L2 regularization:

 - L₁ penalizes $|weight|$. increase sparsity -> reduce the size of a model -> may affect loss
	 - L₁ Regularization turns out to be quite efficient for **wide models**. 
	 - You can think of the derivative of L₁ as a force that subtracts some constant from the weight every time. 
	 - However, thanks to absolute values, L₁ has a discontinuity at 0, which causes subtraction results that cross 0 to become zeroed out.
 - $L_2$ penalizes $weight^2$
	 - You can think of the derivative of $L_2$ as a force that removes x% of the weight every time.

## Neural Network

### Introduction to Neural Networks

Neural networks are a more sophisticated version of feature crosses. In essence, neural networks learn the appropriate feature crosses.

Activation Functions: the nonlinear function transforms the value of each node in Hidden Layer, and then pass on to the weighted sums of the next layer.

 - sigmoid function: such as $F(x) = \frac{1}{1 + e^{-x}}$
 - rectified linear unit activation function (or **ReLU**): $F(x) = max(0, x)$. The **superiority** of ReLU is based on empirical findings, probably driven by ReLU having a more useful range of responsiveness.

Stacking nonlinearities on nonlinearities lets us model very complicated relationships between the inputs and the predicted outputs.

Even with Neural Nets, some amount of feature engineering is often needed to achieve best performance.

You can look at the gap between loss on training data and loss on validation data to help judge if your model is starting to overfit. If the gap starts to grow, that is usually a sure sign of overfitting.

[Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)（[翻译](https://zcao.info/2017/10/09/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E3%80%81%E6%B5%81%E5%BD%A2%E3%80%81%E6%8B%93%E6%89%91/)）

 1. [ConvnetJS demo](https://cs.stanford.edu/people/karpathy/convnetjs//demo/classify2d.html)
 2. Homeomorphisms（同态）: preserves topological properties. Each layer stretches and squishes space, but it never cuts, breaks, or folds it.
 3. ...

### Training Neural Networks

**Backpropagation** makes gradient descent feasible for multi-layer neural networks.

 - [工作原理](https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/)

Best Practices

 - Vanishing Gradients
	 - When the gradients vanish toward 0 for the lower layers, these layers train very slowly, or not at all.
	 - The **ReLU activation function** can help prevent vanishing gradients.
 - Exploding Gradients
	 - If the weights in a network are very large, then the gradients for the lower layers involve products of many large terms.
	 - **Batch normalization** can help prevent exploding gradients, as can lowering the learning rate.
 - Dead ReLU Units
	 - Once the weighted sum for a ReLU unit falls below 0, the ReLU unit can get stuck.
	 - **Lowering the learning rate** can help keep ReLU units from dying.

Yet another form of regularization useful for neural networks, called **Dropout**. It works by randomly "dropping out" unit activations in a network for a single gradient step.

**Normalization Methods**

 - **Linear Scaling** (as a rule of thumb, NN's train best when the input features are roughly on the same scale):
	```python
	def linear_scale(series):
	  min_val = series.min()
	  max_val = series.max()
	  scale = (max_val - min_val) / 2.0
	  return series.apply(lambda x:((x - min_val) / scale) - 1.0)
	```
 - Adagrad optimizer: works great for **convex problems**. The key insight of Adagrad is that it modifies the learning rate adaptively for each coefficient in a model, monotonically lowering the effective learning rate
 - Adam optimizer: for **non-convex  problems**
 - log scaling
 - clipping extreme values

扩展：
[An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/)

 1. 

### Multi-Class Neural Networks

**One vs. All**
Given a classification problem with N possible solutions, a one-vs.-all solution consists of **N separate binary classifiers**—one binary classifier for each possible outcome.

This approach is fairly reasonable when the total number of classes is small, but becomes increasingly inefficient as the number of classes rises.

We can create a significantly more efficient one-vs.-all model with a deep neural network in which **each output node represents a different class**.

**Softmax**

The Softmax equation:
$$
p(y=j | \bm{x}) = \frac{e^{\bm{w}_j^{T}\bm{x} + b_j}}{\sum_{k\in K}e^{\bm{w}_k^T\bm{x} + b_k}}
$$
This formula basically **extends the formula for logistic regression into multiple classes**.

Softmax assigns decimal probabilities to each class in a multi-class problem. Those decimal probabilities must add up to 1.0. This additional constraint helps training converge more quickly than it otherwise would.

Softmax is implemented through a neural network layer just before the output layer.

Variants of Softmax:

 - Full Softmax: calculates a probability for every possible class.
 - Candidate sampling: calculates a probability for **all the positive** labels but only for a **random sample of negative** labels. It can improve efficiency in problems having a large number of classes.

Softmax assumes that each example is a member of exactly one class. For **many-labels** problems, you must rely on multiple logistic regressions.

## Embedding

# ML Engineering

## Static vs. Dynamic Training

## Static (Offline) vs. Dynamic (Online) Inference

## Data Dependencies

## Fairness

### Types of Bias

 - **Reporting** Bias: occurs when the **frequency** of events, properties, and/or outcomes captured in a data set does not accurately reflect their real-world frequency.
 - **Automation** Bias: favor results generated by automated systems over those generated by non-automated systems, irrespective of the error rates of each.
 - **Selection** Bias: occurs if a data set's examples are chosen in a way that is not reflective of their real-world distribution.
	 - Coverage bias: Data is not selected in a representative fashion.
	 - Non-response bias (or participation bias): Data ends up being unrepresentative due to participation gaps in the data-collection process.
	 - Sampling bias: Proper randomization is not used during data collection.
 - **Group Attribution** Bias: generalize what is true of individuals to an entire group to which they belong.
	 - In-group bias
	 - Out-group homogeneity bias
 - **Implicit** Bias: occurs when assumptions are made based on one's own mental models and personal experiences that do not necessary apply more generally.
	 - confirmation bias
	 - experimenter's bias

### Identifying Bias

Missing Feature Values

Unexpected Feature Values

Data Skew

### Evaluating for Bias

# ML Real World Examples

## Guidelines

 - Keep the first model simple
 - Focus on ensuring data pipeline correctness
 - Use a simple, observable metric for training & evaluation
 - Own and monitor your input features
 - Treat your model configuration as code: review it, check it in
 - Write down the results of all experiments, especially "failures"

# Conclusion

[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml/)

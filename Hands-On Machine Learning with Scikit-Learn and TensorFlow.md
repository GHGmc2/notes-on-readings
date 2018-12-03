---


---

<h1 id="hands-on-machine-learning-with-scikit-learn-and-tensorflow">Hands-On Machine Learning with Scikit-Learn and TensorFlow</h1>
<blockquote>
<p><a href="http://shop.oreilly.com/product/0636920052289.do">主页</a>, <a href="https://www.oreilly.com/catalog/errata.csp?isbn=0636920052289">勘误</a><br>
<a href="https://github.com/ageron/handson-ml">Github</a>, <a href="https://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/index.ipynb">Jupyter Viewer</a>, <a href="https://github.com/DeqianBai/Hands-on-Machine-Learning">中文注释</a><br>
<a href="https://book.douban.com/subject/30317874/">douban</a><br>
<a href="https://github.com/apachecn/hands_on_Ml_with_Sklearn_and_TF">中文翻译</a></p>
</blockquote>
<h2 id="preface">Preface</h2>
<p>This book assumes that you know close to nothing about Machine Learning.</p>
<h3 id="prerequisites">Prerequisites</h3>
<p>This book assumes that you have some Python programming experience and that you are familiar with Python’s main scientific libraries, in particular NumPy, Pandas, and Matplotlib.</p>
<h3 id="roadmap">Roadmap</h3>
<p>Don’t jump into deep waters too hastily, you should master the fundamentals first. Moreover, most problems can be solved quite well using simpler techniques.</p>
<h3 id="other-resources">Other Resources</h3>
<p><a href="https://www.quora.com/What-are-the-best-regularly-updated-machine-learning-blogs-or-resources-available">What are the best, regularly updated machine learning blogs or resources available?</a><br>
<a href="http://deeplearning.net/">deeplearning.net</a></p>
<p><a href="https://www.dataquest.io/">Dataquest</a><br>
<a href="https://www.kaggle.com/">Kaggle</a></p>
<h1 id="the-fundamentals-of-machine-learning">The Fundamentals of Machine Learning</h1>
<h2 id="the-machine-learning-landscape">The Machine Learning Landscape</h2>
<p>Machine Learning is about making machines get better at some task by <strong>learning from data, instead of having to explicitly code rules</strong>.</p>
<h3 id="types-of-ml-systems">Types of ML Systems</h3>
<p>based on:</p>
<ul>
<li>Whether or not they are trained with human supervision (supervised, unsupervised, semisupervised, and Reinforcement Learning)</li>
<li>Whether or not they can learn incrementally on the fly (<strong>online vs. batch learning</strong>)</li>
<li>Whether they work by simply comparing new data points to known data points, or instead detect patterns in the training data and build a predictive model, much like scientists do (instance-based versus model-based learning)</li>
</ul>
<h4 id="batch-and-online-learning">Batch and Online Learning</h4>
<p><strong>Batch learning</strong>: it must be trained using <strong>all</strong> the available data.<br>
First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called <strong>offline learning</strong>.</p>
<p><strong>Online learning (incremental learning)</strong>: train the system incrementally by feeding it data instances sequentially, either <strong>individually or by mini-batches</strong>.<br>
A big <strong>challenge</strong> with online learning is that if bad data is fed to the system, the system’s performance will gradually decline. To reduce this risk, you need to <strong>monitor</strong> your system closely.</p>
<h3 id="main-challenges">Main Challenges</h3>
<p>Bad data:</p>
<ul>
<li><strong>Insufficient Quantity</strong> of Training Data</li>
<li><strong>Nonrepresentative</strong> Training Data: sample bias</li>
<li><strong>Poor-Quality</strong> Data: errors, outliers, and noise</li>
<li><strong>Irrelevant Features</strong>: the training data contains too many irrelevant features.<br>
Feature engineering:
<ul>
<li>Feature selection: selecting the most useful features to train on among existing features.</li>
<li>Feature extraction: combining existing features to produce a more useful one.</li>
<li>Creating new features by gathering new data.</li>
</ul>
</li>
</ul>
<p>Bad algorithm:</p>
<ul>
<li><strong>Overfitting</strong> the Training Data<br>
A <strong>hyperparameter</strong> is a parameter of a learning algorithm (not of the model).<br>
Possible solutions:
<ul>
<li><strong>Simplify the model</strong> by selecting one with fewer parameters (e.g., a linear model rather than a high-degree polynomial model), by <strong>reducing the number of attributes</strong> in the training data or by <strong>constraining the model</strong> (regularization)</li>
<li>Gather <strong>more training data</strong></li>
<li>Reduce the noise in the training data (e.g., fix data errors and remove outliers)</li>
</ul>
</li>
<li><strong>Underfitting</strong> the Training Data<br>
Main options:
<ul>
<li>Selecting a more powerful model, with more parameters</li>
<li>Feeding better features to the learning algorithm</li>
<li>Reducing the constraints on the model</li>
</ul>
</li>
</ul>
<h3 id="testing-and-validating">Testing and Validating</h3>
<p>Generalization error (out-of- sample error): the error rate on new cases.</p>
<p>The training set, the validation set, and the test set.</p>
<p><strong>Cross-validation</strong>: the training set is split into <strong>complementary</strong> subsets, and each model is trained against a <strong>different combination</strong> of these subsets and validated against the <strong>remaining</strong> parts.</p>
<p>Once the model type and hyperparameters have been selected, a final model is trained using these hyperparameters on the <strong>full</strong> training set, and the generalized error is measured on the <strong>test</strong> set.</p>
<p>No Free Lunch (NFL) theorem: if you make absolutely no assumption about the data, then there is no reason to prefer one model over any other.</p>
<h2 id="end-to-end-machine-learning-project">End-to-End Machine Learning Project</h2>
<p>The main steps: <a href="http://www.ic.unicamp.br/~sandra/pdf/Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow-427-432.pdf">Machine Learning Project Checklist</a></p>
<ol>
<li>Look at the big picture.</li>
<li>Get the data.</li>
<li>Discover and visualize the data to gain insights.</li>
<li>Prepare the data for Machine Learning algorithms.</li>
<li>Select a model and train it.</li>
<li>Fine-tune your model.</li>
<li>Present your solution.</li>
<li>Launch, monitor, and maintain your system.</li>
</ol>
<h3 id="working-with-real-data">Working with Real Data</h3>
<p>Open datasets</p>
<h3 id="look-at-the-big-picture">Look at the big picture</h3>
<h4 id="frame-the-problem">Frame the Problem</h4>
<p>two questions:</p>
<ul>
<li>what exactly is the business objective?<br>
Getting this right is critical, as it will determine how you frame the problem, what algorithms you will select, what performance measure you will use to evaluate your model, and how much effort you should spend tweaking it.</li>
<li>what the current solution looks like (if any)?<br>
It will often give you a reference performance, as well as insights on how to solve the problem.</li>
</ul>
<p>A sequence of data processing components is called a data <strong>pipeline</strong>.</p>
<h4 id="select-a-performance-measure">Select a Performance Measure</h4>
<p><strong>The higher the norm index, the more it focuses on large values and neglects small ones.</strong> (The RMSE is more sensitive to outliers than the MAE)</p>
<ul>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi mathvariant="normal">ℓ</mi><mn>1</mn></msub></mrow><annotation encoding="application/x-tex">\ell_1</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.84444em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord">ℓ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.301108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">1</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span> norm(Manhattan norm): MAE(Mean Absolute Error, also called the Average Absolute Deviation)</li>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi mathvariant="normal">ℓ</mi><mn>2</mn></msub></mrow><annotation encoding="application/x-tex">\ell_2</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.84444em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord">ℓ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.301108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span> norm(Euclidian norm): RMSE(Root Mean Square Error)</li>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi mathvariant="normal">ℓ</mi><mi>k</mi></msub></mrow><annotation encoding="application/x-tex">\ell_k</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.84444em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord">ℓ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.336108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathit mtight" style="margin-right: 0.03148em;">k</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span> norm</li>
</ul>
<h4 id="check-the-assumptions">Check the Assumptions</h4>
<h3 id="get-the-data">Get the data</h3>
<h4 id="create-the-workspace">Create the Workspace</h4>
<p>Create a workspace directory:</p>
<pre><code>$ export ML_PATH="$HOME/ml"
$ mkdir -p $ML_PATH
</code></pre>
<p>Creating an isolated environment: <strong>virtualenv</strong><br>
It is strongly recommended so you can work on different projects without having conflicting library versions.</p>
<pre><code>$ pip3 install --user --upgrade virtualenv
$ cd $ML_PATH
$ virtualenv env

Now every time you want to activate this environment, just open a terminal and type:
$ cd $ML_PATH
$ source env/bin/activate
</code></pre>
<p>While the environment is active, any package you install using pip will be installed in this isolated environment.</p>
<h4 id="download-the-data">Download the Data</h4>
<p>Automating the process of fetching(downloading and loading) the data with small functions.</p>
<h4 id="take-a-quick-look-at-the-data-structure">Take a Quick Look at the Data Structure</h4>
<p>pandas methods:</p>
<ul>
<li>head(): look at the top five rows</li>
<li>info(): get a quick description</li>
<li><strong>describe()</strong>: shows a summary of the numerical attributes</li>
<li>hist(): plot a histogram for a numerical attribute</li>
</ul>
<h4 id="create-a-test-set">Create a Test Set</h4>
<ul>
<li>random sampling: pick some instances(typically 20% of the dataset) randomly, and set them aside
<ul>
<li>A common solution is to use each instance’s identifier(use the most stable features to build a unique identifier) to decide whether or not it should go in the test set.</li>
</ul>
</li>
<li>stratified sampling: maintain the ratio in the sample.
<ul>
<li>Scikit-Learn’s StratifiedShuffleSplit class</li>
</ul>
</li>
</ul>
<p>Ensure that the test set will remain consistent across multiple runs, even if you refresh the dataset.</p>
<h3 id="discover-and-visualize-the-data-to-gain-insights">Discover and visualize the data to gain insights</h3>
<h3 id="prepare-the-data-for-machine-learning-algorithms">Prepare the data for Machine Learning algorithms</h3>
<h4 id="data-cleaning">Data Cleaning</h4>
<h4 id="feature-scaling">Feature Scaling</h4>
<h3 id="select-a-model-and-train-it">Select a model and train it</h3>
<h4 id="better-evaluation-using-cross-validation">Better Evaluation Using Cross-Validation</h4>
<h3 id="fine-tune-your-model">Fine-tune your model</h3>
<h4 id="grid-search">Grid Search</h4>
<h4 id="randomized-search">Randomized Search</h4>
<h4 id="ensemble-methods">Ensemble Methods</h4>
<h3 id="launch-monitor-and-maintain-your-system">Launch, monitor, and maintain your system</h3>
<h2 id="classification">Classification</h2>
<h3 id="training-a-binary-classifier">Training a Binary Classifier</h3>
<h3 id="performance-measures">Performance Measures</h3>
<h4 id="measuring-accuracy-using-cross-validation">Measuring Accuracy Using Cross-Validation</h4>
<h4 id="confusion-matrix">Confusion Matrix</h4>
<h4 id="precision-and-recall">Precision and Recall</h4>
<h4 id="precisionrecall-tradeoff">Precision/Recall Tradeoff</h4>
<h4 id="the-roc-curve">The ROC Curve</h4>
<h3 id="multiclass-classification">Multiclass Classification</h3>
<h3 id="error-analysis">Error Analysis</h3>
<h3 id="multilabel-classification">Multilabel Classification</h3>
<h3 id="multioutput-classification">Multioutput Classification</h3>
<h2 id="training-models">Training Models</h2>
<blockquote>
<p><a href="https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/ch04.html">safari book</a></p>
</blockquote>
<h3 id="linear-regression">Linear Regression</h3>
<h3 id="gradient-descent">Gradient Descent</h3>
<h4 id="batch-gradient-descent">Batch Gradient Descent</h4>
<h4 id="stochastic-gradient-descent">Stochastic Gradient Descent</h4>
<h4 id="mini-batch-gradient-descent">Mini-batch Gradient Descent</h4>
<h3 id="polynomial-regression">Polynomial Regression</h3>
<h3 id="learning-curves">Learning Curves</h3>
<h3 id="regularized-linear-models">Regularized Linear Models</h3>
<h3 id="logistic-regression">Logistic Regression</h3>
<h2 id="svm">SVM</h2>
<h2 id="decision-trees">Decision Trees</h2>
<h2 id="ensemble-learning-and-random-forests">Ensemble Learning and Random Forests</h2>
<h2 id="dimensionality-reduction">Dimensionality Reduction</h2>
<h3 id="main-approaches-for-dimensionality-reduction">Main Approaches for Dimensionality Reduction</h3>
<h3 id="pca-principal-component-analysis">PCA (Principal Component Analysis)</h3>
<h3 id="kernel-pca">Kernel PCA</h3>
<h3 id="lle-locally-linear-embedding">LLE (Locally Linear Embedding)</h3>
<h3 id="other-dimensionality-reduction-techniques">Other Dimensionality Reduction Techniques</h3>
<h1 id="neural-networks-and-deep-learning">Neural Networks and Deep Learning</h1>
<h2 id="up-and-running-with-tensorflow">Up and Running with TensorFlow</h2>
<h2 id="introduction-to-artificial-neural-networks">Introduction to Artificial Neural Networks</h2>
<h2 id="training-deep-neural-nets">Training Deep Neural Nets</h2>
<h3 id="vanishingexploding-gradients-problems">Vanishing/Exploding Gradients Problems</h3>
<h3 id="reusing-pretrained-layers">Reusing Pretrained Layers</h3>
<h3 id="faster-optimizers">Faster Optimizers</h3>
<h3 id="avoiding-overfitting-through-regularization">Avoiding Overfitting Through Regularization</h3>
<h3 id="practical-guidelines">Practical Guidelines</h3>
<h2 id="distributing-tensorflow-across-devices-and-servers">Distributing TensorFlow Across Devices and Servers</h2>
<h3 id="multiple-devices-on-a-single-machine">Multiple Devices on a Single Machine</h3>
<h4 id="installation">Installation</h4>
<p>TensorFlow uses CUDA and cuDNN to control the GPU cards and accelerate computations.</p>
<ul>
<li>Compute Unified Device Architecture library (CUDA)</li>
<li>CUDA Deep Neural Network library (cuDNN): <strong>a GPU-accelerated library of primitives for DNNs</strong>.</li>
</ul>
<h4 id="managing-the-gpu-ram">Managing the GPU RAM</h4>
<p>By default TensorFlow automatically grabs all the RAM in all available GPUs the first time you run a graph.</p>
<p>Three solutions:</p>
<ul>
<li>run each process on different GPU cards: set the CUDA_VISIBLE_DEVICES environment variable</li>
<li>tell TensorFlow to grab only a fraction of the memory: create a ConfigProto object, set its gpu_options.per_process_gpu_memory_fraction option</li>
<li>tell TensorFlow to grab memory only when it needs it: set config.gpu_options.allow_growth to True</li>
</ul>
<h4 id="placing-operations-on-devices">Placing Operations on Devices</h4>
<ul>
<li>dynamic placer</li>
<li>simple placer</li>
</ul>
<p><strong>Simple placement</strong><br>
The simple placer respects the following rules:</p>
<ul>
<li>If a node was already placed on a device in a previous run of the graph, it is left on that device.</li>
<li>Else, if the user pinned a node to a device, the placer places it on that device.<br>
To pin nodes onto a device, you must create a device block using the device() function.</li>
<li>Else, it defaults to GPU #0, or the CPU if there is no GPU.<br>
<strong>The CPU is shared by all tasks located on the same machine.</strong> The “/cpu:0” device aggregates all CPUs on a multi-CPU system. There is currently no way to pin nodes on specific CPUs.</li>
</ul>
<p><strong>Logging placements</strong><br>
Set the log_device_placement option to True; this tells the placer to log a message whenever it places a node.</p>
<p><strong>Dynamic placement function</strong><br>
Specify a function (instead of a device name) when create a device block. TensorFlow will call this function for each operation it needs to place in the device block, and the function must return the name of the device to pin the operation on.</p>
<p><strong>Operations and kernels</strong><br>
<strong>kernel</strong>: an implementation for a device.</p>
<p><strong>Soft placement</strong>: fall back to the CPU when the operation has no kernel for GPU.<br>
Set the allow_soft_placement configuration option to True.</p>
<h4 id="parallel-execution">Parallel Execution</h4>
<p>graph</p>
<ul>
<li>node (op)</li>
<li>edge (dependency)</li>
</ul>
<p>When TensorFlow runs a graph, it starts by finding out the list of nodes that need to be evaluated, and it counts how many dependencies each of them has. Then it starts evaluating the nodes with zero dependencies.</p>
<p>TensorFlow manages a thread pool on each device to parallelize operations. These are called the <strong>inter-op thread pools</strong>.<br>
Some operations have multi‐ threaded kernels: they can use other thread pools (one per device) called the <strong>intra-op thread pools</strong>.</p>
<h4 id="control-dependencies">Control Dependencies</h4>
<p>To postpone evaluation of some nodes.</p>
<h3 id="multiple-devices-across-multiple-servers">Multiple Devices Across Multiple Servers</h3>
<p>TensorFlow cluster:</p>

<table>
<thead>
<tr>
<th>physical</th>
<th>logical</th>
</tr>
</thead>
<tbody>
<tr>
<td>machine</td>
<td></td>
</tr>
<tr>
<td>CPU/GPU</td>
<td>device</td>
</tr>
<tr>
<td></td>
<td>client</td>
</tr>
<tr>
<td></td>
<td>server/cluster</td>
</tr>
<tr>
<td></td>
<td>task/job</td>
</tr>
</tbody>
</table><p><img src="http://images2.imagebam.com/16/36/b1/efc4fa1024807754.png" alt=""></p>
<p>To run a graph across multiple servers, you first need to define a cluster.<br>
A <strong>cluster</strong> is composed of one or more TensorFlow servers, called <strong>tasks</strong>, typically spread across several machines.</p>
<p>A <strong>job</strong> is just a named group of tasks that typically have a common role.</p>
<ul>
<li>parameter server</li>
<li>worker</li>
</ul>
<p>To start a TensorFlow server, you must create a Server object.</p>
<p>If you have several servers on one machine, you will need to ensure that they don’t all try to grab all the RAM of every GPU.<br>
If you want the process to do nothing other than run the TensorFlow server, you can block the main thread by telling it to wait for the server to finish using the join() method.</p>
<h4 id="opening-a-session">Opening a Session</h4>
<p>You can open a session on any of the servers, from a client located in any process on any machine.</p>
<h4 id="the-master-and-worker-services">The Master and Worker Services</h4>
<p>The client uses the gRPC (Google Remote Procedure Call) protocol to communicate with the server.<br>
Data is transmitted in the form of protocol buffers.</p>
<p>Every TensorFlow server provides two services:</p>
<ul>
<li>the master service: <strong>coordinates</strong> the computations across tasks</li>
<li>the worker service: actually <strong>execute</strong> computations on tasks and get their results</li>
</ul>
<h4 id="pinning-operations-across-tasks">Pinning Operations Across Tasks</h4>
<p>You can use <strong>device blocks</strong> to pin operations on any device managed by any task.</p>
<p>Priority: device &gt; task (or job) &gt; session</p>
<h4 id="sharding-variables-across-multiple-parameter-servers">Sharding Variables Across Multiple Parameter Servers</h4>
<p>To reduce the risk of saturating a single parameter server’s network card.</p>
<p>TensorFlow provides the replica_device_setter() function to distribute variables across all the “ps” tasks in a round-robin fashion.</p>
<p>An inner device block can override the job, task, or device defined in an outer block.</p>
<h4 id="sharing-state-across-sessions-using-resource-containers">Sharing State Across Sessions Using Resource Containers</h4>
<ul>
<li>local session</li>
<li><strong>distributed sessions</strong>: variable state is managed by <strong>resource containers</strong> located on the cluster itself (not by the sessions).</li>
</ul>
<p>If you want to run completely independent computations on the same cluster, use a <strong>container block</strong>. Advantages are variable names remain nice and short, and you can easily reset a named container.</p>
<p>Resource containers also take care of preserving the state of queues and readers.</p>
<h4 id="asynchronous-communication-using-tensorflow-queues">Asynchronous Communication Using TensorFlow Queues</h4>
<p>To exchange data between multiple sessions.</p>
<ul>
<li>FIFO queue</li>
<li>RandomShuffleQueue</li>
<li>PaddingFifoQueue</li>
</ul>
<h4 id="loading-data-directly-from-the-graph">Loading Data Directly from the Graph</h4>
<p><strong>Preload the data into a variable</strong><br>
For datasets that can fit in memory, load the training data once and assign it to a variable, then just use that variable in your graph.</p>
<p><strong>Reading the training data directly from the graph</strong><br>
If the training set does not fit in memory, a good solution is to use <strong>reader operations</strong>.</p>
<p><strong>Multithreaded readers using a Coordinator and a QueueRunner</strong><br>
the Coordinator class and the QueueRunner class</p>
<p><strong>Other convenience functions</strong></p>
<h3 id="parallelizing-neural-networks-on-a-tensorflow-cluster">Parallelizing Neural Networks on a TensorFlow Cluster</h3>
<h4 id="one-neural-network-per-device">One Neural Network per Device</h4>
<p><strong>The speedup is almost linear.</strong></p>
<p>This solution is <strong>perfect for hyperparameter tuning</strong>: each device in the cluster will train a different model with its own set of hyperparameters.</p>
<p>It also works perfectly if you host a web service that receives a large number of queries per second (QPS) and you need your neural network to <strong>make a prediction for each query</strong>.<br>
<a href="https://www.tensorflow.org/serving/">TensorFlow Serving: for model deployment in production</a></p>
<h4 id="in-graph-versus-between-graph-replication">In-Graph Versus Between-Graph Replication</h4>
<ul>
<li><strong>in-graph replication</strong>: create one big graph, containing every neural network
<ul>
<li>simpler to implement since you don’t have to manage multiple clients and multiple queues</li>
</ul>
</li>
<li><strong>between-graph replication</strong>: create one <strong>separate graph for each neural network</strong> and handle synchronization between these graphs yourself
<ul>
<li>easier to organize into well-bounded and easy-to-test modules. Moreover, it gives you more flexibility.</li>
<li>one typical implementation is to <strong>coordinate the execution of these graphs using queues</strong></li>
</ul>
</li>
</ul>
<h4 id="model-parallelism">Model Parallelism</h4>
<p><strong>Run a single neural network across multiple devices.</strong> This requires chopping your model into separate chunks and running each chunk on a different device.</p>
<p>Model parallelism can speed up running or training some types of neural networks(CNN and RNN), but not all(such as fully connected networks), <strong>it really depends on the architecture of your neural network</strong>. And it requires special care and tuning.</p>
<ul>
<li>vertical split for CNN: contain layers that are only <strong>partially</strong> connected to the lower layers</li>
<li>horizontal split for RNN: <strong>placing each layer on a different device</strong>, active one device for each step, and by the time the signal propagates to the output layer all devices will be active simultaneously. The benefit of running multiple cells in parallel often outweighs the communication penalty.</li>
</ul>
<h4 id="data-parallelism">Data Parallelism</h4>
<p>Replicate the neural network on each device, run a training step simultaneously on all replicas <strong>using a different mini-batch for each, and then aggregate the gradients</strong> to update the model parameters.</p>
<p><strong>Synchronous updates</strong><br>
The aggregator waits for <strong>all</strong> gradients to be available before computing the average and applying the result.</p>
<p>The <strong>downside</strong> is that some devices may be slower than others. To reduce the waiting time at each step, you could <strong>ignore the gradients from the slowest few replicas</strong> (typically ~10% spare replicas).</p>
<p><strong>Asynchronous updates</strong><br>
Whenever a replica has finished computing the gradients, it <strong>immediately</strong> uses them to update the model parameters. There is no aggregation and synchronization.</p>
<p><strong>Stale gradients</strong> can slow down convergence, introducing noise and wobble effects, or they can even make the training algorithm diverge.<br>
Ways to reduce the effect of stale gradients:</p>
<ul>
<li>Reduce the learning rate</li>
<li>Drop stale gradients or scale them down</li>
<li>Adjust the mini-batch size.</li>
<li>Start the first few epochs using just one replica (this is called the warmup phase), since stale gradients tend to be more damaging at the beginning of training.</li>
</ul>
<p><a href="https://arxiv.org/pdf/1604.00981v2.pdf">REVISITING DISTRIBUTED SYNCHRONOUS SGD</a> found that <strong>data parallelism with synchronous updates using a few spare replicas was the most efficient</strong>.</p>
<p><strong>Bandwidth saturation</strong></p>
<h2 id="cnn">CNN</h2>
<h3 id="the-architecture-of-the-visual-cortex">The Architecture of the Visual Cortex</h3>
<h3 id="convolutional-layer">Convolutional Layer</h3>
<h3 id="pooling-layer">Pooling Layer</h3>
<h3 id="cnn-architectures">CNN Architectures</h3>
<ul>
<li>LeNet-5 (1998)</li>
<li>AlexNet (2012)</li>
<li>GoogLeNet (2014)</li>
<li>ResNet (2015)</li>
</ul>
<h2 id="rnn">RNN</h2>
<h3 id="recurrent-neurons">Recurrent Neurons</h3>
<h3 id="training-rnns">Training RNNs</h3>
<h3 id="deep-rnns">Deep RNNs</h3>
<h3 id="lstm-cell">LSTM Cell</h3>
<h3 id="gru-cell">GRU Cell</h3>
<h3 id="npl">NPL</h3>
<h2 id="autoencoders">Autoencoders</h2>
<h2 id="rl">RL</h2>
<h1 id="appendix">Appendix</h1>
<h2 id="machine-learning-project-checklist">Machine Learning Project Checklist</h2>
<blockquote>
<p><a href="http://www.ic.unicamp.br/~sandra/pdf/Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow-427-432.pdf">pdf</a></p>
</blockquote>
<h2 id="other-popular-ann-architectures">Other Popular ANN Architectures</h2>


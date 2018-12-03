---


---

<h1 id="machine-learning-crash-course">Machine Learning Crash Course</h1>
<blockquote>
<p><a href="https://developers.google.com/machine-learning/crash-course/">Machine Learning Crash Course</a>, <a href="https://developers.google.cn/machine-learning/crash-course/">中文</a><br>
<a href="https://github.com/amusi/TensorFlow-From-Zero-To-One">笔记</a></p>
</blockquote>
<h1 id="资源">资源</h1>
<blockquote>
<p><a href="https://ai.google/education">Google AI Education</a><br>
<a href="https://developers.google.com/machine-learning/practica/">Pratica</a><br>
<a href="https://developers.google.com/machine-learning/guides/">Guides</a><br>
<a href="https://chinagdg.org/2016/03/machine-learning-recipes-for-new-developers/">面向普通开发者的机器学习应用方案</a></p>
</blockquote>
<h1 id="introduction">Introduction</h1>
<h2 id="prework">Prework</h2>
<h3 id="pandas"><a href="https://pandas.pydata.org/">pandas</a></h3>
<blockquote>
<p><a href="https://colab.research.google.com/">Colaboratory</a><br>
<a href="https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb?utm_source=mlcc&amp;utm_campaign=colab-external&amp;utm_medium=referral&amp;utm_content=pandas-colab&amp;hl=en#scrollTo=rHLcriKWLRe4">Quick Introduction to pandas</a></p>
</blockquote>
<p>The primary data structures in pandas are implemented as two classes:</p>
<ul>
<li><strong>DataFrame</strong>: a relational data table, with rows and named columns</li>
<li><strong>Series</strong>: a single column. A DataFrame contains one or more Series and a name for each Series.</li>
</ul>
<p>By default, at construction, pandas assigns index values(an identifier value to each Series item or DataFrame row) that reflect the ordering of the source data. Once created, the index values are stable; that is, they do not change when data is reordered.</p>
<p><strong>Feature column</strong>s store only a description of the feature data; they do not contain the feature data itself.</p>
<h3 id="tensorflow-basics">TensorFlow basics</h3>
<p><strong>概念</strong></p>
<ul>
<li>张量：任意维度的数组。</li>
<li>指令：创建、销毁和操控张量。</li>
<li>图（也称为计算图或数据流图）：图的节点是指令；图的边是张量。</li>
<li>会话：存储它所运行的图的状态。图必须在 TensorFlow 会话中运行，会话可以将图分发到多个机器上执行</li>
</ul>
<p>TensorFlow 会实现<strong>延迟执行</strong>模型，意味着系统仅会根据相关节点的需求在需要时计算节点。</p>
<p>张量可以作为常量或变量存储在图中。常量和变量都只是图中的一种指令。常量是始终会返回同一张量值的指令。变量是会返回分配给它的任何张量的指令。</p>
<p><strong>流程</strong><br>
TensorFlow 编程本质上是一个两步流程：</p>
<ul>
<li>将常量、变量和指令整合到一个图中。</li>
<li>在一个会话中评估这些常量、变量和指令。</li>
</ul>
<p><strong>组件</strong><br>
TensorFlow consists of the following two <strong>components</strong>:</p>
<ul>
<li>a  <a href="https://www.tensorflow.org/extend/tool_developers/#protocol_buffers">graph protocol buffer</a></li>
<li>a runtime that executes the (distributed) graph</li>
</ul>
<p>These two components are analogous to Python code and the Python interpreter.</p>
<h2 id="key-concepts-and-tools">Key Concepts and Tools</h2>
<h3 id="math">Math</h3>
<ul>
<li>Algebra</li>
<li>Linear algebra</li>
<li>Trigonometry</li>
<li>Statistics</li>
<li>Calculus</li>
</ul>
<h3 id="python">Python</h3>
<h4 id="tutorial"><a href="https://docs.python.org/3/tutorial/">Tutorial</a></h4>
<h4 id="libraries">Libraries</h4>
<ul>
<li>pandas (for data manipulation)</li>
<li>NumPy (for low-level math operations)</li>
<li>scikit-learn (for evaluation metrics)</li>
<li>Matplotlib (for data visualization)</li>
</ul>
<h3 id="command-line">Command line</h3>
<ul>
<li>Bash</li>
<li>Shell</li>
</ul>
<h1 id="ml-concepts">ML Concepts</h1>
<blockquote>
<p>800 min</p>
</blockquote>
<h2 id="framing">Framing</h2>
<h3 id="models">Models</h3>
<p>A model defines the relationship between <strong>features</strong> and <strong>label</strong>. Two phases of a model’s life:</p>
<ul>
<li><strong>Training</strong> means creating or learning the model. That is, you show the model labeled examples and enable the model to gradually learn the relationships between features and label.</li>
<li><strong>Inference</strong> means applying the trained model to unlabeled examples. That is, you use the trained model to make useful predictions (y’).</li>
</ul>
<h3 id="regression-vs.-classification">Regression vs. classification</h3>
<ul>
<li>A <strong>regression</strong> model predicts continuous values.</li>
<li>A <strong>classification</strong> model predicts discrete values.</li>
</ul>
<h2 id="descending-into-ml">Descending into ML</h2>
<h3 id="linear-regression">Linear Regression</h3>
<p><span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msup><mi>y</mi><msup><mrow></mrow><mo mathvariant="normal">′</mo></msup></msup><mo>=</mo><mi>b</mi><mo>+</mo><munder><mo>∑</mo><mrow><mi>i</mi><mo>∈</mo><mi>D</mi></mrow></munder><msub><mi>ω</mi><mi>i</mi></msub><mo>∗</mo><msub><mi>x</mi><mi>i</mi></msub></mrow><annotation encoding="application/x-tex">
y^{&amp;#x27;} = b + \sum_{i\in D} \omega_{i} * x_{i}
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.18692em; vertical-align: -0.19444em;"></span><span class="mord"><span class="mord mathit" style="margin-right: 0.03588em;">y</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.99248em;"><span class="" style="top: -2.99248em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.57948em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight"><span class=""></span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.827829em;"><span class="" style="top: -2.931em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight"><span class="mord mtight">′</span></span></span></span></span></span></span></span></span></span></span></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 0.77777em; vertical-align: -0.08333em;"></span><span class="mord mathit">b</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 2.37171em; vertical-align: -1.32171em;"></span><span class="mop op-limits"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.05001em;"><span class="" style="top: -1.85566em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight">i</span><span class="mrel mtight">∈</span><span class="mord mathit mtight" style="margin-right: 0.02778em;">D</span></span></span></span><span class="" style="top: -3.05em;"><span class="pstrut" style="height: 3.05em;"></span><span class=""><span class="mop op-symbol large-op">∑</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 1.32171em;"><span class=""></span></span></span></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathit" style="margin-right: 0.03588em;">ω</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.03588em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight">i</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">∗</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 0.58056em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathit">x</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight">i</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span></span><br>
where:</p>
<ul>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msup><mi>y</mi><msup><mrow></mrow><mo mathvariant="normal">′</mo></msup></msup></mrow><annotation encoding="application/x-tex">y^{&amp;#x27;}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.13692em; vertical-align: -0.19444em;"></span><span class="mord"><span class="mord mathit" style="margin-right: 0.03588em;">y</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.94248em;"><span class="" style="top: -2.94248em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.57948em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight"><span class=""></span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.827829em;"><span class="" style="top: -2.931em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight"><span class="mord mtight">′</span></span></span></span></span></span></span></span></span></span></span></span></span></span></span></span></span></span></span></span></span> is the predicted label(a desired output).</li>
<li>b  is the bias.</li>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>ω</mi><mi>i</mi></msub></mrow><annotation encoding="application/x-tex">\omega_i</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.58056em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathit" style="margin-right: 0.03588em;">ω</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.03588em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathit mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span>  is the weight of feature i.</li>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>x</mi><mi>i</mi></msub></mrow><annotation encoding="application/x-tex">x_i</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.58056em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathit">x</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathit mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span>  is a feature(a known input).</li>
<li>D  is a data set containing many labeled examples</li>
</ul>
<h3 id="training-and-loss">Training and Loss</h3>
<p>In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called <strong>empirical risk minimization(ERM)</strong>.</p>
<p><strong>Loss</strong> is a number indicating how bad the model’s prediction was on a single example. The goal of training a model is to find a set of weights and biases that have low loss, on average, across all examples.</p>
<p>Loss function: <strong>Squared loss</strong>  (<strong><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>L</mi><mn>2</mn></msub><mi>l</mi><mi>o</mi><mi>s</mi><mi>s</mi></mrow><annotation encoding="application/x-tex">L_2 loss</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.84444em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathit">L</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.301108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mord mathit" style="margin-right: 0.01968em;">l</span><span class="mord mathit">o</span><span class="mord mathit">s</span><span class="mord mathit">s</span></span></span></span></span></strong>) &amp; <strong>Mean square error(MSE)</strong><br>
MSE is the average squared loss per example over the whole dataset:<br>
<span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>M</mi><mi>S</mi><mi>E</mi><mo>=</mo><mfrac><mn>1</mn><mi>N</mi></mfrac><munder><mo>∑</mo><mrow><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo>∈</mo><mi>D</mi></mrow></munder><mo>(</mo><mi>y</mi><mo>−</mo><mi>p</mi><mi>r</mi><mi>e</mi><mi>d</mi><mi>i</mi><mi>c</mi><mi>t</mi><mi>i</mi><mi>o</mi><mi>n</mi><mo>(</mo><mi>x</mi><mo>)</mo><msup><mo>)</mo><mn>2</mn></msup></mrow><annotation encoding="application/x-tex">
MSE = \frac{1}{N}\sum_{x,y\in D} (y - prediction(x))^2
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.68333em; vertical-align: 0em;"></span><span class="mord mathit" style="margin-right: 0.10903em;">M</span><span class="mord mathit" style="margin-right: 0.05764em;">S</span><span class="mord mathit" style="margin-right: 0.05764em;">E</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 2.75188em; vertical-align: -1.43044em;"></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.32144em;"><span class="" style="top: -2.314em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord mathit" style="margin-right: 0.10903em;">N</span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.677em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord">1</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.686em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mop op-limits"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.05001em;"><span class="" style="top: -1.85566em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight">x</span><span class="mpunct mtight">,</span><span class="mord mathit mtight" style="margin-right: 0.03588em;">y</span><span class="mrel mtight">∈</span><span class="mord mathit mtight" style="margin-right: 0.02778em;">D</span></span></span></span><span class="" style="top: -3.05em;"><span class="pstrut" style="height: 3.05em;"></span><span class=""><span class="mop op-symbol large-op">∑</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 1.43044em;"><span class=""></span></span></span></span></span><span class="mopen">(</span><span class="mord mathit" style="margin-right: 0.03588em;">y</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1.11411em; vertical-align: -0.25em;"></span><span class="mord mathit">p</span><span class="mord mathit" style="margin-right: 0.02778em;">r</span><span class="mord mathit">e</span><span class="mord mathit">d</span><span class="mord mathit">i</span><span class="mord mathit">c</span><span class="mord mathit">t</span><span class="mord mathit">i</span><span class="mord mathit">o</span><span class="mord mathit">n</span><span class="mopen">(</span><span class="mord mathit">x</span><span class="mclose">)</span><span class="mclose"><span class="mclose">)</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.864108em;"><span class="" style="top: -3.113em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span></span></span></span></span></span></span></span></span></span><br>
where:</p>
<ul>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo>(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo>)</mo></mrow><annotation encoding="application/x-tex">(x,y)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mopen">(</span><span class="mord mathit">x</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathit" style="margin-right: 0.03588em;">y</span><span class="mclose">)</span></span></span></span></span>  is an example in which
<ul>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>x</mi></mrow><annotation encoding="application/x-tex">x</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathit">x</span></span></span></span></span>  is the set of features that the model uses to make predictions.</li>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>y</mi></mrow><annotation encoding="application/x-tex">y</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.19444em;"></span><span class="mord mathit" style="margin-right: 0.03588em;">y</span></span></span></span></span>  is the example’s label.</li>
</ul>
</li>
<li>prediction(x)  is a function of the weights and bias in combination with the set of features  x.</li>
<li>D  is a data set containing many labeled examples, which are  (x,y)  pairs.</li>
<li>N  is the number of examples in  D.</li>
</ul>
<p>Root Mean Square Error (RMSE):<br>
<span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>R</mi><mi>M</mi><mi>S</mi><mi>E</mi><mo>=</mo><msqrt><mrow><mi>M</mi><mi>S</mi><mi>E</mi></mrow></msqrt></mrow><annotation encoding="application/x-tex">
RMSE = \sqrt{MSE}
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.68333em; vertical-align: 0em;"></span><span class="mord mathit" style="margin-right: 0.00773em;">R</span><span class="mord mathit" style="margin-right: 0.10903em;">M</span><span class="mord mathit" style="margin-right: 0.05764em;">S</span><span class="mord mathit" style="margin-right: 0.05764em;">E</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1.04em; vertical-align: -0.06446em;"></span><span class="mord sqrt"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.97554em;"><span class="svg-align" style="top: -3em;"><span class="pstrut" style="height: 3em;"></span><span class="mord" style="padding-left: 0.833em;"><span class="mord mathit" style="margin-right: 0.10903em;">M</span><span class="mord mathit" style="margin-right: 0.05764em;">S</span><span class="mord mathit" style="margin-right: 0.05764em;">E</span></span></span><span class="" style="top: -2.93554em;"><span class="pstrut" style="height: 3em;"></span><span class="hide-tail" style="min-width: 0.853em; height: 1.08em;"><svg width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,
-10,-9.5,-14c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54c44.2,-33.3,65.8,
-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10s173,378,173,378c0.7,0,
35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429c69,-144,104.5,-217.7,106.5,
-221c5.3,-9.3,12,-14,20,-14H400000v40H845.2724s-225.272,467,-225.272,467
s-235,486,-235,486c-2.7,4.7,-9,7,-19,7c-6,0,-10,-1,-12,-3s-194,-422,-194,-422
s-65,47,-65,47z M834 80H400000v40H845z"></path></svg></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.06446em;"><span class=""></span></span></span></span></span></span></span></span></span></span></p>
<h2 id="reducing-loss">Reducing Loss</h2>
<h3 id="an-iterative-approach">An Iterative Approach</h3>
<p><img src="http://flowtime-linear-regression.soft.today/images/c30/iterative-approach.png" alt=""></p>
<ul>
<li>The “model” takes one or more features as input and returns one prediction (y’) as output.</li>
<li>The “Compute Loss” part of the diagram is the loss function that the model will use.</li>
<li>The “Compute parameter updates” part examines the value of the loss function and generates new values for bias and weight.</li>
</ul>
<p>And then the machine learning system re-evaluates all those features against all those labels, yielding a new value for the loss function, which yields new parameter values. Usually, you iterate until overall loss stops changing or at least changes extremely slowly. When that happens, we say that the model has <strong>converged</strong>.</p>
<h3 id="gradient-descent">Gradient Descent</h3>
<p><strong>Partial derivatives</strong><br>
Intuitively, a partial derivative tells you how much the function changes when you perturb one variable a bit.</p>
<p><strong>Gradients</strong><br>
The gradient of a function is <strong>the vector of partial derivatives with respect to all of the independent variables</strong>, the vector falls within the domain space of the function.</p>
<ul>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi mathvariant="normal">∇</mi><mi>f</mi></mrow><annotation encoding="application/x-tex">\nabla f</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.88888em; vertical-align: -0.19444em;"></span><span class="mord">∇</span><span class="mord mathit" style="margin-right: 0.10764em;">f</span></span></span></span></span>	Points in the direction of greatest <strong>increase</strong> of the function.</li>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo>−</mo><mi mathvariant="normal">∇</mi><mi>f</mi></mrow><annotation encoding="application/x-tex">-\nabla f</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.88888em; vertical-align: -0.19444em;"></span><span class="mord">−</span><span class="mord">∇</span><span class="mord mathit" style="margin-right: 0.10764em;">f</span></span></span></span></span>	Points in the direction of greatest <strong>decrease</strong> of the function. We often try to minimize the loss function by following the negative of the gradient of the function.</li>
</ul>
<h3 id="learning-rate">Learning Rate</h3>
<p>Gradient descent algorithms <strong>multiply the gradient by learning rate (step size)</strong> to determine the next point. The <strong>Goldilocks</strong> learning rate is related to how flat the loss function is.</p>
<p><strong><a href="https://www.quora.com/What-are-hyperparameters-in-machine-learning">Hyperparameters</a></strong> are the knobs that programmers tweak in machine learning algorithms.</p>
<h3 id="stochastic-gradient-descent">Stochastic Gradient Descent</h3>
<p>Stochastic gradient descent (SGD) uses only a single example (a batch size of 1) per iteration. Given enough iterations, SGD works but is very noisy. The term “stochastic” indicates that the one example is chosen at random.</p>
<p>Mini-batch stochastic gradient descent (mini-batch SGD) (typically between 10 and 1,000 examples) reduces the amount of noise in SGD but is still more efficient than full-batch.</p>
<h2 id="generalization">Generalization</h2>
<ul>
<li>We draw examples <strong>independently and identically</strong> (i.i.d) at random from the distribution. In other words, examples don’t influence each other.</li>
<li>The distribution is <strong>stationary</strong>; that is the distribution doesn’t change within the data set.</li>
<li>We draw examples from partitions from <strong>the same distribution</strong>.</li>
</ul>
<p>When we know that any of the preceding three basic assumptions are violated, we must pay careful attention to metrics.</p>
<p>Overfitting<br>
Ockham’s razor in machine learning terms: The less complex an ML model, the more likely that a good empirical result is not just due to the peculiarities of the sample.</p>
<p>Good performance on the test set is a useful indicator of good performance on the new data in general, assuming that the test set is large enough.</p>
<h2 id="training-validation-and-test-sets">Training, Validation and Test Sets</h2>
<p>You can greatly reduce your chances of overfitting by partitioning the data set into the three subsets: training set, validation set, test set.</p>
<p>Use the <strong>validation set</strong> to evaluate results from the <strong>training set</strong>. Then, use the <strong>test set</strong> to double-check your evaluation after the model has “passed” the validation set:<br>
<img src="https://developers.google.com/machine-learning/crash-course/images/WorkflowWithValidationSet.svg" alt=""></p>
<p><strong>Never train on test data!</strong></p>
<p>Debugging in ML is often data debugging rather than code debugging.</p>
<h2 id="representation">Representation</h2>
<h3 id="feature-engineering">Feature Engineering</h3>
<p>Feature engineering means transforming raw data into a feature vector.<br>
Since models cannot multiply strings by the learned weights, we use feature engineering to convert strings to numeric values.</p>
<p>OOV (out-of-vocabulary)</p>
<p><strong>Mapping categorical values</strong><br>
Discrete features are usually converted into families of binary features before training a logistic regression model.<br>
The representation is called <strong>one-hot encoding</strong> when a single value is 1, and a <strong>multi-hot encoding</strong> when multiple values are 1.</p>
<p><strong>Sparse Representation</strong>: only nonzero values are stored.</p>
<h3 id="qualities-of-good-features">Qualities of Good Features</h3>
<ul>
<li>Avoid rarely used discrete feature values. Good feature values should appear more than 5 or so times in a data set.</li>
<li>Prefer clear and obvious meanings.</li>
<li>Don’t mix “magic” values with actual data. Good floating-point features don’t contain peculiar out-of-range discontinuities or “magic” values.</li>
<li>Account for upstream instability. The definition of a feature shouldn’t change over time.</li>
</ul>
<h3 id="cleaning-data">Cleaning Data</h3>
<blockquote>
<p>Good ML relies on good data.<br>
<a href="https://developers.google.com/machine-learning/guides/rules-of-ml/#ml_phase_ii_feature_engineering">Rules of Machine Learning, ML Phase II: Feature Engineering</a></p>
</blockquote>
<p><strong>Scaling feature values</strong><br>
Scaling means converting floating-point feature values from their natural range into a standard range</p>
<ul>
<li><strong>linearly map</strong> to a small scale</li>
<li>calculate the <strong>Z score</strong> of each value: <code>scaledvalue = (value - mean) / stddev</code></li>
</ul>
<p>If a feature set consists of multiple features, then feature scaling provides the following benefits:</p>
<ul>
<li>Helps gradient descent converge more quickly.</li>
<li>Helps avoid the “NaN trap”</li>
<li>Helps the model learn appropriate weights for each feature. Without feature scaling, the model will pay too much attention to the features having a wider range.</li>
</ul>
<p><strong>Handling extreme outliers</strong></p>
<ul>
<li><strong>take the log</strong> of every value</li>
<li><strong>clip the maximum value</strong>. All values that were greater than max value now become max value</li>
</ul>
<p><strong>Binning</strong><br>
With binning, our model can now learn completely different weights for each bin.</p>
<p>Another approach is to <strong>bin by quantile</strong>, which ensures that the number of examples in each bucket is equal. Binning by quantile completely removes the need to worry about outliers.</p>
<p><strong>Scrubbing</strong><br>
Removing bad examples from the data set:</p>
<ul>
<li>Omitted values</li>
<li>Duplicate examples</li>
<li>Bad labels</li>
<li>Bad feature values</li>
</ul>
<h2 id="feature-crosses">Feature Crosses</h2>
<p>A <strong>feature cross</strong> is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together. (The term cross comes from cross product.)</p>
<p><strong>Linear learners scale well to massive data</strong>. Supplementing scaled linear models with feature crosses has traditionally been an efficient way to train on massive-scale data sets.</p>
<h3 id="crossing-one-hot-vectors">Crossing One-Hot Vectors</h3>
<p>Machine learning models do frequently cross one-hot feature vectors. Think of feature crosses of one-hot feature vectors as logical conjunctions.</p>
<p>Using feature crosses on massive data sets is one efficient strategy for <strong>learning highly complex models.</strong> Neural networks provide another strategy.</p>
<p><a href="https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf"><strong>FTRL Algorithm</strong></a></p>
<h2 id="regularization-for-simplicity">Regularization for Simplicity</h2>
<p><strong>Regularization</strong> means penalizing the complexity of a model to reduce overfitting.</p>
<h3 id="l₂-regularization">L₂ Regularization</h3>
<p><strong>Structural risk minimization</strong>: <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>m</mi><mi>i</mi><mi>n</mi><mi>i</mi><mi>m</mi><mi>i</mi><mi>z</mi><mi>e</mi><mo>(</mo><mi>L</mi><mi>o</mi><mi>s</mi><mi>s</mi><mo>(</mo><mi>D</mi><mi>a</mi><mi>t</mi><mi>a</mi><mi mathvariant="normal">∣</mi><mi>M</mi><mi>o</mi><mi>d</mi><mi>e</mi><mi>l</mi><mo>)</mo><mo>+</mo><mi>λ</mi><mo>⋅</mo><mi>c</mi><mi>o</mi><mi>m</mi><mi>p</mi><mi>l</mi><mi>e</mi><mi>c</mi><mi>i</mi><mi>t</mi><mi>y</mi><mo>(</mo><mi>M</mi><mi>o</mi><mi>d</mi><mi>e</mi><mi>l</mi><mo>)</mo><mo>)</mo></mrow><annotation encoding="application/x-tex">minimize(Loss(Data|Model) + \lambda\cdot complecity(Model))</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathit">m</span><span class="mord mathit">i</span><span class="mord mathit">n</span><span class="mord mathit">i</span><span class="mord mathit">m</span><span class="mord mathit">i</span><span class="mord mathit" style="margin-right: 0.04398em;">z</span><span class="mord mathit">e</span><span class="mopen">(</span><span class="mord mathit">L</span><span class="mord mathit">o</span><span class="mord mathit">s</span><span class="mord mathit">s</span><span class="mopen">(</span><span class="mord mathit" style="margin-right: 0.02778em;">D</span><span class="mord mathit">a</span><span class="mord mathit">t</span><span class="mord mathit">a</span><span class="mord">∣</span><span class="mord mathit" style="margin-right: 0.10903em;">M</span><span class="mord mathit">o</span><span class="mord mathit">d</span><span class="mord mathit">e</span><span class="mord mathit" style="margin-right: 0.01968em;">l</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathit">λ</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">⋅</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathit">c</span><span class="mord mathit">o</span><span class="mord mathit">m</span><span class="mord mathit">p</span><span class="mord mathit" style="margin-right: 0.01968em;">l</span><span class="mord mathit">e</span><span class="mord mathit">c</span><span class="mord mathit">i</span><span class="mord mathit">t</span><span class="mord mathit" style="margin-right: 0.03588em;">y</span><span class="mopen">(</span><span class="mord mathit" style="margin-right: 0.10903em;">M</span><span class="mord mathit">o</span><span class="mord mathit">d</span><span class="mord mathit">e</span><span class="mord mathit" style="margin-right: 0.01968em;">l</span><span class="mclose">)</span><span class="mclose">)</span></span></span></span></span></p>
<ul>
<li>the <strong>loss term</strong>, which measures how well the model fits the data</li>
<li>the <strong>regularization term</strong>, which measures model complexity</li>
</ul>
<p>Two common (and somewhat related) ways to think of model complexity:</p>
<ul>
<li>Model complexity as a function of the <strong>weights</strong> of all the features in the model.</li>
<li>Model complexity as a function of the total <strong>number of features</strong> with nonzero weights.</li>
</ul>
<p>L2 regularization formula:<br>
<span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>L</mi><mn>2</mn></msub><mtext>&nbsp;</mtext><mi>R</mi><mi>e</mi><mi>g</mi><mi>u</mi><mi>l</mi><mi>a</mi><mi>r</mi><mi>i</mi><mi>z</mi><mi>a</mi><mi>t</mi><mi>i</mi><mi>o</mi><mi>n</mi><mtext>&nbsp;</mtext><mi>t</mi><mi>e</mi><mi>r</mi><mi>m</mi><mo>=</mo><mo>∥</mo><mi mathvariant="bold-italic">w</mi><msubsup><mo>∥</mo><mn>2</mn><mn>2</mn></msubsup></mrow><annotation encoding="application/x-tex">
L_{2}\space Regularization\space term = \parallel\bm{w}\parallel_{2}^{2}
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.88888em; vertical-align: -0.19444em;"></span><span class="mord"><span class="mord mathit">L</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.301108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">2</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mspace">&nbsp;</span><span class="mord mathit" style="margin-right: 0.00773em;">R</span><span class="mord mathit">e</span><span class="mord mathit" style="margin-right: 0.03588em;">g</span><span class="mord mathit">u</span><span class="mord mathit" style="margin-right: 0.01968em;">l</span><span class="mord mathit">a</span><span class="mord mathit" style="margin-right: 0.02778em;">r</span><span class="mord mathit">i</span><span class="mord mathit" style="margin-right: 0.04398em;">z</span><span class="mord mathit">a</span><span class="mord mathit">t</span><span class="mord mathit">i</span><span class="mord mathit">o</span><span class="mord mathit">n</span><span class="mspace">&nbsp;</span><span class="mord mathit">t</span><span class="mord mathit">e</span><span class="mord mathit" style="margin-right: 0.02778em;">r</span><span class="mord mathit">m</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mrel">∥</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1.11411em; vertical-align: -0.25em;"></span><span class="mord"><span class="mord"><span class="mord boldsymbol" style="margin-right: 0.02778em;">w</span></span></span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel"><span class="mrel">∥</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.864108em;"><span class="" style="top: -2.453em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">2</span></span></span></span><span class="" style="top: -3.113em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">2</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.247em;"><span class=""></span></span></span></span></span></span></span></span></span></span></span></p>
<h3 id="lambdaregularization-rate">Lambda(regularization rate)</h3>
<p>Performing L2 regularization has the following effect on a model:</p>
<ul>
<li>Encourages weight values toward 0 (but not exactly 0)</li>
<li>Encourages the mean of the weights toward 0, with a normal (bell-shaped or Gaussian) distribution.</li>
</ul>
<p>Increasing the lambda value strengthens the regularization effect.</p>
<p>When choosing a lambda value, the goal is to strike the right balance between simplicity and training-data fit:</p>
<ul>
<li>lambda value is too high -&gt; simple model -&gt; underfitting</li>
<li>lambda value is too low -&gt; complex model -&gt; overfitting</li>
</ul>
<p><strong>Early stopping</strong> means ending training before the model fully reaches convergence.</p>
<p>The effects from changes to regularization parameters can be confounded with the effects from changes in learning rate or number of iterations.<br>
One useful practice (when training across a fixed batch of data) is to give yourself a high enough number of iterations that early stopping doesn’t play into things.</p>
<p><strong>Test loss</strong> is the true measure of the model’s ability to make good predictions on new data.</p>
<h2 id="logistic-regression">Logistic Regression</h2>
<p>Many problems require a probability estimate as output. Logistic regression is an extremely efficient mechanism for calculating probabilities.</p>
<p><strong>Loss function</strong><br>
The loss function for linear regression is squared loss. The loss function for logistic regression is <strong>Log Loss</strong>:<br>
<span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>L</mi><mi>o</mi><mi>g</mi><mtext>&nbsp;</mtext><mi>L</mi><mi>o</mi><mi>s</mi><mi>s</mi><mo>=</mo><munder><mo>∑</mo><mrow><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo>∈</mo><mi>D</mi></mrow></munder><mo>−</mo><mi>y</mi><mi>log</mi><mo>⁡</mo><msup><mi>y</mi><mo mathvariant="normal">′</mo></msup><mo>−</mo><mo>(</mo><mn>1</mn><mo>−</mo><mi>y</mi><mo>)</mo><mi>log</mi><mo>⁡</mo><mo>(</mo><mn>1</mn><mo>−</mo><msup><mi>y</mi><mo mathvariant="normal">′</mo></msup><mo>)</mo></mrow><annotation encoding="application/x-tex">
Log\space Loss = \sum_{x,y \in D} -y\log{y&amp;#x27;} - (1-y)\log(1-y&amp;#x27;)
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.87777em; vertical-align: -0.19444em;"></span><span class="mord mathit">L</span><span class="mord mathit">o</span><span class="mord mathit" style="margin-right: 0.03588em;">g</span><span class="mspace">&nbsp;</span><span class="mord mathit">L</span><span class="mord mathit">o</span><span class="mord mathit">s</span><span class="mord mathit">s</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 2.48045em; vertical-align: -1.43044em;"></span><span class="mop op-limits"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.05001em;"><span class="" style="top: -1.85566em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight">x</span><span class="mpunct mtight">,</span><span class="mord mathit mtight" style="margin-right: 0.03588em;">y</span><span class="mrel mtight">∈</span><span class="mord mathit mtight" style="margin-right: 0.02778em;">D</span></span></span></span><span class="" style="top: -3.05em;"><span class="pstrut" style="height: 3.05em;"></span><span class=""><span class="mop op-symbol large-op">∑</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 1.43044em;"><span class=""></span></span></span></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord">−</span><span class="mord mathit" style="margin-right: 0.03588em;">y</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mop">lo<span style="margin-right: 0.01389em;">g</span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord"><span class="mord mathit" style="margin-right: 0.03588em;">y</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.801892em;"><span class="" style="top: -3.113em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">′</span></span></span></span></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mopen">(</span><span class="mord">1</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathit" style="margin-right: 0.03588em;">y</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mop">lo<span style="margin-right: 0.01389em;">g</span></span><span class="mopen">(</span><span class="mord">1</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1.05189em; vertical-align: -0.25em;"></span><span class="mord"><span class="mord mathit" style="margin-right: 0.03588em;">y</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.801892em;"><span class="" style="top: -3.113em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">′</span></span></span></span></span></span></span></span></span><span class="mclose">)</span></span></span></span></span></span></p>
<ul>
<li>every value of <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>y</mi></mrow><annotation encoding="application/x-tex">y</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.19444em;"></span><span class="mord mathit" style="margin-right: 0.03588em;">y</span></span></span></span></span> must either be 0 or 1.</li>
</ul>
<p>Indeed, minimizing the loss function yields a maximum likelihood estimate.</p>
<p><strong>Regularization in Logistic Regression</strong></p>
<p>Without regularization, the asymptotic nature of logistic regression would keep driving loss towards 0 in high dimensions?<br>
Strategies to dampen model complexity:</p>
<ul>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>L</mi><mn>2</mn></msub></mrow><annotation encoding="application/x-tex">L_2</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.83333em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathit">L</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.301108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span> regularization</li>
<li>Early stopping</li>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>L</mi><mn>1</mn></msub></mrow><annotation encoding="application/x-tex">L_1</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.83333em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathit">L</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.301108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">1</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span> regularization</li>
</ul>
<h2 id="classification">Classification</h2>
<h3 id="thresholding">Thresholding</h3>
<p>Logistic regression returns a probability.</p>
<p>In order to map a logistic regression value to a binary category, you must define a classification threshold (also called the decision threshold).</p>
<h3 id="true-vs.-false-and-positive-vs.-negative">True vs. False and Positive vs. Negative</h3>
<p><a href="https://developers.google.com/machine-learning/glossary#confusion_matrix">confusion matrix</a>：row为预测值，column为真实值.</p>
<ul>
<li>A <strong>true positive(TP)</strong> is an outcome where the model correctly predicts the positive class.</li>
<li>A <strong>true negative(TN)</strong> is an outcome where the model correctly predicts the negative class.</li>
<li>A <strong>false positive(FP)</strong> is an outcome where the model incorrectly predicts the positive class.</li>
<li>A <strong>false negative(FN)</strong> is an outcome where the model incorrectly predicts the negative class.</li>
</ul>
<h3 id="accuracy">Accuracy</h3>
<p>Informally, accuracy is the fraction of predictions our model got right.<br>
Accuracy alone doesn’t tell the full story when you’re working with a class-imbalanced data set.</p>
<h3 id="precision-and-recall">Precision and Recall</h3>
<p>Precision: What proportion of positive identifications was actually correct?<br>
<span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>P</mi><mi>r</mi><mi>e</mi><mi>c</mi><mi>i</mi><mi>s</mi><mi>i</mi><mi>o</mi><mi>n</mi><mo>=</mo><mfrac><mrow><mi>T</mi><mi>P</mi></mrow><mrow><mi>T</mi><mi>P</mi><mo>+</mo><mi>F</mi><mi>P</mi></mrow></mfrac><mi mathvariant="normal">（</mi><mi mathvariant="normal">以</mi><mi mathvariant="normal">第</mi><mn>1</mn><mi mathvariant="normal">行</mi><mi mathvariant="normal">为</mi><mi mathvariant="normal">基</mi><mi mathvariant="normal">准</mi><mi mathvariant="normal">）</mi></mrow><annotation encoding="application/x-tex">
Precision = \frac{TP}{TP + FP}（以第1行为基准）
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.68333em; vertical-align: 0em;"></span><span class="mord mathit" style="margin-right: 0.13889em;">P</span><span class="mord mathit" style="margin-right: 0.02778em;">r</span><span class="mord mathit">e</span><span class="mord mathit">c</span><span class="mord mathit">i</span><span class="mord mathit">s</span><span class="mord mathit">i</span><span class="mord mathit">o</span><span class="mord mathit">n</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 2.12966em; vertical-align: -0.76933em;"></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.36033em;"><span class="" style="top: -2.314em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord mathit" style="margin-right: 0.13889em;">T</span><span class="mord mathit" style="margin-right: 0.13889em;">P</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mord mathit" style="margin-right: 0.13889em;">F</span><span class="mord mathit" style="margin-right: 0.13889em;">P</span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.677em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord mathit" style="margin-right: 0.13889em;">T</span><span class="mord mathit" style="margin-right: 0.13889em;">P</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.76933em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span><span class="mord cjk_fallback">（</span><span class="mord cjk_fallback">以</span><span class="mord cjk_fallback">第</span><span class="mord">1</span><span class="mord cjk_fallback">行</span><span class="mord cjk_fallback">为</span><span class="mord cjk_fallback">基</span><span class="mord cjk_fallback">准</span><span class="mord cjk_fallback">）</span></span></span></span></span></span></p>
<p>Recall: What proportion of actual positives was identified correctly?<br>
<span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>R</mi><mi>e</mi><mi>c</mi><mi>a</mi><mi>l</mi><mi>l</mi><mo>=</mo><mfrac><mrow><mi>T</mi><mi>P</mi></mrow><mrow><mi>T</mi><mi>P</mi><mo>+</mo><mi>F</mi><mi>N</mi></mrow></mfrac><mi mathvariant="normal">（</mi><mi mathvariant="normal">以</mi><mi mathvariant="normal">第</mi><mn>1</mn><mi mathvariant="normal">列</mi><mi mathvariant="normal">为</mi><mi mathvariant="normal">基</mi><mi mathvariant="normal">准</mi><mi mathvariant="normal">）</mi></mrow><annotation encoding="application/x-tex">
Recall = \frac{TP}{TP + FN}（以第1列为基准）
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathit" style="margin-right: 0.00773em;">R</span><span class="mord mathit">e</span><span class="mord mathit">c</span><span class="mord mathit">a</span><span class="mord mathit" style="margin-right: 0.01968em;">l</span><span class="mord mathit" style="margin-right: 0.01968em;">l</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 2.12966em; vertical-align: -0.76933em;"></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.36033em;"><span class="" style="top: -2.314em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord mathit" style="margin-right: 0.13889em;">T</span><span class="mord mathit" style="margin-right: 0.13889em;">P</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mord mathit" style="margin-right: 0.13889em;">F</span><span class="mord mathit" style="margin-right: 0.10903em;">N</span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.677em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord mathit" style="margin-right: 0.13889em;">T</span><span class="mord mathit" style="margin-right: 0.13889em;">P</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.76933em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span><span class="mord cjk_fallback">（</span><span class="mord cjk_fallback">以</span><span class="mord cjk_fallback">第</span><span class="mord">1</span><span class="mord cjk_fallback">列</span><span class="mord cjk_fallback">为</span><span class="mord cjk_fallback">基</span><span class="mord cjk_fallback">准</span><span class="mord cjk_fallback">）</span></span></span></span></span></span></p>
<p>To fully evaluate the effectiveness of a model, you must examine <strong>both</strong> precision and recall.</p>
<p><a href="https://wikipedia.org/wiki/F1_score">F1 score</a></p>
<h3 id="roc-and-auc">ROC and AUC</h3>
<p>An ROC(receiver operating characteristic curve) curve plots two parameters:</p>
<ul>
<li>True Positive Rate(TPR): a synonym for recall. The y-axis</li>
<li>False Positive Rate(FPR): <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>F</mi><mi>P</mi><mi>R</mi><mo>=</mo><mfrac><mrow><mi>F</mi><mi>P</mi></mrow><mrow><mi>F</mi><mi>P</mi><mo>+</mo><mi>T</mi><mi>N</mi></mrow></mfrac></mrow><annotation encoding="application/x-tex">FPR = \frac{FP}{FP + TN}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.68333em; vertical-align: 0em;"></span><span class="mord mathit" style="margin-right: 0.13889em;">F</span><span class="mord mathit" style="margin-right: 0.13889em;">P</span><span class="mord mathit" style="margin-right: 0.00773em;">R</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1.27566em; vertical-align: -0.403331em;"></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.872331em;"><span class="" style="top: -2.655em;"><span class="pstrut" style="height: 3em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight" style="margin-right: 0.13889em;">F</span><span class="mord mathit mtight" style="margin-right: 0.13889em;">P</span><span class="mbin mtight">+</span><span class="mord mathit mtight" style="margin-right: 0.13889em;">T</span><span class="mord mathit mtight" style="margin-right: 0.10903em;">N</span></span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.394em;"><span class="pstrut" style="height: 3em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight" style="margin-right: 0.13889em;">F</span><span class="mord mathit mtight" style="margin-right: 0.13889em;">P</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.403331em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span></span></span></span></span>（以第2列为基准）. The x-axis</li>
</ul>
<p>One way of interpreting AUC(Area Under the ROC Curve) is as <strong>the probability that the model ranks a random positive example more highly than a random negative example</strong>.</p>
<p>AUC limits</p>
<ul>
<li>Scale invariance is not always desirable.</li>
<li>Classification-threshold invariance is not always desirable. In cases where there are wide disparities in the cost of false negatives vs. false positives</li>
</ul>
<h3 id="prediction-bias">Prediction Bias</h3>
<p>prediction bias = average of predictions - average of labels in data set</p>
<p>If possible, avoid calibration layers.</p>
<p>A good model will usually have near-zero bias. That said, a low prediction bias does not prove that your model is good. A really terrible model could have a zero prediction bias.</p>
<h2 id="regularization-for-sparsity">Regularization for Sparsity</h2>
<p><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>L</mi><mn>0</mn></msub></mrow><annotation encoding="application/x-tex">L_0</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.83333em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathit">L</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.301108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">0</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span> regularization: penalizes the count of non-zero coefficient values in a model. It would turn our convex optimization problem into a non-convex optimization problem that’s NP-hard. It isn’t something we can use effectively in practice.</p>
<h3 id="l₁-regularization">L₁ Regularization</h3>
<p>L₁ vs L2 regularization:</p>
<ul>
<li>L₁ penalizes <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi mathvariant="normal">∣</mi><mi>w</mi><mi>e</mi><mi>i</mi><mi>g</mi><mi>h</mi><mi>t</mi><mi mathvariant="normal">∣</mi></mrow><annotation encoding="application/x-tex">|weight|</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord">∣</span><span class="mord mathit" style="margin-right: 0.02691em;">w</span><span class="mord mathit">e</span><span class="mord mathit">i</span><span class="mord mathit" style="margin-right: 0.03588em;">g</span><span class="mord mathit">h</span><span class="mord mathit">t</span><span class="mord">∣</span></span></span></span></span>. increase sparsity -&gt; reduce the size of a model -&gt; may affect loss
<ul>
<li>L₁ Regularization turns out to be quite efficient for <strong>wide models</strong>.</li>
<li>You can think of the derivative of L₁ as a force that subtracts some constant from the weight every time.</li>
<li>However, thanks to absolute values, L₁ has a discontinuity at 0, which causes subtraction results that cross 0 to become zeroed out.</li>
</ul>
</li>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>L</mi><mn>2</mn></msub></mrow><annotation encoding="application/x-tex">L_2</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.83333em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathit">L</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.301108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span> penalizes <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>w</mi><mi>e</mi><mi>i</mi><mi>g</mi><mi>h</mi><msup><mi>t</mi><mn>2</mn></msup></mrow><annotation encoding="application/x-tex">weight^2</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.00855em; vertical-align: -0.19444em;"></span><span class="mord mathit" style="margin-right: 0.02691em;">w</span><span class="mord mathit">e</span><span class="mord mathit">i</span><span class="mord mathit" style="margin-right: 0.03588em;">g</span><span class="mord mathit">h</span><span class="mord"><span class="mord mathit">t</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.814108em;"><span class="" style="top: -3.063em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span></span></span></span></span></span></span></span></span>
<ul>
<li>You can think of the derivative of <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>L</mi><mn>2</mn></msub></mrow><annotation encoding="application/x-tex">L_2</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.83333em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathit">L</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.301108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span> as a force that removes x% of the weight every time.</li>
</ul>
</li>
</ul>
<h2 id="neural-network">Neural Network</h2>
<h3 id="introduction-to-neural-networks">Introduction to Neural Networks</h3>
<p>Neural networks are a more sophisticated version of feature crosses. In essence, neural networks learn the appropriate feature crosses.</p>
<p>Activation Functions: the nonlinear function transforms the value of each node in Hidden Layer, and then pass on to the weighted sums of the next layer.</p>
<ul>
<li>sigmoid function: such as <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>F</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>x</mi></mrow></msup></mrow></mfrac></mrow><annotation encoding="application/x-tex">F(x) = \frac{1}{1 + e^{-x}}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathit" style="margin-right: 0.13889em;">F</span><span class="mopen">(</span><span class="mord mathit">x</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1.24844em; vertical-align: -0.403331em;"></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.845108em;"><span class="" style="top: -2.655em;"><span class="pstrut" style="height: 3em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">1</span><span class="mbin mtight">+</span><span class="mord mtight"><span class="mord mathit mtight">e</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.702664em;"><span class="" style="top: -2.786em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight"><span class="mord mtight">−</span><span class="mord mathit mtight">x</span></span></span></span></span></span></span></span></span></span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.394em;"><span class="pstrut" style="height: 3em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">1</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.403331em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span></span></span></span></span></li>
<li>rectified linear unit activation function (or <strong>ReLU</strong>): <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>F</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mi>m</mi><mi>a</mi><mi>x</mi><mo>(</mo><mn>0</mn><mo separator="true">,</mo><mi>x</mi><mo>)</mo></mrow><annotation encoding="application/x-tex">F(x) = max(0, x)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathit" style="margin-right: 0.13889em;">F</span><span class="mopen">(</span><span class="mord mathit">x</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathit">m</span><span class="mord mathit">a</span><span class="mord mathit">x</span><span class="mopen">(</span><span class="mord">0</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathit">x</span><span class="mclose">)</span></span></span></span></span>. The <strong>superiority</strong> of ReLU is based on empirical findings, probably driven by ReLU having a more useful range of responsiveness.</li>
</ul>
<p>Stacking nonlinearities on nonlinearities lets us model very complicated relationships between the inputs and the predicted outputs.</p>
<p>Even with Neural Nets, some amount of feature engineering is often needed to achieve best performance.</p>
<p>You can look at the gap between loss on training data and loss on validation data to help judge if your model is starting to overfit. If the gap starts to grow, that is usually a sure sign of overfitting.</p>
<p><a href="https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/">Neural Networks, Manifolds, and Topology</a>（<a href="https://zcao.info/2017/10/09/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E3%80%81%E6%B5%81%E5%BD%A2%E3%80%81%E6%8B%93%E6%89%91/">翻译</a>）</p>
<ol>
<li><a href="https://cs.stanford.edu/people/karpathy/convnetjs//demo/classify2d.html">ConvnetJS demo</a></li>
<li>Homeomorphisms（同态）: preserves topological properties. Each layer stretches and squishes space, but it never cuts, breaks, or folds it.</li>
<li>…</li>
</ol>
<h3 id="training-neural-networks">Training Neural Networks</h3>
<p><strong>Backpropagation</strong> makes gradient descent feasible for multi-layer neural networks.</p>
<ul>
<li><a href="https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/">工作原理</a></li>
</ul>
<p>Best Practices</p>
<ul>
<li>Vanishing Gradients
<ul>
<li>When the gradients vanish toward 0 for the lower layers, these layers train very slowly, or not at all.</li>
<li>The <strong>ReLU activation function</strong> can help prevent vanishing gradients.</li>
</ul>
</li>
<li>Exploding Gradients
<ul>
<li>If the weights in a network are very large, then the gradients for the lower layers involve products of many large terms.</li>
<li><strong>Batch normalization</strong> can help prevent exploding gradients, as can lowering the learning rate.</li>
</ul>
</li>
<li>Dead ReLU Units
<ul>
<li>Once the weighted sum for a ReLU unit falls below 0, the ReLU unit can get stuck.</li>
<li><strong>Lowering the learning rate</strong> can help keep ReLU units from dying.</li>
</ul>
</li>
</ul>
<p>Yet another form of regularization useful for neural networks, called <strong>Dropout</strong>. It works by randomly “dropping out” unit activations in a network for a single gradient step.</p>
<p><strong>Normalization Methods</strong></p>
<ul>
<li><strong>Linear Scaling</strong> (as a rule of thumb, NN’s train best when the input features are roughly on the same scale):<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">linear_scale</span><span class="token punctuation">(</span>series<span class="token punctuation">)</span><span class="token punctuation">:</span>
  min_val <span class="token operator">=</span> series<span class="token punctuation">.</span><span class="token builtin">min</span><span class="token punctuation">(</span><span class="token punctuation">)</span>
  max_val <span class="token operator">=</span> series<span class="token punctuation">.</span><span class="token builtin">max</span><span class="token punctuation">(</span><span class="token punctuation">)</span>
  scale <span class="token operator">=</span> <span class="token punctuation">(</span>max_val <span class="token operator">-</span> min_val<span class="token punctuation">)</span> <span class="token operator">/</span> <span class="token number">2.0</span>
  <span class="token keyword">return</span> series<span class="token punctuation">.</span><span class="token builtin">apply</span><span class="token punctuation">(</span><span class="token keyword">lambda</span> x<span class="token punctuation">:</span><span class="token punctuation">(</span><span class="token punctuation">(</span>x <span class="token operator">-</span> min_val<span class="token punctuation">)</span> <span class="token operator">/</span> scale<span class="token punctuation">)</span> <span class="token operator">-</span> <span class="token number">1.0</span><span class="token punctuation">)</span>
</code></pre>
</li>
<li>Adagrad optimizer: works great for <strong>convex problems</strong>. The key insight of Adagrad is that it modifies the learning rate adaptively for each coefficient in a model, monotonically lowering the effective learning rate</li>
<li>Adam optimizer: for <strong>non-convex  problems</strong></li>
<li>log scaling</li>
<li>clipping extreme values</li>
</ul>
<p>扩展：<br>
<a href="http://ruder.io/optimizing-gradient-descent/">An overview of gradient descent optimization algorithms</a></p>
<ol>
<li></li>
</ol>
<h3 id="multi-class-neural-networks">Multi-Class Neural Networks</h3>
<p><strong>One vs. All</strong><br>
Given a classification problem with N possible solutions, a one-vs.-all solution consists of <strong>N separate binary classifiers</strong>—one binary classifier for each possible outcome.</p>
<p>This approach is fairly reasonable when the total number of classes is small, but becomes increasingly inefficient as the number of classes rises.</p>
<p>We can create a significantly more efficient one-vs.-all model with a deep neural network in which <strong>each output node represents a different class</strong>.</p>
<p><strong>Softmax</strong></p>
<p>The Softmax equation:<br>
<span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>p</mi><mo>(</mo><mi>y</mi><mo>=</mo><mi>j</mi><mi mathvariant="normal">∣</mi><mi mathvariant="bold-italic">x</mi><mo>)</mo><mo>=</mo><mfrac><msup><mi>e</mi><mrow><msubsup><mi mathvariant="bold-italic">w</mi><mi>j</mi><mi>T</mi></msubsup><mi mathvariant="bold-italic">x</mi><mo>+</mo><msub><mi>b</mi><mi>j</mi></msub></mrow></msup><mrow><munder><mo>∑</mo><mrow><mi>k</mi><mo>∈</mo><mi>K</mi></mrow></munder><msup><mi>e</mi><mrow><msubsup><mi mathvariant="bold-italic">w</mi><mi>k</mi><mi>T</mi></msubsup><mi mathvariant="bold-italic">x</mi><mo>+</mo><msub><mi>b</mi><mi>k</mi></msub></mrow></msup></mrow></mfrac></mrow><annotation encoding="application/x-tex">
p(y=j | \bm{x}) = \frac{e^{\bm{w}_j^{T}\bm{x} + b_j}}{\sum_{k\in K}e^{\bm{w}_k^T\bm{x} + b_k}}
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathit">p</span><span class="mopen">(</span><span class="mord mathit" style="margin-right: 0.03588em;">y</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathit" style="margin-right: 0.05724em;">j</span><span class="mord">∣</span><span class="mord"><span class="mord"><span class="mord boldsymbol">x</span></span></span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 2.87645em; vertical-align: -1.15091em;"></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.72554em;"><span class="" style="top: -2.2247em;"><span class="pstrut" style="height: 3.04854em;"></span><span class="mord"><span class="mop"><span class="mop op-symbol small-op" style="position: relative; top: -5e-06em;">∑</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.186398em;"><span class="" style="top: -2.40029em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight" style="margin-right: 0.03148em;">k</span><span class="mrel mtight">∈</span><span class="mord mathit mtight" style="margin-right: 0.07153em;">K</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.32708em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathit">e</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.933835em;"><span class="" style="top: -3.05081em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight"><span class="mord mtight"><span class="mord mtight"><span class="mord boldsymbol mtight" style="margin-right: 0.02778em;">w</span></span></span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.832893em;"><span class="" style="top: -2.15277em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathit mtight" style="margin-right: 0.03148em;">k</span></span></span><span class="" style="top: -2.8448em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathit mtight" style="margin-right: 0.13889em;">T</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.347229em;"><span class=""></span></span></span></span></span></span><span class="mord mtight"><span class="mord mtight"><span class="mord boldsymbol mtight">x</span></span></span><span class="mbin mtight">+</span><span class="mord mtight"><span class="mord mathit mtight">b</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.3448em;"><span class="" style="top: -2.34877em; margin-left: 0em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathit mtight" style="margin-right: 0.03148em;">k</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.151229em;"><span class=""></span></span></span></span></span></span></span></span></span></span></span></span></span></span></span></span><span class="" style="top: -3.27854em;"><span class="pstrut" style="height: 3.04854em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.72554em;"><span class="pstrut" style="height: 3.04854em;"></span><span class="mord"><span class="mord"><span class="mord mathit">e</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 1.04853em;"><span class="" style="top: -3.10517em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight"><span class="mord mtight"><span class="mord mtight"><span class="mord boldsymbol mtight" style="margin-right: 0.02778em;">w</span></span></span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.919093em;"><span class="" style="top: -2.214em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathit mtight" style="margin-right: 0.05724em;">j</span></span></span><span class="" style="top: -2.931em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight"><span class="mord mathit mtight" style="margin-right: 0.13889em;">T</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.424886em;"><span class=""></span></span></span></span></span></span><span class="mord mtight"><span class="mord mtight"><span class="mord boldsymbol mtight">x</span></span></span><span class="mbin mtight">+</span><span class="mord mtight"><span class="mord mathit mtight">b</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.328086em;"><span class="" style="top: -2.357em; margin-left: 0em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathit mtight" style="margin-right: 0.05724em;">j</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.281886em;"><span class=""></span></span></span></span></span></span></span></span></span></span></span></span></span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 1.15091em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span></span></span></span></span></span><br>
This formula basically <strong>extends the formula for logistic regression into multiple classes</strong>.</p>
<p>Softmax assigns decimal probabilities to each class in a multi-class problem. Those decimal probabilities must add up to 1.0. This additional constraint helps training converge more quickly than it otherwise would.</p>
<p>Softmax is implemented through a neural network layer just before the output layer.</p>
<p>Variants of Softmax:</p>
<ul>
<li>Full Softmax: calculates a probability for every possible class.</li>
<li>Candidate sampling: calculates a probability for <strong>all the positive</strong> labels but only for a <strong>random sample of negative</strong> labels. It can improve efficiency in problems having a large number of classes.</li>
</ul>
<p>Softmax assumes that each example is a member of exactly one class. For <strong>many-labels</strong> problems, you must rely on multiple logistic regressions.</p>
<h2 id="embedding">Embedding</h2>
<h1 id="ml-engineering">ML Engineering</h1>
<blockquote>
<p>30 min</p>
</blockquote>
<h1 id="ml-real-world-examples">ML Real World Examples</h1>
<blockquote>
<p>15 min</p>
</blockquote>
<h1 id="conclusion">Conclusion</h1>


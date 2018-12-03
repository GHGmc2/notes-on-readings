---


---

<h1 id="深入理解tensorflow：架构设计与实现原理">深入理解TensorFlow：架构设计与实现原理</h1>
<blockquote>
<p><a href="http://www.ituring.com.cn/book/2397">勘误</a>, <a href="https://github.com/DjangoPeng/tensorflow-in-depth">Github</a><br>
<a href="https://book.douban.com/subject/30205343/">douban</a></p>
</blockquote>
<h2 id="tensorflow">TensorFlow</h2>
<p><a href="https://www.tensorflow.org/guide/extend/architecture">TensorFlow Architecture</a></p>
<p><a href="https://www.tensorflow.org/guide/low_level_intro">TensorFlow Core (Low-level APIs)</a></p>
<p><a href="https://github.com/tqchen/tinyflow">TinyFlow</a>: It demonstrates how can we build a clean, minimum and powerful computational graph based deep learning system with same API as TensorFlow.</p>
<p><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dist_test/python/mnist_replica.py">分布式MNIST</a></p>
<h1 id="基础">基础</h1>
<h2 id="系统综述">系统综述</h2>
<h3 id="设计目标">设计目标</h3>
<p>高性能全栈优化</p>
<ul>
<li>对高端和专用硬件的支持：StreamExecutor；支持RDMA</li>
<li>系统层优化：XLA；通信；数据流图优化</li>
<li>算法层优化：内置优化后的基础算子和模型组件</li>
</ul>
<h3 id="基本架构">基本架构</h3>
<p>组件结构：<br>
<img src="http://epub.ituring.com.cn/api/storage/getbykey/screenshow?key=1804fa8f70ba3bf482bc" alt=""></p>
<h2 id="环境准备">环境准备</h2>
<h3 id="依赖项">依赖项</h3>
<ul>
<li><a href="https://bazel.build/">Bazel</a></li>
<li><a href="https://developers.google.com/protocol-buffers/">Protocol Buffers</a>：生成代码</li>
<li><a href="http://eigen.tuxfamily.org">Eigen</a>：线性代数计算库</li>
<li><a href="https://developer.nvidia.com/cuda-zone">CUDA</a></li>
</ul>
<h3 id="源码结构">源码结构</h3>
<ul>
<li><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python">python</a>：Python API的多数模块内部通过<a href="http://www.swig.org/">SWIG</a>工具生成的胶合代码调用C API，进而使用C++核心库的功能
<ul>
<li><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py">framework/ops.py</a>：定义了Tensor、Graph、Opreator类等</li>
<li><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/variables.py">ops/variables.py</a>：定义了Variable类</li>
<li><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/training">training</a>：各种优化器</li>
</ul>
</li>
<li><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core">core</a>
<ul>
<li>framework</li>
<li>common_runtime</li>
<li>distributed_runtime</li>
</ul>
</li>
<li><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla">compiler/xla</a></li>
<li><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/stream_executor">stream_executor</a></li>
</ul>
<h2 id="基础概念">基础概念</h2>
<h3 id="编程范式：数据流图dataflow-graph">编程范式：数据流图Dataflow Graph</h3>
<ul>
<li>节点Node
<ul>
<li>计算节点Operation</li>
<li>存储节点Variable</li>
<li>数据节点Placeholder</li>
</ul>
</li>
<li>有向边Edges
<ul>
<li>数据边：传输数据</li>
<li>控制边：定义控制依赖</li>
</ul>
</li>
</ul>
<p>执行原理</p>
<ol>
<li>以<strong>节点名称为K，入度为V</strong>，创建散列表并将数据流图上节点放入散列表；</li>
<li>为此数据流图创建一个可执行节点队列，将散列表中入度为0的节点加入该队列，并从散列表中删除；</li>
<li>依次执行该队列中每个节点，成功后将该节点输出指向的节点入度减1，并更新散列表；</li>
<li>重复步骤2和3，直到可执行队列为空</li>
</ol>
<p>先编译得到完整的数据流图，然后根据用户选择的子图，输入数据进行计算。因此可以实现预编译优化。</p>
<p><img src="http://www.tensorfly.cn/images/tensors_flowing.gif" alt=""></p>
<h3 id="数据载体：张量">数据载体：张量</h3>
<p>张量的阶（rank）表示它所描述数据的最大维度。<br>
张量的形状（shape）用列表表示，每个值表示张量各阶的的长度。</p>
<p>张量被实现为一个文件句柄，它存储张量的元信息及<strong>指向张量数据的内存缓冲区指针</strong>，以便实现内存复用。</p>
<p>TensorFlow内部通过<strong>引用计数</strong>方式判断是否应该释放张量数据的内存缓冲区。</p>
<h4 id="稀疏张量sparsetensor">稀疏张量SparseTensor</h4>
<p>以键值对形式表示高维稀疏数据，包含三个属性：</p>
<ul>
<li>indices：形状为[N, ndims]的张量实例，N表示非零元素个数，ndims表示张量阶数</li>
<li>values：保存indices中指定的非零元素</li>
<li>dense_shape：表示对应稠密张量的形状</li>
</ul>
<h3 id="模型载体：操作op">模型载体：操作Op</h3>
<ul>
<li>计算节点Operator：无状态的计算或控制操作</li>
<li>存储节点Variable：有状态的变量操作</li>
<li>数据节点Placeholder：占位符操作。定义待输入数据的属性。</li>
</ul>
<h3 id="运行环境：会话session">运行环境：会话Session</h3>
<p>会话通过提取和切分数据流图、调度并执行操作结点，将抽象的计算拓扑转化为设备上的执行流。是发放任务的客户端。</p>
<p>with语句会隐式调用新建的Session对象的 <code>_enter_</code> 方法，将当前会话实例注册为默认会话。<br>
交互式会话InteractiveSession的构造方法会将会话实例注册为默认会话。</p>
<p>Session类的reset方法主要用于分布式会话的资源释放。</p>
<h3 id="训练工具：优化器optimizer">训练工具：<a href="https://www.tensorflow.org/api_docs/python/tf/train/Optimizer">优化器Optimizer</a></h3>
<ul>
<li>minimize()：最小化损失函数</li>
<li>compute_gradients()：计算梯度</li>
<li>apply_gradients()：应用梯度</li>
</ul>
<p><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/sync_replicas_optimizer.py">同步优化器Synchronize replicas</a>封装各种单机优化器，用于分布式训练。</p>
<h1 id="关键模块">关键模块</h1>
<h2 id="数据处理">数据处理</h2>
<h3 id="输入数据集">输入数据集</h3>
<h3 id="模型参数">模型参数</h3>
<h3 id="命令行参数">命令行参数</h3>
<p>指用户启动TensorFlow程序时输入的可选项，包含模型超参数和集群参数等。</p>
<h2 id="编程框架">编程框架</h2>
<h3 id="单机">单机</h3>
<p>处理的三类数据：</p>

<table>
<thead>
<tr>
<th>数据类别</th>
<th>数据来源</th>
<th>数据载体</th>
<th>数据消费者</th>
</tr>
</thead>
<tbody>
<tr>
<td>输入数据源</td>
<td>文件系统</td>
<td>张量</td>
<td>操作</td>
</tr>
<tr>
<td>模型参数</td>
<td>checkpoint文件</td>
<td>变量</td>
<td>Saver</td>
</tr>
<tr>
<td>模型超参数</td>
<td>命令行</td>
<td>FLAGS名字空间</td>
<td>优化器</td>
</tr>
</tbody>
</table><h3 id="分布式">分布式</h3>
<p>TensorFlow将PS-worker架构作为推荐的、标准的分布式编程框架。<br>
多worker间操作的依赖控制和模型参数更新模式由同步优化器的apply_gradients方法实现。</p>
<p>步骤：</p>
<ol>
<li>创建集群</li>
<li>创建分布式数据流图</li>
<li>创建并运行分布式会话</li>
</ol>
<p><img src="http://images2.imagebam.com/50/ac/43/dcfdb91036181014.jpg" alt=""></p>
<p><strong>同步训练机制</strong>基于同步优化器，涉及两个重要组件：</p>
<ul>
<li>梯度聚合器（gradients accumulator）：存储梯度值得队列。每个模型参数拥有一个单独的队列</li>
<li>同步标记队列（sync_token_queue）：存储同步标记的队列。同步标记决定worker是否能够执行梯度计算任务</li>
</ul>
<p><strong>异步训练机制</strong>：当不同的worker同时进行参数更新和拉取操作时，TensorFlow内部的锁机制保证模型参数的数据一致性。</p>
<p><strong>Supervisor管理模型训练</strong></p>
<ul>
<li>训练过程中定期保存模型参数到checkpoint文件</li>
<li>重启时从checkpoint文件读取和恢复模型参数，并继续训练</li>
<li>异常发生时，处理程序关闭和异常退出，同时完成内存回收等清理工作</li>
</ul>
<p>Supervisor本质上是对Saver（参数存储和恢复）、Coordinator（多线程服务的生命周期管理）和SessionManager（会话管理）三个类的封装。</p>
<p>Supervisor将大部分定期操作和异常处理封装成接口。用户只需在初始化时设置好相关参数即可。</p>
<h2 id="可视化tensorboard">可视化TensorBoard</h2>
<blockquote>
<p><a href="https://www.tensorflow.org/guide/summaries_and_tensorboard">TensorBoard: Visualizing Learning</a></p>
</blockquote>
<h2 id="模型托管tensorflow-serving">模型托管TensorFlow Serving</h2>
<blockquote>
<p><a href="https://www.tensorflow.org/serving/">Serving</a></p>
</blockquote>
<p>支持将多个模型组合为一个完备服务进行发布，也能够管理一个模型服务的多个版本。使得模型的在线学习和增量学习成为可能。</p>
<h3 id="系统架构">系统架构</h3>
<p><img src="https://cdn-images-1.medium.com/max/1600/1*A40ylIbkgsiO66XZ8sJBrQ.png" alt=""></p>
<p>插件模块：</p>
<ul>
<li>Servable：提供计算和查询等服务，<strong>每个Servable都拥有一个唯一的Servable Stream队列</strong>，队列中元素按版本号大小升序排列，并共享同一个Servable名称。</li>
<li>SavedModel：模型持久化存储的通用序列化格式</li>
<li>VersionPolicy：服务版本更新策略</li>
</ul>
<p><strong>Source</strong>监控和处理Servable加载的数据。一个Source可以同时维护多个不同的Servable Stream。当有新版本模型或模型有参数更新时，它会向对应的Servable Stream中添加新版本的Servable，并向DynamicManager发起一个加载该Servable的请求。</p>
<p><strong>Loader</strong>加载和卸载Servable，以及评估系统资源是否足够加载Servable。同一时刻，每个Loader只能加载对应Servable Stream中一个特定版本的Servable。</p>
<p><strong>DynamicManager</strong>是服务管理器。根据VersionPolicy决定是否允许Loader加载或卸载Servable Stream中的Servable。</p>
<p><strong>ServableHandle</strong>响应客户端访问请求，与已加载的Servable一一对应。</p>
<h1 id="核心揭秘">核心揭秘</h1>
<h2 id="运行时">运行时</h2>
<h3 id="运行时框架">运行时框架</h3>
<p><img src="http://images2.imagebam.com/c6/78/f5/3e97e41036832674.jpg" alt=""></p>
<h3 id="关键数据结构">关键数据结构</h3>
<ul>
<li>张量</li>
<li>设备</li>
<li>数据流图：大多为.proto文件，代码构建时由Protocol Buffers编译器自动生成对应C++源文件和头文件</li>
</ul>
<h3 id="公共基础机制">公共基础机制</h3>
<ul>
<li>内存分配：Allocator基类。屏蔽异构设备内存分配接口差异；针对不同设备和工作场景，实现相对高效的内存分配算法</li>
<li>线程管理：ThreadPool类。以动态、异步、事件驱动的任务并行为主要特征</li>
<li>多语言接口：以C API为中介的接口机制（C++接口无需通过C转发）。Python到C的接口链接通过SWIG工具实现</li>
<li>XLA</li>
<li>单元测试</li>
</ul>
<h3 id="外部环境接口">外部环境接口</h3>
<ul>
<li>加速器硬件接口</li>
<li>系统软件接口</li>
</ul>
<h4 id="加速器硬件接口">加速器硬件接口</h4>
<p>对CUDA GPU支持方面，TensorFlow使用StreamExecutor库实现对GPU的地层次、细粒度控制。StreamExecutor是一套异构并行计算库，调度核函数（Kernel）在流（stream）上执行。</p>
<p><img src="http://images2.imagebam.com/83/37/ae/d9b2c71036833254.jpg" alt=""></p>
<p>TensorFlow集成的模块为StreamExecutor项目的子集，主要组件：</p>
<ul>
<li>设备适配层：将CUDA和主机原有的编程抽象映射到流抽象，并实现设备特有的内存管理函数。</li>
<li>运行时引擎：在流上调度执行核函数</li>
<li>对上层开发者提供的API，主要抽象在Stream等类中</li>
<li>通用算子库：简化深度学习系统的开发。如DNN等</li>
</ul>
<p>TensorFlow对CUDA GPU的支持主要体现在：</p>
<ul>
<li>通用运行时库：封装StreamExecutor接口。主要涉及内存管理、设备管理、任务调度、跟踪调试等</li>
<li>核函数：Op – 核函数 – Stream对象 – GPU</li>
</ul>
<h4 id="系统软件接口">系统软件接口</h4>
<blockquote>
<p><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/platform">core/platform</a></p>
</blockquote>
<ul>
<li>本地操作系统：Env类以统一的接口集成了多类系统调用，如文件系统操作、线程管理、计时等
<ul>
<li>实现了POSIX标准的类Unix系统</li>
<li>Windows</li>
</ul>
</li>
<li>第三方PaaS：如文件系统</li>
</ul>
<h2 id="通信">通信</h2>
<p>进程内通信和进程间通信使用统一的消息传递模型及一致的编程接口。<br>
发送和接收操作一般是TensorFlow在图提取和切分过程中隐式地加入到数据流图中的。</p>
<h3 id="进程内通信">进程内通信</h3>
<p>汇合点（rendezvous）机制：协调异步执行的消息发送与接收操作，保证操作的正确配对和通信的顺利执行。</p>
<h3 id="进程间通信">进程间通信</h3>
<p>gRPC通信机制</p>
<h3 id="rdma模块">RDMA模块</h3>
<p>将进程间数据通信的实现迁移到支持RDMA特性的InfiniBand协议栈。</p>
<p>基于<strong>缓冲区预分配</strong>和<strong>消息交互</strong>的方案</p>
<h2 id="数据流图计算">数据流图计算</h2>
<p><img src="http://images2.imagebam.com/31/25/cf/f1b19d1036862774.jpg" alt=""></p>
<h3 id="创建">创建</h3>
<ol>
<li>全图构造</li>
<li>子图提取</li>
<li>图切分</li>
<li>图优化</li>
</ol>
<h3 id="单机会话运行">单机会话运行</h3>
<ol>
<li>执行器获取</li>
<li>输入数据填充</li>
<li>图运行输出数据获取</li>
<li>张量保存</li>
</ol>
<h3 id="分布式会话运行">分布式会话运行</h3>
<p>主-从模型</p>
<p>分布式会话抽象在TensorFlow核心层被分解为：</p>
<ul>
<li>GrpcSession：上层（调用层）抽象</li>
<li>MasterSession：下层（执行层）抽象</li>
<li>WorkerSession：封装会话执行时所需调用的外部对象的指针，便于Worker对象在会话运行时调用外部对象的方法</li>
</ul>
<h3 id="操作节点执行">操作节点执行</h3>
<p>核函数抽象</p>
<p>CPU上执行流程</p>
<p>CUDA GPU上执行流程</p>
<h1 id="生态发展">生态发展</h1>
<h2 id="keras">Keras</h2>
<blockquote>
<p><a href="https://www.tensorflow.org/guide/keras">tf.keras</a></p>
</blockquote>
<h2 id="kubernetes">Kubernetes</h2>
<p>TensorFlow两点不足：</p>
<ul>
<li>Serving挂载模型提供<strong>推理</strong>服务的方式无法应对高并发访问。人工添加负载均衡成本高。伸缩性差</li>
<li>没有一次性启动整个集群的方案。分布式<strong>训练</strong>作业需要在每台机器上手动启动进程，且显示指定每台机器主机地址和端口号</li>
</ul>
<p>Kubernetes优势：</p>
<ul>
<li>推理时提供伸缩性；</li>
<li>训练时提供统一的资源调度；统一任务启动；统一资源回收</li>
</ul>
<h2 id="spark">Spark</h2>
<p>TensorFlowOnSpark</p>
<h2 id="通信优化">通信优化</h2>
<p><a href="https://github.com/NVIDIA/nccl">NCCL: Optimized primitives for collective multi-GPU communication</a></p>
<p><a href="https://github.com/baidu-research/tensorflow-allreduce">tensorflow-allreduce</a></p>
<h2 id="tpu及asic">TPU及ASIC</h2>
<h2 id="nntm模块化深度学习组件">NNTM模块化深度学习组件</h2>
<p>NNVM本身主要关注计算图的中间表示，它使用TVM组件实现张量算子在不同硬件平台上的编译优化。</p>
<h2 id="tfx">TFX</h2>


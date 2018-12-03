---


---

<h1 id="tensorflow内核剖析">TensorFlow内核剖析</h1>
<blockquote>
<p><a href="https://github.com/horance-liu/tensorflow-internals">Github</a>, <a href="http://www.itdks.com/dakalive/detail/7719">视频</a>, <a href="https://myslide.cn/slides/8361">PPT</a><br>
<a href="http://www.itdks.com/dakalive/detail/4084">视频：TensorFlow分布式原理与应用</a></p>
</blockquote>
<h1 id="i-基础知识">I 基础知识</h1>
<h2 id="介绍">介绍</h2>
<p>TensorFlow 使用<strong>节点</strong>表示抽象的数学计算，并使用 OP 表达计算的逻辑；而<strong>边</strong>表示节点间传递的数据流， 并使用 Tensor 表达数据的表示。</p>
<p>设计原则：</p>
<ul>
<li>延迟计算：图的构造与执行分离，并推迟计算图的执行过程；</li>
<li>原子 OP：OP 是最小的抽象计算单元，支持构造复杂的网络模型；</li>
<li>抽象设备：支持 CPU, GPU, ASIC 多种异构计算设备类型；</li>
<li>抽象任务：基于任务的 PS，对新的优化算法和网络模型具有良好的可扩展性。</li>
</ul>
<h1 id="ii-系统架构">II 系统架构</h1>
<h2 id="系统架构">系统架构</h2>
<ul>
<li>Client</li>
<li>Master</li>
<li>Worker</li>
<li>Kernel</li>
</ul>
<p><img src="https://img2018.cnblogs.com/blog/1161096/201809/1161096-20180905150132923-1424753096.png" alt=""></p>
<h3 id="图控制">图控制</h3>
<h4 id="图构造">图构造</h4>
<h4 id="图执行">图执行</h4>
<ul>
<li>图分裂</li>
<li>子图注册</li>
<li>子图运算：</li>
</ul>
<h3 id="会话管理">会话管理</h3>
<h4 id="创建会话">创建会话</h4>
<h4 id="迭代运行">迭代运行</h4>
<ul>
<li>注册子图</li>
<li>运行子图</li>
<li>交换数据</li>
</ul>
<h4 id="关闭会话">关闭会话</h4>
<h2 id="c-api：分水岭">C API：分水岭</h2>
<p>Swig代码生成</p>
<h3 id="会话生命周期">会话生命周期</h3>
<ul>
<li>创建会话</li>
<li>创建/扩展图</li>
<li>迭代运行</li>
<li>关闭会话</li>
<li>销毁会话</li>
</ul>
<h3 id="性能调优">性能调优</h3>
<ul>
<li>共享图实例：在 Graph 实例上维持 Session 的引用计数器</li>
<li>消除序列化：在图的构造器，前端 Python 在构造每个 OP 时，直接通过 C API 将其追加至后端 C++ 的图实例中，从而避免了图实例在前后端的序列化和反序列化的开销。</li>
</ul>
<h1 id="iii-编程模型">III 编程模型</h1>
<h2 id="计算图graph">计算图Graph</h2>
<h3 id="python前端">Python前端</h3>
<ul>
<li>Operation</li>
<li>Tensor</li>
<li>TensorShape</li>
<li>Graph</li>
<li>图构造</li>
</ul>
<h3 id="c后端">C++后端</h3>
<ul>
<li>边</li>
<li>节点</li>
<li>图</li>
<li>OpDef仓库</li>
</ul>
<h2 id="设备device">设备Device</h2>
<h3 id="设备规范">设备规范</h3>
<h4 id="形式">形式</h4>
<h4 id="上下文管理">上下文管理</h4>
<ul>
<li>合并</li>
<li>覆盖</li>
<li>重置</li>
</ul>
<h2 id="会话session">会话Session</h2>
<h3 id="资源管理">资源管理</h3>
<ul>
<li>关闭会话</li>
<li>上下文管理器</li>
<li>图实例</li>
</ul>
<h3 id="默认会话">默认会话</h3>
<h3 id="会话类型">会话类型</h3>
<ul>
<li>Session</li>
<li>InteractiveSession</li>
<li>BaseSession</li>
</ul>
<h2 id="变量variable">变量Variable</h2>
<h3 id="初始化模型">初始化模型</h3>
<p>Variable 是一个特殊的 OP，它拥有状态 (Stateful)。</p>
<h3 id="变量分组">变量分组</h3>
<ul>
<li>全局变量</li>
<li>本地变量</li>
<li>训练变量</li>
<li>global_step</li>
</ul>
<h2 id="队列queue">队列Queue</h2>
<h3 id="队列">队列</h3>
<p>Queue是一种特殊的 OP，是一类有状态的 OP。<br>
Queue 有与之关联的 OP，例如 Enqueue，Dequeue，EnqueueMany，DequeueMany 等 OP，它们都能直接修改 Queue 的状态。</p>
<h3 id="协调器coordinator">协调器Coordinator</h3>
<p>Coordinator 提供了一种同时停止一组线程执行的简单机制。它拥有 3 个重要的方法:</p>
<ul>
<li>should_stop: 判断当前线程是否应该退出</li>
<li>request_stop: 请求所有线程停止执行</li>
<li>join: 等待所有线程停止执行</li>
</ul>
<h3 id="queuerunner">QueueRunner</h3>
<p>一个 QueueRunner 实例持有一个或多个 Enqueue 的入队 OP，它为每个 Enqueue OP 启动一个线程。</p>
<h2 id="op本质论">OP本质论</h2>
<h3 id="op的注册">OP的注册</h3>
<p>OP 的注册是通过 REGISTER_OP 宏完成的。</p>
<h1 id="iv-运行模型">IV 运行模型</h1>
<h2 id="本地执行">本地执行</h2>
<h3 id="本地模式">本地模式</h3>
<p>在本地模式下，Client, Master, Worker 部署在同一台机器同一进程内，并由 DirectSession 同时扮演这三个角色。</p>
<p>执行过程：</p>
<ul>
<li>部分执行：Master 收到计算图执行命令后，启动计算图的剪枝操作。它根据计算图的输入输出反向遍历图，寻找一个最小依赖的子图，常称为 <strong>ClientGraph</strong>。</li>
<li>并发执行：运行时按照当前设备集完成图的分裂，生成了很多子图，每个子图称为 <strong>PartitionGraph</strong>；然后触发各个 Worker 并发地执行每个 PartitionGraph；对于每一个 PartitionGraph，运行时将启动一个 Executor，按照其拓扑排序完成 PartitionGraph 的执行。<br>
<img src="http://images2.imagebam.com/48/01/9e/db5afa1045812294.png" alt=""></li>
</ul>
<h3 id="会话控制">会话控制</h3>
<p>DirectSession领域模型：<br>
<img src="http://images2.imagebam.com/46/e9/0f/1c596c1045821244.png" alt=""></p>
<h3 id="剪枝">剪枝</h3>
<p>DirectSession::Run 执行时，首先完成 ClientGraph 的构造：主要完成 FullGraph 的剪枝算法，并生成 ClientGraph。</p>
<p>外部运行时与输入/输出节点可以使用两种媒介交换数据：</p>
<ul>
<li>FunctionCallFrame</li>
<li>Rendezvous：用于 Send/Recv 消息发送的 OP，适用于分布式的运行时环境。</li>
</ul>
<p><strong>剪枝算法</strong></p>
<ol>
<li>追加输入节点</li>
<li>追加输出节点</li>
<li>反向剪枝：DAG 反向的宽度优先遍历</li>
</ol>
<p>经过剪枝后，将形成若干 DAG 子图。将入度为 0 的节点，与 Source 节点通过控制依赖边相连接；出度为 0 的节点，与 Sink 节点通过控制依赖边相连接，最终形成一个完整的 DAG 图。</p>
<h3 id="分裂">分裂</h3>
<p><strong>分裂算法</strong>实现：也是一个反向遍历图的算法。对于当前遍历的节点，将其标记为 dst；然后再寻找 dst 的所有输入边；遍历所有输入边，从而找到与改边相连的源节点，将其标记为 src。</p>
<p>回调函数：在 PartitionOptions 中，存在两个重要的回调函数。NodeToLocFunc 用于图分裂；NewNameFunc 用于给新增加的节点命名。</p>
<h3 id="执行">执行</h3>
<p>每个 PartitionGraph 启动一个 Executor，实现并发执行图的计算。每个 Executor 将执行 PartitionGraph 的拓扑排序算法，将入度为 0 的 OP 追加到 ready_queue 之中，并将其关联的 OP 的入度减 1。调度器调度 ready_queue 之中 OP ，并 将其放入 ThreadPool 中执行对应的 Kernel 实现。</p>
<ul>
<li>在所有 Partition 开始并发执行之前，需要外部将其输入传递给相应的 Arg 节点；当所有 Partition 完成计算后，外部再从 RetVal 节点中取走数据。其中，Arg/RetVal 节点之间的数据时通过 FunctionCallFrame 完成交互的。</li>
<li>如果 PartitionGraph 之间需要跨设备交换数据，生产者将其放在 Send 节点，消费者通过 Recv 节点获取数据。其中，发送方不阻塞；接收方如果数据未到，则发生阻塞直至超时。此外，Send/Recv 节点之间的数据是通过 Rendezvous 完成交互的。<br>
<img src="http://images2.imagebam.com/01/83/30/d31f691045845194.png" alt=""></li>
</ul>
<h3 id="设备间通信">设备间通信</h3>
<ul>
<li>SendOp</li>
<li>RecvOp</li>
</ul>
<h2 id="分布式tensorflow">分布式TensorFlow</h2>
<h3 id="分布式模式">分布式模式</h3>
<p>图的两级分裂过程：</p>
<ul>
<li>一级分裂：由 MasterSession 完成，按照 SplitByWorker 或 SplitByTask 完成图分裂过程；</li>
<li>二级分裂：由 WorkerSession 完成，按照 SplitByDevice 完成图分裂过程。</li>
</ul>
<p><img src="http://images2.imagebam.com/95/64/a1/068af91045852334.png" alt=""></p>
<h3 id="master服务">Master服务</h3>
<p>当 Client 根据 target 接入 Server 实例后，Server 扮演了 Master 的角色，对外提供 MasterService 服务。MasterService 定义了 Client 接入 Master 的公共契约，负责协调和控制多个 WorkerService 的执行过程。</p>
<h3 id="worker服务">Worker服务</h3>
<p>WorkerService 负责调度本地设备集执行本地子图。<br>
Master 根据 ClusterSpec 信息，找到集群中其他的 Server 实例，此时这些 Server 实例将扮演 Worker 的角色。Master 与 Worker 之间、Worker 与 Worker 之间的交互遵循 WorkerService 定义的接口规范。</p>
<h3 id="服务器server">服务器Server</h3>
<p>Server 负责管理本地设备集。具有同时扮演 Master 和 Worker 的角色。</p>
<h4 id="领域模型">领域模型</h4>
<p>Master 可以接入多个 Client，而一个 Client 则只能接入一个特定的 Master。</p>
<p>每个 Worker 可以为多个 Master 提供计算服务，它为每个向它请求计算服务的 MasterSession 生成一个相应的 WorkerSession 实例，等待相应的 MasterSession 下发计算图的<strong>注册</strong>和<strong>执行</strong>命令。</p>
<p>在同一个 Server 内，Master 与 Worker 可以部署在同一进程内。此时Master 与 Worker 之间直接使用函数调用。</p>
<h4 id="状态机">状态机</h4>
<p>创建服务<br>
GrpcServer::Init 将完成 GrpcServer 领域对象的初始化， 主要包括如下 3 个基本过程：</p>
<ol>
<li>初始化 MasterEnv 实例；</li>
<li>初始化 WorkerEnv 实例；</li>
<li>创建并启动 grpc::Server
<ul>
<li>初始化 MasterService</li>
<li>初始化 WorkerService</li>
</ul>
</li>
</ol>
<p>启动服务</p>
<p>等待终止服务</p>
<p>终止服务</p>
<h4 id="创建-workercacheinterface">创建 WorkerCacheInterface</h4>
<h4 id="创建-worker-的-rpc-通道">创建 Worker 的 RPC 通道</h4>
<h4 id="创建-workerinterface">创建 WorkerInterface</h4>
<h3 id="会话控制-1">会话控制</h3>
<p>会话控制是 TensorFlow 分布式运行时的核心。</p>
<h4 id="会话协同">会话协同</h4>
<p>在分布式模式中，会话控制通过 GrpcSession, MasterSession, WorkerSession 之间的协同实现的，它们分别驻留在 Client, Master, Worker 上，使 用同一个 session_handle 实现协同工作的。</p>
<p>可能存在多个 Client 同时接入一个 Master，为了区分不同的 Client 的计算服务，使用不同的 session_handle 区分。</p>
<h4 id="生命周期">生命周期</h4>
<h4 id="会话过程">会话过程</h4>
<p>创建会话</p>
<ol>
<li>创建 GrpcSession</li>
<li>获取远端设备集</li>
<li>创建 MasterSession</li>
<li>创建 WorkerSession</li>
</ol>
<p>迭代执行</p>
<ol>
<li>启动执行</li>
<li>图剪枝</li>
<li>图分裂</li>
<li>注册子图</li>
<li>运行子图</li>
</ol>
<p>关闭会话</p>
<ol>
<li>关闭 GrpcSession</li>
<li>关闭 MasterSession</li>
<li>关闭 WorkerSession</li>
</ol>
<h3 id="创建会话-1">创建会话</h3>
<h3 id="迭代执行">迭代执行</h3>
<h3 id="关闭会话-1">关闭会话</h3>
<h1 id="v-模型训练">V 模型训练</h1>
<h2 id="bp算法">BP算法</h2>
<h2 id="数据加载">数据加载</h2>
<h2 id="saver">Saver</h2>
<h2 id="monitoredsession">MonitoredSession</h2>
<h1 id="附录">附录</h1>
<h2 id="代码阅读">代码阅读</h2>
<h2 id="持续学习">持续学习</h2>


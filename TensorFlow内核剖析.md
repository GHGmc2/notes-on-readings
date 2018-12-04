# TensorFlow内核剖析
> 基于1.2版本
> [Github](https://github.com/horance-liu/tensorflow-internals), [视频](http://www.itdks.com/dakalive/detail/7719), [PPT](https://myslide.cn/slides/8361)
> [视频：TensorFlow分布式原理与应用](http://www.itdks.com/dakalive/detail/4084)

# I 基础知识

## 介绍

TensorFlow 使用**节点**表示抽象的数学计算，并使用 OP 表达计算的逻辑；而**边**表示节点间传递的数据流， 并使用 Tensor 表达数据的表示。

设计原则：

 - 延迟计算：图的构造与执行分离，并推迟计算图的执行过程；
 - 原子 OP：OP 是最小的抽象计算单元，支持构造复杂的网络模型；
 - 抽象设备：支持 CPU, GPU, ASIC 多种异构计算设备类型；
 - 抽象任务：基于任务的 PS，对新的优化算法和网络模型具有良好的可扩展性。

## 编程环境

### 代码结构

 - Python
 - Core
 - Compiler
 - StreamExecutor

### 工程构建

#### 环境准备

TensorFlow 使用 C++11 语法实现。
TensorFlow 使用 Bazel 的构建工具， 可以将其视为更高抽象的 Make 工具。
TensorFlow 使用 Swig 构建多语言的编程环境，自动生成相关编程语言的包装器。

#### 配置
```
$ ./configure
```

#### 构建
```
编译：
$ bazel build -c opt --config=cuda tensorflow/tools/pip_package:build_pip_package
构建Wheel包：
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

#### 安装
```
$ sudo pip install /tmp/tensorflow_pkg/tensorflow-1.4.0-py2-none-any.whl
```

### 代码生成

在构建 TensorFlow 系统时，Bazel 或 CMake 会自动生成部分源代码。理解代码生成器的输出结果，可以加深理解系统的行为模式。

# II 系统架构

## 系统架构

### 系统架构

 - 前端：构造计算图；
 - 后端：提供运行时环境，负责执行计算图。
	 - 运行时：分别提供本地模式和分布式模式，并共享大部分设计和实现。
		 - 表达图：构造计算图，但不执行图；
		 - 编排图：将计算图的节点以最佳的执行方案部署在集群中各个计算设备上执行；
		 - 运行图：按照拓扑排序执行图中的节点，并启动每个 OP 的 Kernel 计算。
	 - 计算层：由各个 OP 的 Kernel 实现组成；运行时Kernel 执行 OP 的具体数学运算;
	 - 通信层：基于 gRPC 实现组件间的数据交换，并能够在支持 IB 网络的节点间实现 RDMA 通信;
	 - 设备层：OP 执行的主要载体。

![](https://img2018.cnblogs.com/blog/1161096/201809/1161096-20180905150132923-1424753096.png)

 - Client：执行 Session.run 将 Protobuf 格式的 GraphDef 序列化后传给 Master
 - Master
	 1. 反向遍历 Full Graph，依照依赖关系进行**剪枝**，得到 Client Graph；
	 2. 将 Client Graph 按 SpiltByTask **分裂**为多个 Graph Partition（与Worker一一对应）；
	 3. 将Graph Partition **注册**到相应 Worker；
	 4. **通知** Worker 启动执行
 - Worker
	 1. 对注册的 Graph Partition （对于Worker来说也称Full Graph）按 SplitByDevice 进行**二次分裂**为多个 Graph Partition（与Device一一对应）；
	 2. 启动所有的 Graph Partition 执行；
	 3. 对每一个 Device，执行拓扑排序，依次**调用 OP 的 Kernel 实现**完成运算。
对于Worker间的数据交换：
	 - 本地 CPU 与 GPU 之间，使用 cudaMemcpyAsync 实现异步拷贝；
	 - 本地 GPU 之间，使用端到端的 DMA 操作，避免主机端 CPU 的拷贝。
对于任务间的通信，TensorFlow 支持多种通信协议：
	 - gRPC over TCP；
	 - RDMA over Converged Ethernet。
 - Kernel：OP 在某种硬件设备的特定实现
	 - 大多数 Kernel 基于 Eigen::Tensor 实现。TensorFlow 也可以灵活地直接使用 cuDNN, cuNCCL, cuBLAS 实现更高效的 Kernel。
	 - TensorFlow 实现了矢量化技术；支持更高效的 Kernel 注册。

### 图控制

### 会话（Session）管理

#### 创建会话

Master 创建一个 MasterSession 实例，并用全局唯一的 handle 标识。

#### 迭代运行

![](http://images2.imagebam.com/be/17/21/7795aa1052279074.png)

 - 注册子图：每个子图使用 graph_handle 唯一标识。
 - 运行子图：Worker 根据 graph_handle 索引相应的子图。每个子图放置在单独的 Executor 中执行。
 - 交换数据：Device间通过插入 Send/Recv 节点完成；Worker间需要通过接收端主动发送 RecvTensorRequest 消息到发送方，再从发送方的信箱里取出对应的 Tensor，并通过 RecvTensorResponse 返回。

#### 关闭会话

## C API：分水岭
> 会话生命周期源码解读

### Swig

TensorFlow 使用 Bazel 的构建工具，在系统编译之前启动 Swig 的代码生成过程，通过 tensorflow.i 自动生成了两个适配 (Wrapper) 文件:

 - pywrap_tensorflow_internal.py：负责对接上层 Python 调用；
 - pywrap_tensorflow_internal.cc：负责对接下层 C API 调用。

pywrap_tensorflow_internal.py 模块首次被导入时，自动地加载 _pywrap_tensorflow_internal.so 的动态链接库，其中包含了整个 TensorFlow 运行时的所有符号。
在 pywrap_tensorflow_internal.cc 的实现中， 静态注册了一个**函数符号表**，实现了 Python 函数名到 C 函数名的二元关系。

### 会话控制

在实际运行时环境中，tensorflow::Session 可能存在多种实现。例如，DirectSession 负责本地模式的会话控制。而 GrpcSession 负责基于 gRPC 协议的分布式模式的会话控制。

### 会话生命周期

 - 创建会话
 - 创建/扩展图
 - 迭代运行
 - 关闭会话
 - 销毁会话

### 性能调优

 - 共享图实例：在 Graph 实例上维持 Session 的引用计数器
 - 消除序列化：在图的构造器，前端 Python 在构造每个 OP 时，直接通过 C API 将其追加至后端 C++ 的图实例中，从而避免了图实例在前后端的序列化和反序列化的开销。

# III 编程模型
> 领域模型源码解读

## 计算图Graph

### Python前端

 - Operation
 - Tensor
 - TensorShape
 - Graph
 - 图构造

### C++后端

 - 边
 - 节点
 - 图
 - OpDef仓库

## 设备Device

### 设备规范

#### 形式

#### 上下文管理

 - 合并
 - 覆盖
 - 重置

## 会话Session

### 资源管理

 - 关闭会话
 - 上下文管理器
 - 图实例

### 默认会话

### 会话类型

 - Session
 - InteractiveSession
 - BaseSession

## 变量Variable

### 初始化模型

Variable 是一个特殊的 OP，它拥有状态 (Stateful)。

### 变量分组

 - 全局变量
 - 本地变量
 - 训练变量
 - global_step

## 队列Queue

### 队列

Queue是一种特殊的 OP，是一类有状态的 OP。
Queue 有与之关联的 OP，例如 Enqueue，Dequeue，EnqueueMany，DequeueMany 等 OP，它们都能直接修改 Queue 的状态。

### 协调器Coordinator

Coordinator 提供了一种同时停止一组线程执行的简单机制。它拥有 3 个重要的方法:

 - should_stop: 判断当前线程是否应该退出
 - request_stop: 请求所有线程停止执行
 - join: 等待所有线程停止执行

### QueueRunner

一个 QueueRunner 实例持有一个或多个 Enqueue 的入队 OP，它为每个 Enqueue OP 启动一个线程。

## OP本质论

### OP的注册

OP 的注册是通过 REGISTER_OP 宏完成的。

# IV 运行模型
> 执行过程源码解读

## 本地执行

### 本地模式

在本地模式下，Client, Master, Worker 部署在同一台机器同一进程内，并由 DirectSession 同时扮演这三个角色。

执行过程：

 - 部分执行：Master 收到计算图执行命令后，启动计算图的剪枝操作。它根据计算图的输入输出反向遍历图，寻找一个最小依赖的子图，常称为 **ClientGraph**。
 - 并发执行：运行时按照当前设备集完成图的分裂，生成了很多子图，每个子图称为 **PartitionGraph**；然后触发各个 Worker 并发地执行每个 PartitionGraph；对于每一个 PartitionGraph，运行时将启动一个 Executor，按照其拓扑排序完成 PartitionGraph 的执行。
![](http://images2.imagebam.com/48/01/9e/db5afa1045812294.png)

### 会话控制

DirectSession领域模型：
![](http://images2.imagebam.com/46/e9/0f/1c596c1045821244.png)

### 剪枝

DirectSession::Run 执行时，首先完成 ClientGraph 的构造：主要完成 FullGraph 的剪枝算法，并生成 ClientGraph。

外部运行时与输入/输出节点可以使用两种媒介交换数据：

 - FunctionCallFrame
 - Rendezvous：用于 Send/Recv 消息发送的 OP，适用于分布式的运行时环境。

**剪枝算法**

 1. 追加输入节点
 2. 追加输出节点
 3. 反向剪枝：DAG 反向的广度优先遍历

经过剪枝后，将形成若干 DAG 子图。将入度为 0 的节点，与 Source 节点通过控制依赖边相连接；出度为 0 的节点，与 Sink 节点通过控制依赖边相连接，最终形成一个完整的 DAG 图。

### 分裂

**分裂算法**实现：也是一个反向遍历图的算法。对于当前遍历的节点，将其标记为 dst；然后再寻找 dst 的所有输入边；遍历所有输入边，从而找到与改边相连的源节点，将其标记为 src。


回调函数：在 PartitionOptions 中，存在两个重要的回调函数。NodeToLocFunc 用于图分裂；NewNameFunc 用于给新增加的节点命名。

### 执行

每个 PartitionGraph 启动一个 Executor，实现并发执行图的计算。每个 Executor 将执行 PartitionGraph 的拓扑排序算法，将入度为 0 的 OP 追加到 ready_queue 之中，并将其关联的 OP 的入度减 1。调度器调度 ready_queue 之中 OP ，并 将其放入 ThreadPool 中执行对应的 Kernel 实现。

 - 在所有 Partition 开始并发执行之前，需要外部将其输入传递给相应的 Arg 节点；当所有 Partition 完成计算后，外部再从 RetVal 节点中取走数据。其中，Arg/RetVal 节点之间的数据时通过 FunctionCallFrame 完成交互的。
 - 如果 PartitionGraph 之间需要跨设备交换数据，生产者将其放在 Send 节点，消费者通过 Recv 节点获取数据。其中，发送方不阻塞；接收方如果数据未到，则发生阻塞直至超时。此外，Send/Recv 节点之间的数据是通过 Rendezvous 完成交互的。
![](http://images2.imagebam.com/01/83/30/d31f691045845194.png)

### 设备间通信

 - SendOp
 - RecvOp

## 分布式TensorFlow

### 分布式模式

图的两级分裂过程：

 - 一级分裂：由 MasterSession 完成，按照 SplitByWorker 或 SplitByTask 完成图分裂过程；
 - 二级分裂：由 WorkerSession 完成，按照 SplitByDevice 完成图分裂过程。

![](http://images2.imagebam.com/95/64/a1/068af91045852334.png)

### Master服务

当 Client 根据 target 接入 Server 实例后，Server 扮演了 Master 的角色，对外提供 MasterService 服务。MasterService 定义了 Client 接入 Master 的公共契约，负责协调和控制多个 WorkerService 的执行过程。

### Worker服务

WorkerService 负责调度本地设备集执行本地子图。
Master 根据 ClusterSpec 信息，找到集群中其他的 Server 实例，此时这些 Server 实例将扮演 Worker 的角色。Master 与 Worker 之间、Worker 与 Worker 之间的交互遵循 WorkerService 定义的接口规范。

### 服务器Server

Server 负责管理本地设备集。具有同时扮演 Master 和 Worker 的角色。

#### 领域模型

Master 可以接入多个 Client，而一个 Client 则只能接入一个特定的 Master。

每个 Worker 可以为多个 Master 提供计算服务，它为每个向它请求计算服务的 MasterSession 生成一个相应的 WorkerSession 实例，等待相应的 MasterSession 下发计算图的**注册**和**执行**命令。

在同一个 Server 内，Master 与 Worker 可以部署在同一进程内。此时Master 与 Worker 之间直接使用函数调用。

#### 状态机

创建服务
GrpcServer::Init 将完成 GrpcServer 领域对象的初始化， 主要包括如下 3 个基本过程：

 1. 初始化 MasterEnv 实例；
 2. 初始化 WorkerEnv 实例；
 3. 创建并启动 grpc::Server
	 - 初始化 MasterService
	 - 初始化 WorkerService

启动服务

等待终止服务

终止服务

#### 创建 WorkerCacheInterface

#### 创建 Worker 的 RPC 通道

#### 创建 WorkerInterface

### 会话控制

会话控制是 TensorFlow 分布式运行时的核心。

#### 会话协同

在分布式模式中，会话控制通过 GrpcSession, MasterSession, WorkerSession 之间的协同实现的，它们分别驻留在 Client, Master, Worker 上，使 用同一个 session_handle 实现协同工作的。

可能存在多个 Client 同时接入一个 Master，为了区分不同的 Client 的计算服务，使用不同的 session_handle 区分。

#### 生命周期

#### 会话过程

创建会话

 1. 创建 GrpcSession
 2. 获取远端设备集
 3. 创建 MasterSession
 4. 创建 WorkerSession

迭代执行

 1. 启动执行
 2. 图剪枝
 3. 图分裂
 4. 注册子图
 5. 运行子图

关闭会话

 1. 关闭 GrpcSession
 2. 关闭 MasterSession
 3. 关闭 WorkerSession

### 创建会话

在 Client 端创建 GrpcSession 实例，在 Master 端创建 MasterSession 实例；在各个 Worker 上创建 WorkerSession 实例，三者通过 MasterSession 的 session_handle 实现协同。

### 迭代执行

### 关闭会话

# V 模型训练

## BP算法

## 数据加载

## Saver

## MonitoredSession

# 附录

## 代码阅读

## 持续学习

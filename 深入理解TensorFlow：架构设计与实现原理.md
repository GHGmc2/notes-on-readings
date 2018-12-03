# 深入理解TensorFlow：架构设计与实现原理
> [勘误](http://www.ituring.com.cn/book/2397), [Github](https://github.com/DjangoPeng/tensorflow-in-depth)
> [douban](https://book.douban.com/subject/30205343/)

## TensorFlow
[TensorFlow Architecture](https://www.tensorflow.org/guide/extend/architecture)

[TensorFlow Core (Low-level APIs)](https://www.tensorflow.org/guide/low_level_intro)

[TinyFlow](https://github.com/tqchen/tinyflow): It demonstrates how can we build a clean, minimum and powerful computational graph based deep learning system with same API as TensorFlow.

[分布式MNIST](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dist_test/python/mnist_replica.py)

# 基础

## 系统综述

### 设计目标

高性能全栈优化

 - 对高端和专用硬件的支持：StreamExecutor；支持RDMA
 - 系统层优化：XLA；通信；数据流图优化
 - 算法层优化：内置优化后的基础算子和模型组件

### 基本架构

组件结构：
![](http://epub.ituring.com.cn/api/storage/getbykey/screenshow?key=1804fa8f70ba3bf482bc)

## 环境准备

### 依赖项

 - [Bazel](https://bazel.build/)
 - [Protocol Buffers](https://developers.google.com/protocol-buffers/)：生成代码
 - [Eigen](http://eigen.tuxfamily.org)：线性代数计算库
 - [CUDA](https://developer.nvidia.com/cuda-zone)

### 源码结构

 - [python](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python)：Python API的多数模块内部通过[SWIG](http://www.swig.org/)工具生成的胶合代码调用C API，进而使用C++核心库的功能
	 - [framework/ops.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py)：定义了Tensor、Graph、Opreator类等
	 - [ops/variables.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/variables.py)：定义了Variable类
	 - [training](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/training)：各种优化器
 - [core](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core)
	 - framework
	 - common_runtime
	 - distributed_runtime
 - [compiler/xla](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla)
 - [stream_executor](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/stream_executor)

## 基础概念

### 编程范式：数据流图Dataflow Graph

 - 节点Node
	 - 计算节点Operation
	 - 存储节点Variable
	 - 数据节点Placeholder
 - 有向边Edges
	 - 数据边：传输数据
	 - 控制边：定义控制依赖

执行原理

 1. 以**节点名称为K，入度为V**，创建散列表并将数据流图上节点放入散列表；
 2. 为此数据流图创建一个可执行节点队列，将散列表中入度为0的节点加入该队列，并从散列表中删除；
 3. 依次执行该队列中每个节点，成功后将该节点输出指向的节点入度减1，并更新散列表；
 4. 重复步骤2和3，直到可执行队列为空

先编译得到完整的数据流图，然后根据用户选择的子图，输入数据进行计算。因此可以实现预编译优化。

![](http://www.tensorfly.cn/images/tensors_flowing.gif)

### 数据载体：张量

张量的阶（rank）表示它所描述数据的最大维度。
张量的形状（shape）用列表表示，每个值表示张量各阶的的长度。

张量被实现为一个文件句柄，它存储张量的元信息及**指向张量数据的内存缓冲区指针**，以便实现内存复用。

TensorFlow内部通过**引用计数**方式判断是否应该释放张量数据的内存缓冲区。

#### 稀疏张量SparseTensor

以键值对形式表示高维稀疏数据，包含三个属性：

 - indices：形状为[N, ndims]的张量实例，N表示非零元素个数，ndims表示张量阶数
 - values：保存indices中指定的非零元素
 - dense_shape：表示对应稠密张量的形状

### 模型载体：操作Op

 - 计算节点Operator：无状态的计算或控制操作
 - 存储节点Variable：有状态的变量操作
 - 数据节点Placeholder：占位符操作。定义待输入数据的属性。

### 运行环境：会话Session

会话通过提取和切分数据流图、调度并执行操作结点，将抽象的计算拓扑转化为设备上的执行流。是发放任务的客户端。

with语句会隐式调用新建的Session对象的 `_enter_` 方法，将当前会话实例注册为默认会话。
交互式会话InteractiveSession的构造方法会将会话实例注册为默认会话。

Session类的reset方法主要用于分布式会话的资源释放。

### 训练工具：[优化器Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/Optimizer)

 - minimize()：最小化损失函数
 - compute_gradients()：计算梯度
 - apply_gradients()：应用梯度

[同步优化器Synchronize replicas](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/sync_replicas_optimizer.py)封装各种单机优化器，用于分布式训练。

# 关键模块

## 数据处理

### 输入数据集

### 模型参数

### 命令行参数

指用户启动TensorFlow程序时输入的可选项，包含模型超参数和集群参数等。

## 编程框架

### 单机

处理的三类数据：
| 数据类别 | 数据来源 | 数据载体 | 数据消费者 |
|--|--|--|--|
| 输入数据源 | 文件系统 | 张量 | 操作 |
| 模型参数 | checkpoint文件 | 变量 | Saver |
| 模型超参数 | 命令行 | FLAGS名字空间 | 优化器 |

### 分布式

TensorFlow将PS-worker架构作为推荐的、标准的分布式编程框架。
多worker间操作的依赖控制和模型参数更新模式由同步优化器的apply_gradients方法实现。

步骤：

 1. 创建集群
 2. 创建分布式数据流图
 3. 创建并运行分布式会话

![](http://images2.imagebam.com/50/ac/43/dcfdb91036181014.jpg)

**同步训练机制**基于同步优化器，涉及两个重要组件：

 - 梯度聚合器（gradients accumulator）：存储梯度值得队列。每个模型参数拥有一个单独的队列
 - 同步标记队列（sync_token_queue）：存储同步标记的队列。同步标记决定worker是否能够执行梯度计算任务

**异步训练机制**：当不同的worker同时进行参数更新和拉取操作时，TensorFlow内部的锁机制保证模型参数的数据一致性。

**Supervisor管理模型训练**

 - 训练过程中定期保存模型参数到checkpoint文件
 - 重启时从checkpoint文件读取和恢复模型参数，并继续训练
 - 异常发生时，处理程序关闭和异常退出，同时完成内存回收等清理工作

Supervisor本质上是对Saver（参数存储和恢复）、Coordinator（多线程服务的生命周期管理）和SessionManager（会话管理）三个类的封装。

Supervisor将大部分定期操作和异常处理封装成接口。用户只需在初始化时设置好相关参数即可。

## 可视化TensorBoard
> [TensorBoard: Visualizing Learning](https://www.tensorflow.org/guide/summaries_and_tensorboard)

## 模型托管TensorFlow Serving
> [Serving](https://www.tensorflow.org/serving/)

支持将多个模型组合为一个完备服务进行发布，也能够管理一个模型服务的多个版本。使得模型的在线学习和增量学习成为可能。

### 系统架构

![](https://cdn-images-1.medium.com/max/1600/1*A40ylIbkgsiO66XZ8sJBrQ.png)

插件模块：

 - Servable：提供计算和查询等服务，**每个Servable都拥有一个唯一的Servable Stream队列**，队列中元素按版本号大小升序排列，并共享同一个Servable名称。
 - SavedModel：模型持久化存储的通用序列化格式
 - VersionPolicy：服务版本更新策略

**Source**监控和处理Servable加载的数据。一个Source可以同时维护多个不同的Servable Stream。当有新版本模型或模型有参数更新时，它会向对应的Servable Stream中添加新版本的Servable，并向DynamicManager发起一个加载该Servable的请求。

**Loader**加载和卸载Servable，以及评估系统资源是否足够加载Servable。同一时刻，每个Loader只能加载对应Servable Stream中一个特定版本的Servable。

**DynamicManager**是服务管理器。根据VersionPolicy决定是否允许Loader加载或卸载Servable Stream中的Servable。

**ServableHandle**响应客户端访问请求，与已加载的Servable一一对应。

# 核心揭秘

## 运行时

### 运行时框架

![](http://images2.imagebam.com/c6/78/f5/3e97e41036832674.jpg)

### 关键数据结构

 - 张量
 - 设备
 - 数据流图：大多为.proto文件，代码构建时由Protocol Buffers编译器自动生成对应C++源文件和头文件

### 公共基础机制

 - 内存分配：Allocator基类。屏蔽异构设备内存分配接口差异；针对不同设备和工作场景，实现相对高效的内存分配算法
 - 线程管理：ThreadPool类。以动态、异步、事件驱动的任务并行为主要特征
 - 多语言接口：以C API为中介的接口机制（C++接口无需通过C转发）。Python到C的接口链接通过SWIG工具实现
 - XLA
 - 单元测试

### 外部环境接口

 - 加速器硬件接口
 - 系统软件接口

#### 加速器硬件接口

对CUDA GPU支持方面，TensorFlow使用StreamExecutor库实现对GPU的地层次、细粒度控制。StreamExecutor是一套异构并行计算库，调度核函数（Kernel）在流（stream）上执行。

![](http://images2.imagebam.com/83/37/ae/d9b2c71036833254.jpg)

TensorFlow集成的模块为StreamExecutor项目的子集，主要组件：

 - 设备适配层：将CUDA和主机原有的编程抽象映射到流抽象，并实现设备特有的内存管理函数。
 - 运行时引擎：在流上调度执行核函数
 - 对上层开发者提供的API，主要抽象在Stream等类中
 - 通用算子库：简化深度学习系统的开发。如DNN等

TensorFlow对CUDA GPU的支持主要体现在：

 - 通用运行时库：封装StreamExecutor接口。主要涉及内存管理、设备管理、任务调度、跟踪调试等
 - 核函数：Op -- 核函数 -- Stream对象 -- GPU

#### 系统软件接口
> [core/platform](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/platform)

 - 本地操作系统：Env类以统一的接口集成了多类系统调用，如文件系统操作、线程管理、计时等
	 - 实现了POSIX标准的类Unix系统
	 - Windows
 - 第三方PaaS：如文件系统

## 通信

进程内通信和进程间通信使用统一的消息传递模型及一致的编程接口。
发送和接收操作一般是TensorFlow在图提取和切分过程中隐式地加入到数据流图中的。

### 进程内通信

汇合点（rendezvous）机制：协调异步执行的消息发送与接收操作，保证操作的正确配对和通信的顺利执行。

### 进程间通信

gRPC通信机制

### RDMA模块

将进程间数据通信的实现迁移到支持RDMA特性的InfiniBand协议栈。

基于**缓冲区预分配**和**消息交互**的方案

## 数据流图计算

![](http://images2.imagebam.com/31/25/cf/f1b19d1036862774.jpg)

### 创建

 1. 全图构造
 2. 子图提取
 3. 图切分
 4. 图优化

### 单机会话运行

 1. 执行器获取
 2. 输入数据填充
 3. 图运行输出数据获取
 4. 张量保存

### 分布式会话运行

主-从模型

分布式会话抽象在TensorFlow核心层被分解为：

 - GrpcSession：上层（调用层）抽象
 - MasterSession：下层（执行层）抽象
 - WorkerSession：封装会话执行时所需调用的外部对象的指针，便于Worker对象在会话运行时调用外部对象的方法

### 操作节点执行

核函数抽象

CPU上执行流程

CUDA GPU上执行流程

# 生态发展

## Keras
> [tf.keras](https://www.tensorflow.org/guide/keras)

## Kubernetes

TensorFlow两点不足：

 - Serving挂载模型提供**推理**服务的方式无法应对高并发访问。人工添加负载均衡成本高。伸缩性差
 - 没有一次性启动整个集群的方案。分布式**训练**作业需要在每台机器上手动启动进程，且显示指定每台机器主机地址和端口号

Kubernetes优势：

 - 推理时提供伸缩性；
 - 训练时提供统一的资源调度；统一任务启动；统一资源回收

## Spark

TensorFlowOnSpark

## 通信优化

[NCCL: Optimized primitives for collective multi-GPU communication](https://github.com/NVIDIA/nccl)

[tensorflow-allreduce](https://github.com/baidu-research/tensorflow-allreduce)

## TPU及ASIC

## NNTM模块化深度学习组件

NNVM本身主要关注计算图的中间表示，它使用TVM组件实现张量算子在不同硬件平台上的编译优化。

## TFX


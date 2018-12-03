# 深入剖析k8s
> [课程信息](https://time.geekbang.org/column/intro/116)
> [k8s技能图谱](https://time.geekbang.org/column/article/17841)
> [Kubernetes项⽬与基础设施“⺠主化”的探索](https://static001.geekbang.org/con/30/pdf/3953244280/file/AS%E6%B7%B1%E5%9C%B32018-%E3%80%8AKubernetes%E9%A1%B9%E7%9B%AE%E4%B8%8E%26ldquo%3B%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD%E6%B0%91%E4%B8%BB%E5%8C%96%26rdquo%3B%E7%9A%84%E6%8E%A2%E7%B4%A2%E3%80%8B-%E5%BC%A0%E7%A3%8A.pdf)

## 背景

Docker 一举走红的重要原因

 - 解决了应用打包这个根本性的问题
 - 同开发者与生俱来的的亲密关系
 - PaaS 概念已经深入人心的完美契机

## 容器基础

 1. 对 Docker 项目来说，它最核心的原理实际上就是为待创建的用户进程：
	 * 隔离：启用 Linux Namespace 配置；
	 * 约束：设置指定的 Cgroups 参数；
	 * 切换进程的根目录（Change Root）
 2. Namespace
	 * 优势：“敏捷”和“高性能”是容器相较于虚拟机最大的优势
	 * 不足：隔离不彻底 -> 约束 -> Cgroups
 3. Linux Cgroups 的全称是 Linux Control Group。它最主要的作用，就是限制一个进程组能够使用的资源上限，包括 CPU、内存、磁盘、网络带宽等等。
 4. 容器是一个“单进程”模型。容器本身的设计，就是希望容器和应用能够“同生命周期”，这个概念对后续的容器编排非常重要
 5. 镜像：
	 * chroot -> [mount namespace](https://segmentfault.com/a/1190000006912742)，得到rootfs
	 * rootfs（亦即容器镜像）：挂载在容器根目录上、用来为容器进程提供隔离后执行环境的文件系统
		 * rootfs 只是一个操作系统所包含的文件、配置和目录，并不包括操作系统内核。实际上，同一台机器上的所有容器，都共享宿主机操作系统的内核。
		 * rootfs 里打包的不只是应用，而是整个操作系统的文件和目录，也就意味着，应用以及它运行所需要的所有依赖，都被封装在了一起。这样容器才有了一个被反复宣传至今的重要特性：一致性。
	 * 增量rootfs：层（layer）
		 * AuFS
			 * 可读写层(rw)：存放修改 rootfs 后产生的增量。用户执行 docker commit 只会提交可读写层
			 * Init层(ro+wh)：Init 层是 Docker 项目单独生成的一个内部层，专门用来存放一些只对当前的容器有效的如/etc/hosts、/etc/resolv.conf 等信息
			 * 只读层(ro+wh)：这些层都以增量的方式分别包含了 Ubuntu 操作系统的一部分
 6. Docker容器：通过操作系统进程相关的知识逐步剖析 Docker 容器的方法，是理解容器的一个关键思路
	 * Linux Namespace 创建的隔离空间虽然看不见摸不着，但一个进程的 Namespace 信息在宿主机上是确确实实存在的，并且是以一个文件的方式存在
	 * docker exec原理：一个进程可以选择加入到某个进程已有的 Namespace 当中，从而“进入”这个进程所在容器
	 * Volume（数据卷）：允许将宿主机上指定的目录或者文件，挂载到容器里面进行读取和修改操作
		 * 过程：在 rootfs 准备好之后、执行 chroot 之前，把 Volume 指定的宿主机目录，挂载到指定的容器目录在宿主机上对应的目录上。此时由于容器进程（指初始化进程，不是应用进程）已经创建了，Mount Namespace 已经开启，所以这个挂载事件只在这个容器里可见，保证了容器的隔离性不会被 Volume 打破
		 * 绑定挂载（bind mount）
		 * 容器 Volume 里的信息并不会被 docker commit 提交掉；但这个挂载点目录 /test 本身则会出现在新的镜像当中。因为容器的镜像操作如 docker commit，都是发生在宿主机空间的。而由于 Mount Namespace 的隔离作用，宿主机并不知道这个绑定挂载的存在。所以在宿主机看来，容器中可读写层的 /test 目录始终为空
		 * 应用运行在由 Linux Namespace 和 Cgroups 构成的隔离环境里；而它运行所需要的各种文件及整个操作系统文件，则由多个联合挂载在一起的 rootfs 层提供（容器声明的Volume挂载点在可读写层）。
			 * ![全景图](https://static001.geekbang.org/resource/image/2b/18/2b1b470575817444aef07ae9f51b7a18.png)
 7. k8s本质
	 * 容器运行时：由Namespace+Cgroups构成的隔离环境
	 * 容器镜像：一组联合挂载在 /var/lib/docker/aufs/mnt 上的 rootfs
	 * 在整个“开发 - 测试 - 发布”的流程中，真正承载着容器信息进行传递的，是容器镜像，而不是容器运行时。只要能够将用户提交的 Docker 镜像以容器的方式运行起来，就能将整个容器技术栈上的价值沉淀在我的这个节点上。更重要的是，只要从我这个承载点向 Docker 镜像制作者和使用者方向回溯，整条路径上的各个服务节点，比如 CI/CD、监控、安全、网络、存储等等，和潜在用户（开发者）直接关联起来
	 * 架构图：![架构图](https://static001.geekbang.org/resource/image/8e/67/8ee9f2fa987eccb490cfaa91c6484f67.png)
		 * Master控制结点
			 * Controller Manager：容器编排
			 * API Server：API服务。而整个集群的持久化数据，由 kube-apiserver 处理后保存在 Ectd 中
			 * Scheduler：调度
		 * Node计算结点
			 * Kubelet
				 * 通过CRI与容器运行时交互，容器运行时通过OCI与Linux系统进行交互
				 * 通过gRPC与Device Plugin插件交互，管理宿主机GPU等物理设备
				 * 通过CNI为容器配置网络
				 * 通过CSI配置持久化存储
    * 声明式API。
	    * API对象
		    * 编排对象：如 Pod、Job、CronJob 等，来描述你试图管理的应用
		    * 服务对象：如 Service、Secret、Horizontal Pod Autoscaler等，负责具体的平台级功能
	    * Kubernetes 以统一的方式来定义任务之间的各种关系
		    * Pod
			    * 在 Kubernetes 项目中，容器会被划分为一个“Pod”，Pod 里的容器共享同一个 Network Namespace、同一组数据卷，从而达到高效率交换信息的目的
				* Kubernetes 给 Pod 绑定一个 Service 服务，而 Service 服务声明的 IP 地址等信息是“终生不变”的。这个Service 服务的主要作用，就是作为 Pod 的代理入口（Portal），从而代替 Pod 对外暴露一个固定的网络地址
		 * 除了应用与应用之间的关系外，**应用运行的形态**是影响“如何容器化这个应用”的第二个重要因素
			 * Kubernetes 定义了新的、基于 Pod 改进后的对象。如Job，DeamonSet

## 搭建与实践

### kubeadm

 1. kubelet 在配置容器网络、管理容器数据卷时，都需要直接操作宿主机。（如果 kubelet 本身就运行在一个容器里，对于网络配置来说还好，kubelet 容器可以通过不开启 Network Namespace 的方式，直接共享宿主机的网络栈。可是，要让 kubelet 隔着容器的 Mount Namespace 和文件系统，操作宿主机的文件系统，就有点儿困难了）
	 * kubeadm妥协方案：把kubelet运行在宿主机，然后用容器部署其他k8s组件
 2. kubeadm init工作流程
	 * Preflight Checks
	 * 生成 Kubernetes 对外提供服务所需的各种证书和对应的目录。证书放在 Master 节点的 /etc/kubernetes/pki 目录下
	 * 为其他组件生成访问 kube-apiserver 所需的配置文件。路径是/etc/kubernetes/xxx.conf
	 * 为 Master 组件生成 Pod 配置文件（还会生成一个 Etcd 的 Pod YAML 文件）。路径在 /etc/kubernetes/manifests
		 * static pod启动方式
		 * Master 容器启动后，kubeadm 会通过检查 localhost:6443/healthz 这个 Master 组件的健康检查 URL，等待 Master 组件完全运行起来
	  * 为集群生成一个 bootstrap token。后面只要持有这个 token，任何一个安装了 kubelet 和 kubadm 的节点，都可以通过 kubeadm join 加入到这个集群当中
	  * 将 ca.crt 等 Master 节点的重要信息，通过 ConfigMap(cluster-info) 的方式保存在 Etcd 当中，供后续部署 Node 节点使用
	  * 安装默认插件。k8s 默认 kube-proxy 和 DNS 这两个插件是必须安装的，分别用来提供服务发现和 DNS 功能
 3. kubeadm join工作流程
	  * 任何一台机器想要成为 Kubernetes 集群中的一个节点，就必须在集群的 kube-apiserver 上注册。可是，要想跟 apiserver 打交道，这台机器就必须要获取到相应的证书文件（CA 文件）
	  * 所以，kubeadm 至少需要发起一次“不安全模式”的访问到 kube-apiserver，从而拿到保存在 ConfigMap 中的 cluster-info（它保存了 APIServer 的授权信息）
	  * 有了 cluster-info 里的 kube-apiserver 的地址、端口、证书，kubelet 就可以以“安全模式”连接到 apiserver 上，这样一个新的节点就部署完成了
 4. kubeadm 目前最欠缺的是，一键部署一个高可用的 Kubernetes 集群，即：Etcd、Master 组件都应该是多节点集群，而不是现在这样的单点

### [搭建k8s集群](https://time.geekbang.org/column/article/39724)

 1. 

### 部署应用

 1. 一个 YAML 文件，对应到 Kubernetes 中就是一个 API Object（API 对象）。Kubernetes 会负责创建出这些对象所定义的容器或者其他类型的 API 资源
	 * 例：
		```
			 apiVersion: apps/v1
			 kind: Deployment
			 metadata:
			   name: nginx-deployment
			 spec:
			   selector:
			     matchLabels:
			       app: nginx
			   replicas: 2
			   template:
			     metadata:
			       labels:
			         app: nginx
			     spec:
			       containers:
			       - name: nginx
			         image: nginx:1.7.9
			         ports:
			         - containerPort: 80
			         volumeMounts:
			         - mountPath: "/usr/share/nginx/html"
			           name: nginx-vol
			       volumes:
			       - name: nginx-vol
			         emptyDir: {}
		```

	 * Deployment 是一个定义多副本应用的对象，还负责在 Pod 定义发生变化时，对每个副本进行滚动更新
	 * Labels 是一组 key-value 格式的标签。而像 Deployment 这样的控制器对象，就可以通过这个 Labels 字段从 Kubernetes 中过滤出它所关心的被控制对象
	 * Volume 属于 Pod 对象的一部分
		 * EmptyDir类型即不显式声明宿主机目录的 Volume。Kubernetes 会在宿主机上创建一个临时目录，将来会被绑定挂载到容器所声明的 Volume 目录上
		 * volumeMounts 字段来声明自己要挂载哪个 Volume，并通过 mountPath 字段来定义容器内的 Volume 目录
  2. Pod 就是 Kubernetes 世界里的“应用”；一个应用可以由多个容器组成
  3. 像这样使用一种 API 对象（Deployment）管理另一种 API 对象（Pod）的方法，在 Kubernetes 中叫作“控制器”模式
  4. 命令
	    ```
	    
		$ kubectl create -f 我的配置文件
		$ kubectl get：从 Kubernetes 里面获取（GET）指定的 API 对象
		$ kubectl describe：查看API对象细节
		$ kubectl apply：进行 Kubernetes 对象的创建和更新操作
		$ kubectl delete
		```
  5. 快速熟悉 Kubernetes 练习流程
	  * 本地通过Docker测试代码，制作镜像
	  * 选择合适的k8s API对象，制作yaml
	  * 在k8s上部署yaml（接下来的所有操作，要么通过 kubectl 来执行，要么通过修改 YAML 文件来实现，尽量不要再碰 Docker 命令行了）

## 编排与作业管理
### Pod

 1. Pod是 Kubernetes 项目中最小的 API 对象和原子调度单位。
 2. 模型映射：
	 * Kubernetes —— OS
	 * Pod —— 进程组
	 * 容器 —— 进程
 3. 容器的“单进程模型”，并不是指容器里只能运行“一个”进程，而是指容器没有管理多个进程的能力。（因为容器里 PID=1 的进程就是应用本身，其他的进程都是这个 PID=1 进程的子进程。用户编写的应用并不能像正常操作系统里的 init 进程或者 systemd 那样拥有进程管理的功能）
 4. “超亲密关系”特征包括但不限于：互相之间会发生直接的文件交换、使用 localhost 或者 Socket 文件进行本地通信、会发生非常频繁的远程调用、需要共享某些 Linux Namespace（比如，一个容器要加入另一个容器的 Network Namespace）等
 5. Pod实现原理
	 * Pod只是个逻辑概念。
	 * Pod 里的所有容器，共享的是同一个 Network Namespace（一个 Pod 只有一个 IP 地址），并且可以声明共享同一个 Volume
	 * Pod 的实现需要使用一个中间容器——Infra 容器。在这个 Pod 中，Infra 容器永远都是第一个被创建的容器，而其他用户定义的容器，则通过 Join Network Namespace 的方式，与 Infra 容器关联在一起，所有用户容器的进出流量也可以认为都是通过 Infra 容器完成的
		 * Pod 的生命周期只跟 Infra 容器一致，而与用户容器无关
 6. 容器设计模式Sidecar：指的是我们可以在一个 Pod 中，启动一个辅助容器，来完成一些独立于主进程（主容器）之外的工作
	 * [Design Patterns for Container-based Distributed Systems](https://www.usenix.org/conference/hotcloud16/workshop-program/presentation/burns)
	 * [管理设计篇之"边车模式"](https://time.geekbang.org/column/article/5909)
	 * [Designing Distributed Systems: Patterns and Paradigms for Scalable, Reliable Services](https://book.douban.com/subject/27050608/)
 7. Pod 扮演的是传统部署环境里“虚拟机”的角色。可以把 Pod 看成传统环境里的“机器”、把容器看作是运行在这个“机器”里的“用户程序”
	 * Pod级别：调度、网络、存储、安全相关的属性；跟容器的 Linux Namespace 相关的属性；Pod 中的容器要共享宿主机的 Namespace
 8. Pod字段（源码见kubernetes/vendor/k8s.io/api/core/v1/types.go）
	 * NodeSelector：供用户将 Pod 与 Node 进行绑定
	 * NodeName：这个字段一般由调度器负责设置，被赋值说明经过了调度，调度的结果就是赋值的节点名字
	 * HostAliases：定义了 Pod 的 hosts 文件里的内容。如果要设置 hosts 文件里的内容，一定要通过这种方法。否则，如果直接修改了 hosts 文件的话，在 Pod 被删除重建之后，kubelet 会自动覆盖掉被修改的内容
	 * Containers
		 * ImagePullPolicy：镜像拉取策略
		 * Lifecycle：定义 Container Lifecycle Hooks，在容器状态发生变化时触发“钩子”，如postStart 和 preStop
	 * Status：生命周期
		 * Pending
		 * Running
		 * Succeeded
		 * Failed
		 * Unknown
 9. Projected Volume：为容器提供预先定义好的数据
	 * Secret：把 Pod 想要访问的加密数据存放到 Etcd 中，然后通过在 Pod 的容器里挂载 Volume 的方式访问这些 Secret 里保存的信息。像这样通过挂载方式进入到容器里的 Secret，一旦其对应的 Etcd 里的数据被更新，这些 Volume 里的文件内容同样也会被更新（kubelet 组件在定时维护这些 Volume）
		 * 场景：存放数据库的 Credential 信息
		 * ServiceAccountToken：Service Account（Kubernetes 系统内置的一种“服务账户”，它是 Kubernetes 进行权限分配的对象）的授权信息和文件，实际上保存在它所绑定的一个特殊的 Secret 对象里，即ServiceAccountToken。任何运行在 Kubernetes 集群上的应用，都必须使用这个 ServiceAccountToken 里保存的授权信息，也就是 Token，才可以合法地访问 API Server
			 * InClusterConfig：把 Kubernetes 客户端以容器的方式运行在集群里，然后使用 default Service Account 自动授权的方式。推荐进行 Kubernetes API 编程的授权方式
	 * ConfigMap：非加密配置
	 * Downward API：让 Pod 里的容器能够直接获取到这个 Pod API 对象本身的信息。
		 * [支持的字段](https://kubernetes.io/docs/tasks/inject-data-application/downward-api-volume-expose-pod-information/#capabilities-of-the-downward-api)
		 * 能够获取到是容器**进程启动之前**就能够确定下来的信息
 10. 容器健康检查和恢复
	 * 可以为 Pod 里的容器定义一个健康检查“探针”（Probe），kubelet 根据这个 Probe 的返回值决定这个容器的状态（而不是直接以容器进行是否运行作为依据）
	 * restartPolicy：Pod恢复机制。Pod 的恢复过程永远都是发生在当前节点上，而不会跑到别的节点上去。事实上，一旦一个 Pod 与一个节点（Node）绑定，除非这个绑定发生了变化（pod.spec.node 字段被修改），否则它永远都不会离开这个节点（宿主机宕机也不会）
		 * Always：在任何情况下，只要容器不在运行状态，就自动重启容器
		 * OnFailure: 只在容器 异常时才自动重启容器
		 * Never: 从来不重启容器
		 * 只要 Pod 的 restartPolicy 指定的策略允许重启异常的容器，那么这个 Pod 就会保持 Running 状态，并进行容器重启（实际上是重新创建）
		 * 对于包含多个容器的 Pod，只有它里面所有的容器都进入异常状态后，Pod 才会进入 Failed 状态
 11. PodPreset：Pod预设置。自动给Pod填充字段
	 * PodPreset 里定义的内容，只会在 Pod API 对象被创建之前追加在这个对象本身上，而不会影响任何 Pod 的控制器的定义

### 控制器模型

控制循环（control loop）：也被称作“Reconcile Loop”（调谐循环）或“Sync Loop”（同步循环）

 - 通过**循环**比较期望状态和实际状态，**将实际状态调整为期望状态**
 - 实际状态往往来自于 Kubernetes 集群本身；而期望状态，一般来自于用户提交的 YAML 文件

控制器是由控制器定义（包括期望状态），加上被控制对象的模板（template）组成的
	   ![](https://static001.geekbang.org/resource/image/72/26/72cc68d82237071898a1d149c8354b26.png)

Deployment实现
Deployment 是一个两层控制器。它通过**ReplicaSet 的个数**来描述**应用的版本**，通过ReplicaSet 的**属性**保证 Pod 的副本数量。Deployment 控制器实际操纵的正是ReplicaSet 对象，而不是 Pod 对象。
 - ![](https://static001.geekbang.org/resource/image/79/f6/79dcd2743645e39c96fafa6deae9d6f6.png)
 - ReplicaSet 负责通过“控制器模式”，保证系统中 Pod 的个数永远等于指定的个数。（Deployment 只允许容器的 restartPolicy=Always ：只有在容器能保证自己始终是 Running 状态的前提下，ReplicaSet 调整 Pod 的个数才有意义）
 - Deployment 同样通过“控制器模式”，来操作 ReplicaSet 的个数和属性，进而实现“水平扩展 / 收缩”和“滚动更新”这两个编排动作
 - 应用版本和 ReplicaSet 一一对应

[Kubernetes deployment strategies](https://github.com/ContainerSolutions/k8s-deployment-strategies)

### StatefulSet

Kubernetes对 **有状态应用（Stateful Application）** 编排功能的支持，就是StatefulSet。
StatefulSet 的核心功能，就是通过某种方式记录这些状态，然后在 Pod 被重新创建时，能够为新 Pod 恢复这些状态。它把应用状态抽象成两种情况：
 - 拓扑状态：应用的多个实例之间不是完全对等的关系。
 - 存储状态：应用的多个实例分别绑定了不同的存储数据。

**拓扑状态**
Service 是 Kubernetes 项目中用来将一组 Pod 暴露给外界访问的一种机制。访问Service的方式有：
 - 以 Service 的 VIP（Virtual IP，即：虚拟 IP）方式：Service VIP -> Pod IP
 - 以 Service 的 DNS 方式（）
	 - Normal Service：DNS -> Service VIP -> Pod IP
	 - Headless Service：DNS -> Pod IP（以 DNS 记录直接解析出被代理 Pod 的 IP 地址，不需要VIP）
	 它所代理的所有 Pod 的 IP 地址，都会被绑定一个 DNS 记录：
	 ```<pod-name>.<svc-name>.<namespace>.svc.cluster.local```
	 有了这个“可解析身份”，只需知道 Pod 的名字和它对应 Service 的名字，就可以访问到 Pod 的 IP 地址（Pod 的 DNS 记录本身不会变，但它解析到的 Pod 的 IP 地址并不是固定的）。

StatefulSet 通过给它所管理的所有 Pod 进行编号（Pod 的“名字 + 编号”），严格按照编号顺序进行创建。

**存储状态**
PV/PVC：Kubernetes 中 PVC（Persistent Volume Claim） 和 PV（Persistent Volume） 的设计，实际上类似于“接口”和“实现”的思想。开发者只要知道并会使用 PVC ，而运维人员则负责给 PVC 绑定具体的实现，即 PV。
即使 Pod 被删除，它所对应的 PVC 和 PV 依然会保留下来。

StatefulSet 的工作原理：
 - StatefulSet 的控制器直接管理的是 Pod
 - Kubernetes 通过 Headless Service，为这些有编号的 Pod，在 DNS 服务器中生成带有同样编号的 DNS 记录
 - StatefulSet 还为每一个 Pod 分配并创建一个同样编号的 PVC

**实践**

[一个“主从复制”（Maser-Slave Replication）的 MySQL 集群](https://time.geekbang.org/column/article/41217)

### DaemonSet

DaemonSet 的主要作用，是在 Kubernetes 集群里运行一个 Daemon Pod（比如网络、存储、日志等插件的Agent）。

在DaemonSet 的控制循环中，只需要遍历Etcd所有节点，然后根据节点上是否有被管理 Pod 的情况，来决定是否要创建或者删除一个 Pod。

DaemonSet 只管理 Pod 对象，然后通过 nodeAffinity 和 Toleration 保证了每个节点上有且只有一个 Pod：
 - 在创建每个 Pod 的时候，DaemonSet 会自动给这个 Pod 加上一个 **nodeAffinity**，从而保证这个 Pod 只会在指定节点上启动。
 - 同时，它还会自动给这个 Pod 加上一个 **Toleration**，从而忽略节点的 unschedulable“污点”（比如网络插件尚未安装）

DaemonSet 使用 **ControllerRevision**来保存和管理自己的“版本”（StatefulSet也是）。

注：v1.11 之前版本DaemonSet 所管理的 Pod 的调度过程，实际上都是由 DaemonSet Controller 自己，而不是由调度器完成的。这种方式很快会被废除

### Job

Deployment、StatefulSet，以及 DaemonSet 主要编排的对象，都是“在线业务”，即 Long Running Task（长作业）。比如 Nginx、Tomcat，以及 MySQL 等等。
离线业务，或叫 Batch Job（计算业务）。这种业务在计算完成后就直接退出了。

Job 对象在创建后，它的 Pod 模板被自动加上了一个 controller-uid=< 一个随机字符串 > 这样的 Label。而这个 Job 对象本身，则被自动加上了这个 Label 对应的 Selector，从而保证了 Job 与它所管理的 Pod 之间的匹配关系。

restartPolicy 在 Job 对象里只允许被设置为 Never（失败后创建新Pod）和 OnFailure（失败后重启Pod里的容器）；而在 Deployment 对象里，restartPolicy 则只允许被设置为 Always。

Job Controller 直接管理Pod。它控制了作业执行的**并行度**，以及总共需要完成的**任务数**这两个重要参数。在 Job 对象中，负责**并行控制**的参数有两个：
 - spec.parallelism：Job 最多可以同时运行的 Pod 数
 - spec.completions：Job 的最小完成数

使用 Job 对象的方法（大多数情况下用户更倾向于自己控制 Job 对象）：
 - 外部管理器 + Job 模板：把 Job 的 YAML 文件定义为一个“模板”，然后用一个外部工具控制这些“模板”来生成 Job。在这种模式下，completions 和 parallelism 这两个字段都应该使用默认值 1，作业 Pod 的并行控制应该完全交由外部工具来进行管理。
	 - 典型应用：KubeFlow
 - 拥有固定任务数目的并行 Job
 - 指定并行度（parallelism），但不设置固定的 completions 的值

**CronJob**（定时任务）是一个专门用来管理 Job 对象的控制器。它创建和删除 Job 的依据，是 schedule 字段定义的“Unix Cron”表达式。

### 声明式API

声明式 API，才是 Kubernetes 项目编排能力“赖以生存”的核心所在。Kubernetes“声明式 API”的独特之处：

 - 所谓“声明式”，指只需要提交一个定义好的 API 对象来“声明”我所期望的状态是什么样子
 - “声明式 API”允许有多个 API 写端，以 PATCH 的方式对 API 对象进行修改，而无需关心本地原始 YAML 文件的内容
 - 有了上述两个能力，Kubernetes 项目才可以基于对 API 对象的增、删、改、查，在完全无需外界干预的情况下，完成对“实际状态”和“期望状态”的调谐（Reconcile）过程

kubectl apply 执行一个对原有 API 对象的 PATCH 操作，一次能处理多个写操作，并且具备 **Merge 能力**。

**Istio**
架构
![架构](https://static001.geekbang.org/resource/image/d3/1b/d38daed2fedc90e20e9d2f27afbaec1b.jpg)

Istio 最根本的组件，是运行在每一个应用 Pod 里的 Envoy 容器（以 sidecar 容器的方式）。
Envoy 容器就能够通过配置 Pod 里的 iptables 规则，把整个 Pod 的进出流量接管下来。这时候，Istio 的控制层（Control Plane）里的 Pilot 组件，就能够通过调用每个 Envoy 容器的 API，对这个 Envoy 代理进行配置，从而实现微服务治理。

Istio 使用“热插拔”式的 Dynamic Admission Control（也叫 Initializer）功能，实现在应用 Pod YAML 被提交给 Kubernetes 之后，在它对应的 API 对象里自动加上 Envoy 容器的配置（Admission Controller 的代码可以选择性地被编译进 APIServer 中，在 API 对象创建之后会被立刻调用到）。

Istio 要做的，就是编写一个用来为 Pod“自动注入”Envoy 容器的 Initializer：

 1. 首先，Istio 会将这个 Envoy 容器本身的定义，以 ConfigMap 的方式保存在 Kubernetes 当中
 2. 接下来，Istio 将一个编写好的 Initializer，作为一个 Pod 部署在 Kubernetes 中
 3. 在 Initializer 控制器的工作逻辑里，它首先会从 APIServer 中拿到这个 ConfigMap；然后把这个 ConfigMap 里存储的 containers 和 volumes 字段，直接添加进一个空的 Pod 对象里；
 4. 使用新旧两个 Pod 对象，生成一个 TwoWayMergePatch，Initializer 的代码使用这个 patch 的数据发起一个 PATCH 请求。这样一个用户提交的 Pod 对象里，就会被自动加上 Envoy 容器相关的字段

**声明式API工作原理**
在 Kubernetes 项目中，一个 API 对象在 Etcd 里的完整资源路径，是由：Group（Group 的分类是以对象**功能**为依据的，核心 API 对象不需要 Group）、Version（apiVersion后半段）和 Resource（kind字段）三个部分组成的。

API 对象创建流程：
![](https://static001.geekbang.org/resource/image/df/6f/df6f1dda45e9a353a051d06c48f0286f.png)

**CRD(Custom Resource Definition)机制**

**自定义对象**包括两部分内容（[代码示例](https://github.com/resouer/k8s-controller-custom-resource)）：

 - API 描述，包括：组（Group）、版本（Version）、资源类型（Resource）等，即CRD 的 YAML 文件
 - 对象描述，包括：Spec、Status 等

[Kubernetes Deep Dive: Code Generation for CustomResources](https://blog.openshift.com/kubernetes-deep-dive-code-generation-customresources/), [翻译](https://www.cn18k.com/2018/04/04/Kubernetes-Deep-Dive-Code-Generation-for-CustomResources/)

**[自定义控制器](https://time.geekbang.org/column/article/42076)**
![工作原理](https://static001.geekbang.org/resource/image/32/c3/32e545dcd4664a3f36e95af83b571ec3.png)

Informer：带有本地缓存 Store 和索引 Index 机制的、可以注册 EventHandler 的 client。职责：

 - 同步本地缓存
 - 根据事件的类型，触发事先注册好的 ResourceEventHandler

Informer 使用了 Reflector 包，它是一个可以通过 ListAndWatch 机制获取并监视 API 对象变化的客户端封装。

Reflector 和 Informer 之间，用到了一个“增量先进先出队列”进行协同。而 Informer 与你要编写的控制循环之间，则使用了一个工作队列来进行协同。

### RBAC

基本概念

 - Role（角色）：定义了一组对 Kubernetes API 对象的操作权限
 - Subject（被作用者）
 - RoleBinding：定义了“被作用者”和“角色”的绑定关系

Role 和 RoleBinding 对象都是 **Namespaced 对象**（Namespaced Object），它们对权限的限制规则仅在它们自己的 Namespace 内有效，roleRef 也只能引用当前 Namespace 里的 Role 对象。
对于**非 Namespaced对象**（比如：Node），或者某一个 Role 想要作用于所有的 Namespace 的时候，可以使用 ClusterRole 和 ClusterRoleBinding 这两个组合。
在 Kubernetes 中已经内置了很多个为系统保留的 ClusterRole，它们的名字都以 system: 开头。一般来说，这些**系统 ClusterRole**是绑定给 Kubernetes 系统组件对应的 ServiceAccount 使用的。
除此之外，Kubernetes 还提供了四个预先定义好的 ClusterRole 来供用户直接使用：cluster-admin、admin、edit、view。

**Role**对象rules字段定义权限规则。
**RoleBinding**对象“subjects”字段即“被作用者”；“roleRef”字段通过名字来引用前面定义的 Role 对象，从而定义了 Subject 和 Role 之间的绑定关系。

### Operator工作原理
利用 Kubernetes 的自定义 API 资源（CRD），来描述我们想要部署的“有状态应用”；然后在自定义控制器里，根据自定义 API 对象的变化，来完成具体的部署和运维工作。

**Etcd Operator工作原理**
![](https://static001.geekbang.org/resource/image/e7/36/e7f2905ae46e0ccd24db47c915382536.jpg)
Etcd Operator 的特殊之处在于，它为每一个 EtcdCluster 对象都启动了一个控制循环，“并发”地响应这些对象的变化。这种做法不仅可以简化 Etcd Operator 的代码实现，还有助于提高它的响应速度。

Cluster 对象具体负责：

 - 创建一个单节点的种子集群。Bootstrap只在该 Cluster 对象第一次被创建的时候才会执行
 - 启动该集群所对应的控制循环

Etcd Operator 把一个 Etcd 集群抽象成了一个具有一定“自治能力”的整体。而当这个“自治能力”本身不足以解决问题的时候，我们可以通过两个专门负责备份和恢复的 Operator 进行修正。

Operator 和 StatefulSet 并不是竞争关系。你完全可以编写一个 Operator，然后在 Operator 的控制循环里创建和控制 StatefulSet 而不是 Pod。比如[prometheus-operator](https://github.com/coreos/prometheus-operator)

[etcd：从应用场景到实现原理的全方位解读](http://www.infoq.com/cn/articles/etcd-interpretation-application-scenario-implement-principle)

## 存储

## 网络

## 作业调度与资源管理

### 资源模型与资源管理

#### 资源模型设计

**可压缩资源**（compressible resources）：当可压缩资源不足时，Pod 只会“饥饿”，但不会退出。如CPU
**不可压缩资源**（uncompressible resources）：当不可压缩资源不足时，Pod 就会被内核杀掉。如内存

其中，Kubernetes 里为 CPU 设置的单位是“CPU 的个数”。具体“1 个 CPU”在宿主机上如何解释，完全取决于宿主机的 CPU 实现方式。可以是 1 个 CPU 核心、 1 个 vCPU，或 1 个 CPU 的超线程（Hyperthread）。

Kubernetes 里 Pod 的 CPU 和内存资源，分为 **limits** 和 **requests** 两种情况。在调度的时候，kube-scheduler 只会按照 requests 的值进行计算。在真正设置 Cgroups 限制的时候，kubelet 则会按照 limits 的值来进行设置。

Kubernetes 用户在提交 Pod 时，可以声明一个相对较小的 requests 值供调度器使用，而 Kubernetes 真正设置给容器 Cgroups 的，则是相对较大的 limits 值。这种对 CPU 和内存资源**限额**的设计，参考了 Borg 论文对“动态资源边界”的定义。因为在实际场景中，大多数作业使用到的资源其实远小于它所请求的资源限额。

#### QoS模型

在 Kubernetes 中，不同的 requests 和 limits 的设置方式，会将 Pod 划分到不同的 **QoS 级别**当中：

 - Guaranteed：Pod 里的每一个 Container 都同时设置了 requests 和 limits，并且 requests 和 limits 值相等；
 - Burstable：Pod 不满足 Guaranteed 的条件，但至少有一个 Container 设置了 requests；
 - BestEffort：Pod 既没有设置 requests，也没有设置 limits

QoS 划分的主要应用场景，是当宿主机资源紧张的时候，kubelet 对 Pod 进行 Eviction时需要用到的。Eviction 在 Kubernetes 里其实分为 Soft 和 Hard 两种模式：

 - Soft Eviction 允许你为 Eviction 过程设置一段“优雅时间”，当不足的阈值超过设定的时间后，kubelet 才会开始 Eviction 的过程。
 - Hard Eviction 模式下，Eviction 过程就会在阈值达到之后立刻开始。

当宿主机的 Eviction 阈值达到后，就会进入 MemoryPressure 或者 DiskPressure 状态，从而避免新的 Pod 被调度到这台宿主机上。
而当 Eviction 发生的时候，kubelet 会参考这些 Pod 的 QoS 类别挑选 Pod 进行删除操作。顺序：BestEffort -> Burstable -> Guaranteed。
Kubernetes 会保证只有当 Guaranteed 类别的 Pod 的资源使用量超过了其 limits 的限制，或者宿主机本身正处于 Memory Pressure 状态时，Guaranteed 的 Pod 才可能被选中进行 Eviction 操作。对于同 QoS 类别的 Pod 来说，Kubernetes 还会根据 Pod 的优先级来进行进一步地排序和选择。

**cpuset** 的设置要求：Pod 必须是 Guaranteed 的 QoS 类型；Pod 的 CPU 资源的 requests 和 limits 值相等。
通过设置 cpuset 把容器绑定到某个 CPU 的核上，而不是像 cpushare 那样共享 CPU 的计算能力。由于操作系统在 CPU 之间进行上下文切换的次数大大减少，容器里应用的性能会得到大幅提升。

### 默认调度器

Kubernetes 默认调度器的主要职责，就是为一个新创建出来的 Pod，寻找一个最合适的节点（Node）

 - 从集群所有的节点中，根据调度算法挑选出所有可以运行该 Pod 的节点；
 - 从第一步的结果中，再根据调度算法挑选一个最符合条件的节点作为最终结果。

在具体的调度流程中，默认调度器会首先调用一组叫作 Predicate 的调度算法，来检查每个 Node。然后，再调用一组叫作 Priority 的调度算法，来给上一步得到的结果里的每个 Node 打分。最终的调度结果，就是得分最高的那个 Node。

Kubernetes 的调度器的核心，实际上就是两个相互独立的控制循环：
![](https://static001.geekbang.org/resource/image/90/9b/90343a090a8242ad46d2f82cb6b99b9b.png)

 - Informer Path的主要目的，是启动一系列 Informer，用来监听（Watch）Etcd 中 Pod、Node、Service 等与调度相关的 API 对象的变化。比如，当一个待调度 Pod（即：它的 nodeName 字段是空的）被创建出来之后，调度器就会通过 Pod Informer 的 Handler，将这个待调度 Pod 添加进调度队列。
在默认情况下，Kubernetes 的调度队列是一个 PriorityQueue（优先级队列），并且当某些集群信息发生变化的时候，调度器还会对调度队列里的内容进行一些特殊操作。这里主要是出于调度优先级和抢占的考虑。
 - Scheduling Path 的主要逻辑，就是不断地从调度队列里出队一个 Pod。然后，调用 Predicates 算法进行“过滤”。这一步“过滤”得到的一组 Node，就是所有可以运行这个 Pod 的宿主机列表。当然，Predicates 算法需要的 Node 信息，都是从 Scheduler Cache 里直接拿到的，这是调度器保证算法执行效率的主要手段之一。
接下来，调度器就会再调用 Priorities 算法为上述列表里的 Node 打分，分数从 0 到 10。得分最高的 Node，就会作为这次调度的结果。

**乐观绑定**
为了不在关键调度路径里远程访问 APIServer，Kubernetes 的默认调度器在 Bind 阶段（将 Pod 对象的 nodeName 字段的值修改为选出 Node 的名字），只会更新 Scheduler Cache 里的 Pod 和 Node 的信息。这种基于“乐观”假设的 API 对象更新方式，在 Kubernetes 里被称作 **Assume**。
Assume 之后，调度器才会创建一个 Goroutine 来异步地向 APIServer 发起更新 Pod 的请求，来真正完成 Bind 操作。如果这次异步的 Bind 过程失败了，等 Scheduler Cache 同步之后一切就会恢复正常。
当一个新的 Pod 完成调度需要在某个节点上运行起来之前，该节点上的 kubelet 还会通过一个叫作 Admit 的操作（把一组叫作 GeneralPredicates 的、最基本的调度算法再执行一遍，作为 kubelet 端的二次确认）来再次验证该 Pod 是否确实能够运行在该节点上。

**无锁化**
在 Scheduling Path 上，调度器会启动多个 Goroutine 以节点为粒度**并发**执行 Predicates 算法，从而提高这一阶段的执行效率。而与之类似的，Priorities 算法也会以 **MapReduce** 的方式并行计算然后再进行汇总。而在这些所有需要并发的路径上，调度器会避免设置任何全局的竞争资源，从而免去了使用锁进行同步带来的巨大的性能损耗。Kubernetes 调度器只有对调度队列和 Scheduler Cache 进行操作时，才需要加锁。而这两部分操作，都不在 Scheduling Path 的算法执行路径上。

Kubernetes 默认调度器的**可扩展机制 Scheduler Framework**：
![](https://static001.geekbang.org/resource/image/e9/17/e9e00d60f14bc125e46caf02c01f7817.png)
这些可插拔式逻辑，都是标准的 [Go plugin](https://golang.org/pkg/plugin/) 机制，也就是说，你需要在编译的时候选择把哪些插件编译进去。

#### 调度策略

Predicates 在调度过程中的作用，可以理解为 Filter，它按照调度策略，从当前集群的所有节点中，“过滤”出一系列符合条件的节点。这些节点，都是可以运行待调度 Pod 的宿主机。
默认的调度策略有：

 - GeneralPredicates：负责最基础的调度策略。比如 PodFitsResources 计算的就是宿主机的 CPU 和内存资源等是否够用。
 - Volume 相关的过滤规则：负责跟容器持久化 Volume 相关的调度策略。
 - 宿主机相关的过滤规则：考察待调度 Pod 是否满足 Node 本身的某些条件。如“污点”机制
 - Pod 相关的过滤规则：跟 GeneralPredicates 大多数重合。而比较特殊的是 PodAffinityPredicate，检查待调度 Pod 与 Node 上的已有 Pod 之间的亲密（affinity）和反亲密（anti-affinity）关系

当开始调度一个 Pod 时，Kubernetes 调度器会同时启动 16 个 Goroutine，来并发地为集群里的所有 Node 计算 Predicates，最后返回可以运行这个 Pod 的宿主机列表。需要注意的是，在为每个 Node 执行 Predicates 时，调度器会按照固定的顺序来进行检查。这个顺序，是按照 Predicates 本身的含义来确定的。

在 Predicates 阶段完成了节点的“过滤”之后，Priorities 阶段的工作就是为这些节点打分。
常用打分规则：

 - LeastRequestedPriority：选择空闲资源（CPU 和 Memory）最多的宿主机；
 - BalancedResourceAllocation：选择调度完成后，所有节点里各种资源分配最均衡的那个节点。从而避免一个节点上 CPU 被大量分配、而 Memory 大量剩余的情况。
 - NodeAffinityPriority、TaintTolerationPriority 和 InterPodAffinityPriority 三种 Priority：一个 Node 满足上述规则的字段数目越多，它的得分就会越高。
 - ImageLocalityPriority：如果待调度 Pod 需要使用的镜像很大，并且已经存在于某些 Node 上，那么这些 Node 的得分就会比较高。为了避免引发调度堆叠，调度器在计算得分的时候还会根据镜像的分布进行优化，即：如果大镜像分布的节点数目很少，那么这些节点的权重就会被调低，从而“对冲”掉引起调度堆叠的风险。

在实际的执行过程中，调度器里关于集群和 Pod 的信息都已经缓存化，所以这些算法的执行过程还是比较快的。

#### 优先级与抢占机制

优先级和抢占机制，解决的是 Pod 调度失败时该怎么办的问题。

Kubernetes 规定，优先级是一个 32 bit 的整数，最大值不超过10 亿（1 billion），并且值越大代表优先级越高。而超出 10 亿的值，其实是被 Kubernetes 保留下来分配给系统 Pod 使用的。

调度器里维护着一个调度队列。当 Pod 拥有了优先级之后，高优先级的 Pod 就可能会比低优先级的 Pod 提前出队，从而尽早完成调度过程。

而当一个高优先级的 Pod 调度失败的时候，调度器的抢占能力就会被触发。这时，调度器就会试图从当前集群里寻找一个节点，使得当这个节点上的一个或者多个低优先级 Pod 被删除后，待调度的高优先级 Pod 就可以被调度到这个节点上。

当抢占过程发生时，调度器只会将抢占者的 spec.nominatedNodeName 字段，设置为被抢占的 Node 的名字。然后，抢占者会重新进入下一个调度周期，然后在新的调度周期里来决定是不是要运行在被抢占的节点上。
把抢占者交给下一个调度周期再处理。主要有两方面原因：被抢占节点“优雅退出”期间集群的可调度性可能会发生变化；抢占节点在等待调度过程中，允许更高优先级节点抢占。

而 Kubernetes 调度器实现抢占算法的一个最重要的设计，就是在调度队列的实现里使用了两个不同的队列：

 - activeQ：凡是在 activeQ 里的 Pod，都是下一个调度周期需要调度的对象。
 - unschedulableQ：专门用来存放调度失败的 Pod。当一个 unschedulableQ 里的 Pod 被更新之后，调度器会自动把这个 Pod 移动到 activeQ 里。

调度失败之后，抢占者就会被放进 unschedulableQ 里面。然后，这次失败事件就会触发调度器为抢占者寻找牺牲者的流程：

 1. 调度器会检查这次失败事件的原因，来确认抢占是不是可以帮助抢占者找到一个新节点。这是因为有很多 Predicates 的失败是不能通过抢占来解决的。
 2. 如果确定抢占可以发生，那么调度器就会把自己缓存的所有节点信息复制一份，然后使用这个副本来模拟抢占过程。

调度器会检查缓存副本里的每一个节点，然后从该节点上最低优先级的 Pod 开始，逐一“删除”这些 Pod。而每删除一个低优先级 Pod，调度器都会检查一下抢占者是否能够运行在该 Node 上。一旦可以运行，调度器就记录下这个 Node 的名字和被删除 Pod 的列表，这就是一次抢占过程的结果了。当遍历完所有的节点之后，调度器会在上述模拟产生的所有抢占结果里做一个选择，找出最佳结果。而这一步的判断原则，就是**尽量减少抢占对整个系统的影响**。

在得到了最佳的抢占结果之后，调度器就可以真正开始抢占的操作：

 1. 调度器会检查牺牲者列表，清理这些 Pod 所携带的 nominatedNodeName 字段。
 2. 调度器会把抢占者的 nominatedNodeName，设置为被抢占的 Node 的名字。这里对抢占者 Pod 的更新操作，就会触发让抢占者在下一个调度周期重新进入调度流程。
 3. 调度器会开启一个 Goroutine，异步地删除牺牲者。

在为某一对 Pod 和 Node 执行 Predicates 算法的时候，如果待检查的 Node 是一个即将被抢占的节点。那么调度器就会对这个 Node ，将同样的 Predicates 算法运行两遍：

 1. 调度器会假设上述“潜在的抢占者”已经运行在这个节点上，然后执行 Predicates 算法，这一步只需要考虑那些优先级等于或者大于待调度 Pod 的抢占者；这一步的原因是由于 InterPodAntiAffinity 规则的存在。
 2. 调度器会正常执行 Predicates 算法，即：不考虑任何“潜在的抢占者”。这一步是因为“潜在的抢占者”最后不一定会运行在待考察的 Node 上。

只有这两遍 Predicates 算法都能通过时，这个 Pod 和 Node 才会被认为是可以绑定（bind）的。

#### Device Plugin机制

以 NVIDIA 的 GPU 设备为例，当用户的容器被创建之后，容器里必须出现如下两部分：

 - GPU 设备路径，正是该容器启动时的 Devices 参数；
 - 驱动目录，则是该容器启动时的 Volume 参数。

kubelet 将上述两部分内容设置在了创建该容器的 CRI参数里面。这样，等到该容器启动之后，对应的容器里就会出现 GPU 设备和驱动的路径了。

Kubernetes 在 Pod 的 API 对象里，并没有为 GPU 专门设置一个资源类型字段，而是使用了一种叫作 Extended Resource（ER）的特殊字段来负责传递 GPU 的信息。
在 Kubernetes 中，对所有硬件加速设备进行管理的功能，都是由一种叫作 Device Plugin 的插件来负责的。其中也就包括了对该硬件的 Extended Resource 进行汇报的逻辑。

Kubernetes 的 Device Plugin 机制：
![](https://static001.geekbang.org/resource/image/5d/85/5db13d33cb647f33c62837e9cccdfb85.png)
对于每一种硬件设备，都需要有它所对应的 Device Plugin （如[FPGA](https://github.com/intel/intel-device-plugins-for-kubernetes)、[SRIOV](https://github.com/intel/sriov-network-device-plugin)、[RDMA](https://github.com/hustcat/k8s-rdma-device-plugin)）进行管理，这些 Device Plugin，都通过 gRPC 的方式，同 kubelet 连接起来。

kubelet 会负责从它所持有的硬件设备列表中，为容器挑选一个硬件设备，然后调用 Device Plugin 的 Allocate API 来完成这个分配操作。

Kubernetes 里对硬件设备的管理，**只能处理“设备个数”这唯一一种情况**。一旦你的设备是异构的、不能简单地用“数目”去描述具体使用需求的时候，Device Plugin 就完全不能处理了。在很多场景下，我们其实希望在调度器进行调度的时候，就可以根据整个集群里的某种硬件设备的全局分布，做出一个最佳的调度选择。
Kubernetes 里缺乏一种能够对 Device 进行描述的 API 对象。如果你的硬件设备本身的属性比较复杂，并且 Pod 也关心这些硬件的属性的话，那么 Device Plugin 也是完全没有办法支持的。


## 运行时

## 监控与日志

## 社区


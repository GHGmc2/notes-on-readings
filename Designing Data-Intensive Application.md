# Designing Data-Intensive Application
> [官网](https://dataintensive.net/), [references](https://github.com/ept/ddia-references), [errata](https://www.oreilly.com/catalog/errata.csp?isbn=0636920032175)
> [翻译](https://github.com/Vonng/ddia)
> [douban](https://book.douban.com/subject/26197294/)
> dig deeper than buzzwords!

## Preface

Fortunately, behind the rapid changes in technology, there are enduring principles that remain true. If you understand those principles, you’re in a position to see where each tool fits in, how to make good use of it, and how to avoid its pitfalls.

We look primarily at the **architecture** of data systems and the ways they are integrated into data-intensive applications.

# Foundations of Data System

## Reliable, Scalable, and Maintainable Applications

Many applications today are data-intensive. Raw CPU power is rarely a limiting factor for these applications—bigger problems are usually **the amount of data, the complexity of data, and the speed at which it is changing**.

The fundamentals of what we are trying to achieve: reliable, scalable, and maintainable data systems.

 - Reliability: The system should continue to work correctly (performing the correct function at the desired level of performance) even in the face of adversity (hardware or software faults, and even human error).
 - Scalability: As the system grows (in data volume, traffic volume, or complexity), there should be reasonable ways of dealing with that growth.
 - Maintainability: Over time, many different people will work on the system (engineering and operations, both maintaining current behavior and adapting the system to new use cases), and they should all be able to work on it productively.

### Thinking About Data Systems

### Reliability

The things that can go wrong are called **faults**, and systems that anticipate faults and can cope with them are called **fault-tolerant** or **resilient**.

A **fault** is usually defined as **one component** of the system deviating from its spec, whereas a **failure** is when the **system as a whole** stops providing the required service to the user.
It is impossible to reduce the probability of a fault to zero; therefore it is usually best to **design fault-tolerance mechanisms that prevent faults from causing failures**.

[Chaos Monkey: a resiliency tool that helps applications tolerate random instance failures](https://github.com/Netflix/chaosmonkey)

We generally prefer **tolerating** faults over **preventing** faults, but there are cases (with security matters) where prevention is better than cure.

This book mostly deals with the kinds of faults that can be cured.

#### Hardware Faults

Ways:

 - add **redundancy** to the individual **hardware** components
 - **software fault-tolerance**

We usually think of hardware faults as being random and independent from each other.

#### Software Errors

Systematic errors within the system are harder to anticipate, and because they are **correlated** across nodes, they tend to cause many more system failures than uncorrelated hardware faults.

#### Human Errors

How do we make our systems reliable, in spite of unreliable humans? The best systems combine several approaches:

 - Design systems in a way that **minimizes opportunities for error**.
 - **Decouple** the places where people make the most mistakes from the places where they can cause failures.
 - **Test thoroughly** at all levels, from unit tests to whole-system integration tests and manual tests.
 - Allow **quick and easy recovery** from human errors, to minimize the impact in the case of a failure.
 - Set up detailed and clear **monitoring**, such as performance metrics and error rates.
 - Implement good **management** practices and training.

#### How Important Is Reliability?

### Scalability

Scalability is the term we use to describe a system’s **ability to cope with increased load**.

#### Describing Load

load parameters.

#### Describing Performance

The **response time** is what the client sees: besides the actual time to process the request (the service time), it includes network delays and queueing delays.
**Latency** is the duration that a request is waiting to be handled—during which it is latent, await‐ing service.

Usually it is better to use **percentiles**. If you take your list of response times and sort it from fastest to slowest, then the median is the halfway point, this makes the median a good metric if you want to know how long users typically have to wait.

High percentiles of response times, also known as tail latencies, are important because they directly affect users’ experience of the service.

#### Approaches for Coping with Load

Scaling up: vertical scaling, moving to a more powerful machine.
Scaling out: horizontal scaling, distributing the load across multiple smaller machines.

Distributing load across multiple machines is also known as a **shared-nothing architecture**.

While distributing **stateless** services across multiple machines is fairly straightforward, taking **stateful** data systems from a single node to a distributed setup can introduce a lot of additional complexity.

If you are working on a fast-growing service, it is therefore likely that you will need to rethink your architecture on every order of magnitude load increase — or perhaps even more often than that.
In an early-stage startup or an unproven product it’s usually more important to be able to iterate quickly on product features than it is to scale to some hypothetical future load.

### Maintainability

three design principles for software systems:

 - Operability: make it easy for operations teams to keep the system running smoothly.
 - Simplicity: make it easy for new engineers to understand the system, by removing as much complexity as possible from the system. (Note this is not the same as simplicity of the user interface.)
 - Evolvability: make it easy for engineers to make changes to the system in the future, adapting it for unanticipated use cases as requirements change. Also known as extensibility, modifiability, or plasticity.

#### Operability: Making Life Easy for Operations

Good operability means making routine tasks easy, allowing the operations team to focus their efforts on high-value activities.
Data systems can do various things to make routine tasks easy, including:

 - Providing visibility into the runtime behavior and internals of the system, with good monitoring.
 - Providing good support for automation and integration with standard tools.
 - Avoiding dependency on individual machines.
 - Providing good documentation and an easy-to-understand operational model.
 - Providing good default behavior, but also giving administrators the freedom to override defaults when needed.
 - Self-healing where appropriate, but also giving administrators manual control over the system state when needed.
 - Exhibiting predictable behavior, minimizing surprises.

#### Simplicity: Managing Complexity

Making a system simpler does not necessarily mean reducing its functionality; it can also mean removing accidental complexity.

One of the best tools we have for removing accidental complexity is **abstraction**.

#### Evolvability: Making Change Easy

## Data Models and Query Languages

### Relational Model Versus Document Model

### Query Languages for Data

#### Declarative Queries on the Web

#### MapReduce Querying

### Graph-Like Data Models

#### Property Graphs

#### The Cypher Query Language

#### Graph Queries in SQL

#### Triple-Stores and SPARQL

#### The Foundation: Datalog

### Summary

Each data model comes with its own query language or framework.

## Storage and Retrieval

### Data Structures That Power Your Database

#### Hash Indexes

#### SSTables and LSM-Trees

#### B-Trees

#### Comparing B-Trees and LSM-Trees

#### Other Indexing Structures

### Transaction Processing or Analytics?

#### Data Warehousing

#### Stars and Snowflakes: Schemas for Analytics

### Column-Oriented Storage

#### Column Compression

#### Sort Order in Column Storage

#### Writing to Column-Oriented Storage

#### Aggregation: Data Cubes and Materialized Views

### Summary

Storage engines fall into two broad categories: those optimized for transaction processing (OLTP), and those optimized for analytics (OLAP).

 - OLTP systems are typically user-facing, which means that they may see a huge volume of requests. In order to handle the load, applications usually only touch a small number of records in each query. **Disk seek time** is often the bottleneck here.
 - Data warehouses and similar analytic systems are primarily used by business analysts, not by end users. They handle a much lower volume of queries than OLTP systems, but each query is typically very demanding, requiring many millions of records to be scanned in a short time. **Disk bandwidth** is often the bottleneck here, and column- oriented storage is an increasingly popular solution for this kind of workload.

On the OLTP side, we saw **storage engines** from two main schools of thought:

 - The **log-structured** school, which only permits appending to files and deleting obsolete files, but never updates a file that has been written. Their key idea is that they systematically turn random-access writes into **sequential writes** on disk, which enables higher write throughput due to the performance characteristics of hard drives and SSDs.
 - The **update-in-place** school, which treats the disk as a set of fixed-size pages that can be overwritten. **B-trees** are the biggest example of this philosophy.

## Encoding and Evolution

### Formats for Encoding Data

### Modes of Dataflow

#### Dataflow Through Databases

#### Dataflow Through Services: REST and RPC

##### Web services

**REST** is not a protocol, but rather a design philosophy that builds upon the principles of HTTP. It emphasizes simple data formats, **using URLs for identifying resources** and **using HTTP features for cache control, authentication, and content type negotiation**.

##### The problems with RPCs

RPC is fundamentally flawed. A network request is very different from a local function call:

 - 

##### Current directions for RPC

This new generation of RPC frameworks is more explicit about the fact that a remote request is different from a local function call.

Custom RPC protocols with a binary encoding format can achieve better performance than something generic like JSON over REST.
However, a RESTful API has other significant advantages: it is good for experimentation and debugging, it is supported by all mainstream programming languages and platforms, and there is a vast ecosystem of tools available.

REST seems to be the predominant style for public APIs.
The main focus of RPC frameworks is on requests between services owned by the same organization, typically within the same datacenter.

##### Data encoding and evolution for RPC

#### Message-Passing Dataflow

### Summary

The details of encodings affect not only their efficiency, but more importantly also the architecture of applications and your options for deploying them.

# Distributed Data

Reasons why you might want to distribute a database across multiple machines:

 - Scalability
 - Fault tolerance/high availability
 - Latency

architecture:

 - Shared-memory architecture: may offer limited fault tolerance, but it is definitely limited to a single geographic location.
 - Shared-disk architecture: is used for some data warehousing workloads, but contention and the overhead of locking limit the scalability of the shared-disk approach.
 - **Shared-nothing architecture**: each node uses its CPUs, RAM, and disks independently. Any coordination between nodes is done at the software level, using a conventional network.

No special hardware is required by a shared-nothing system, so you can use whatever machines have the best price/performance ratio. You can potentially distribute data across multiple geographic regions, and thus reduce latency for users and potentially be able to survive the loss of an entire datacenter.

In this part of the book, we focus on shared-nothing architectures.

There are two common ways data is distributed across multiple nodes, they often go hand in hand:

 - Replication: keeping a copy of the **same data** on several different nodes, potentially in different locations.
 - Partitioning (sharding): splitting a big database into smaller **subsets** called partitions so that different partitions can be assigned to different nodes.

## Replication

### Leaders and Followers

#### Synchronous Versus Asynchronous Replication

#### Setting Up New Followers

#### Handling Node Outages

#### Implementation of Replication Logs

### Problems with Replication Lag

#### Reading Your Own Writes

#### Monotonic Reads

#### Consistent Prefix Reads

#### Solutions for Replication Lag

### Multi-Leader Replication

#### Use Cases for Multi-Leader Replication

#### Handling Write Conflicts

#### Multi-Leader Replication Topologies

### Leaderless Replication

#### Writing to the Database When a Node Is Down

#### Limitations of Quorum Consistency

#### Sloppy Quorums and Hinted Handoff

#### Detecting Concurrent Writes

### Summary

Replication can serve several **purposes**:

 - High availability
 - Disconnected operation: allowing an application to continue working when there is a network interruption.
 - Latency
 - Scalability

Three main **approaches** to replication:

 - **Single-leader replication**: clients send all **writes** to a single node (the leader), which sends a stream of data change events to the other replicas (followers). **Reads** can be performed on any replica, but reads from followers might be stale.
 - Multi-leader replication: clients send each **write** to one of several leader nodes, any of which can accept writes. The leaders send streams of data change events to each other and to any follower nodes.
 - Leaderless replication: clients send each write to several nodes, and read from several nodes **in parallel** in order to detect and correct nodes with stale data.

Consistency models:

 - Read-after-write consistency: users should always see data that they submitted themselves.
 - Monotonic reads: after users have seen the data at one point in time, they shouldn’t later see the data from some earlier point in time.
 - Consistent prefix reads: users should see the data in a state that makes causal sense.

## Partitioning

The main reason for wanting to partition data is **scalability**.

### Partitioning and Replication

### Partitioning of Key-Value Data

#### Partitioning by Key Range

#### Partitioning by Hash of Key

#### Skewed Workloads and Relieving Hot Spots

### Partitioning and Secondary Indexes

#### Partitioning Secondary Indexes by Document

#### Partitioning Secondary Indexes by Term

### Rebalancing Partitions

#### Strategies for Rebalancing

#### Operations: Automatic or Manual Rebalancing

### Request Routing

#### Parallel Query Execution

### Summary

The **goal** of partitioning is to spread the data and query load evenly across multiple machines, avoiding hot spots.

Two main approaches to partitioning:

 - Key range partitioning: keys are **sorted**, and **a partition owns all the keys from some minimum up to some maximum**. Sorting has the advantage that efficient range queries are possible.
 - Hash partitioning: a hash function is applied to each key, and **a partition owns a range of hashes**.

Hybrid approaches like **compound key**: using one part of the key to identify the partition and another part for the sort order.

A secondary index also needs to be partitioned, and there are two methods:

 - Document-partitioned indexes (local indexes): the secondary indexes are stored in the **same partition** as the primary key and value.
 - Term-partitioned indexes (global indexes): the secondary indexes are partitioned **separately**, using the indexed values.

routing

## Transactions

Transactions were created with a **purpose**, namely to simplify the programming model for applications accessing a database.

### The Slippery Concept of a Transaction

#### ACID

#### Single-Object and Multi-Object Operations

### Weak Isolation Levels

#### Read Committed

#### Snapshot Isolation and Repeatable Read

#### Preventing Lost Updates

#### Write Skew and Phantoms

A **write** in one transaction changes the result of a **search** query in another transaction, is called a **phantom**.

### Serializability

#### Actual Serial Execution

#### Two-Phase Locking (2PL)

#### Serializable Snapshot Isolation (SSI)

### Summary

Several widely used isolation levels: 

 - read committed
 - snapshot isolation (sometimes called repeatable read)
 - serializable

We characterized those isolation levels by discussing various examples of race conditions:

 - Dirty reads
 - Dirty writes
 - Read skew (nonrepeatable reads)
 - Lost updates
 - Write skew
 - Phantom reads

Only serializable isolation protects against all of these issues. Three different approaches to implementing serializable transactions:

 - Literally executing transactions in a serial order
 - Two-phase locking
 - Serializable snapshot isolation (SSI)

## The trouble with Distributed Systems

### Faults and Partial Failures

#### Cloud Computing and Supercomputing

### Unreliable Networks

#### Network Faults in Practice

#### Detecting Faults

#### Timeouts and Unbounded Delays

#### Synchronous Versus Asynchronous Networks

### Unreliable Clocks

#### Monotonic Versus Time-of-Day Clocks

#### Clock Synchronization and Accuracy

#### Relying on Synchronized Clocks

#### Process Pauses

### Knowledge, Truth, and Lies

#### The Truth Is Defined by the Majority

#### Byzantine Faults

#### System Model and Reality

### Summary

Problems that can occur in distributed systems:

 - A **packet** over the network may be **lost or arbitrarily delayed**
 - A node’s **clock** may be significantly **out of sync** with other nodes
 - A **process** may pause for a substantial amount of time at any point in its execution

The fact that such **partial failures** can occur is the defining characteristic of distributed systems.

In distributed systems, we try to **build tolerance of partial failures into software**, so that the system as a whole may continue functioning even when some of its constituent parts are broken.

To tolerate faults, the first step is to **detect** them, but even that is hard. Most distributed algorithms rely on timeouts to determine whether a remote node is still available. However, **timeouts can’t distinguish between network and node failures**.

Once a fault is detected, making a system **tolerate** it is not easy either. Major decisions cannot be safely made by a single node, so we require protocols that enlist help from other nodes and try to get a quorum to agree.

## Consistency and Consensus

### Consistency Guarantees

### Linearizability

#### What Makes a System Linearizable?

#### Relying on Linearizability

#### Implementing Linearizable Systems

#### The Cost of Linearizability

### Ordering Guarantees

#### Ordering Guarantees

#### Sequence Number Ordering

#### Total Order Broadcast

### Distributed Transactions and Consensus

#### Atomic Commit and Two-Phase Commit (2PC)

#### Distributed Transactions in Practice

#### Fault-Tolerant Consensus

#### Membership and Coordination Services

### Summary

**Linearizability** puts all operations in a single, totally ordered timeline. It has the downside of being **slow**, especially in environments with large network delays.

**Causality** provides us with a weaker consistency model: some things can be concurrent, so the version history is like a timeline with branching and merging. Causal consistency does not have the coordination overhead of linearizability and is **much less sensitive to network problems**.

It turns out that a wide range of problems are actually reducible to consensus and are equivalent to each other. Such equivalent problems include:

 - Linearizable compare-and-set registers
 - Atomic transaction commit
 - Total order broadcast
 - Locks and leases
 - Membership/coordination service
 - Uniqueness constraint

However, if the only single leader fails, or if a network interruption makes the leader unreachable, such a system becomes unable to make any progress. There are three ways of handling that situation:

 - **Wait** for the leader to recover, and accept that the system will be blocked in the meantime.
 - **Manually** fail over by getting humans to choose a new leader node and reconfigure the system to use it.
 - Use a **consensus algorithm** to **automatically** choose a new leader.

# Derived Data

On a high level, systems that store and process data can be grouped into two broad categories:

 - Systems of record
 - Derived data systems

The distinction between system of record and derived data system depends not on the tool, but on **how you use it in your application**.

## Batch Processing

Three different types of systems:

 - Services (online systems): response time is usually the primary measure of performance of a service, and availability is often very important.
 - Batch processing systems (offline systems): the primary performance measure of a batch job is usually throughput.
 - Stream processing systems (near-real-time systems): a stream processor consumes inputs and produces outputs (rather than responding to requests) with lower latency than the equivalent batch systems.

### Batch Processing with Unix Tools

#### Simple Log Analysis

#### The Unix Philosophy

##### A uniform interface

##### Separation of logic and wiring

##### Transparency and experimentation

### MapReduce and Distributed Filesystems

#### MapReduce Job Execution

#### Reduce-Side Joins and Grouping

##### Sort-merge joins

##### Bringing related data together in the same place

##### GROUP BY

##### Handling skew

#### Map-Side Joins

##### Broadcast hash joins

##### Partitioned hash joins

##### Map-side merge joins

##### MapReduce workflows with map-side joins

#### The Output of Batch Workflows

#### Comparing Hadoop to Distributed Databases

### Beyond MapReduce

#### Materialization of Intermediate State

The process of writing out this intermediate state to files is called **materialization**.

##### Dataflow engines

They **handle an entire workflow as one job**, rather than breaking it up into independent subjobs.

##### Fault tolerance

Spark uses the RDD abstraction for tracking the ancestry of data, while Flink checkpoints operator state, allowing it to resume running an operator that ran into a fault during its execution.

When recomputing data, it is important to know whether the computation is **deterministic**.

##### Discussion of materialization

#### Graphs and Iterative Processing

##### The Pregel processing model

##### Fault tolerance

##### Parallel execution

#### High-Level APIs and Languages

### Summary

The two main problems that distributed batch processing frameworks need to solve are:

 - Partitioning: in MapReduce, **mappers** are partitioned according to input file blocks. The output of mappers is repartitioned, sorted, and merged into a configurable number of **reducer** partitions.
 - Fault tolerance: 

How partitioned algorithms work:

 - Sort-merge joins
 - Broadcast hash joins: start a mapper for each partition of the **large** join input, load the hash table for the **small** input into each mapper.
 - Partitioned hash joins: the hash table approach is used independently for each partition.

Distributed batch processing engines have a **deliberately restricted programming model**: callback functions (such as mappers and reducers) are assumed to be **stateless** and to have no externally visible side effects besides their designated output.
This restriction allows the framework to hide some of the hard distributed systems problems behind its abstraction: in the face of crashes and network issues, tasks can be **retried safely**, and the output from any failed tasks is discarded.

## Stream Processing

### Transmitting Event Streams

#### Messaging Systems

#### Partitioned Logs

##### Using logs for message storage

A **log** is simply an **append-only** sequence of records on disk.

A producer sends a message by appending it to the end of the log, and a consumer receives messages by reading the log sequentially.

The log can be **partitioned**. Different partitions can then be hosted on different machines, making each partition a separate log that can be read and written **independently** from other partitions. A **topic** can then be defined as a group of partitions that all carry messages of the same type.

Within each partition, the broker assigns a monotonically increasing sequence number, or **offset**, to every message. A partition is append-only, so the messages within a partition are totally ordered. There is no ordering guarantee across different partitions.

##### Logs compared to traditional messaging

### Databases and Streams

#### Keeping Systems in Sync

#### Change Data Capture

#### Event Sourcing

#### State, Streams, and Immutability

### Processing Streams

#### Uses of Stream Processing

#### Reasoning About Time

#### Stream Joins

#### Fault Tolerance

### Summary

Two types of message brokers:

 - AMQP/JMS-style message broker
 - Log-based message broker: 

**Representing databases as streams** opens up powerful opportunities for integrating systems. You can keep derived data systems such as search indexes, caches, and analytics systems continually up to date by consuming the log of changes and applying them to the derived system. You can even build fresh views onto existing data by starting from scratch and consuming the log of changes from the beginning all the way to the present.

The facilities for **maintaining state as streams and replaying messages** are also the basis for the techniques that enable stream joins and fault tolerance in various stream processing frameworks.

Three types of joins that may appear in stream processes:

 - Stream-stream joins
 - Stream-table joins
 - Table-table joins

## The Future of Data Systems

### Data Integration

#### Combining Specialized Tools by Deriving Data

#### Batch and Stream Processing

##### The lambda architecture

##### Unifying batch and stream processing

Unifying batch and stream processing in one system requires the following features:

 - The ability to replay historical events **through the same processing engine** that handles the stream of recent events.
 - **Exactly-once semantics** for stream processors — that is, ensuring that the output is the same as if no faults had occurred, even if faults did in fact occur.
 - Tools for **windowing by event time**, not by processing time.

### Unbundling Databases

#### Composing Data Storage Technologies

#### Designing Applications Around Dataflow

#### Observing Derived State

### Aiming for Correctness

#### The End-to-End Argument for Databases

#### Enforcing Constraints

#### Timeliness and Integrity

#### Trust, but Verify

### Doing the Right Thing

#### Predictive Analytics

#### Privacy and Tracking

### Summary

By making derivations and transformations asynchronous and loosely coupled, a problem in one area is prevented from spreading to unrelated parts of the system, increasing the robustness and fault-tolerance of the system as a whole.

Derived state can be updated by observing changes in the underlying data.

**Strong integrity guarantees** can be implemented scalably with asynchronous event processing, by using **end-to-end operation identifiers** to make operations idempotent and by **checking constraints asynchronously**. This approach is much more scalable and robust than the traditional approach of using distributed transactions, and fits with how many business processes work in practice.

By **structuring applications around dataflow and checking constraints asynchronously**, we can avoid most coordination and create systems that maintain integrity but still perform well, even in geographically distributed scenarios and in the presence of faults.


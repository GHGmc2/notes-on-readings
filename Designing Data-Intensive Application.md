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

### Graph-Like Data Models

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

## Encoding and Evolution

### Formats for Encoding Data

### Modes of Dataflow

#### Dataflow Through Databases

#### Dataflow Through Services: REST and RPC

#### Message-Passing Dataflow

# Distributed Data

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

## Partitioning

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

## Transactions

### The Slippery Concept of a Transaction

#### ACID

#### Single-Object and Multi-Object Operations

### Weak Isolation Levels

#### Read Committed

#### Snapshot Isolation and Repeatable Read

#### Preventing Lost Updates

#### Write Skew and Phantoms

### Serializability

#### Actual Serial Execution

#### Two-Phase Locking (2PL)

#### Serializable Snapshot Isolation (SSI)

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

# Derived Data

## Batch Processing

### Batch Processing with Unix Tools

#### Simple Log Analysis

#### The Unix Philosophy

### MapReduce and Distributed Filesystems

#### MapReduce Job Execution

#### Reduce-Side Joins and Grouping

#### Map-Side Joins

#### The Output of Batch Workflows

#### Comparing Hadoop to Distributed Databases

### Beyond MapReduce

#### Materialization of Intermediate State

#### Graphs and Iterative Processing

#### High-Level APIs and Languages

## Stream Processing

### Transmitting Event Streams

#### Messaging Systems

#### Partitioned Logs

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

## The Future of Data Systems

### Data Integration

#### Combining Specialized Tools by Deriving Data

#### Batch and Stream Processing

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

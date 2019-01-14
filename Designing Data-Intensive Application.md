# Designing Data-Intensive Application
> [官网](https://dataintensive.net/), [references](https://github.com/ept/ddia-references), [errata](https://www.oreilly.com/catalog/errata.csp?isbn=0636920032175)
> [翻译](https://github.com/Vonng/ddia)
> [douban](https://book.douban.com/subject/26197294/)

## Preface



# Foundations of Data System

## Reliable, Scalable, and Maintainable Applications

### Reliability

#### Hardware Faults

#### Software Errors

#### Human Errors

### Scalability

#### Describing Load

#### Describing Performance

#### Approaches for Coping with Load

### Maintainability

#### Operability

#### Simplicity

#### Evolvability

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

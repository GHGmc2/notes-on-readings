# A Tour of C++
> [官网](http://www.stroustrup.com/Tour.html)
> [douban](https://book.douban.com/subject/30430874/)

### Preface

This book gives an overview of C++ as defined by C++17 and implemented by the major C++ suppliers. In addition, it mentions concepts and modules, as defined in ISO Technical Specifications and in current use, but not scheduled for inclusion into the standard until C++20.

## The Basics

### Programs

For a program to run, its source text has to be processed by a compiler, producing object files, which are combined by a linker yielding an executable program.

The ISO C++ standard defines two kinds of entities:

 - *Core language features*, such as built-in types and loops
 - *Standard-library components*, such as containers and I/O operations

C++ is a statically typed language.

Every C++ program must have exactly one global function named main(). The program starts by executing that function.

A nonzero value from main() indicates failure.

### Functions

A function cannot be called unless it has been previously declared.

Unless the argument declaration is also a function definition, the compiler simply ignores such names.

The type of a function consists of its return type and the sequence of its argument types.

A function can be the member of a class. For such a *member function*, the name of its class is also part of the function type.

Defining multiple functions with the same name is known as *function overloading* and is one of the essential parts of generic programming.

### Types, Variables and Arithmetic

The size of a type is implementation-defined and can be obtained by the `sizeof` operator.

#### Arithmetic

*The usual arithmetic conversions* aim to ensure that expressions are computed at the highest precision of its operands.

#### Initialization

Before an object can be used, it must be given a value.

The problems caused by implicit *narrowing conversions* are a price paid for C compatibility.

We use `auto` where we don’t have a specific reason to mention the type explicitly. “Specific reasons” include:

 - The definition is in a large scope where we want to make the type clearly visible to readers of our code.
 - We want to be explicit about a variable’s range or precision.

### Scope and lifetime

Scope:

 - *Local scope*: A name declared in a function or lambda is called a local name.
 - *Class scope*: A name is called a *member name* (or a class member name) if it is defined in a class, outside any function, lambda, or `enum class`.
 - *Namespace scope*: A name is called a *namespace member name* if it is defined in a namespace outside any function, lambda, class, or `enum class`.

An object created by `new` “lives” until destroyed by `delete`.

### Constants

C++ supports two notions of immutability:

 - `const`: This is used primarily to specify interfaces so that data can be passed to functions using pointers and references without fear of it being modified. The value of a `const` **can** be calculated at **run time**.
 - `constexpr`: This is used primarily to specify constants, to allow placement of data in read-only memory, and for performance. The value of a `constexpr` **must** be calculated by the **compiler**.

For a function to be usable in a *constant expression*, that is, in an expression that will be evaluated by the compiler, it must be defined `constexpr`.

We allow a `constexpr` function to be called with non-constant-expression arguments in contexts that do not require constant expressions.

To be `constexpr`, a function must be rather simple and cannot have side effects and can only use information passed to it as arguments. In particular, it cannot modify non-local variables.

### Pointers, Arrays and References

In declarations, `[ ]` means “array of”, `*` means “pointer to" and `&` means “reference to”.

In an expression, prefix unary `*` means “contents of” and prefix unary `&` means “address of.”

A **reference** is similar to a **pointer**, except that you don’t need to use a prefix `*` to access the value referred to by the reference. Also, a reference cannot be made to refer to a different object after its initialization.

When we don’t want to modify an argument but still don’t want the cost of copying, we use a `const` reference.

When used in declarations, operators (such as `&`, `*`, and `[ ]`) are called *declarator operators*.

#### The Null Pointer

There is only one `nullptr` shared by all pointer types.

There is no “null reference.” A reference must refer to a valid object (and implementations assume that it does).

### Tests

### Mapping to Hardware

#### Assignment

An assignment of a built-in type is a simple machine copy operation.

A reference and a pointer both refer/point to an object and both are represented in memory as a machine address. However, the language rules for using them differ. Assignment to a **reference** does not change what the reference refers to but **assigns to the referenced object**.

#### Initialization

### Advice
> [The C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines)

 - You don’t have to know every detail of C++ to write good programs.
 - Focus on programming techniques, not on language features.

## User-Defined Types

### Structures

### Classes

### Unions

### Enumerations

### Advice

## Modularity

### Separate Compilation

### Modules (C++20)

### Namespaces

### Error Handling

### Function Arguments and Return Values

### Advice

## Classes

### Concrete Types

### Abstract Types

### Virtual Functions

### Class Hierarchies

### Advice

## Essential Operations

### Copy and Move

### Resource Management

### Conventional Operations

### Advice

## Templates

### Parameterized Types

### Parameterized Operations

### Template Mechanisms

### Advice

## Concepts and Generic Programming

### Concepts

### Generic Programming

### Variadic Templates

### Template Compilation Model

### Advice

## Library Overview

### Standard-Library Components

### Standard-Library Headers and Namespace

### Advice

## Strings and Regular Expressions

### Strings

### String Views

### Regular Expressions

### Advice

## Input and Output

### Output

### Input

### I/O State

### I/O of User-Defined Types

### Formatting

### File Streams

### String Streams

### C-style I/O

### File System

### Advice

## Containers

### `vector`

### `list`

### `map`

### `unordered_map`

### Container Overview

### Advice

## Algorithms

### Use of Iterators

### Iterator Types

### Stream Iterators

### Predicates

### Algorithm Overview

### Concepts

### Container Algorithms

### Parallel Algorithms

### Advice

## Utilities

### Resource Management

### Range Checking: `span`

### Specialized Containers

### Alternatives

### Time

### Function Adaption

### Allocators

### Type Functions

### Advice

## Numerics

### Mathematical Functions

### Numerical Algorithms

### Complex Numbers

### Random Numbers

### Vector Arithmetic

### Numeric Limits

### Advice

## Concurrency

### Tasks and `thread`s

### Passing Arguments

### Returning Results

### Sharing Data

### Waiting for Events

### Communicating Tasks

### Advice

## History and Compatibility

### C/C++ Compatibility

### Advice

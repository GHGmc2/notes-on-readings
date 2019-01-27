# Effective Modern C++
> [emc++ errata](http://www.aristeia.com/BookErrata/emc++-errata.html)
> [douban原版](https://book.douban.com/subject/25923597/)
> [翻译1](https://vivym.gitbooks.io/effective-modern-cpp-zh/content/), [翻译2](https://github.com/racaljk/EffectiveModernCppChinese)

## 资源
> [读书笔记1](https://zhuanlan.zhihu.com/p/21102748), [2](https://zhuanlan.zhihu.com/p/21264013), [3](https://zhuanlan.zhihu.com/p/21722362), [4](https://zhuanlan.zhihu.com/p/22002842), [5](https://zhuanlan.zhihu.com/p/25057478)

# Notes
> 动机 -> 语法 -> 利弊 -> 案例

## 1 类别推导

## 2 auto

### 优先选用auto，而非显式类型声明

## 3 转向现代C++

### 创建对象时区分()和{}

### 优先选用nullptr，而非0或NULL

### 优先选用别名声明，而非typedef

### 优先选用限定作用域的枚举类型，而非不限定作用域的枚举类型

### 为在意改写的函数添加override声明

### 优先选用const_iterator，而非iterator

### 保证const成员函数的线程安全性

## 4 智能指针

裸指针的坑：TODO.

C++11 中共有四种，都是为管理动态分配对象的生命周期而设计的。通过保证这样的对象在适当的时机以适当的方式析构（包括发生异常的场合），来防止资源泄漏。

 - std::auto_ptr
 - std::unique_ptr：用std::unique_ptr替换std::auto_ptr
 - std::shared_ptr
 - std::weak_ptr

### 使用std::unique_ptr管理具备*专属*所有权的资源

### 使用std::shared_ptr管理具备*共享*所有权的资源

### 对于类似std::shared_ptr但有可能空悬的指针使用std::weak_ptr

## 5 右值引用、移动语义和完美转发

移动语义：使得编译器能使用不那么昂贵的**移动**操作，来替换昂贵的**复制**操作。

完美转发：使可以撰写**接受任意实参**的函数模板，并将其转发到其他函数。

右值引用：将移动语义和完美转发**胶合**起来的底层语言机制。

**形参总是左值**，即使其类型是右值引用。

## 6 lambda表达式

## 7 并发API

## 8 微调

# Python数据科学手册
> [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/), [Github](https://github.com/jakevdp/PythonDataScienceHandbook)
> [douban](https://book.douban.com/subject/27667378/)

# [SciPy](https://www.scipy.org/)

Core packages:
 - **NumPy**: Base N-dimensional array package
 - SciPy library: Fundamental library for scientific computing
 - **Matplotlib**: Comprehensive 2D Plotting
 - **IPython**: Enhanced Interactive Console
 - Sympy: Symbolic mathematics
 - **pandas**: Data structures & analysis

## [IPython](https://ipython.org/)
> [Docs](https://ipython.readthedocs.io/en/stable/)
> [API](https://ipython.readthedocs.io/en/stable/api/index.html)

使用 IPython（interactive Python）的两种方式：

 - IPython shell
 - IPython Notebook

IPython Notebook 其实只是通用 Jupyter Notebook 结构的特例（IPython shell 基于浏览器的图形界面），而 Jupyter Notebook 不仅支持 Python，还包括用于 Julia、R 和其他编程语言的 Notebook。

**IPython shell命令**

## [NumPy](http://www.numpy.org/)
> [Quickstart tutorial](https://docs.scipy.org/doc/numpy/user/quickstart.html)
> [NumPy functions by category](https://docs.scipy.org/doc/numpy/reference/routines.html)

NumPy（Numerical Python 的简称）提供了高效存储和操作密集 数据缓存的接口。

**NumPy 数组**几乎是整个 Python 数据科学工具生态系统的**核心**。

### 数据类型

创建数组（[Array creation routines](https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html)）

标准数据类型（[Data types](https://docs.scipy.org/doc/numpy/user/basics.types.html)）

### 数组基础
数组基本操作（[Array manipulation routines](https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html#basic-operations)）：

 - 属性（[The Basics](https://docs.scipy.org/doc/numpy/user/quickstart.html#the-basics)）
	 - nidm：维度
	 - shape：每个维度的大小（The length of the *shape* tuple is therefore the number of *ndim*）
	 - size：数组的总大小
	 - dtype：数据类型
	 - itemsize：每个数组元素字节大小
	 - nbytes：数组总字节大小。一般可以认为 nbytes = itemsize * size
 - 索引
 - 切分
 - 变形
 - 拼接和分裂

### 计算通用函数

NumPy 中的**向量**操作是通过通用函数（ufunc）实现的（将作用于数组中的每一个元素）。通用函数的主要目的是对 NumPy 数组中的值执行更快的重复操作。

通用函数（[Mathematical functions](https://docs.scipy.org/doc/numpy/reference/routines.math.html)）

 - 数组运算（算术运算符都是 NumPy 内置函数的简单封装器）
 - 绝对值abs(x)
 - 三角函数
 - 指数和对数：exp(x)；log(x)
 - 专用通用函数：双曲三角函数、比特位运算、比较运算符、弧度转化为角度的运算、取整和求余运算等
	 - scipy.special 模块

通用函数特性

 - 指定输出：指定out参数
 - 聚合
	 - reduce()：对给定的元素和操作重复执行，直至得到单个的结果
	 - accumulate()：存储每次计算的中间结果
 - 外积：outer(x, y)

### 聚合
> [Statistics](https://docs.scipy.org/doc/numpy/reference/routines.statistics.html)

 - 求和sum(x)
 - 最小值和最大值：min()和max()，可指定axis参数

### 广播

广播可理解为用于**不同大小数组**的二进制通用函数（加、减、乘等）的一组规则（p57例）：

 - 维度不相同，那么小维度数组的形状将会在**最左边维度补 1**
 - 如果两个数组的形状在任何一个维度上都不匹配，那么数组的形状会**沿着维度为 1 的维度扩展**，以匹配另外一个数组的形状
 - 如果两个数组的形状在任何一个维度上都不匹配并且没有任何一个维度等于 1，那么会引发异常

应用

 - 数组归一化（normalization，也叫标准化）
 - 基于二维函数显示图像

### 布尔掩码
> （[Masked arrays arithmetics](https://docs.scipy.org/doc/numpy/reference/routines.ma.html#masked-arrays-arithmetics)）

 - 比较操作（[comparision](https://docs.scipy.org/doc/numpy/reference/routines.logic.html#comparison)）：比较运算操作在 NumPy 中也是借助通用函数来实现的。可以用于任意形状、大小的数组
 - 操作布尔数组
	 - count_nonzero()统计个数
	 - 逐位逻辑运算符（[Logical operations](https://docs.scipy.org/doc/numpy/reference/routines.logic.html#logical-operations)）：注意and 和 or 判断整个对象是真或假，而 & 和 | 是指每个对象中的比特位
 - 将布尔数组作为掩码（masking operation: index on Boolean array，如 arr[arr > 0]）：通过该掩码选择数据的子数据集

### 花哨索引（fancy indexing）

花哨的索引传递一个**索引数组**来一次性获得多个数组元素，让我们能够快速获得并修改复杂的数组值的子数据集。

在花哨的索引中，索引值的配对遵循广播的规则。结果的形状与**广播后的索引数组的形状**一致（而不是与**被**索引数组的形状一致）。

**组合索引**：花哨的索引可以和其他索引方案（简单索引、切片、掩码）结合起来形成更强大的索引操作。

花哨的索引也可以被用于修改部分数组。

应用

 - 快速分割数据

### 排序

 - sort()：默认快排。可指定 axis 参数
 - argsort()：返回排序后的索引值
 - 分隔partition(x, k)：输出一个新数组，最左边是第 k 小的 k 个值，往右是任意顺序的其他值。类似还有argpartition() 计算分隔的索引值

应用：k 个最近邻（p78）

## pandas

可以把 pandas 对象看成增强版的 NumPy 结构化数组，行列都不再只是简单的整数索引，还可以带上标签。

### 对象简介

 - Series：**带索引**数据构成的**一维数组**，可以通过values 和 index 属性获取
	 - 看成 NumPy 一维数组：Series 对象**用一种显式定义的索引与数值关联，索引可以是任意想要的类型**。支持数组形式的操作，如切片
	 - 看成特殊字典：可直接用字典创建Series，索引默认按照顺序排列
 - DataFrame：既有灵活的**行索引**，又有灵活**列名**的二维数组。
	 - 看成 NumPy 二维数组：是**有序排列的若干 Series 对象**。index 属性可以获取（行）索引标签；columns 属性是存放列标签的 Index 对象
	 - 看成特殊字典：一列映射一个 Series 的数据
 - Index
	 - 看做不可变数组：（注：Index 对象的**索引不可变**，Index 对象的不可变特征使得多个 DataFrame 和数组之间进行**索引共享**时更加安全）
	 - 看做有序集合（set）：Index 对象遵循 Python 集合的许多习惯用法，如并、交、 差等

### 数据选择



### 数值运算

### 处理缺失值

### 层级索引

### 合并数据集

### 累计与分组

### 数据透视表

### 字符串操作

### 处理时间序列

### 高性能

## Matplotlib

### 线形图

### 散点图

### 密度图与等高线图

### 区间划分和分布密度

### 多子图

### 三维图

## Scikit-Learn

### 数据表示

### 评估器API

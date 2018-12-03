# Python编程快速上手
> [官网](https://automatetheboringstuff.com/)、[douban](https://book.douban.com/subject/26836700/)
> [PDF](https://studentportalen.uu.se/uusp-webapp/auth/webwork/filearea/download.action?nodeId=2053259&toolAttachmentId=475830&uusp.userId=guest)、[中文PDF](https://download.csdn.net/download/qq_25281937/9717244)

# Python编程基础

## 基础

数学操作符
| 操作符 | 操作 | 例子 | 值 |
|--|--|--|--|
| ** | 指数 | 2 ** 3 | 8 |
| // | 取商 | 22 // 8 | 2 |

变量名**区分大小写**，惯例小写开头.

#注释

## 控制流

布尔值 True 和 False 以**大写**开头.

布尔操作符：and、or和not（**优先级 not > and > or**）.

根据代码行的**缩进**判断代码块的开始和结束.

控制流语句都以冒号结尾，后跟代码块. 如
```python
if name == 'Alice':
    print('Hi, Alice.')
```

range()函数

 - range(stop)，**左闭右开**
 - range(start, stop[, step])

sys.exit()退出程序

## 函数（Functions）

def语句

None（N大写）值：表示没有值，是NoneType类型的唯一值. （类似其他语言的 null 或 nil）

关键字参数

作用域：局部、全局. 可用 global 语句使变量成为全局变量.

异常处理：try - except语句
```python
try:
    raise NameError('Hi, here.')
except NameError:
    do sth
```

## 列表（Lists）
> 方括号[]

负数下标：-1指倒数第1个下标，以此类推.

切片（slice）：返回新列表，如spam[1:4]，左闭右开.

连接：“+”操作符. 如[1, 2] + [3] 得 [1, 2, 3]

复制：“*”操作符. 如[1, 2] * 2 得 [1, 2, 1, 2]

删除值：del语句. 如del spam[2]，被删除值前移一个下标. del语句作用于变量时是“取消赋值”.

in 和 not in 操作符判断值是否在列表中，返回 True 或 False.

多重赋值：用列表中的值为多个变量赋值（变量数目需和列表长度相等）. 如
```python
cat = ['fat', 'black']
size, color = cat
```

列表方法：

 - index(x)：返回第一次出现x值的下标. 不存在则报ValueError
 - insert(i, x)：注意返回值是None
 - remove(x)：删除第一次出现的x值
 - sort()：支持关键字参数 key 和 reverse

**字符串**
字符串不可变.

**元组（tuple）**
> 圆括号(). 表示下标仍然用方括号

元组不可变.

若元组只有一个值，在值后面跟一个逗号. 如：('hello',)

可用 list() 和 tuple() 函数转换类型.

**引用**
当函数被调用时，参数值被复制给变元.

copy.copy()复制列表或字典.
copy.deepcopy()将同时复制列表内部的列表.

## 字典（Dictionaries）和结构化数据

字典：大括号{}，无序键-值对. 键可以为任意值？

字典方法

 - keys(), values(), items(): 分别返回**不能被修改**的键、值和键-值对列表
 - get(key[, default])
 - setdefault(key[, default])

## 字符串操作

字面量：单引号（''）或双引号（""）. 使用双引号时，字符串内可以有单引号字符.

转义字符：倒斜杠（\）

原始字符串：引号之前加上r，如：
```python
print(r'That is Carol\'s cat.')
```
多行字符串：三重引号（单、双均可）. 之间的引号、制表符或换行，都被认为是字符串一部分. 可用在多行注释.

字符串方法

 - isX()类
 - bytes.join(iterable): 返回字符串. 注意bytes是分隔符
 - split(sep): 返回列表
 - strip(), rstrip(), lstrip(): 删除空白字符

# 自动化任务

## 正则表达式

步骤

 1. import re
 2. re.compile() 函数创建Regex对象
 3. Regex的search()方法返回Match对象
 4. Match的group()方法返回第一次匹配的字符串；findall()方法返回所有匹配的字符串列表
```python
import re

phoneNumRegex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
mo = phoneNumRegex.search('My number is 415-555-4242')
print('Phone number found: ' + mo.group())
```

正则表达式符号（p128）

## 读写文件

OS X和Linux下，路径分隔符是正斜杠（/）

os方法

 - getcwd()：当前工作目录的字符串
 - makedirs(name)创建目录
 - os.path模块的文件名和文件路径函数：

读写文件步骤

 1. 调用open()返回File对象
 2. 调用File的read()或write()方法
 3. 调用File的close()关闭文件

## 调试

抛出异常：raise语句
```python
raise Exception('This is the error message')
```

断言格式：
```
assert condition, 'message when False'
```

日志：logging模块

## 时间

time和datatime模块

多线程：threading模块

启动其他程序：subprocess.Popen()

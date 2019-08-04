# Effective C++
> [errata](http://www.aristeia.com/BookErrata/ec++3e-errata.html)
> [豆瓣](https://book.douban.com/subject/5387403/)
> [pdf](https://doc.lagout.org/programmation/C/Addison.Wesley.Effective.CPP.3rd.Edition.May.2005.pdf)

## 资源
> [如何系统地学习 C++ 语言？](https://www.zhihu.com/question/23447320)
> [leveldb](https://github.com/google/leveldb), [LevelDB 实现分析](http://taobaofed.org/blog/2017/07/05/leveldb-analysis/)

# Notes

语法 -> 语义 -> 深层思维 -> 对象模型

### 导读

条款分两类
 - 一般性的设计策略
 - 带具体细节的特定语言特性

一个函数的签名等同于该函数的类型。C++对签名式的官方定义并不包括函数的返回类型。

`explicit`可用来阻止隐式类型转换。

*Pass-by-value*意味“调用copy构造函数”。

## 让自己习惯C++

### 01 视C++为一个语言联邦

C++是个多重范型编程语言，同时支持procedural、object-oriented、functional、generic、metaprogramming。

当从一个次语言移往另一个次语言，守则可能改变。C++主要的次语言：
 - C
 - Object-Oriented C++
 - Template C++
 - STL

对内置（即C-like）类型而言，*pass-by-value*通常比*pass-by-reference*高效。

### 02 尽量以 `const, enum, inline`替换`#define`
> prefer 编译器 to 预处理器

以常量替换`#define`有两种特殊情况：
 - 定义常量指针
 - class专属常量

enum hack

template incline函数：可以获得宏带来的效率以及一般函数的所有可预料行为和类型安全性（type safe）。
```c++
template<typename T>
inline void callWithMax(const T& a, const T& b) {
  f(a > b ? a : b);
}
```

对于单纯常量，最好以`const`对象或enums替换`#defines`。
对于形似函数的宏，最好改用inline函数替换`#defines`。

### 03 尽可能使用 `const`

`const`出现在星号`*`左边，表示被指物是常量；出现在星号`*`右边，表示指针自身是常量（`const`写在类型之前，或写在类型后、星号`*`前均可）：
```c++
char greeting[] = "hello";
const char* p = greeting; // non-const pointer, const data
char* const p = greeting; // const pointer, non-const data
```

STL迭代器的作用就像个 T* 指针。const_iterator就像声明 T* const，表示该迭代器不得指向不同的东西，但它所指的值是可以改变的。

`const`成员函数可作用于`const`对象身上：
```c++
class TextBlock {
  public:
    const char& operator[] (std::size_t position) const { // operator[] for const对象
      return text[position];
    }
    char& operator[] (std::size_t position) { // operator[] for non-const对象
      return text[position];
    }
};
```

bitwise const

### 04 确定对象被使用前已先被初始化

别混淆了赋值（assignment）和初始化（initialization）。
```c++
class PhoneNumber {...};
class ABEntry {
  public:
    ABEntry(const std::string& name, const std::list<PhoneNumber> &phones);
  private:
    std::string theName;
    std::list<PhoneNumber> thePhones;
};

// assignments
ABEntry::ABEntry(const std::string &name, const std::list<PhoneNumber> &phones) {
  theName = name;
  thePhones = phones;
}

// member initialization list
ABEntry::ABEntry(const std::string &name, const std::list<PhoneNumber> &phones): theName(name), thePhones(phones) {}
```

C++规定，对象的成员变量的初始化动作，发生在进入构造函数本体之前。

总是使用成员初始列。

成员初始化次序：base classes更早于其derived classes被初始化，而class的成员变量总是以其声明次序被初始化（即使它们在成员初值列中以不同的次序出现）。

C++对“定义于不同编译单元内的non-local static对象”的初始化顺序并无明确定义。请以local static对象替换non-local static对象。

## 构造/析构/赋值运算

### 05 了解C++默认生成并调用哪些函数

编译器可暗自为class创建（被调用时才会创建）：
 - *default*构造函数
 - *copy*构造函数
 - *copy assignment*操作符
 - 析构函数。是non-virtual的，除非base class声明了virtual析构函数。
所有这些函数都是public且incline的。

C++不允许让reference改指向不同对象。
若要在内含“reference成员”或“const成员”的class内支持assignment操作，必须自定义*copy assignment*操作符。


### 06 若不想使用编译器自动生成的函数，就明确拒绝

所有编译器产出的函数都是public。

藉由明确声明一个成员函数，可阻止编译器暗自创建其专属版本。
trick：将相应的member function声明为private且不实现它们。或使用类似Uncopyable这样的base class。

### 07 为多态基类声明`virtual`析构函数

vptr（virtual table pointer）指针指向一个由函数指针构成的数组，称为vtbl（virtual table），每一个带有virtual函数的class都有一个相应的vtbl。

带多态性质的base class应该声明一个virtual析构函数。只有当class内含至少一个virtual函数，才为它声明virtual析构函数。
反之，class的设计目的如果不是作为base classes使用，或不是为了具备多态性，就不该声明virtual析构函数。

拒绝继承不带virtual析构函数的class（如std::string，所有的STL容器如vector, list, set, unordered_map等）：
```c++
class SpecialString: public std::string {...};

SpecialString* pss = new SpecialString("Impending Doom");
std::string* ps;
ps = pss; // SpecialStirng* => std::string*
delete ps; // 错误！SpecialString的析构函数没被调用
```

pure virtual函数导致不能被实体化的abstract classes。

若你希望拥有抽象class，而手上没有任何pure virtual函数，可声明一个pure virtual析构函数：
```c++
class AWOV {
  public:
    virtual ~AWOV() = 0; // 声明pure virtual析构函数
}
// 你必须为这个pure virtual析构函数提供一份定义
AWOV::~AWOV() {}
```

析构函数的运作方式是：最深层派生的那个class析构函数最先被调用，然后其每一个base class的析构函数被调用。

### 08 别让异常逃离析构函数

在两个异常同时存在的情况下，程序若不是结束执行就是导致不明确行为。

析构函数绝对不要抛出异常。（可选择吞掉或结束程序）

### 09 绝不在构造和析构过程中调用`virtual`函数

在derived class对象的base class构造或析构期间，对象的类型是base class而不是derived class。virtual函数不会下降到derived class层。

### 10 令赋值操作符`operator=`返回一个 *reference to* `*this`

为了实现“连锁赋值”？

这个协议不仅适用于标准赋值，也适用于所有赋值相关运算，如`operator+=`等。

### 11 在`operator=`中处理“自我赋值”

trick包括：比较“来源对象”和目标对象的地址、精心周到的语句顺序、以及copy-and-swap。

让`operator=`具备“异常安全性”往往自动获得“自我赋值安全”的回报。

### 12 复制对象时勿忘其每个成分

*copying*函数：*copy*构造函数和*copy assignment*操作符。

用derived class的*copying*函数调用相应的base class函数。

assignment操作符只施行于已初始化对象身上。你不该令copy assignment操作符调用*copy*构造函数，或令*copy*构造函数调用*copy assignment*操作符。

## 资源管理

### 13 以对象管理资源

RAII（Resource Acquisition Is Initialization）

智能指针

### 14 在资源管理类中小心 *copying* 行为

复制RAII对象必须一并复制它所管理的资源。

普遍而常见的RAII class *copying*行为是：
 - 抑制*copying*
 - 施行引用计数法
 - 

### 15 在资源管理类中提供对原始资源的访问

每一个RAII class应该提供一个“取得其所管理之资源”的办法，可能经由显示转换（比较安全）或隐式转换（对客户比较方便）。

### 16 成对使用`new`和`delete`时要采取相同形式

`new` —— `delete`
`new ...[]` —— `delete[]`

### 17 在单独语句内将`new`ed对象置入智能指针

因为在“资源被创建”和“资源被转换为资源管理对象”两个时间点之间有可能发生异常。
```c++
std::shared_ptr<Widget> pw(new Widget);
```

编译器对于“跨越语句的各项操作”没有重新排列的自由（只有在语句内它才拥有那个自由度）。

## 设计与声明

### 18 让接口容易被正确使用，不易被误用

“促进正确使用”的办法：接口的一致性、与内置类型的行为兼容。
“阻止误用”的办法：建立新类型、限制类型上的操作，束缚对象值，以及消除客户的资源管理责任。

cross-DLL problem

### 19 设计 class 犹如设计 type

### 20 宁以 pass-by-reference-to-const 替换 pass-by-value

缺省情况下C++以*by value*方式传递对象至/来自函数。

slicing（对象切割）问题：derived class对象以*by value*方式传递并被视为一个base class对象，derived class对象的特化信息都会被切除。用*by reference*方式传递参数可以避免。

C++编译器的底层往往以指针来实现references，因此*pass by reference*通常意味真正传递的是指针。因此内置类型、STL的迭代器和函数对象用*pass by value*往往比*pass by reference*效率高些。

### 21 必须返回对象时，别妄想返回其reference

绝不要（never）：
 - 返回pointer或reference指向一个local stack对象；
 - 返回reference指向一个heap-allocated对象？
 - 返回pointer或reference指向一个local static对象而有可能同时需要多个这样的对象。

赋值的成本：对许多types而言它相当于调用一个析构函数加上一个构造函数。

### 22 将成员变量声明为 private

protected并不比public更具封装性。

### 23 宁以 non-member、non-friend 替换 member 函数

friends函数对class private成员的访问权力和member函数相同。

将所有便利函数放在多个头文件内但隶属于同一个命名空间。

### 24 若所有参数皆需类型转换，请为此采用 non-member 函数

只有当参数被列于参数列（parameter list）内，这个参数才是隐式类型转换的合格参与者。

无论何时能避免friend函数就该避免。

### 25 考虑写出一个不抛异常的`swap`函数

"template<>"表示全特化（total template specialization）。

C++只允许对class templates偏特化，在function templates身上偏特化是行不通的（惯常做法是添加一个重载版本）。

缺省版本的`swap`是以*copy*构造函数和*copy assignment*操作符为基础，这两者都允许抛出异常。

如果`swap`缺省实现版的效率不足（几乎总是意味着你的class或template使用了某种pimpl（pointer to implementation）手法），尝试：
 1. 提供一个public swap成员函数实现高效置换，这个函数绝不该抛出异常；
 2. 在你的class或template所在的命名空间内提供一个non-member `swap`，并令它调用上述`swap`成员函数；
 3. 如果你正在编写一个class（而非class template），为你的class特化`std::swap`。并令它调用你的`swap`成员函数。

## 实现

### 26 尽可能延后变量定义式的出现时间

### 27 尽量少做转型动作

转型语法
 - C风格：`(T) expression`
 - 函数风格：`T(expression)`
 - 新式转型
	 - `const_cast<T> (expression)`：将对象的常量性转除（cast away the constness）
	 - `dynamic_cast<T>(expression)`：执行“安全向下转型”（safe downcasting），效率差
	 - `reinterpret_cast<T>(expression)`：执行低级转型，不可移植
	 - `static_cast<T>(expression)`：强迫隐式转换（implicit conversions）

成员函数都有个隐藏的`this`指针。

尽量避免转型；宁可使用新式转型，不要使用旧式转型。

### 28 避免返回 handles 指向对象内部成分

References、指针和迭代器统统都是*handles*。

遵守这个条约可增加封装性。

### 29 为“异常安全”努力是值得的

异常安全性函数（exception-safe functions）在发生异常时：
 - 不泄漏任何资源
 - 不允许数据败坏

异常安全函数提供以下三个保证之一：
 - 基本承诺：如果异常被抛出，程序内的任何事物仍然保持在有效状态下。
 - 强烈保证：如果函数成功，就是完全成功；如果函数失败，程序会恢复到“调用函数之前”的状态。
 - 不抛掷（nothrow）保证：承诺绝不抛出异常。作用于内置类型身上的所有操作都提供nothrow保证。

强烈保证往往能够以copy and swap实现，即：在副本上做修改，然后在一个不抛出异常的操作中置换（swap）。实现上通常采用pimpl idiom。

### 30 透彻了解 inlining 的里里外外

inline函数背后的整体观念是，将“对此函数的每一个调用”都以函数本体替换之。

`inline`只是对编译器的一个申请，编译器可以忽略。大部分编译器拒绝将太过复杂的函数inline。申请可以明确提出，也可以隐喻提出（将函数定义于class定义式内）。

inline在大多数C++程序中是编译器行为（少数连接期甚至运行期）。而`virtual`意味“直到运行期才确定调用哪个函数”，所有对virtual函数的调用（除非最平淡无奇的？）也都会使inlining落空。

编译器通常**不**对“通过函数指针而进行的调用”实施inlining。

大部分调试器面对inline函数都束手无策。

80-20法则：将大多数inlining限制在小型 、被频繁调用的函数身上。

### 31 将文件间的编译依存关系降至最低

编译依存性最小化的本质：以“声明的依存性”替换“定义的依存性”。基于此构想的两个手段是handle class和interface class
 - 如果使用object references或object pointers可以完成任务，就不要使用objects；
 - 尽量以class声明式替换class定义式；
 - 为声明式和定义式提供不同的头文件。同时这些文件必须保持一致性。

让头文件尽可能自我满足。

头文件应该以“完全且仅有声明式”的形式存在。

## 继承与面向对象设计

### 32 确定你的`public`继承塑模出 is-a 关系

公开继承（public inheritance）意味“is-a”的关系。

### 33 避免遮掩（hiding）继承而来的名称

name-hiding-rules：遮掩名称。尽管名称是不同的类型，或函数有不同的参数类型、是否为`virtual`一体适用。

### 34 区分接口继承和实现继承

pure virtual函数有两个最突出的特性：它必须被任何继承了它们的具象class重新声明，而且它们在抽象class中通常没有定义。

 - pure virtual函数只具体指定**接口**继承；
 - impure virtual函数具体指定**接口及缺省的实现**继承；
 - non-virtual函数具体指定**接口及强制的实现**继承。

### 35 考虑`virtual`函数以外的其他选择

non-virtual interface（NVI）手法：以public non-virtual成员函数包裹较低访问性（private或protected）的virtual函数。

### 36 绝不重新定义继承而来的 non-virtual 函数

### 37 绝不重新定义继承而来的缺省参数值

virtual函数系动态绑定（dynamically bound），而缺省参数值是静态绑定（statically bound）。静态绑定下函数并不从其base继承缺省参数值。

对象的静态类型：被声明时所采用的类型。
对象的动态类型：目前所指对象的类型。

### 38 通过组合（composition）塑模出 has-a 或“根据某物实现出”

组合意味has-a（在应用域）或is-implemented-in-terms-of（在实现域）。

### 39 明智而审慎地使用`private`继承

对于private继承：
 - 编译器不会将一个derived class对象自动转换为一个base class对象；
 - 继承而来的所有成员在derived class中都会变成private属性，纵使它们在base class中原本是protected或public属性。

private继承意味implemented-in-terms-of。只有实现部分被继承，接口部分应略去。其意义只及于实现层面。

尽可能使用复合，必要时才使用private继承。
当derived class需要访问`protected` base class的成员，或需要重新定义继承而来的virtual函数时，private继承极有可能成为正统设计策略。

private继承可以造成EBO（empty base optimization），即空白基类最优化。

### 40 明智而审慎地使用多重继承

virtual继承会增加大小、速度、初始化（及赋值）复杂度等成本。

virtual base的初始化责任是由继承体系中的最底层（most derived）class负责。

非必要不要使用virtual bases。平常请使用non-virtual继承。
如果必须使用virtual base classes，尽可能避免在其中放置数据。

多重继承的一个应用场景：将“public继承自某接口”和“private继承自某实现”结合起来。

## 模板与泛型编程

C++ template自身是一部完整的图灵机（Turing-complete）：它可以被用来计算任何可计算的值。于是导出了模板元编程，创造出“在C++编译器内执行并于编译完成时停止执行”的程序。

### 41 了解隐式接口和编译期多态

template具现化（instantiated）

编译期多态（compile-time polymorphism）：以不同的template参数具现化function templates，会导致调用不同的函数。

加诸于template参数身上的隐式接口，就像加诸于class对象身上的显示接口一样，且两者都在编译期完成检查。

对class而言，接口是显式（explicit）的，以函数签名为中心。多态则是通过virtual函数发生于**运行**期。
对template参数而言，接口是隐式（implicit）的，奠基于有效表达式（valid expressions）。多态则是通过template具现化和函数重载解析（function overloading resolution）发生于**编译**期。

### 42 了解`typename`的双重意义

template内出现的名称如果相依于某个template参数，称为从属名称（dependent names）。如果从属名称在class内呈嵌套状，称为嵌套从属名称（nested dependent name）。
```c++
template<typename C>
void print2nd(const C& container) {
  if (container.size() >= 2) {
    C::const_iterator iter(container.begin()); // iter为从属名称，C::const_iterator为嵌套从属名称
    ++iter;
    int value = *iter;
    std::cout << value;
  }
}
```

如果解析器在template中遭遇一个嵌套从属名称，它变假设这个名称不是个类型，除非你告诉它是（在前面加上`typename`）：
```c++
typename C::const_iterator iter(container.begin());
```

请使用关键字typename标识嵌套从属类型名称；但不得在base class lists或member initialization list内以它作为base class修饰符。

### 43 学习处理模块化基类内的名称

模板全特化（total template specialization）：`template<>`
```c++
template<typename Company>
class MsgSender {
  public:
    void sendClear(const MsgInfo& info) {...}
    void sendSecret(const MsgInfo& info) {...}
}

template<>
class MsgSender<CompanyZ> { // template MsgSender针对类型CompanyZ的特化，删掉了sendClear()
  public:
    void sendSecret(const MsgInfo& info) {...}
};
```

C++不进入模板化基类（templatized base classes）观察。三个解决办法：
 - 在base class函数调用动作前加上`this->`；
 - 使用`using`声明式；
 - 明白指出被调用的函数位于base class内。不推荐，若被调用的是virtual函数，会关闭virtual绑定行为。
```c++
template<typename Company>
class LoggingMsgSender: public MsgSender<Company> {
  public:
    // 方法1
    void sendClearMsg(const MsgInfo& info) {
      this->sendClear(info);
    }
    // 方法2
    using MsgSender<Company>::sendClear;
    void sendClearMsg(const MsgInfo& info) {
      sendClear(info);
    }
    // 方法3
    void sendClearMsg(const MsgInfo& info) {
      MsgSender<Company>::sendClear(info);
    }
};
```



### 44 将参数无关的代码抽离 templates

非类型参数（non-type parameter）

因非类型模板参数造成的代码膨胀往往可消除，做法是：以函数参数或class成员变量替换template参数。
因类型参数造成的代码膨胀往往可降低，做法是：让带有完全相同二进制表述的具现类型共享实现码。

### 45 运用成员函数模板（member function templates）接受所有兼容类型

member templates作用是为class生成函数：
```c++
template<typename T>
class SmartPtr {
  public:
    template<typename U> // member template
    SmartPtr(const SmartPtr<U>& other); // 泛化copy构造函数，根据SmartPtr<U>生成SmartPtr<T>
};
```
在class内声明泛化*copy*构造函数并不会阻止编译器生成它们自己的*copy*构造函数。相同规则也适用于赋值操作。

### 46 需要类型转换时请为模板定义非成员函数

template实参推导过程中从不将隐式类型转换函数纳入考虑。

当我们编写class template，而它所提供的“与此template相关的”函数 支持“所有参数之隐式类型转换”时，请将那些函数定义为“class template内部的friend函数”。

### 47 请使用 traits classes 表现类型信息

Traits技术要求之一是：它对内置类型和用户自定义类型的表现必须一样好。

习惯上traits总是被实现为structs，但它们却又往往被称为traits classes。

Traits classes使得“类型相关信息”在编译器可用。它们以templates和“templates特化”完成实现。

### 48 认识`template`元编程

Template metaprogramming（TMP，模板元编程）是编写template-based C++程序并执行于编译期的过程。

TMP可将工作由运行期移往编译器。可被用来生成based on combinations of policy choices的客户定制代码，避免生成对某些特殊类型并不适合的代码。

## 定制`new`和`delete`

STL容器所使用的heap内存是由容器所拥有的分配器对象管理，不是被`new`和`delete`直接管理。

### 49 了解 new-handler 的行为

### 50 了解`new`和`delete`的合理替换时机

### 51 编写`new`和`delete`时需固守常规

`operator new`应该内含一个无穷循环，并在其中尝试分配内存，如果它无法满足内存需求，就应该调用new-handler。它也应该有能力处理0 bytes申请。Class专属版本则还应该处理“比正确大小更大的（错误）申请”。

`operator delete`应该在收到null指针时不做任何事。Class专属版本则还应该处理“比正确大小更大的（错误）申请”。

### 52 写了 *placement* `new`也要写 *placement* `delete`

## 杂项讨论

### 53 不要轻忽编译器的警告

### 55 让自己熟悉 Boost



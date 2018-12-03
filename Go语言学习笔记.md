---


---

<h1 id="go语言学习笔记">Go语言学习笔记</h1>
<blockquote>
<p><a href="https://book.douban.com/subject/26832468/">Douban</a><br>
<a href="https://github.com/qyuhen/book">作者Github</a></p>
</blockquote>
<h2 id="语言详解">语言详解</h2>
<blockquote>
<p>基于 Go 1.6（2016/02/17）</p>
</blockquote>
<h3 id="第1章-概述">第1章 概述</h3>
<ol>
<li>特征
<ul>
<li>语法</li>
<li>并发模型</li>
<li>内存分配：<a href="https://github.com/gperftools/gperftools">tcmalloc</a></li>
<li>垃圾回收</li>
<li>静态链接</li>
<li>标准库</li>
<li>工具链</li>
</ul>
</li>
<li>简介</li>
</ol>
<h3 id="第2章-类型">第2章 类型</h3>
<ol>
<li>变量 var
<ul>
<li>Go是静态类型语言</li>
<li>运行时内存分配操作确保变量自动初始化为zero value</li>
<li>简短模式：只能用在函数内部。注意重名遮蔽问题</li>
<li>退化赋值前提是最少有一个新变量，且同一scope</li>
<li>多变量赋值</li>
</ul>
</li>
<li>命名
<ul>
<li>区分大小写</li>
<li>导出变量首字母大写，包外可见</li>
</ul>
</li>
<li>常量 const
<ul>
<li>运行时恒定不可改变。通常会被编译器在预处理阶段直接展开</li>
<li>常量组中如不指定类型和初始化值，则与上一行非空常量右值（类型、值）相同</li>
<li>枚举：Go没有enum定义，可借助iota标识符（希腊字母第9个）实现一组自增常量值实现枚举（可理解为行索引）</li>
</ul>
</li>
<li>基本类型
<ul>
<li>byte（uint8的别名）</li>
<li>rune（int32的别名）</li>
</ul>
</li>
<li>引用类型：sclice, map, channel
<ul>
<li>必须使用make函数创建</li>
</ul>
</li>
<li>类型转换
<ul>
<li>如果转换的目标是<strong>指针、单向通道或没有返回值的函数</strong>，必须使用括号</li>
</ul>
</li>
<li>自定义类型 type
<ul>
<li>拥有相同基础类型的不同自定义类型，仍然属于两种类型</li>
<li>自定义类型只会继承操作符，不会继承基础类型的其他信息，包括方法</li>
<li>未命名类型：相同声明的判定（p35）；转换规则（p36）</li>
</ul>
</li>
</ol>
<h3 id="第3章-表达式">第3章 表达式</h3>
<ol>
<li>保留字
<ul>
<li>25个keyword（p38）</li>
</ul>
</li>
<li>运算符
<ul>
<li>一元运算符优先级最高，二元优先级5个级别（p39）</li>
<li>位运算符：bit clear（AND NOT: &amp;^）</li>
<li>++和–不再是运算符。只能作为独立语句，不能用于表达式</li>
<li>指针：不能做加减法运算和类型转换</li>
</ul>
</li>
<li>初始化
<ul>
<li>复合类型初始化：多行时每行必须以逗号或花括号结束</li>
</ul>
</li>
<li>流控制
<ul>
<li>if…else…：局部变量scope包含整个if/else block</li>
<li>switch：case a, b相当于case (a OR b)；case无需显式break；fallthrough后执行下一case，不再匹配条件</li>
<li>for：for…range会复制目标数据（数组受影响，可用数组指针或slice）；若range目标表达式是函数，仅执行一次</li>
<li>goto, continue, break：标签label区分大小写；不能跳转到其他函数或内层代码块内</li>
</ul>
</li>
</ol>
<h3 id="第4章-函数-func">第4章 函数 func</h3>
<ol>
<li>定义
<ul>
<li>签名：包括参数及返回值列表</li>
<li>不支持和nil以外的比较操作</li>
</ul>
</li>
<li>参数
<ul>
<li>参数都是值拷贝传递（pass-by-value）</li>
</ul>
</li>
<li>返回值
<ul>
<li>支持多返回值和命名返回值</li>
</ul>
</li>
<li>匿名函数
<ul>
<li>可直接调用。保存到变量，作为参数或返回值（也可作为结构体字段，或经通道传递）</li>
<li>闭包（closure）：函数和引用环境的组合体</li>
<li>延迟求值问题：p74</li>
</ul>
</li>
<li>延迟调用 defer
<ul>
<li>defer向<strong>当前函数</strong>注册稍后执行的函数调用，待当前函数执行结束前才执行（如return或panic语句会引发执行）。执行参数在注册时被复制并缓存起来。多个defer按FILO顺序执行</li>
<li>常用于资源释放、解除锁定及错误处理</li>
<li>性能和直接调用比相差数倍</li>
</ul>
</li>
<li>错误处理
<ul>
<li>error</li>
<li>panic, recover：是内置函数
<ul>
<li>defer函数中recover可捕获并返回panic提交的错误对象</li>
<li>连续调用panic，仅最后一个会被recover捕获。recover之后panic，可被再次捕获</li>
<li>除非是不可恢复性、导致系统无法正常工作的错误，否则不建议使用panic</li>
</ul>
</li>
</ul>
</li>
</ol>
<h3 id="第5章-数据">第5章 数据</h3>
<ol>
<li>字符串 string
<ul>
<li>不可变字节（byte）序列。通常在堆上分配内存。默认以UTF-8编码存储Unicode字符。默认值是""</li>
<li>用“`”定义不作转义的原始字符串</li>
<li>for遍历分为byte和rune两种方式</li>
<li>拼接可用strings.Join或bytes.Buffer函数</li>
<li>使用单引号“’”的字面量，默认类型就是rune。可用utf8.RuneCountInString(s)返回Unicode字符数量</li>
</ul>
</li>
<li>数组
<ul>
<li>长度是类型组成部分</li>
<li>多维数组仅第一维度允许使用“…”，len和cap也都返回第一维度长度</li>
<li>如元素类型支持“==、!=”，则数组也支持</li>
<li>数组指针可直接用来操作元素</li>
<li>Go数组是值类型，<strong>赋值和传参都会复制整个数组数据</strong>。可用指针或切片避免数据复制</li>
</ul>
</li>
<li>切片 slice：属性和数组比较
<ul>
<li>只读对象。内部通过指针引用底层数组</li>
<li>不支持比较操作，仅能判断是否为nil</li>
<li>不能直接用指针访问元素内容</li>
<li>reslice操作创建的新切片依旧指向原底层数组</li>
<li>append将数据追加到原底层数组。如超过切片cap限制，则为新切片重新分配数组</li>
<li>copy：切片间复制数据，允许指向同一底层数组，最终复制长度以较短切片长度len为准（作用在底层数组）</li>
</ul>
</li>
<li>字典 map
<ul>
<li>要求key必须支持相等运算符</li>
<li>访问不存在的key不会报错；cap不接受map类型；迭代无序；nil字典能读不能写</li>
<li>not addressable，不能直接修改value成员</li>
<li>迭代过程中可增删。运行时会对并发操作进行检测，若正在写，其他任务无法进行并发操作（读、写、删），否则会导致进程奔溃</li>
<li>字段对象本身就是指针包装，无需取址</li>
</ul>
</li>
<li>结构体 struct
<ul>
<li>字段名、排列<strong>顺序、标签</strong>属于类型组成部分</li>
<li>所有字段都支持时，才能做相等操作</li>
<li>可用指针直接操作字段，但不能是多级指针</li>
<li>空结构 struct{} 可作为channel元素类型，用于事件通知</li>
<li>匿名字段
<ul>
<li>隐式地以类型名作为字段名（指针使用基础类型作为字段名），注意重名遮蔽问题</li>
<li>除接口指针和多级指针外的任何命名类型都可作为匿名字段</li>
</ul>
</li>
<li>内存布局：内存一次性分配，各字段在相邻地址空间按定义顺序排列，通常以最长的基础类型宽度为准作对齐处理</li>
<li>“长度”为零的对象通常都指向runtime.zerobase变量</li>
</ul>
</li>
</ol>
<h3 id="第6章-方法-method">第6章 方法 method</h3>
<ol>
<li>定义
<ul>
<li>method有前置实例接收参数（receiver），func没有</li>
<li>可用实例值或指针（多级指针不行）调用方法</li>
<li>方法的receiver类型选择（p133）：无需修改状态的小对象或固定值建议用T，其他都用*T</li>
</ul>
</li>
<li>匿名字段
<ul>
<li>利用方法的同名遮蔽特性，可实现类似override的操作</li>
</ul>
</li>
<li>方法集 method set：决定了类型是否实现某个接口
<ul>
<li>*T方法集包含所有receiver T + *T方法</li>
<li>匿名嵌入，方法集也被包含（匿名嵌入 S 或 *S，*T 方法集包含所有receiver S + *S 方法）</li>
<li>组合优于继承：模块单元通过匿名嵌入方式组合到一起，共同实现对外接口</li>
</ul>
</li>
<li>表达式：也可赋值给变量，或作为参数传递
<ul>
<li>Method Expression：通过类型（T或*T）引用，调用时须显式传参。会被还原成func(t T, …)，即receiver是第一参数</li>
<li>Method Value：基于实例或指针引用。参数签名不变
<ul>
<li>被赋值给变量或作为参数传递时，会立即计算并复制执行所需的receiver对象，与其绑定，稍后执行（若receiver为指针类型，仅复制指针，见p139例）</li>
<li>作为参数时，会复制receiver</li>
</ul>
</li>
</ul>
</li>
</ol>
<h3 id="第7章-接口-interface">第7章 接口 interface</h3>
<ol>
<li>定义
<ul>
<li>目标类型方法集内包含接口声明的全部方法，即视为实现了该接口</li>
<li>接口：不能有字段；可嵌入其他接口类型（相当于方法集导入，要求不能有同名方法）</li>
<li>空接口interface{} 可被赋值为任何类型的对象（类似Object）</li>
<li>超集接口变量可隐式转换为子集，反之不行</li>
</ul>
</li>
<li>执行机制
<ul>
<li>接口使用名为itab的结构存储运行时所需相关类型信息</li>
<li>将对象赋值给接口变量时，会复制该对象，且该复制品是unaddressable的</li>
</ul>
</li>
<li>类型转换
<ul>
<li>type switch不支持fallthrough</li>
</ul>
</li>
<li>技巧
<ul>
<li></li>
</ul>
</li>
</ol>
<h3 id="第8章-并发">第8章 并发</h3>
<ol>
<li>goroutine
<ul>
<li>go关键字创建并发任务单元放到系统队列中，等待调度器安排系统线程获取执行权</li>
<li>goroutine自定义栈按需分配大小</li>
<li>延迟执行：会立即计算并复制执行参数</li>
<li>Wait：进程退出时不会等待并发任务结束，可用channel阻塞；如需等待多个任务结束用sync.WaitGroup</li>
<li>GOMAXPROCS</li>
<li>Local Storage：goroutine无法设置优先级</li>
<li>runtime.Gosched()：暂停。当前任务被放回队列，等待下次调度恢复</li>
<li>runtime.Goexit()：立即终止整个调用栈。不影响defer执行，不会引发panic</li>
</ul>
</li>
<li>通道：并发安全队列
<ul>
<li>CSP(Communicating Sequential Process)，通信代替内存共享</li>
<li>同步通道：要求收、发方配对。失败等待，成功唤醒。可用cap为0判断</li>
<li>异步通道：要求有数据缓冲槽。不符等待，满足唤醒</li>
<li>操作限制
<ul>
<li>已关闭通道：发送引发panic，接收返回已缓冲数据或零值</li>
<li>nil通道：收发都会阻塞；关闭引发panic</li>
<li>重复关闭通道会引发panic</li>
</ul>
</li>
<li>通道默认双向。通常用类型转换获取单向通道，并分别赋予操作双方。无法将单向通道转换回去
<ul>
<li>接收端不能close</li>
</ul>
</li>
<li>select会随机选取非nil通道做收发操作，若都不可用执行default</li>
<li>模式：通常用工厂方法将goroutine和channel绑定</li>
<li>性能：将发往通道的数据打包，减少传输次数</li>
<li>资源泄漏：垃圾回收器不收集阻塞状态的goroutine</li>
</ul>
</li>
<li>同步
<ul>
<li>注意复制导致锁机制失效问题（使方法返回*T）</li>
<li>Mutex不支持递归锁，即便在同一goroutine下也会导致死锁</li>
</ul>
</li>
</ol>
<h3 id="第9章-包结构">第9章 包结构</h3>
<ol>
<li>工作空间
<ul>
<li>组成：src, bin, pkg</li>
<li>环境变量</li>
</ul>
</li>
<li>导入包
<ul>
<li>导入方式：默认、简便、别名、初始化（让目标包初始化函数得以执行）</li>
<li>相对路径</li>
<li>自定义路径</li>
</ul>
</li>
<li>组织结构
<ul>
<li>特殊包：main, all, std, cmd, documentation</li>
<li>所有初始化函数都由编译器自动生成的一个包装函数进行调用，在单一线程上执行且仅执行一次。编译器不保证初始化函数执行顺序，因此初始化函数之间不应有逻辑关联。初始化函数不能调用</li>
<li>初始化顺序：全局变量 -&gt; 初始化函数 -&gt; main.main入口函数</li>
<li>内部包：internal目录</li>
</ul>
</li>
<li>依赖管理：vender目录存放第三方包</li>
</ol>
<h3 id="第10章-反射">第10章 反射</h3>
<ol>
<li>类型</li>
<li>值</li>
<li>方法</li>
<li>构建</li>
<li>性能</li>
</ol>
<h3 id="第11章-测试">第11章 测试</h3>
<ol>
<li>单元测试</li>
<li>性能测试</li>
<li>代码覆盖率</li>
<li>性能监控</li>
</ol>
<h3 id="第12章-工具链">第12章 工具链</h3>
<ol>
<li>工具
<ul>
<li>go build</li>
<li>go install</li>
<li>go get</li>
<li>go env</li>
<li>go clean</li>
</ul>
</li>
<li>编译</li>
</ol>
<h2 id="源码剖析">源码剖析</h2>
<blockquote>
<p>基于 Go 1.5.1（2015/09/08）</p>
</blockquote>
<h3 id="第16章-内存分配">第16章 内存分配</h3>
<h3 id="第17章-垃圾回收">第17章 垃圾回收</h3>
<h3 id="第18章-并发调度">第18章 并发调度</h3>
<h3 id="第19章-通道">第19章 通道</h3>
<h3 id="第20章-延迟">第20章 延迟</h3>
<h3 id="section"></h3>


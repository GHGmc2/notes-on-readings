---


---

<h1 id="python数据科学手册">Python数据科学手册</h1>
<blockquote>
<p><a href="https://jakevdp.github.io/PythonDataScienceHandbook/">Python Data Science Handbook</a>, <a href="https://github.com/jakevdp/PythonDataScienceHandbook">Github</a><br>
<a href="https://book.douban.com/subject/27667378/">douban</a></p>
</blockquote>
<h1 id="scipy"><a href="https://www.scipy.org/">SciPy</a></h1>
<p>Core packages:</p>
<ul>
<li><strong>NumPy</strong>: Base N-dimensional array package</li>
<li>SciPy library: Fundamental library for scientific computing</li>
<li><strong>Matplotlib</strong>: Comprehensive 2D Plotting</li>
<li><strong>IPython</strong>: Enhanced Interactive Console</li>
<li>Sympy: Symbolic mathematics</li>
<li><strong>pandas</strong>: Data structures &amp; analysis</li>
</ul>
<h2 id="ipython"><a href="https://ipython.org/">IPython</a></h2>
<blockquote>
<p><a href="https://ipython.readthedocs.io/en/stable/">Docs</a><br>
<a href="https://ipython.readthedocs.io/en/stable/api/index.html">API</a></p>
</blockquote>
<p>使用 IPython（interactive Python）的两种方式：</p>
<ul>
<li>IPython shell</li>
<li>IPython Notebook</li>
</ul>
<p>IPython Notebook 其实只是通用 Jupyter Notebook 结构的特例（IPython shell 基于浏览器的图形界面），而 Jupyter Notebook 不仅支持 Python，还包括用于 Julia、R 和其他编程语言的 Notebook。</p>
<p><strong>IPython shell命令</strong></p>
<h2 id="numpy"><a href="http://www.numpy.org/">NumPy</a></h2>
<blockquote>
<p><a href="https://docs.scipy.org/doc/numpy/user/quickstart.html">Quickstart tutorial</a><br>
<a href="https://docs.scipy.org/doc/numpy/reference/routines.html">NumPy functions by category</a></p>
</blockquote>
<p>NumPy（Numerical Python 的简称）提供了高效存储和操作密集 数据缓存的接口。</p>
<p><strong>NumPy 数组</strong>几乎是整个 Python 数据科学工具生态系统的<strong>核心</strong>。</p>
<h3 id="数据类型">数据类型</h3>
<p>创建数组（<a href="https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html">Array creation routines</a>）</p>
<p>标准数据类型（<a href="https://docs.scipy.org/doc/numpy/user/basics.types.html">Data types</a>）</p>
<h3 id="数组基础">数组基础</h3>
<p>数组基本操作（<a href="https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html#basic-operations">Array manipulation routines</a>）：</p>
<ul>
<li>属性（<a href="https://docs.scipy.org/doc/numpy/user/quickstart.html#the-basics">The Basics</a>）
<ul>
<li>nidm：维度</li>
<li>shape：每个维度的大小（The length of the <em>shape</em> tuple is therefore the number of <em>ndim</em>）</li>
<li>size：数组的总大小</li>
<li>dtype：数据类型</li>
<li>itemsize：每个数组元素字节大小</li>
<li>nbytes：数组总字节大小。一般可以认为 nbytes = itemsize * size</li>
</ul>
</li>
<li>索引</li>
<li>切分</li>
<li>变形</li>
<li>拼接和分裂</li>
</ul>
<h3 id="计算通用函数">计算通用函数</h3>
<p>NumPy 中的<strong>向量</strong>操作是通过通用函数（ufunc）实现的（将作用于数组中的每一个元素）。通用函数的主要目的是对 NumPy 数组中的值执行更快的重复操作。</p>
<p>通用函数（<a href="https://docs.scipy.org/doc/numpy/reference/routines.math.html">Mathematical functions</a>）</p>
<ul>
<li>数组运算（算术运算符都是 NumPy 内置函数的简单封装器）</li>
<li>绝对值abs(x)</li>
<li>三角函数</li>
<li>指数和对数：exp(x)；log(x)</li>
<li>专用通用函数：双曲三角函数、比特位运算、比较运算符、弧度转化为角度的运算、取整和求余运算等
<ul>
<li>scipy.special 模块</li>
</ul>
</li>
</ul>
<p>通用函数特性</p>
<ul>
<li>指定输出：指定out参数</li>
<li>聚合
<ul>
<li>reduce()：对给定的元素和操作重复执行，直至得到单个的结果</li>
<li>accumulate()：存储每次计算的中间结果</li>
</ul>
</li>
<li>外积：outer(x, y)</li>
</ul>
<h3 id="聚合">聚合</h3>
<blockquote>
<p><a href="https://docs.scipy.org/doc/numpy/reference/routines.statistics.html">Statistics</a></p>
</blockquote>
<ul>
<li>求和sum(x)</li>
<li>最小值和最大值：min()和max()，可指定axis参数</li>
</ul>
<h3 id="广播">广播</h3>
<p>广播可理解为用于<strong>不同大小数组</strong>的二进制通用函数（加、减、乘等）的一组规则（p57例）：</p>
<ul>
<li>维度不相同，那么小维度数组的形状将会在<strong>最左边维度补 1</strong></li>
<li>如果两个数组的形状在任何一个维度上都不匹配，那么数组的形状会<strong>沿着维度为 1 的维度扩展</strong>，以匹配另外一个数组的形状</li>
<li>如果两个数组的形状在任何一个维度上都不匹配并且没有任何一个维度等于 1，那么会引发异常</li>
</ul>
<p>应用</p>
<ul>
<li>数组归一化（normalization，也叫标准化）</li>
<li>基于二维函数显示图像</li>
</ul>
<h3 id="布尔掩码">布尔掩码</h3>
<blockquote>
<p>（<a href="https://docs.scipy.org/doc/numpy/reference/routines.ma.html#masked-arrays-arithmetics">Masked arrays arithmetics</a>）</p>
</blockquote>
<ul>
<li>比较操作（<a href="https://docs.scipy.org/doc/numpy/reference/routines.logic.html#comparison">comparision</a>）：比较运算操作在 NumPy 中也是借助通用函数来实现的。可以用于任意形状、大小的数组</li>
<li>操作布尔数组
<ul>
<li>count_nonzero()统计个数</li>
<li>逐位逻辑运算符（<a href="https://docs.scipy.org/doc/numpy/reference/routines.logic.html#logical-operations">Logical operations</a>）：注意and 和 or 判断整个对象是真或假，而 &amp; 和 | 是指每个对象中的比特位</li>
</ul>
</li>
<li>将布尔数组作为掩码（masking operation: index on Boolean array，如 arr[arr &gt; 0]）：通过该掩码选择数据的子数据集</li>
</ul>
<h3 id="花哨索引（fancy-indexing）">花哨索引（fancy indexing）</h3>
<p>花哨的索引传递一个<strong>索引数组</strong>来一次性获得多个数组元素，让我们能够快速获得并修改复杂的数组值的子数据集。</p>
<p>在花哨的索引中，索引值的配对遵循广播的规则。结果的形状与<strong>广播后的索引数组的形状</strong>一致（而不是与<strong>被</strong>索引数组的形状一致）。</p>
<p><strong>组合索引</strong>：花哨的索引可以和其他索引方案（简单索引、切片、掩码）结合起来形成更强大的索引操作。</p>
<p>花哨的索引也可以被用于修改部分数组。</p>
<p>应用</p>
<ul>
<li>快速分割数据</li>
</ul>
<h3 id="排序">排序</h3>
<ul>
<li>sort()：默认快排。可指定 axis 参数</li>
<li>argsort()：返回排序后的索引值</li>
<li>分隔partition(x, k)：输出一个新数组，最左边是第 k 小的 k 个值，往右是任意顺序的其他值。类似还有argpartition() 计算分隔的索引值</li>
</ul>
<p>应用：k 个最近邻（p78）</p>
<h2 id="pandas">pandas</h2>
<p>可以把 pandas 对象看成增强版的 NumPy 结构化数组，行列都不再只是简单的整数索引，还可以带上标签。</p>
<h3 id="对象简介">对象简介</h3>
<ul>
<li>Series：<strong>带索引</strong>数据构成的<strong>一维数组</strong>，可以通过values 和 index 属性获取
<ul>
<li>看成 NumPy 一维数组：Series 对象<strong>用一种显式定义的索引与数值关联，索引可以是任意想要的类型</strong>。支持数组形式的操作，如切片</li>
<li>看成特殊字典：可直接用字典创建Series，索引默认按照顺序排列</li>
</ul>
</li>
<li>DataFrame：既有灵活的<strong>行索引</strong>，又有灵活<strong>列名</strong>的二维数组。
<ul>
<li>看成 NumPy 二维数组：是<strong>有序排列的若干 Series 对象</strong>。index 属性可以获取（行）索引标签；columns 属性是存放列标签的 Index 对象</li>
<li>看成特殊字典：一列映射一个 Series 的数据</li>
</ul>
</li>
<li>Index
<ul>
<li>看做不可变数组：（注：Index 对象的<strong>索引不可变</strong>，Index 对象的不可变特征使得多个 DataFrame 和数组之间进行<strong>索引共享</strong>时更加安全）</li>
<li>看做有序集合（set）：Index 对象遵循 Python 集合的许多习惯用法，如并、交、 差等</li>
</ul>
</li>
</ul>
<h3 id="数据选择">数据选择</h3>
<h3 id="数值运算">数值运算</h3>
<h3 id="处理缺失值">处理缺失值</h3>
<h3 id="层级索引">层级索引</h3>
<h3 id="合并数据集">合并数据集</h3>
<h3 id="累计与分组">累计与分组</h3>
<h3 id="数据透视表">数据透视表</h3>
<h3 id="字符串操作">字符串操作</h3>
<h3 id="处理时间序列">处理时间序列</h3>
<h3 id="高性能">高性能</h3>
<h2 id="matplotlib">Matplotlib</h2>
<h3 id="线形图">线形图</h3>
<h3 id="散点图">散点图</h3>
<h3 id="密度图与等高线图">密度图与等高线图</h3>
<h3 id="区间划分和分布密度">区间划分和分布密度</h3>
<h3 id="多子图">多子图</h3>
<h3 id="三维图">三维图</h3>
<h2 id="scikit-learn">Scikit-Learn</h2>
<h3 id="数据表示">数据表示</h3>
<h3 id="评估器api">评估器API</h3>


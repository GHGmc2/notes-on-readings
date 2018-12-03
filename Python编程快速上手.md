---


---

<h1 id="python编程快速上手">Python编程快速上手</h1>
<blockquote>
<p><a href="https://automatetheboringstuff.com/">官网</a>、<a href="https://book.douban.com/subject/26836700/">douban</a><br>
<a href="https://studentportalen.uu.se/uusp-webapp/auth/webwork/filearea/download.action?nodeId=2053259&amp;toolAttachmentId=475830&amp;uusp.userId=guest">PDF</a>、<a href="https://download.csdn.net/download/qq_25281937/9717244">中文PDF</a></p>
</blockquote>
<h1 id="python编程基础">Python编程基础</h1>
<h2 id="基础">基础</h2>
<p>数学操作符</p>

<table>
<thead>
<tr>
<th>操作符</th>
<th>操作</th>
<th>例子</th>
<th>值</th>
</tr>
</thead>
<tbody>
<tr>
<td>**</td>
<td>指数</td>
<td>2 ** 3</td>
<td>8</td>
</tr>
<tr>
<td>//</td>
<td>取商</td>
<td>22 // 8</td>
<td>2</td>
</tr>
</tbody>
</table><p>变量名<strong>区分大小写</strong>，惯例小写开头.</p>
<p>#注释</p>
<h2 id="控制流">控制流</h2>
<p>布尔值 True 和 False 以<strong>大写</strong>开头.</p>
<p>布尔操作符：and、or和not（<strong>优先级 not &gt; and &gt; or</strong>）.</p>
<p>根据代码行的<strong>缩进</strong>判断代码块的开始和结束.</p>
<p>控制流语句都以冒号结尾，后跟代码块. 如</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">if</span> name <span class="token operator">==</span> <span class="token string">'Alice'</span><span class="token punctuation">:</span>
    <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">'Hi, Alice.'</span><span class="token punctuation">)</span>
</code></pre>
<p>range()函数</p>
<ul>
<li>range(stop)，<strong>左闭右开</strong></li>
<li>range(start, stop[, step])</li>
</ul>
<p>sys.exit()退出程序</p>
<h2 id="函数（functions）">函数（Functions）</h2>
<p>def语句</p>
<p>None（N大写）值：表示没有值，是NoneType类型的唯一值. （类似其他语言的 null 或 nil）</p>
<p>关键字参数</p>
<p>作用域：局部、全局. 可用 global 语句使变量成为全局变量.</p>
<p>异常处理：try - except语句</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">try</span><span class="token punctuation">:</span>
    <span class="token keyword">raise</span> NameError<span class="token punctuation">(</span><span class="token string">'Hi, here.'</span><span class="token punctuation">)</span>
<span class="token keyword">except</span> NameError<span class="token punctuation">:</span>
    do sth
</code></pre>
<h2 id="列表（lists）">列表（Lists）</h2>
<blockquote>
<p>方括号[]</p>
</blockquote>
<p>负数下标：-1指倒数第1个下标，以此类推.</p>
<p>切片（slice）：返回新列表，如spam[1:4]，左闭右开.</p>
<p>连接：“+”操作符. 如[1, 2] + [3] 得 [1, 2, 3]</p>
<p>复制：“*”操作符. 如[1, 2] * 2 得 [1, 2, 1, 2]</p>
<p>删除值：del语句. 如del spam[2]，被删除值前移一个下标. del语句作用于变量时是“取消赋值”.</p>
<p>in 和 not in 操作符判断值是否在列表中，返回 True 或 False.</p>
<p>多重赋值：用列表中的值为多个变量赋值（变量数目需和列表长度相等）. 如</p>
<pre class=" language-python"><code class="prism  language-python">cat <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token string">'fat'</span><span class="token punctuation">,</span> <span class="token string">'black'</span><span class="token punctuation">]</span>
size<span class="token punctuation">,</span> color <span class="token operator">=</span> cat
</code></pre>
<p>列表方法：</p>
<ul>
<li>index(x)：返回第一次出现x值的下标. 不存在则报ValueError</li>
<li>insert(i, x)：注意返回值是None</li>
<li>remove(x)：删除第一次出现的x值</li>
<li>sort()：支持关键字参数 key 和 reverse</li>
</ul>
<p><strong>字符串</strong><br>
字符串不可变.</p>
<p><strong>元组（tuple）</strong></p>
<blockquote>
<p>圆括号(). 表示下标仍然用方括号</p>
</blockquote>
<p>元组不可变.</p>
<p>若元组只有一个值，在值后面跟一个逗号. 如：(‘hello’,)</p>
<p>可用 list() 和 tuple() 函数转换类型.</p>
<p><strong>引用</strong><br>
当函数被调用时，参数值被复制给变元.</p>
<p>copy.copy()复制列表或字典.<br>
copy.deepcopy()将同时复制列表内部的列表.</p>
<h2 id="字典（dictionaries）和结构化数据">字典（Dictionaries）和结构化数据</h2>
<p>字典：大括号{}，无序键-值对. 键可以为任意值？</p>
<p>字典方法</p>
<ul>
<li>keys(), values(), items(): 分别返回<strong>不能被修改</strong>的键、值和键-值对列表</li>
<li>get(key[, default])</li>
<li>setdefault(key[, default])</li>
</ul>
<h2 id="字符串操作">字符串操作</h2>
<p>字面量：单引号（’’）或双引号（""）. 使用双引号时，字符串内可以有单引号字符.</p>
<p>转义字符：倒斜杠（\）</p>
<p>原始字符串：引号之前加上r，如：</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">print</span><span class="token punctuation">(</span>r<span class="token string">'That is Carol\'s cat.'</span><span class="token punctuation">)</span>
</code></pre>
<p>多行字符串：三重引号（单、双均可）. 之间的引号、制表符或换行，都被认为是字符串一部分. 可用在多行注释.</p>
<p>字符串方法</p>
<ul>
<li>isX()类</li>
<li>bytes.join(iterable): 返回字符串. 注意bytes是分隔符</li>
<li>split(sep): 返回列表</li>
<li>strip(), rstrip(), lstrip(): 删除空白字符</li>
</ul>
<h1 id="自动化任务">自动化任务</h1>
<h2 id="正则表达式">正则表达式</h2>
<p>步骤</p>
<ol>
<li>import re</li>
<li>re.compile() 函数创建Regex对象</li>
<li>Regex的search()方法返回Match对象</li>
<li>Match的group()方法返回第一次匹配的字符串；findall()方法返回所有匹配的字符串列表</li>
</ol>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> re

phoneNumRegex <span class="token operator">=</span> re<span class="token punctuation">.</span><span class="token builtin">compile</span><span class="token punctuation">(</span>r<span class="token string">'\d\d\d-\d\d\d-\d\d\d\d'</span><span class="token punctuation">)</span>
mo <span class="token operator">=</span> phoneNumRegex<span class="token punctuation">.</span>search<span class="token punctuation">(</span><span class="token string">'My number is 415-555-4242'</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">'Phone number found: '</span> <span class="token operator">+</span> mo<span class="token punctuation">.</span>group<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<p>正则表达式符号（p128）</p>
<h2 id="读写文件">读写文件</h2>
<p>OS X和Linux下，路径分隔符是正斜杠（/）</p>
<p>os方法</p>
<ul>
<li>getcwd()：当前工作目录的字符串</li>
<li>makedirs(name)创建目录</li>
<li>os.path模块的文件名和文件路径函数：</li>
</ul>
<p>读写文件步骤</p>
<ol>
<li>调用open()返回File对象</li>
<li>调用File的read()或write()方法</li>
<li>调用File的close()关闭文件</li>
</ol>
<h2 id="调试">调试</h2>
<p>抛出异常：raise语句</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">raise</span> Exception<span class="token punctuation">(</span><span class="token string">'This is the error message'</span><span class="token punctuation">)</span>
</code></pre>
<p>断言格式：</p>
<pre><code>assert condition, 'message when False'
</code></pre>
<p>日志：logging模块</p>
<h2 id="时间">时间</h2>
<p>time和datatime模块</p>
<p>多线程：threading模块</p>
<p>启动其他程序：subprocess.Popen()</p>


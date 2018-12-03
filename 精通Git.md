---


---

<h1 id="精通git">精通Git</h1>
<blockquote>
<p><a href="https://git-scm.com/book/zh/v2">Pro Git 2</a><br>
<a href="https://v.youku.com/v_show/id_XMzg5MjIzODM3Mg==.html">Linus 介绍 Git 的特点和设计思路</a></p>
</blockquote>
<h2 id="资源">资源</h2>
<p><a href="https://coding.net/help/doc/practice/git-principle.html">使用原理视角看 Git</a><br>
<a href="https://marklodato.github.io/visual-git-guide/index-zh-cn.html">图解Git</a></p>
<h1 id="notes">Notes</h1>
<h2 id="vcs及git基础知识">1 VCS及Git基础知识</h2>
<h3 id="版本控制系统">版本控制系统</h3>
<ul>
<li>本地：RCS通过叠加补丁集的方式恢复文件</li>
<li>集中式：有一个包含文件所有修订版本的单一服务器，如SVN</li>
<li>分布式：客户端是对代码仓库进行完整的镜像。如Git</li>
</ul>
<h3 id="git基础">Git基础</h3>
<p>快照、而非差异：将数据视为微型文件系统的快照流。<br>
每次提交或在Git中保存项目状态时，Git抓取一张所有文件当前状态的快照，然后存储一个指向该快照的引用。</p>
<p>几乎所有操作本地执行：快；不依赖网络。</p>
<p>完整性：所有数据在存储前进行校验和计算（用SHA-1根据文件内容或目录结构计算得到），随后以校验和（用信息的散列值而不是文件名）来引用对应的数据。</p>
<p>通常只增加数据。</p>
<p>文件的三种状态：committed、modified、staged。<br>
三个工作区域：</p>
<ul>
<li>Git目录：保存项目元数据和对象元数据</li>
<li>暂存区（索引）：保存下次所要提交内容的相关信息</li>
<li>工作目录：项目某个版本的单次检出</li>
<li><img src="https://git-scm.com/book/en/v2/images/areas.png" alt=""></li>
</ul>
<h3 id="首次配置">首次配置</h3>
<p>用户身份：若希望不同项目用不同的用户名或邮件，不带"–global"即可</p>
<pre><code>$ git config --global user.name "Gmc2"
$ git config --global user.email "gmc2.each@gmail.com"
</code></pre>
<p>默认文本编辑器：</p>
<pre><code>$ git config --global core.editor emacs
</code></pre>
<p>检查个人设置：</p>
<pre><code>$ git config --list
$ git config &lt;key&gt;
</code></pre>
<h2 id="git基本用法">2 Git基本用法</h2>
<p><img src="https://unwiredlearning.com/wp-content/uploads/2018/07/git-flow.png" alt=""></p>
<h3 id="获取git仓库">获取Git仓库</h3>
<p>从现有目录初始化：</p>
<pre><code>$ git init
</code></pre>
<p>克隆现有：</p>
<pre><code>$ git clone [url]
</code></pre>
<h3 id="记录变更">记录变更</h3>
<pre><code>$ git status
$ git add：添加内容到下一次提交中
$ git diff [-staged]
$ git commit [-a跳过暂存区]
$ git rm
$ git mv
</code></pre>
<h3 id="查看提交历史">查看提交历史</h3>
<pre><code>$ git log
</code></pre>
<h3 id="远程仓库">远程仓库</h3>
<p>显示远程仓库：默认名称是origin</p>
<pre><code>$ git remote
$ git remote [-v显示仓库的url]
</code></pre>
<p>添加远程仓库：</p>
<pre><code>$ git remote add [shortname] [url]
</code></pre>
<p>将数据推送到远程仓库：</p>
<pre><code>$ git push [remote-name] [branch-name]，如：$ git push origin master
</code></pre>
<p>删除和重命名：</p>
<pre><code>$ git remote rm [remote-name]
$ git remore rename [oldname] [newname]
</code></pre>
<h3 id="标记">标记</h3>
<ul>
<li>轻量（lightweight）标签</li>
<li>注释（annotated）标签</li>
</ul>
<p>列举标签：</p>
<pre><code>$ git tag
</code></pre>
<p>创建标签：</p>
<pre><code>$ git tag [name]
$ git tag [-a注释标签]，如：$ git tag -a v1.4 -m "my version"
</code></pre>
<p>检出标签：</p>
<pre><code>$ git checkout -b [branchname] [tagname]
</code></pre>
<h3 id="别名">别名</h3>
<p>如：</p>
<pre><code>$ git config --global alias.co checkout
</code></pre>
<h2 id="git分支模型">3 Git分支模型</h2>
<h3 id="简述">简述</h3>
<p>commit object, tree object and blob object:<br>
<img src="https://git-scm.com/book/en/v2/images/commit-and-tree.png" alt=""><br>
<img src="https://git-scm.com/book/en/v2/images/commits-and-parents.png" alt=""></p>
<p>Git分支是一个指向某次提交的轻量级的可移动指针。默认分支名是master。</p>
<p>创建分支：只创建，不会切换到该分支</p>
<pre><code>$ git branch [name]
</code></pre>
<p>切换分支：会改变HEAD指针和工作目录</p>
<pre><code>$ git checkout [name]
</code></pre>
<p>Git维护着一个名为HEAD的特殊指针，<strong>指向当前所在的本地分支</strong>（指向指针的指针）。<br>
<img src="https://git-scm.com/book/en/v2/images/head-to-testing.png" alt=""></p>
<h3 id="分支和合并基本操作">分支和合并基本操作</h3>
<p>创建并切换到新分支：注意工作区和暂存区是否存在未提交的更改且和新切换的分支冲突</p>
<pre><code>$ git checkout -b [name]
</code></pre>
<p>删除分支：</p>
<pre><code>$ git branch -d [name]
</code></pre>
<p>合并name分支到当前分支：</p>
<pre><code>$ git merge [name]
</code></pre>
<p>合并提交：基于三方合并的结果<strong>创建新的快照</strong>，然后再创建新的提交，指向新建的快照。Git会判断最优公共祖先并将其作为合并基础。<br>
合并冲突后，可以用git add把文件标记为冲突已解决。</p>
<h3 id="分支管理">分支管理</h3>
<p>查看每个分支上最新提交：</p>
<pre><code>$ git branch -v
</code></pre>
<p>筛选已并入和尚未并入当前分支的所有分支：</p>
<pre><code>$ git branch --merged
$ git branch --no-merged
</code></pre>
<h3 id="工作流">工作流</h3>
<p><img src="http://nvie.com/img/git-model@2x.png" alt=""></p>
<h3 id="远程分支">远程分支</h3>
<p>远程分支：指向远程仓库分支的指针，存在于本地且无法被移动。<strong>当本地与服务器进行通信时自动更新</strong>。<br>
<strong>表示形式为(remote)/(branch)</strong>（即分支前多了个远程仓库名）。执行git clone时远程仓库默认名称为origin（可用git clone -o [name]修改），和master分支一样都仅是默认名称，并无特殊。</p>
<p>与服务器同步：</p>
<pre><code>$ git fetch [remote]
</code></pre>
<p>添加远程服务器到现有项目：</p>
<pre><code>$ git remote add [remote] [url]
</code></pre>
<p><img src="https://git-scm.com/book/en/v2/images/remote-branches-5.png" alt=""></p>
<p>推送到远程分支：</p>
<pre><code>git push (remote) (branch)
</code></pre>
<p>跟踪远程分支：</p>
<pre><code>$ git chckout -b [branch] [remotename]/[branch]
$ git checkout --track [remotename]/[branch]：使当前分支跟踪远程分支
$ git branch -u [remotename]/[branch]：更改本地分支对应的远程分支
$ git branch -vv：查看分支跟踪信息。若要查看最新领先或落后提交次数的信息，先执行git fetch -- all
</code></pre>
<p>拉取：</p>
<pre><code>$ git pull = git fetch + git merge
</code></pre>
<p>删除远程分支：</p>
<pre><code>$ git push [branch] --delete [remotebranch]
</code></pre>
<h2 id="git高级命令">7 Git高级命令</h2>
<h3 id="搜索">搜索</h3>
<pre><code>$ git grep
</code></pre>
<h3 id="合并高级用法">合并高级用法</h3>
<p>退出合并：</p>
<pre><code>$ git merge --abort
</code></pre>
<p>还原提交：</p>
<pre><code>$ git revert
</code></pre>
<h3 id="子模块">子模块</h3>
<p>子模块允许你将一个 Git 仓库作为另一个 Git 仓库的子目录。 它能让你将另一个仓库克隆到自己的项目中，同时还保持提交的独立。</p>
<p>可以在 git submodule add 命令后面加上想要跟踪的项目 URL 来添加新的子模块。</p>
<pre><code>$ git submodule add [URL] [path]
</code></pre>
<p>克隆一个含有子模块的项目时，默认会包含该子模块目录，但其中没有任何文件。<br>
可运行两个命令：<code>git submodule init</code> 用来初始化本地配置文件，而 <code>git submodule update</code> 则从该项目中抓取所有数据并检出父项目中列出的合适的提交。<br>
还有更简单一点的方式：给 git clone 命令传递 --recursive 选项，它就会自动初始化并更新仓库中的每一个子模块。</p>
<p>发布子模块变更：git push 命令接受可以设置为 “check” 或 “on-demand” 的 --recurse-submodules 参数。</p>
<pre><code>$ git push --recurse-submodules=check
或
$ git push --recurse-submodules=on-demand
</code></pre>
<h2 id="git原理">10 Git原理</h2>
<h2 id="附录-git命令">附录 <a href="https://git-scm.com/book/zh/v2/%E9%99%84%E5%BD%95-C%3A-Git-%E5%91%BD%E4%BB%A4-%E8%AE%BE%E7%BD%AE%E4%B8%8E%E9%85%8D%E7%BD%AE">Git命令</a></h2>


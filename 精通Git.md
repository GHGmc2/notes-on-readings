# 精通Git
> [Pro Git 2](https://git-scm.com/book/zh/v2)
> [Linus 介绍 Git 的特点和设计思路](https://v.youku.com/v_show/id_XMzg5MjIzODM3Mg==.html)

## 资源
[使用原理视角看 Git](https://coding.net/help/doc/practice/git-principle.html)
[图解Git](https://marklodato.github.io/visual-git-guide/index-zh-cn.html)

# Notes

## 1 VCS及Git基础知识

### 版本控制系统

 - 本地：RCS通过叠加补丁集的方式恢复文件
 - 集中式：有一个包含文件所有修订版本的单一服务器，如SVN
 - 分布式：客户端是对代码仓库进行完整的镜像。如Git

### Git基础

快照、而非差异：将数据视为微型文件系统的快照流。
每次提交或在Git中保存项目状态时，Git抓取一张所有文件当前状态的快照，然后存储一个指向该快照的引用。

几乎所有操作本地执行：快；不依赖网络。

完整性：所有数据在存储前进行校验和计算（用SHA-1根据文件内容或目录结构计算得到），随后以校验和（用信息的散列值而不是文件名）来引用对应的数据。

通常只增加数据。

文件的三种状态：committed、modified、staged。
三个工作区域：

 - Git目录：保存项目元数据和对象元数据
 - 暂存区（索引）：保存下次所要提交内容的相关信息
 - 工作目录：项目某个版本的单次检出
 - ![](https://git-scm.com/book/en/v2/images/areas.png)

### 首次配置

用户身份：若希望不同项目用不同的用户名或邮件，不带"--global"即可
```
$ git config --global user.name "Gmc2"
$ git config --global user.email "gmc2.each@gmail.com"
```

默认文本编辑器：
```
$ git config --global core.editor emacs
```

检查个人设置：
```
$ git config --list
$ git config <key>
```

## 2 Git基本用法

![](https://unwiredlearning.com/wp-content/uploads/2018/07/git-flow.png)

### 获取Git仓库

从现有目录初始化：
```
$ git init
```

克隆现有：
```
$ git clone [url]
```

### 记录变更
```
$ git status
$ git add：添加内容到下一次提交中
$ git diff [-staged]
$ git commit [-a跳过暂存区]
$ git rm
$ git mv
```

### 查看提交历史

```
$ git log
```

### 远程仓库

显示远程仓库：默认名称是origin
```
$ git remote
$ git remote [-v显示仓库的url]
```

添加远程仓库：
```
$ git remote add [shortname] [url]
```

将数据推送到远程仓库：
```
$ git push [remote-name] [branch-name]，如：$ git push origin master
```

删除和重命名：
```
$ git remote rm [remote-name]
$ git remore rename [oldname] [newname]
```

### 标记

 - 轻量（lightweight）标签
 - 注释（annotated）标签

列举标签：
```
$ git tag
```

创建标签：
```
$ git tag [name]
$ git tag [-a注释标签]，如：$ git tag -a v1.4 -m "my version"
```

检出标签：
```
$ git checkout -b [branchname] [tagname]
```

### 别名

如：
```
$ git config --global alias.co checkout
```

## 3 Git分支模型

### 简述

commit object, tree object and blob object:
![](https://git-scm.com/book/en/v2/images/commit-and-tree.png)
![](https://git-scm.com/book/en/v2/images/commits-and-parents.png)

Git分支是一个指向某次提交的轻量级的可移动指针。默认分支名是master。

创建分支：只创建，不会切换到该分支
```
$ git branch [name]
```

切换分支：会改变HEAD指针和工作目录
```
$ git checkout [name]
```
Git维护着一个名为HEAD的特殊指针，**指向当前所在的本地分支**（指向指针的指针）。
![](https://git-scm.com/book/en/v2/images/head-to-testing.png)

### 分支和合并基本操作

创建并切换到新分支：注意工作区和暂存区是否存在未提交的更改且和新切换的分支冲突
```
$ git checkout -b [name]
```

删除分支：
```
$ git branch -d [name]
```

合并name分支到当前分支：
```
$ git merge [name]
```
合并提交：基于三方合并的结果**创建新的快照**，然后再创建新的提交，指向新建的快照。Git会判断最优公共祖先并将其作为合并基础。
合并冲突后，可以用git add把文件标记为冲突已解决。

### 分支管理

查看每个分支上最新提交：
```
$ git branch -v
```

筛选已并入和尚未并入当前分支的所有分支：
```
$ git branch --merged
$ git branch --no-merged
```

### 工作流

![](http://nvie.com/img/git-model@2x.png)

### 远程分支

远程分支：指向远程仓库分支的指针，存在于本地且无法被移动。**当本地与服务器进行通信时自动更新**。
**表示形式为(remote)/(branch)**（即分支前多了个远程仓库名）。执行git clone时远程仓库默认名称为origin（可用git clone -o [name]修改），和master分支一样都仅是默认名称，并无特殊。

与服务器同步：
```
$ git fetch [remote]
```

添加远程服务器到现有项目：
```
$ git remote add [remote] [url]
```

![](https://git-scm.com/book/en/v2/images/remote-branches-5.png)

推送到远程分支：
```
git push (remote) (branch)
```

跟踪远程分支：
```
$ git chckout -b [branch] [remotename]/[branch]
$ git checkout --track [remotename]/[branch]：使当前分支跟踪远程分支
$ git branch -u [remotename]/[branch]：更改本地分支对应的远程分支
$ git branch -vv：查看分支跟踪信息。若要查看最新领先或落后提交次数的信息，先执行git fetch -- all
```

拉取：
```
$ git pull = git fetch + git merge
```

删除远程分支：
```
$ git push [branch] --delete [remotebranch]
```

## 7 Git高级命令

### 搜索

```
$ git grep
```

### 合并高级用法

退出合并：
```
$ git merge --abort
```

还原提交：
```
$ git revert
```

### 子模块

子模块允许你将一个 Git 仓库作为另一个 Git 仓库的子目录。 它能让你将另一个仓库克隆到自己的项目中，同时还保持提交的独立。

可以在 git submodule add 命令后面加上想要跟踪的项目 URL 来添加新的子模块。
```
$ git submodule add [URL] [path]
```

克隆一个含有子模块的项目时，默认会包含该子模块目录，但其中没有任何文件。
可运行两个命令：`git submodule init` 用来初始化本地配置文件，而 `git submodule update` 则从该项目中抓取所有数据并检出父项目中列出的合适的提交。
还有更简单一点的方式：给 git clone 命令传递 --recursive 选项，它就会自动初始化并更新仓库中的每一个子模块。

发布子模块变更：git push 命令接受可以设置为 “check” 或 “on-demand” 的 --recurse-submodules 参数。 
```
$ git push --recurse-submodules=check
或
$ git push --recurse-submodules=on-demand
```

## 10 Git原理

## 附录 [Git命令](https://git-scm.com/book/zh/v2/%E9%99%84%E5%BD%95-C%3A-Git-%E5%91%BD%E4%BB%A4-%E8%AE%BE%E7%BD%AE%E4%B8%8E%E9%85%8D%E7%BD%AE)

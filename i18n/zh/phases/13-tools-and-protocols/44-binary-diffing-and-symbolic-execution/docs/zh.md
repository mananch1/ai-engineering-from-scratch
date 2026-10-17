# 双元分化和象征执行:匹配函数和输入解决

> 两个函数之间的匹配是一个具有一个质量的签名所做的索赔,一个解决的输入是一个路径条件的模型;这个课程构建了两者,并对抗了 radiff2,Ghidra,angr和z3.

**Type:** Build
**Languages:** Python, C, x86-64 assembly, Java (one Ghidra script)
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), 34 (lifting to pseudocode), 36 (rebuild and prove equal)
**Time:** ~150 minutes

## 学习目标

- 从对象及其列表中构建五个函数签名:精确字节,隐藏的移动字节,mnemonic序列,流量图形和符号名称.
- 运行一个以BinDiff风格的阶段的贪匹配器,一个签名只有在两侧唯一时匹配,并记录每个比赛背后的阶段和质量.
- 复制两个字节相同函数在加载图像上失败字节对称器的情况,然后通过掩盖后,使用移动算法解释了四个改变的字节.
- 根据莱文施泰因距离复制radiff2的相似度比,并解释为什么只有在比较使用达到的82字节雷达2时才出现一个比率.
- 列出函数的路径在一个小 IR 上,每条路径运行一个条件,并通过有限搜索来解决每个输入单页.
- 测量具有象征性界限的循环路径爆炸,并与Angr 10.0.3和z3 5.1.0的数量,类型和模型进行比较.

## 问题

你重新建造了`ledger.c`在`-O2`问一个代理人,自从事件发生后,有没有发生任何变化?`-O1`经纪人进行了分离,报告说`grade`由于其字节在两个加载的程序中不同,因此它被修改了.`grade`两个对象文件中的图像是相同的.`.rodata`在                                                                                                           `-O2`计划,所以一个移动在`grade`增长了0x70

同一个下午你问了什么分数`grade`函数返回4给每一个分数从90到109,问题没有结局,所以代理的答案是一个样本,你需要一个集合.

两个失败都来自同一间隙.一个匹配和一个解决输入是每个要求的方法背后.这个课程在标准图书馆Python中构建了两个方法,并运行了它们旁边的生产工具.工具和代码一致的地方,你有两个衍生.他们不同意的地方,你有一个地址,字节,或一个被绑定的查看.

```figure
re-44-matches-and-paths
```

## 功能签名是具有质量的要求

2004年,Flake 设置了二进制分化,作为两个类似执行式函数组之间的同形结构构建. 2005年,Dullien和Rolls将其扩展到基本块和指令. 根据BinDiff概念页面, 计算两个侧面的每个函数的一个属性. 值出现在每一个侧面的确切一次是匹配的. 出现一次以上的值是模糊的,并且在同等级类内下一个属性被钻下来. 没有对应的值保持无与伦比. 每次传递后,新的匹配的调用者和调用者形成小组,下一个属性可能是独特的,并且过程重复下一个属性.

每个匹配算法都有自己的品质. 页面的分数是原始函数字节和名称的哈希匹配非常好,流量图MD指数和元签名匹配 (每个mnemonic一个小元,产品超过函数) 都很好,字符串引用是中等,循环数量或地址序列是差的. 匹配的信心是产生它的算法的平均质量, 函数的相似性是约25%的相匹配的流量图边缘,15%的相匹配的基本块,10%的相匹配的指示,以及50%的流量图 MD指数的差异,乘以信心.

这一课保留了五个属性,并保留了唯一重要的规则:在两侧都是唯一的,或者没有匹配.它将MD指数丢弃,因为概念页面没有公式的名称,并使用一个简单的级别序列.质量标签仍然存在,因为匹配只能像产生它最弱的事实一样好.

## 单个功能记录的五个签名

对于每个对象,差异读取两个东西.`.rela.*`其他国家`llvm-objdump -d`列表给出指令界限和记, 第33课证明,这些对象的每个指令都具有相同的解码器.`function_record()`检查列出的指令的字节均等于该地址的对象的字节,并增加其他情况.

接下来`grade`在`ledger-x86_64-O1.o`符号表示偏移 0x10和尺寸 46,所以记录涵盖0x10到0x3e,排除了2字节`nop`它们将其次的函数排列在线.

密封字节签名是46字节中的SHA-256字节. 面具字节签名首先将每个移动字段进行零化. 有一个字段:记录`R_X86_64_PC32 .rodata+0x10`在 0x36 的偏移时,宽度为4字节,`leaq (%rip), %rcx`在RELA对象中,这四个字节已经保持了零,所以两个签名在每个对象的每个函数上一致.

记符号签名将14个记符的序列分为:

```text
movslq imulq movq sarq shrq addl addl xorl cmpl ja movl leaq movl retq
```

流量图形签名来自第33课程的基本块和键字边缘,限制在输入可以达到的块. `grade`在`-O1`区块0x10结束在`ja 0x3d`通过一个取边和一个倒边,区块0x31倒入到0x3d,并返回0x3d. 度序列列表 (进出) 每个区块以排序顺序,所以签名读`3b 3e [0/2,1/1,2/0]`为了`grade`在`-O0`记录解决了通过五的间接跳跃`.rela.rodata`它们的入口是33课程的样子,`table`边缘.

标志名字是标志名字. 它在存在时是最强大的事实,而在构建被剥夺时是什么.

函数的函数在一个函数中具有一个`lea`转移到`.rodata`检查边界,以及索引载荷,`referenced_table()`课程36的方法:`cmpl $0x4`在之前`ja`给了五个输入和负载`movl (%rcx,%rax,4), %eax`给每一个字节4个字节.`grade`在`-O1`其他问题`.rodata+0x14`读书`1, 2, 3, 4, 4`现在,我们要去.

## 贪的子

`match_functions()`对于每个阶段,它将每个阶段的尚未匹配的函数由该阶段的签名组合,并接受每个每个方面都有一个函数的每个值.

```python
for stage in order:
    for signature, candidates in left_groups.items():
        partners = right_groups.get(signature, [])
        if len(candidates) != 1 or len(partners) != 1:
            continue
        a, b = candidates[0], partners[0]
        agreeing = [s.id for s in STAGES if a.signatures()[s.id] == b.signatures()[s.id]]
        matches.append(Match(a, b, stage, agreeing, radare2_ratio(a.raw, b.raw), data_equal))
```

课程顺序首先运行四个结构阶段,而名称是最后的,因此结构阶段进行测试,然后名称进行评级.`-O1`反对`-O2`其他:

| Function | Stage | Agreeing signatures | Similarity |
|---|---|---|---|
| `percent_change` | exact bytes | all five | 1.000 |
| `grade` | exact bytes | all five | 1.000 |
| `total` | name | name only | 0.171 |
| `grade_label` | exact bytes | all five | 1.000 |

`total`由于Clang将循环向向性化为SSE2并保持了 Skalar 剩余循环,因此它从31字节和13个指令增加到140字节和41个指令.它的形状从五个块和五个边缘到十个块和十三个边缘.没有结构性存活,只有符号连接了这两个.

为了`-O0`反对`-O1`相同的顺序产生一个错误的匹配:

| Function at -O0 | Matched at -O1 | Stage | Names agree |
|---|---|---|---|
| `percent_change` | `grade_label` | flow-graph shape | no |
| `grade` | `grade` | name | yes |
| `total` | `total` | name | yes |
| `grade_label` | unmatched | | |

`percent_change`在`-O0`是一个四块钻石:一个分支的入口`before == 0`它们的分数序列是`0/2, 1/1, 1/1, 2/0`现在,我们要去.`grade_label`在`-O1`没有`grade`石的形状是两侧的独特,规则也接受.中质量标签是警告,而名称阶段,运行后,与它相矛盾.`grade_label`现在,我们要把它放在右边,`percent_change`其他`grade_label`两对是无与伦比的.

作为真正名称存在的第二阶段,BinDiff命令运行名称匹配,`BINDIFF_ORDER`按照这个顺序,四个都一样.`-O0`功能与它们相匹配`-O1`根据Ghidra的数据,我们可以看到两个单元的流程图形,即四个节点的流程图形.`if`其他`else`在回来之前,他会加入.

## 相同的字节,转移操作

由于Ghidra在进口物体时应用迁移,因此其字节对称器可以比较加载的图像,因此可以将隐藏的迁移字段进行比较. `load_image()`按标题顺序排序,将每个部分与其标题一致.`sh_addralign`并且从0x100000开始,然后它应用每一个记录:`S + A - P`作为签署的32位值写`R_X86_64_PC32`其他`R_X86_64_PLT32`现在`S + A`作为64位的`R_X86_64_64`现在,我们要去.

测试套件对捕获的版本追踪输出进行检查.

| Section | `-O1` program | `-O2` program |
|---|---|---|
| `.text` | 0x100000, 159 bytes | 0x100000, 271 bytes |
| `.rodata` | 0x1000a0 | 0x100110 |
| `.rodata.str1.1` | 0x1000f0 | 0x100160 |

`.text`是 112 字节长于 `-O2`增长率`total`现在,`.rodata`开始 0x70字节以上. `grade`在两个程序中,它在0x100010上.`leaq`在偏移 0x33 的移动场为 0x36 ,结束为 0x3a.移动加值 0x10 加上通常的4 将表设在 `.rodata+0x14`现在,我们要去.

在`-O1`图片:`0x1000a0 + 0x14 - 0x10003a = 0x7a = 122`在"`-O2`图片:`0x100110 + 0x14 - 0x10003a = 0xea = 234`字节字节是`7a 00 00 00`在一个图像中`ea 00 00 00`另一种方法是: 单字节的差异为46个字节,莱文施泰因比率为0.978,而精确字节的阶段失败. 面具字节的阶段将该领域缩小到零,然后通过.`grade_label`移动的 0x70 与表一样,所以它的移动保持相同,它在两个视角传递精确的字节.

吉德拉报道了同样的分裂.`-O1`反对`-O2`查找了精确函数字节匹配`percent_change`其他`grade_label`查看了两个加上`grade`课程图像视图为Ghidra的字节对对称发现的两个函数和对称对称对称的两个函数的完全匹配.测试套件证实了两个等式.

实际情况 6.5.0 `compare_functions`在组装尺寸上,它附加了限制"移动正常化不可".本课程中的载体和面具提供了正常化.

## 什么措施

雷达2 6.2.4 舰艇`radiff2`随着`-AC`它分析文件,每函数打印一行:左边尺寸和地址, MATCH或 UNMATCH,比例,右边尺寸和地址.

| Pair | `percent_change` | `grade` | `total` | `grade_label` |
|---|---|---|---|---|
| `-O0` vs `-O1` | UNMATCH 0.130435 | UNMATCH 0.426829 | UNMATCH 0.106061 | UNMATCH 0.269841 |
| `-O1` vs `-O2` | MATCH 1.000000 | MATCH 1.000000 | UNMATCH 0.171429 | MATCH 1.000000 |
| `-O1` vs `-O1` | MATCH 1.000000 | MATCH 1.000000 | MATCH 1.000000 | MATCH 1.000000 |

雷达源中的比率是`1 - distance / max(len_a, len_b)`函数的基本块的连接链接是函数的指纹和指纹之间的莱文施泰因距离.`radare2_ratio()`在`code/main.py`运行相同的公式在原始函数字节上,并复制0.130435,0.106061,0.269841,和0.171429到六个十分之一.

没有复制 0.426829 `grade`在`-O0`在函数的118字节上,比率为0.296610.`118`没有应用移动,它不能解决跳跃表,所以它的分析达到入口块,该块结束在`jmpq *%rax`根据第33课程的简单递归下降要求的82字节,这些82字节和46字节之间的莱文施泰因距离在`-O1`占47个,`1 - 47/82 = 0.426829`随着`-e bin.relocs.apply=true`雷达2 发现了所有118字节,并打印了0.288136,这等于课程的比率对移动图像,其中四个移动字节不同于对象的零字段.两个工具只有当他们比较相同的字节时才会一致,并且大小列不说明哪些字节进入.

`-O`限制指纹到opcode字节,并将操作数缩小到零.`-O0`其他`-O1`比例上升到0.239130,0.512195,0.303030,0.523810,以及`total`在`-O1`其他`-O2`整个文件距离也存在: `-s`两体之间的距离为162.`-ss`那些数字包括标题,符号表,以及移动部分,没有说任何函数.

## 格希达版本跟踪相关性

基德拉的版本跟踪工具将源程序与目的地程序匹配, 函数相关器是: 精确函数字节匹配,精确函数说明匹配 (掩盖操作符),精确函数内蒙学匹配,并重复函数说明匹配 (1:N和N:M). 数据相关性是精确,重复和相似数据匹配. 符号相关符号的名称与它们的重复或相似性完全一致. 参考相关器使用已接受的匹配来找到更多. 函数必须在目的地程序中定义,并且最小函数长度默认为10字节. 自动版本跟踪运行了精确符号名称,精确数据,精确函数字节,精确函数说明,精确函数内蒙学,重复函数说明,然后是参考对称,并为接受的函数创建暗示匹配. 在11.2中添加了BSim对应器.

课程将相关者无头脑捕捉.`capture_ghidra_vt.sh`进口三个项目,并运行`ListVersionTrackingMatches.java`简短的脚本`AbstractGhidraVersionTrackingScript`通过"自动版本跟踪"来运行六个非引用相关符,并打印每块,函数,相关符和匹配的一行.本节中的每个数字来自这两个文本文件.

| Correlator | `-O0` to `-O1` | `-O1` to `-O2` |
|---|---|---|
| Exact Symbol Name Match | 4 functions | 4 functions, 5 data items (`labels.rel`, `.L.str.1` to `.L.str.4`) |
| Exact Data Match | 0 | 0 |
| Exact Function Bytes Match | 0 | 2 (`percent_change`, `grade_label`) |
| Exact Function Instructions Match | 0 | 3 (adds `grade`) |
| Exact Function Mnemonics Match | 0 | 3 |
| Duplicate Function Instructions Match | 0 | 0 |

在此期间,每个函数都发生了变化.`-O0`其他`-O1`根据BinDiff顺序,每一个匹配都打印出了1.000的相似性和10.000的信心. 帮助页面描述了正确的相似性1.0和信心1.0,所以打印的信心规模是一个开放的问题.

工具类别的程序差异是不同的.它比较了两个程序,在字节,标签,代码单元,引用,评论和函数上具有相同的地址的地址空间.它是地址一致的差异,并与任何移动的东西都不匹配.

## 形和二元忍者,从他们的文档中

迪亚福拉是IDA Pro插件,首次发布于2015年,并于2026-08-18发布3.4.2版本. 它是2.0以来的AGPL-3.0,需要IDAPython,所以它运行在IDAHome或IDA Pro而不是IDAFree. 它将每个数据库出口到SQLite,并将其论作为SQL查询. 每个数都有一个名称,一个最佳,部分或不可靠的类别,一个比率类型,以及像SAME_CPU或SLOW这样的可选标志. 目前的来源中的名称看起来像这个课程的阶段,其中更多是:字节哈希,相同的地址和mnemonics,相同的清洁组合,Mnemonics小质量产品,相同的节点,边缘,循环和密切连接的组件,拓类型哈希,以及需要 Hex-Rays解译器的伪代码模糊哈希. 它的发布说明显示了调节历史:慢的论仅在2,000个函数以下运行, "同样的图表"转移到不可靠的类别. 这一举动是这课中的四块钻石,

双色球忍者6.0,发布于2026-09-03,将双色球类似性作为提供商和解决器的框架. 它将BinDiff作为结构提供商和WARP发送,以提供可移动操作数的精确函数匹配,其指南说使用WARP是当函数与移动指令不同匹配时,以及BinDiff是当它们在重新编译后保持结构相关时. 功能仅在Ultimate版中. 双色球忍者没有安装包管理器, 并没有运行, 所以这段落来自其文档, 没有其他.

自2023-09-25以来,BinDiff自动开源 Apache-2.0下,最新版本是v8.它需要拆卸器和BinExport插件,支持IDA Pro 9.4或更晚版本,二元忍者和Ghidra.学习者免费的路径是Ghidra加上BinExport加上BinDiff命令行.课程没有运行它,其实践实验室要求你这样做.

## 同样的代码,不同的常数

课36保持了候选人重建的`grade`表以3为最后一个输入,原始则以4为最后一个输入.`grade_case10_wrong.o`具有相同的46个字节码.`grade`在`-O1`匹配器将它们对准的字节,每个签名都同意.`1, 2, 3, 4, 4`通过一个迁移和`1, 2, 3, 4, 3`通过其他报告`dataEqual: false`其他三个人`-O1`函数在左边是无与伦比的,因为候选对象拥有一个函数.

这就是补丁分析的形状.一个停止在字节的函数不同表示没有任何变化.Ghidra的精确数据匹配和BinDiff的字符串引用匹配存在于同一个原因.课程的版本很小:对于每个匹配的对,比较每个函数的转移点的字节,检查从边界和从负载的宽度.

## 象征性执行:路径,条件,解决方案

国王在1976年描述了象征执行.输入成为象征.每个语句更新一个象征性存储.每个分支添加了一个条件到路径,一个解决者决定路径是否可行并产生一个具体的输入.一个具有分支的函数成为路径的树,一个叶子是附有条件的回归.

波尔多尼,科帕,德埃利亚,德米特雷斯库和菲诺奇在2018年调查了该领域. 它们的第5节将路径爆炸称为中心成本:执行器可以在每个分支上叉,因此状态数量可以在分支数量中呈指数增长,而一个循环的边界取决于符号输入给出无限的路径. 他们列出的减轻措施是通过在每个分支解决不可实现的路径, 曲执行同时运行了具体和象征性的程序,记录了整个具体运行的条件,否定一个,并要求解决器下一个输入. 达特 (2005),CUTE (2005),SAGE (2008) 是命名系统.

克莱 (Cadar, Dunbar, Engler, OSDI 2008) 是一个象征性的虚拟机,`clang -emit-llvm`产品的产品是产品的产品,而不是二进制.它运输了POSIX环境模型和重播库.在89个GNU核心程序中,它达到84.5%的线路覆盖率,而开发人员测试的覆盖率为67.7%,并发现了56个严重的错误.KLEE 3.2在2025-12-23年发布在NCSA许可下,LLVM 16推.

格在二进制中工作.`angr.Project`具有一个对象映射和移动的CLE加载器,一个将块转化为VEX IR的升降器 (`block.vex`) 作为替代代代,并为各州提供工厂.`SimState`存储记录,内存和解决器.`SimulationManager`列出了国家名字的步骤库`active`现在`deadended`现在`found`现在`avoided`其他`errored`其他`explore(find=..., avoid=...)`按步骤,发动机按顺序运行:故障,系统调用,,单角用于混凝土数据,以及VEX作为倒退.`from angr import claripy`版本的版本是BSD-2-Clause,需要Python 3.12或更晚版本. 在MIT下发布的Z3 5.1.0,是Angr,Triton和KLEE的可选后端的解决方案.

## 一个小的执行者在课程上 34 IR

执行官在`code/main.py`使用小 IR 文本的说明形状34: `Assign`现在`Branch`现在`Jump`现在`Return`另外一个`Switch`课34的实践实验室要求的终结器.`Const`现在`Var`现在`Bin`宽度`Cast`现在`Load`从一个命名的混凝土桌子,`Not`通过手动提升四个固定函数,在简化后,课程34达到水平,`0x66666667`乘法已经成为了`score / 10`任何一个 IR 区块都列出了它来自的指令地址.`check_ir_against_listing()`声明地址完全覆盖函数的列表.

`grade`在`-O0`在该IR中:

```text
0x30: local_8 = score; local_10 = score / 10 - 6; if (local_10 >u 4) goto 0x9a else 0x62
0x62: switch local_10 -> 0x91, 0x88, 0x7f, 0x76, 0x76
0x76: local_4 = 4; goto 0xa1        0x7f: local_4 = 3; goto 0xa1
0x88: local_4 = 2; goto 0xa1        0x91: local_4 = 1; goto 0xa1
0x9a: local_4 = 0                    0xa1: return local_4
```

交换目标来自`jump_table_targets()`现在,我们在线学习的方法,

`execute()`状态包含当前的区块,区块的痕迹,条件列表和一个环境,将每个变量映射到输入中的表达式. `Assign`环境的来源,并存储结果.`Branch`答案是,如果一个问题被解决后,它会被解决后,它会被解决后,然后两次问解决器:一次是加上条件,一次是否定条件.`Switch`通过分离入口等式,将表的入口按目标和分叉进行一次分别的目标,这也是Angr对象式跳跃目标的处理方式. `Return`步骤限制以上的痕迹标志着截断的路径,阻止了它.

走上了走向的道路`return 4`字符号的位置:`local_10 = score / 10 - 6`分公司的情况`local_10 >u 4`现在,我们可以看到一个问题.`!(score / 10 - 6 >u 4)`切换输入3和4都在0x76点,所以叉子到0x76增加了`score / 10 - 6 == 3 || score / 10 - 6 == 4`区块0x76分配`local_4 = 4`转换为 0xa1`4`叶子是这个条件和这个价值的对.

解决方案是`BoundedSearch`设置一个每个输入的有限域,列出域的产量,并通过C语义来评估每个限制:32位包装,缩小分区,签署和未签署的比较,以及算术或逻辑转变.零分数或表表外的表索引使该分配失败了限制.`score`在200至1200的五个路径中`grade`通过解决每个页面的输入集,得到:

| Leaf | Returns | Condition | Scores in the bound |
|---|---|---|---|
| 0x76 | 4 | `!(q >u 4) && (q == 3 \|\| q == 4)` | 90 to 109, 20 inputs |
| 0x7f | 3 | `!(q >u 4) && q == 2` | 80 to 89, 10 inputs |
| 0x88 | 2 | `!(q >u 4) && q == 1` | 70 to 79, 10 inputs |
| 0x91 | 1 | `!(q >u 4) && q == 0` | 60 to 69, 10 inputs |
| 0x9a | 0 | `q >u 4` | 1,351 inputs, the rest of the bound |

`q`代表`score / 10 - 6`那些同样的课程来自`grade`在`-O1`图表路径返回 `lookup_14[q]`问题成为返回值的限制,因为没有一个案例块.`lookup_14`读取从对象的数据中`.rodata`在0x14时.

边界是方法的边界.边界搜索在其域内完成,在其外保持沉默.在 -200至 1200 之间,60至 109 之间的任何分数都不会返回0 并且没有分数为 110 或以上的分数都会返回非零值. z3 证明所有2^32 分数的两种分数,下一节显示两个方法返回不同的证人.

## 测量路径爆炸

`total`在`-O1`课程33的循环:`count`圆的体积增加了`values[rdx]`增加`rdx`其他`jne`返回身体`rdx != rcx`在IR中`rcx`是`zext64(count)`其他`rdx`开始于0 通过后边叉子.`rdx != count`随着`rdx`现在,我们可以看到一个常数,所以路径为`count = c`控制身体`c`随着时间和出口.`count`执行器在0到2之间找到3个路径.在0到4之间找到5个路径,在0到8之间找到9个路径.每个路径返回其数量的三角形总数: (0,0), (1, 1), (2, 3),直到 (8, 36). angr报告3,5和9个已关闭状态和相同的9个数和总数对在所有三个优化级别,包括`-O2`并且从未改变了该范围中的路径数量.

没有绑定执行器永远不会完成,因为每次反复数量是一个可行的路径.`count`报告给出了这三个数字. 解决成本也增长: 33 项评估在线 2, 115 时 4, 531 时 8,和 6,900 项的限运行.

## 关于同样的问题

`capture_angr.py`对于每一个可转移的对象,CLE将ELF后端加载到0x400000,`-O0`其他四个符号,`grade`具有一个象征性的64位参数,`run()`五个国家已结束`-O0`两位`-O1`其他`-O2`现在,我们需要一个小时的时间,`eax == v`答案是,每一个已达成的状态,并要求解决者签署的最低和最高,`v`在每一个级别上,从1到4次,并且 -2147483648到2147483647为0.

通过CFGFast使用了 Reach-a-block表格`grade`在`-O0`列出了发现的八个块课33`explore(find=0x400076)`对于存储4的区块,它发现一个状态,其中3个状态在0.47秒内仍然活跃,在参数和91,93,94,97,101的样本上,至少90和最大109.

为了`percent_change`两国有限制`before[31:0] != 0`其他`before[31:0] == 0`要求非零状态`eax == 50`给了`before = 21055744, after = 332315210`在`-O0`现在`40566804, 834296877`在`-O1`其他`-24484867, 993934694`在`-O2`限量搜索在 - 16 到 16 的答案`before = -10, after = -15`课34列表,答案`before = -25092599, after = 305712160`测试组通过IR评价器评估了所有五个对,每个对都给出了50个. 一个模型证明输入存在. 它不是输入,三个正确的工具给出了三个不同的.

`capture_z3.py`模型`-O1`列出`grade`在32位位位向量中乘以,两个转移,和表,并问了四个问题.列表与源不同`switch`列表与课程36的门重建不同:在706346536的分数上设置,其中`score / 10`只有在0到109之间,相同的差异是不满足的,这就是为什么课36的格格无法看到它.`percent_change`现在`before == 3`结果 -133 得到了`after = 1073741823`经过的搜索发现了第二个证人.`after = -1`在哪里`(-1 - 3) * 100 / 3`缩小到133号,没有过度.

## 建立它

运行演示:

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution
python3 code/main.py
```

输出是一个JSON报告.`diffs`包含对象文件和加载图像匹配的文件`-O0`反对`-O1`其他`-O1`反对`-O2`另外一个`-O0`分数的数量.`toolComparison`对于每个函数,将课程的阶段放在 radiff2 判决和比率和Ghidra 相对符的旁边. `patch`是什么?`grade_case10_wrong.o`其他地方`radare2Ratios`复制了 radiff2 数字,`gradeRelocationCase`显示布局和两个移动,`symbolic`着路径,课程,爆炸数量.

阅读`code/main.py`在此顺序:

1. `ElfObject`阅读部分,符号,`.rela.*`记录`parse_listing()`阅读`llvm-objdump`给我一个短信.
2. `jump_table_targets()`其他`build_cfg()`提供可达的块和打字边缘.`FunctionRecord.signatures()`建立了五个签名.
3. `load_image()`规定分配的部分,并适用`S + A - P`其他`S + A`现在,我们要去.
4. `match_functions()`们的子,`levenshtein()`其他`radare2_ratio()`它们的相似性.
5. `parse_radiff2()`其他`parse_ghidra_vt()`阅读捕获的工具输出,以便进行比较.
6. `evaluate()`现在`substitute()`其他`BoundedSearch`它们是表达式的语义和解决器.
7. `execute()`列出了路径.`solve_returns()`要求每个页面都提供输入集.
8. `grade_ir_o0()`现在`grade_ir_o1()`现在`percent_change_ir_o1()`其他`total_ir_o1()`它们的指令地址.

现在还有三个命令.`python3 code/main.py diff A.o B.o`区分两个物体,它们的列表位于它们旁边,`--image`它们的载荷图像不同.`python3 code/main.py paths grade O0`打印一个函数的路径和类,`paths total 8`子的环节.`python3 code/main.py figure-data`打印了互动图像嵌入的数据.

## 用它

在两个构建自己的程序上使用不同和执行器.

1. 捕获每个对象的列表,然后让`function_record()`检查它与字节相比, 修复列表之前任何其他东西.
2. 首先,要分辨对象文件,阅读每个匹配的阶段,并把流量图形匹配作为建议,直到第二个签名同意.
3. 根据数据的移动相对,从精确字节下降到隐藏字节的函数,移动的数值将移动的部分命名.
4. 对于每个匹配,比较引用表.相同的代码与不同的常数是修改函数.
5. 运行名称阶段和比较. 当构建有符号时,运行BinDiff顺序,只使用结构结果来分类阶段.
6. 当工具报告比率时, 了解它中的字节,
7. 对于您想要的输入函数,写一个每个语句的IR地址,运行`execute()`通过每一个符号循环计数器上一个边框,并解读每一张.
8. 在可用时,通过angr或z3运行相同的查询,并比较类界限.一个样本是一个模型.

## 互动实验室

### 实验室A:手动签名

关于`code`开放字符串:

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution/code
python3 -q
```

运行:

```python
from main import *
grade = {r.name: r for r in fixture_records("ledger-x86_64-O1.o")}["grade"]
grade.raw.hex(" ")
grade.masked_bytes() == grade.raw
grade.mnemonics
grade.cfg_shape()
grade.signatures()
```

在打印之前,写下函数的四个字节在 0x26 的偏移位置.`grade.raw`然后继续运行相同的线路.`grade`其他`ledger-x86_64-O0.o`并且计算`table`边缘`grade.edges`现在,我们要去.

### 实验室B:观察移动破解字节哈希

```python
case = grade_relocation_case()
case["layout"]
case["displacements"], case["rodataShift"]
case["objectBytesEqual"], case["imageBytesEqual"], case["maskedImageBytesEqual"]
```

计算`0x100110 + 0x14 - 0x10003a`在读前,你就写在纸上.`displacements`然后换个基地.`load_image()`转移到0x400000并确认转移不会发生变化,因为两个操作体都在一起移动.

### 实验室C:错误匹配和避免它的顺序

```python
report = diff_fixtures("ledger-x86_64-O1.o", "ledger-x86_64-O2.o")
[(m.left.name, m.right.name, m.stage) for m in report.matches]
report = diff_fixtures("ledger-x86_64-O0.o", "ledger-x86_64-O1.o")
[(m.left.name, m.right.name, m.stage, m.names_agree) for m in report.matches]
report = diff_fixtures("ledger-x86_64-O0.o", "ledger-x86_64-O1.o", stages=BINDIFF_ORDER)
[(m.left.name, m.right.name, m.stage) for m in report.matches]
```

打印`records(...)["percent_change"].cfg_shape()`为了`-O0`其他`grade_label`为了`-O1`然后写一个第六阶段,将分类的内蒙学分类,所以顺序不再重要,看看它是否分开它们.

### 实验室D:路径和边界

```python
report = grade_paths("O0")
[(p["trace"], p["condition"], p["returns"]) for p in report["paths"]]
report["returns"]["4"]
total_paths(2), total_paths(4), total_paths(8)
total_paths(64, step_limit=24)["truncated"]
```

改变`SCORE_DOMAIN`为了`range(0, 110)`并且再进行`grade_paths("O0")`按0的类缩小到60个输入,两个交叉检查仍然保持.`range(90, 100)`读一读哪些叶子根本没有模型.

## 实践实验室

运行BinDiff在固定和比较你已经有三个结果.

要求:

- 安装BinExport扩展在Ghidra 12.1.4中,然后将三个账本程序导出.记录BinExport发布标签和Ghidra版本.
- 安装BinDiff 8命令行,并生成`-O0`其他`-O1`其他`-O1`其他`-O2`只有类似性选项的结果.
- 捕捉每个函数的相似性和信心`code/fixtures/`准确的命令在`build_fixtures.sh`现在,我们要去.
- 延长时间`tool_comparison()`具有一个`bindiff`通过测试,将BinDiff的匹配集合进行比较.`-O1`其他`-O2`课程的隐形字节.
- 写一个段落,以BinDiff使用的算法命名.`grade`并且是否比较加载的图像.

成功意味着演示程序每一个函数打印了四个工具列,新的测试指出一个等式或一个解释的差异.

## 教训艺术品

这一课包括`outputs/skill-binary-diffing-and-symbolic-execution.md`通过它将两个构建和函数转换为一个阶段差异,每个匹配的定质,对每一个未匹配或改变的函数的解释,以及一个由每页和产生它的边界的解决输入集合的路径报告.

## 检查

运行演示和套件:

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

验证应证明:

- 符号表中,有4个函数,它们的尺寸为46,118,66,39`-O0`其他地方`-O1`其他国家`-O2`
- 其他`.rela.text`记录`-O1`描述对象为`llvm-objdump -r`打印它们
- 每个列出的指令都在符号范围内,其字节等于对象,而改变的字节增加
- `total`在`-O1`它有五个块,五个边缘,一个自环,它的形状字符串是决定性的
- `grade`在`-O0`恢复了5个跳板目标`.rela.rodata`它有八块,有十一条边缘.
- 没有桌子,`grade`在`-O0`在四个区块内达到82字节
- `percent_change`在`-O0`其他`grade_label`在`-O1`具有相同的四块形状
- 精确和隐藏的哈希符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符
- `-O1`其他`-O2`分享3个函数的字节,`-O0`其他`-O1`没有分享
- 参考表`grade`读书`1, 2, 3, 4, 4`考生的阅读`1, 2, 3, 4, 3`
- 装载器的位置`.text`现在`.rodata`其他`.rodata.str1.1`在两项计划中,Ghidra将他们放在
- `R_X86_64_PC32`字母是`S + A - P`并且给出了122次流离失所,`R_X86_64_64`列表中的报表`.data`接收字符串地址
- `grade`字节在对象中一致,在图像中差异于0x70的转移,并且掩盖恢复了匹配
- `-O1`其他`-O2`按精确字节匹配三个函数,`total`名称;图像视图相匹配`grade`通过类似的掩盖字节 1 - 1/46
- 课程顺序产生一个错误的流量图形相匹配`-O0`其他`-O1`,BinDiff的命令与四个都匹配
- 补丁候选对字节匹配,并未在引用数据上进行
- 利文施泰因比重重复0.130435,0.106061,0.269841,0.171429,0.426829在82字节上,以及转移字节比率的 radiff2打印,使用转移
- radiff2 的 MATCH 集合等于精确字节阶段,而Ghidra 的字节和指示相关符等于图像视图的精确和掩盖的集合
- 吉德拉只发现了名字.`-O0`其他`-O1`对于每场比赛,
- 手动升高的IR一次覆盖列出的每一个指令,评估员遵循C语义
- `grade`只有五个路径.`-O0`两个在`-O1`同类类的
- 边界类型60至69,70至79,80至89,90至109在三个级别和z3等,有5和2个截止状态,以及90至109的区间
- 两种交叉检查没有在 bound,angr,或z3 中的模型,z3证明列表等于源开关
- 每一个`percent_change`从边界搜索,angr和z3的模型评估到所需的值
- `total`给出三角形总数的3,5,9路径,等于angr,步顶返回22,并缩小2
- 图形数据包含两个视图,移动号码和五个页面及其输入集,站点模块嵌入相同的数据

## 生产失败模式

| Failure | What the weak tool reports | What the analysis must prove |
|---|---|---|
| Byte hash on a loaded image | "grade modified" between two builds with identical code | The field offset, the two displacements, and the section shift; the masked hash matches |
| Masked hash on a relocatable object | "masking made no difference" | The fields were already zero; the relocation records and the referenced constants carry the change |
| Flow-graph shape alone | A confident pair of unrelated functions | A second signature agrees, or the names contradict and the match is dropped |
| Names before structure on a stripped build | Nothing matches | The structural stages with their qualities, and the unmatched list |
| A ratio compared across tools | "radiff2 says 0.43 and the lesson says 0.30" | Which bytes each side fingerprinted; 82 bytes without relocations, 118 with |
| Same code, different constant | "exact match" | The table read through the relocation and the bounds check |
| One model reported as the input | "the score is 90" | The class 90 to 109 with its count, and the bound it was found in |
| Bounded search reported as unsat | "no input gives -133" | Unsat inside the bound only; z3 finds 1073741823 outside it |
| Symbolic loop without a bound | A timeout | The bound or the cap, the paths returned, truncated, and never started |
| Closed tool described as run | Numbers with no capture | A capture file with the command and version, or a sentence saying documentation only |

## 运营规则

按一下一个签名匹配,接受两侧唯一的值,并与匹配保持舞台和其质量. 区分对象和加载的图像,并解释两者之间的每个阶段变化的函数. 在每个分支上用解决器编写路径,绑定每个符号循环,并报告一个输入集合及其边界.一个没有边界的单一模型是样本.

## 进一步阅读

- [BinDiff concepts: matching strategy, algorithms, and quality ratings](https://github.com/google/bindiff/blob/main/docs/concepts.md)
- [BinDiff repository and the v8 open-source release](https://github.com/google/bindiff/releases/tag/v8)
- [BinExport README: IDA Pro, Binary Ninja, and Ghidra exporters](https://github.com/google/binexport)
- [Flake, "Structural Comparison of Executable Objects" (DIMVA 2004) and Dullien, Rolles, "Graph-based comparison of Executable Objects" (SSTIC 2005), shipped in the BinDiff repository](https://github.com/google/bindiff/tree/main/docs/papers)
- [Diaphora README and heuristics source](https://github.com/joxeankoret/diaphora)
- [Ghidra Version Tracking correlators help page](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/VersionTracking/src/main/help/help/topics/VersionTrackingPlugin/VT_Correlators.html)
- [Ghidra Auto Version Tracking help page](https://www.ghidradocs.com/11.0_PUBLIC/help/VersionTracking/help/topics/VersionTrackingPlugin/VT_AutoVT.html)
- [Ghidra 12.1.4 release](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Binary Ninja 6.0 "Krypton" release notes](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html)其他[Binary Similarity user guide](https://docs.binary.ninja/guide/similarity.html)
- [radare2 book: radiff2](https://book.rada.re/tools/radiff2/intro.html)其他[radare2 6.2.4 source](https://github.com/radareorg/radare2/releases/tag/6.2.4)
- [King, "Symbolic execution and program testing" (CACM 1976)](https://dl.acm.org/doi/10.1145/360248.360252)
- [Baldoni et al., "A Survey of Symbolic Execution Techniques" (ACM Computing Surveys 2018)](https://dl.acm.org/doi/10.1145/3182657)
- [Cadar, Sen, "Symbolic execution for software testing: three decades later" (CACM 2013)](https://dl.acm.org/doi/10.1145/2408776.2408795)
- [Cadar, Dunbar, Engler, "KLEE: Unassisted and Automatic Generation of High-Coverage Tests for Complex Systems Programs" (OSDI 2008)](https://www.usenix.org/legacy/events/osdi08/tech/full_papers/cadar/cadar.pdf)其他[KLEE 3.2 release](https://github.com/klee/klee/releases/tag/v3.2)
- [Shoshitaishvili et al., "SoK: (State of) The Art of War: Offensive Techniques in Binary Analysis" (IEEE S&P 2016)](https://oaklandsok.github.io/papers/shoshitaishvili2016.pdf)
- [angr documentation: loading, simulation managers, and the simulation pipeline](https://docs.angr.io/en/latest/core-concepts/toplevel.html)其他[angr 10.0.3 on PyPI](https://pypi.org/project/angr/)
- [Z3 5.1.0 release](https://github.com/Z3Prover/z3/releases/tag/z3-5.1.0)
- [Triton dynamic binary analysis library](https://github.com/JonathanSalwan/Triton)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, Table 4.9: Relocation Types](https://gitlab.com/x86-psABIs/x86-64-ABI)

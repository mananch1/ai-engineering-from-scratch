# 升级到伪代码:IR,SSA,从零开始构建

> 解译输出是一个隐藏的衍生值的索赔. 当你自己构建衍生值时,

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow)
**Time:** ~150 minutes

## 学习目标

- 升降器将x86-64的指示解码到一个小的IR,明确的标志,负载,存储,分支,电话和回报.
- 将框架插槽转化为本地变量,系统V参数注册器转化为参数.
- 在统治边界上建立最小的SSA,然后替换,简化和删除死码.
- 恢复`if`现在`if`随着`else`现在`while`其他`do while`控制流程图,后控制器和后边.
- 解释乘以`0x66666667`作为10的分数,证明身份.
- 读取Ghidra伪代码,以排列一行,并给每个区别背后的指示或猜测.

## 问题

你问了 rea 的伪代码`percent_change`在`ledger-x86_64-O1.o`几达12.1.4在10秒钟内回答说:

```c
undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)

{
  long lVar1;
  undefined1 auVar2 [16];
  undefined1 auVar3 [16];
  
  if (param_1 != 0) {
    lVar1 = (long)((param_2 - param_1) * 100);
    auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
    auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
    return auVar3;
  }
  auVar2._8_8_ = 0;
  auVar2._0_8_ = param_3;
  return auVar2 << 0x40;
}
```

源长四行,需要两行.`int`值和回报`int`吉德拉报告了16字节返回值和源头没有的第三个参数.机器代码的16字节是正确的,吉德拉正确解码了它们.签名是猜测,文本中没有什么标志它为一个.

rea将此文本返回为`procedure_pseudo_code`根据该数据的数据,该数据的数据源和数据源都会被删除,并且将一个限制连接到记录中. 限制是"伪代码是Ghidra解译器输出,不是原始源,并不是Hopper输出的文本等等级".`compare_functions`工具不同于字体行.它的自己的维度记录说"组合是不透明的供应器文本",并且重新定位正常化是不可用的. rea不能看看文本内部,也不能只读取它的代理.

雷亚的DX-Ball记录显示了成本.`FUN_00406400(void)`电话`__ftol()`导向视图显示了堆输入和x87表达式. 代理只因为两个视图不同而继续运行.

您从第33课解码的56个指令开始,将它们提高到IR,将其改名为SSA,简化和结构.然后您将您的伪代码放在Ghidra的旁边.`percent_change`现在`grade`其他`total`测试将输出数据编译成Clang,然后通过输入网运行到源头.

```figure
lifting-stages-percent-change
```

## 固定物和神话

`code/fixtures/ledger.c`是该课程部分的共享设置:

```c
int percent_change(int before, int after) {
    if (before == 0) return 0;
    return (after - before) * 100 / before;
}

int grade(int score) {
    switch (score / 10) {
    case 10:
    case 9: return 4;
    case 8: return 3;
    case 7: return 2;
    case 6: return 1;
    default: return 0;
    }
}

int total(const int *values, int count) {
    int sum = 0;
    for (int i = 0; i < count; i++) sum += values[i];
    return sum;
}
```

`grade_label`现在,我们可以看到一个文件和一个对象中的一个函数,

项目是用`clang -target x86_64-unknown-linux-gnu -O1 -fno-asynchronous-unwind-tables -c ledger.c`现在,我们要去.`-O0`对于堆槽部分,对象坐在他旁边. `code/fixtures/README.md`列出了两个命令,每个对象的SHA-256和Ghidra固定器的 rea命令.

解码器的预言是`llvm-objdump -d -r`两项对象的文本是承诺的`ledger-x86_64-O1.objdump.txt`其他`ledger-x86_64-O0.objdump.txt`测试分析这些文件,并根据地址,字节和AT&T文本进行每个指令的比较.`-O1`在每一个地方,`-O0`什么时候?`llvm-objdump`测试后,第二次测试将其进行直播,然后再进行比较.

这里`percent_change`预言书所说:

```text
0000000000000000 <percent_change>:
       0: 85 ff                        	testl	%edi, %edi
       2: 74 09                        	je	0xd <percent_change+0xd>
       4: 29 fe                        	subl	%edi, %esi
       6: 6b c6 64                     	imull	$0x64, %esi, %eax
       9: 99                           	cltd
       a: f7 ff                        	idivl	%edi
       c: c3                           	retq
       d: 31 c0                        	xorl	%eax, %eax
       f: c3                           	retq
```

## 阶段1: 命令的字节

课33建立了解码器.这个课保持一个小副本,所以代码自动运行.副本完全覆盖了发射的光码.`ledger.c`在`-O0`其他`-O1`对于其他任何东西,它会提升`DecodeError`包含地址和字节字节.

解码器遵循英特尔SDM的字节顺序,卷 2,第2.1节.前置首先是opcode,然后是ModR/M,然后是SIB,然后是移动,然后是即时.`0x40`为了`0x4F`选择一个64位操作符,REX.R扩展了 `reg`根据 REX.X 扩展SIB指数,REX.B 扩展`r/m`根据2.1.3节,定义了三个ModR/M字段.`mod`现在`reg`其他`r/m`其他三种领域`scale`现在`index`其他`base`在64位模式下,`mod = 00`随着`r/m = 101`意思是32位移动与下一个指令相比 (第2.2.1.6节).

| Opcode bytes | Form | Fixture example |
|---|---|---|
| `01 29 31 39 85 89` | `op Ev, Gv` | `29 fe` `subl %edi, %esi` |
| `03 2b 3b 8b 8d` | `op Gv, Ev` | `03 04 97` `addl (%rdi,%rdx,4), %eax` |
| `63` with REX.W | `movsxd Gv, Ev` | `48 63 c7` `movslq %edi, %rax` |
| `69` and `6b` | `imul Gv, Ev, imm` | `48 69 c0 67 66 66 66` |
| `83 /0 /5 /7` | `add sub cmp Ev, imm8` | `83 c1 fa` `addl $-0x6, %ecx` |
| `c1 /5 /7` | `shr sar Ev, imm8` | `48 c1 f9 22` `sarq $0x22, %rcx` |
| `f7 /7` | `idiv Ev` | `f7 ff` `idivl %edi` |
| `ff /0 /4` | `inc Ev`, `jmp Ev` | `48 ff c2` `incq %rdx` |
| `70` to `7f`, `eb`, `e8` | `jcc rel8`, `jmp rel8`, `call rel32` | `74 09` `je 0xd` |
| `99`, `c3`, `90`, `0f 1f` | `cltd`, `retq`, `nop` | `66 0f 1f 44 00 00` `nopw (%rax,%rax)` |

其他`leaq (%rip), %rcx`在`grade+0x23`移动的速度是7字节,移动速度是0字节.`R_X86_64_PC32.rodata+0x10`在抵消`0x36`升降器在第三阶段解决了这个问题.

## 第二阶段:区块和主导者

基本块是一个列表,上面有一个输入,下一个输出.这是艾伦 (1970) 的定义,每个解译纸都会重复使用它.`build_blocks()`集合领导者:函数输入,函数内部的每个跳跃目标,以及跳跃或返回后的每个地址.每个块从最后的指示中获得继任者.一个条件跳跃有两个继任者和一个`jmp`没有一个.`ret`没有任何,而一个以普通命令结束的块, 传递给下一个领导者.

`total`在`-O1`具有五个区块:

| Block | Instructions | Successors |
|---|---|---|
| `0x40` | `testl %esi, %esi` then `jle 0x5c` | `0x5c`, `0x44` |
| `0x44` | `movl %esi, %ecx`, `xorl %edx, %edx`, `xorl %eax, %eax`, `nopw` | `0x50` |
| `0x50` | `addl (%rdi,%rdx,4), %eax`, `incq %rdx`, `cmpq %rdx, %rcx`, `jne 0x50` | `0x50`, `0x5b` |
| `0x5b` | `retq` | exit |
| `0x5c` | `xorl %eax, %eax` then `retq` | exit |

`immediate_dominators()`通过"反向后序"来完成区块,并切断处理前任的统治者,直到没有任何变化.在这样小的图表上,它在两个通过中完成.`total`这里写着`0x44`占据主导地位`0x50`它还说`0x50`通过后边来控制自己, 这就是第8阶段的循环.

两个更长的函数.`dominance_frontiers()`在5阶段使用SSA. `post_dominators()`通过虚拟出口节点运行在反向图表上,以找到8阶段的结合.

## 阶段3:升级到一个小 IR

 IR有六种声明类型:`Assign`现在`Store`现在`Branch`现在`Jump`现在`IndirectJump`其他`Return`表达式是常数,变量,宽度二进制操作, (`sext`现在`zext`现在`trunc`五个形状还存在,但后来才被简化:`Flags`现在`Cond`现在`Concat`现在`SignFill`其他`Symbolic`现在,我们要去.

记录是以64位家族命名的变量.`%eax`是一个`Assign`为了`rax`宽32个,读数为`%eax`是一个读取的`rax`在64位模式下,32位写字扩展到完整的登记器中,因此以后的64位读取相同值为`zext`升降机不知道到达的定义,所以它记录了读数宽度,让SSA在5阶段插入.

| Instruction | IR statements |
|---|---|
| `subl %edi, %esi` | `flags = cmp(esi, edi)` then `esi = esi - edi` |
| `testl %edi, %edi` | `flags = test(edi, edi)` |
| `je 0xd` | `if je(flags) goto 0xd else goto 0x4` |
| `xorl %eax, %eax` | `eax = 0` |
| `cltd` | `edx = signfill(eax)` |
| `idivl %edi` | `dividend = edx:eax`, `eax = dividend / edi`, `edx = dividend % edi` |
| `movslq %edi, %rax` | `rax = sext64(edi)` |
| `sarq $0x22, %rcx` | `rcx = rcx >>s 34` |
| `shrq $0x3f, %rax` | `rax = rax >>u 63` |
| `addl (%rdi,%rdx,4), %eax` | `flags = add(eax, load32(rdi + rdx * 4))` then `eax = eax + load32(rdi + rdx * 4)` |
| `leaq (%rip), %rcx` with its relocation | `rcx = &.rodata+0x14` |
| `pushq %rbp`, `movq %rsp, %rbp`, `subq $0x10, %rsp`, `popq %rbp` | nothing, frame bookkeeping |
| `retq` | `return rax` |

### 旗是变量

每个设置条件代码的指示都会写一个变量,命名为`flags`任何条件跳跃都会读到.`cmp a, b`写道`cmp(a, b)`其他`test a, b`写道`test(a, b)`其他`sub`写着同样的东西.`cmp`子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子`jcc`成为`Branch(Cond(cc, flags), target, fallthrough)`条件代码决定了比较运算器在6阶段遇到时:

| Code | After `cmp(a, b)` | After `test(a, a)` |
|---|---|---|
| `je` | `a == b` | `a == 0` |
| `jne` | `a != b` | `a != 0` |
| `jl`, `jge`, `jle`, `jg` | signed `<`, `>=`, `<=`, `>` | signed compare with 0 |
| `jb`, `jae`, `jbe`, `ja` | unsigned `<`, `>=`, `<=`, `>` | unsigned compare with 0 |

`xorl %eax, %eax`直接升到`eax = 0`它设置的旗写为`add`没有跳跃的标志.`xor`语言问题在`grade`吉德拉失去了它.

### 公司和公司

`cltd`填充`%edx`标签 `%eax`现在,我们要去.`idivl %edi`分开64位对`%edx:%eax`通过`%edi`输入了分数`%eax`其他部分`%edx`升降器将这对子写入一个临时的名称`dividend`剩下的句子,然后是旧的.`%eax`在SSA中,高半是`signfill`简化器将对减少到低半,分数变成32位.`/`现在,我们要去.

### 堆积槽在 -O0

在`-O0`将每个局部人放在中.`percent_change`商店`%edi`为了`-0x8(%rbp)`其他`%esi`为了`-0xc(%rbp)`升降机看着`pushq %rbp`接着是`movq %rsp, %rbp`在前讲之后,一个记忆操作符和基础`%rbp`没有指数,一个移动变量变得一个命名的变量`local_<hex>`现在,我们要去.`-0x8(%rbp)`是`local_8`现在`-0xc(%rbp)`是`local_c`现在,我们要去.

框架指令本身没有产生IR,`subq`其他`addq`现在`%rsp`任何其他记忆操作符都保持一个`Load`或`Store`现在,我们要去.

### 转移

其他`leaq (%rip), %rcx`在`grade`移动在抵消的位置.`0x36`,,,`R_X86_64_PC32.rodata+0x10`链接器会写`S + A - P`进入场地.`S`是 `.rodata`现在`A`是添加剂`0x10`其他`P`命令然后计算`next_ip + disp = (P + 4) + S + A - P = S + A + 4`现在,我们要去.

升降器使用相同的算法,没有链接器.目标抵消是添加加加到指令结束的距离,`0x10 + 4 = 0x14`电气报说`rcx = &.rodata+0x14`现在,这张桌子里,有五个.`int`价值`1, 2, 3, 4, 4`现在,我们要去.

## 第四阶段:召开会议的参数

系统V AMD64 psABI 3.2.3 节通过INTEGER类参数`%rdi`现在`%rsi`现在`%rdx`现在`%rcx`现在`%r8`现在`%r9`结果返回在`%rax`然后`%rdx`3.2.1节说:`%rbp`现在`%rbx`其他`%r12`通过`%r15`其他一切都属于召唤者.

`collect_params()`应用一个规则:在函数中任何写入之前读取的参数注册表是一个参数.`%rsi`没有什么.`%rdi`现在还有两个参数.`percent_change`读书`%edi`其他`%esi`所以它已经`param_1`其他`param_2`现在,我们要去.`grade`只有读音`%edi`现在,我们要去.`total`读书`%rdi`作为64位的基础和`%esi`作为一个32位计数.

现在是Ghidra的`param_3`没有解释.`idivl`写道`%rdx`并且,psABI允许128位的INTEGER返回`%rax:%rdx`作为返回值,Ghidra 保存了两个登记器.

在零路线上,`xorl %eax, %eax`只有写作`%eax`吉德拉读`%rdx`函数的输入,并且该输入成为`param_3`指示是正确的,签名是猜测指示留下开放的部分.

## 阶段5:SSA,每定义一个名称

在升起后,`grade`写道`rax`两次和四次`rcx`没有什么. 赛特龙,费兰特,罗森,韦格曼和扎德克 (1991) 定义了静态单个任务形式. 每个变量都被分配一次,并将在接点上的phi函数从达到它们的定义中选择. 它们的建造有两个阶段. 5.1节将使用统治界限的phi函数, 5.2节将每个定义和使用重新命名,通过走过统治树.`insert_phis()`其他`rename()`按照这两个部分.

美国国家安全局列表`total`显示建筑物产生的产量:

```text
block 0x44 -> 0x50
  ecx_1 = esi_0
  edx_1 = 0
  eax_1 = 0
block 0x50 -> 0x50, 0x5b
  eax_2 = phi(0x44: eax_1, 0x50: eax_3)
  rdx_2 = phi(0x44: edx_1, 0x50: rdx_3)
  flags_2 = phi(0x44: flags_1, 0x50: flags_5)
  flags_3 = add(eax_2, load32(rdi_0 + rdx_2 * 4))
  eax_3 = eax_2 + load32(rdi_0 + rdx_2 * 4)
  flags_4 = inc(rdx_2, 1)
  rdx_3 = rdx_2 + 1
  flags_5 = cmp(zext64(ecx_1), rdx_3)
  if jne(flags_5) goto 0x50 else goto 0x5b
```

登记册的版本0是登记时的值. `esi_0`其他`rdi_0`它们是参数.`0x50`设置区块和它本身,所以每个定义的变量都得到了phi.`flags_2`鱼在抵达时已经死了.`flags`由于没有任何读取,所以第6阶段删除了它.

改名步骤还修复了宽度. `edx_1 = 0`是32位定义的`rdx_3 = rdx_2 + 1`现在,我们可以说是64位,所以`rdx_2`读取更广泛的.`%rcx`在`cmpq %rdx, %rcx`是64位的,而它的定义`ecx_1 = esi_0`现在,我们可以使用它.`zext64(ecx_1)`最后的伪代码中的每一个角色都来自这个规则.

简单的部分是SSA的退出.`coalesce_webs()`通过编译,它可以分成并行拷贝,其中两个phi在一个结合交换值,而课程不处理这种情况. 赛特龙的第7节描述了通过彩色的一般出口.

## 阶段6:简化,取代,删除已死代码

`optimize()`简单化器将表达式重写下面到上面,用以下规则:

| Before | After |
|---|---|
| `Cond(je, cmp(a, b))` | `a == b`, and the same for the other nine codes |
| `Cond(jle, test(a, a))` | `a <= 0` |
| `x + (-6)` | `x - 6` |
| `trunc32(zext64(x))` where `x` is 32-bit | `x` |
| `trunc32(a + b)` on 64-bit operands | `trunc32(a) + trunc32(b)` |
| `(signfill(x): x) / d` | `x / d` at width 32 |
| `load32(base + idx * 4)` | `base[idx]` |
| `trunc32(p >>s 34) + trunc32(p >>u 63)` with `p = sext64(x) * 0x66666667` | `x / 10` |

转换将定义转换为唯一使用.规则有三个条件.使用在同一块中.使用是声明而不是phi.定义包含负载时,没有存储或调用在定义和使用之间.一个常数或部分地址在任何地方都被取代,除了为phi,因为phi参数必须保持为网络的变量.

移除死码删除了一个任务或一个 phi,其目的地没有用处,其表达没有调用.`percent_change`剩下的`edx_2 = dividend_1 % edi_0`现在,我们要把它放在第一位.`edx_1 = signfill(eax_1)`片的使用量为1次,因此被替换为`dividend_1 = edx_1:eax_1`现在,我们要去.

双子规则则则将减少`signfill(eax_1):eax_1`为了`eax_1`它们的分化是`eax_1 / edi_0`现在`eax_1 = esi_1 * 100`具有一个用途,`esi_1 = esi_0 - edi_0`两者都进入返回声明.`flags`它们的定义被取代为分支条件,并且消失.

```c
return (param_2 - param_1) * 100 / param_1;
```

### 乘以0x66666667

`grade`通过10分分分数, 声发出了不`idiv`它发射`imulq $0x66666667, %rax, %rax`现在`sarq $0x22, %rcx`现在,我可以写一本.`shrq $0x3f, %rax`格兰隆和蒙哥马利"乘法使用不变整数分数" (PLDI 1994),描述了该方法.编译器选择一个乘法.`m`,我也在.`s`对于每一个`x`在范围内,`(x * m) >> s`必须等于`x / d`现在,我们要去.

现在`m = 0x66666667 = 1717986919`其他`s = 34`算法证明:`2^34 / 10 = 1717986918.4`顶层是`1717986919`错误是`m * 10 - 2^34 = 6`让每一个32位的身份都能保持.`x`错误时间`2^31`必须在下面.`2^34`错误最大是8`(x * m) >> 34`相当于`floor(x / 10)`对于每一个非负面的`int`现在,我们要去.

对于负值`x`子从零转移,而C向零转移.`shrq $0x3f`产品的标志位,即1`x`增加它就会纠正圆的情况.

`magic_divisor(m, s)`测试一个候选对.`d`作为整数分数`2^s`通过`m`试图`d`其他`d + 1`它接受了`d`只有当`m`等于 `2^s / d`并且错误绑定保持.`s = 33`给出`d = 5`由于`0x66666667`的高峰值`2^33 / 5`检查身份与`c_div(x, 10)`包含两端的 2,406 值`int`距离.

随着我们的认可,`grade`读书`ecx_1 = param_1 / 10 - 6;`随着无识,演示程序打印了原始形式:

```c
rax_1 = (long)param_1 * 0x66666667;
ecx_1 = (int)(rax_1 >> 34) + (int)((unsigned long)rax_1 >> 63) - 6;
```

编译和计算的值都相同. 保持开关. 当你读到未经研究的编译器的代码时,原始形式是诚实的.

## 第七阶段:使用的类型

机器代码有宽度,没有类型.`infer_types()`网络的定义是32位宽的.`int`并且一个64位网络是`long`网络是元素访问的基础`base[idx]`具有4字节元素的`int *`标志型在复印和图像中传播,所以`local_8 = param_1`在`-O0`让他们都能`int *`返回类型是值的宽度在 `return`声明

结果是`total`是`int total(int *param_1, int param_2)`吉德拉写道`int total(long param_1,uint param_2)`提升员使用了量载荷作为指标证据. 吉德拉保持了登记器宽度,并写入为`*(int *)(param_1 + uVar2 * 4)`没有看到.`const`没有一个人能说`int *`其他`unsigned *`现在,我们要去.

李,阿弗杰诺斯和布鲁姆利 (TIE, NDSS 2011) 框架类型恢复作为一个网格上的约束解决.结果是一个顶部和下边界而不是一个猜测.诺南,洛基诺夫和科克 (Retypd,PLDI 2016) 增加了多形和复制性.他们报告了98%的回忆.`const`编译器留下来的语句中指标注释. 这一课使用了这两条最便宜的规则,并给了他们看不到的名称.

没有签名的算法在表达式中保存,而不是变量类型中. `ja`之后`cmpl $0x4, %ecx`图像`(unsigned int)ecx_1 > 4`其他`shrq`图像`(unsigned long)rax_1 >> 63`变量保持签名,以便读者看到注册表值和指令处理它未签名的唯一位置.

## 第八阶段:从图中构建

图表的区域与高层语言的形状相匹配. 循环首先出现 (6.6.1),然后是双向条件 (6.6.2),然后是 n 方向条件 (6.6.3). 雅克丹,埃施韦勒,格哈德斯-帕迪拉和史密斯 (NDSS 2015) 走了另一条道路. 它们从这些条件中得出每个节点执行的条件,并从这些条件中构建结构,没有固定的形状. 他们的动机是测量. 在一个样本上产生了1571个六射线`goto`它们的结构在1,530个核心函数中没有产生任何 49,514 个行中的声明,每32 个行的声明.

`structure()`需要从第二阶段开始的两个事实.`Branch`后边是一个控制其源的块的边缘.自然循环是连接到后边源的块的集合,而不会通过标题.

为了一个`Branch`合`j`其他:

- 如果`j`发射`if (!cond) { fallthrough region }`继续下去`j`现在,我们要去.
- 如果`j`射,发射`if (cond) { target region }`继续下去`j`现在,我们要去.
- 如果`j`输出小臂,然后输出小臂.`if (cond) {... }`后面的另一只手臂,没有.`else`这是早期的回归.`percent_change`其他`total`现在,我们要去.
- 否则发射`if (cond) { target } else { fallthrough }`继续下去`j`这是`percent_change`在`-O0`两支武器都存储在哪里`local_4`我们会在幕上见面.

对于循环标题:

- 如果标题以一个`Branch`发射 `while (cond) { body }`现在,我们要去.`total`在`-O0`标题的形状是这样的:标题比较`local_14`随着`local_c`叶子`jge`现在,我们要去.
- 如果标题也是后边的块,或者锁的出口,则发射`do { body } while (cond);`现在,我们要去.`total`在`-O1`具有这样的形状:块`0x50`增加,增加,比较,然后跳到自己.

`IndirectJump`没有继任者.`grade`在`-O0`是一个真正的跳板桌,`jmpq *%rax`通过5个相对抵消`.rodata+0x0`现在,举起者打印了一张纸.`0x76`现在`0x7f`现在`0x88`其他`0x91`实验室补充说, 鱼只能通过间接跳跃而达到,`switch`现在,我们要去.

### 什么 - O1 对开关做了

在`-O1`没有留下一个跳桌`grade`它计算了`score / 10 - 6`然后它从一个五个输入表中加载了返回值.`int`在`.rodata+0x14`没有签名的比较是一个指令,涵盖了两者.`score / 10 < 6`其他`score / 10 > 10`负差是一个大未签字的值,所以它失败了相同的检查.

桌子可以住`1, 2, 3, 4, 4`对于6到10的定位数.`switch`五个`return`声明成为一个`if`电梯发射的就是这样,并给表命名.`lookup_14`在其部分的抵消之后.

桌子的长度是论的.`table_values()`从开始到下一个表`.rodata`任何一个`.text`在此对象中,下一个引用的偏移是 `0x28`其他类型`grade_label`现在,`lookup_14`没有`(0x28 - 0x14) / 4 = 5`没有签名的比较与4相符.Ghidra也不知道长度.它写道`*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)`读者可以查看五个值.

## 格希德拉的身边

演示片每一对都打印了.`code/fixtures/ghidra/*.json`证据记录说,雷亚在2026年10-14日返回.`npx -y rea-agents@6.5.0 decompile ledger-x86_64-O1.o <function> --provider ghidra --format json`每张记录都包含`confidence: observed`其他`authority: shipped-artifact`文本中所述的说法是从文本中得到的,而记录中没有说明哪些.

### 百分比_变化

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
int percent_change(int param_1, int param_2)   | undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)
{                                              | {
  if (param_1 == 0) {                          |   long lVar1;
    return 0;                                  |   undefined1 auVar2 [16];
  }                                            |   undefined1 auVar3 [16];
  return (param_2 - param_1) * 100 / param_1;  |   if (param_1 != 0) {
}                                              |     lVar1 = (long)((param_2 - param_1) * 100);
                                               |     auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
                                               |     auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
                                               |     return auVar3;
                                               |   }
                                               |   auVar2._8_8_ = 0;
                                               |   auVar2._0_8_ = param_3;
                                               |   return auVar2 << 0x40;
                                               | }
```

| Difference | Instruction or guess |
|---|---|
| Return type `undefined1 [16]` against `int` | Guess. `idivl` writes `%rax` and `%rdx`, and no caller fixes the return width. Both registers were kept. |
| `param_3` | Guess that follows from the first. On the zero path only `%eax` is written, so `%rdx` is read as an input. |
| `lVar1 / (long)param_1 & 0xffffffff` against `/ param_1` | Instruction. `cltd` plus `idivl` is a 64-bit division of a sign-extended 32-bit value, truncated to 32 bits. Both texts describe it. Ours reduced the pair with the `signfill` rule. |
| `lVar1 % (long)param_1` kept | Follows from the return type. The remainder in `%rdx` is dead in our analysis and alive in Ghidra's. |
| `if (param_1 != 0)` with the zero case after it | Instruction order. `je 0xd` jumps to the zero return. Ghidra put the computed arm first, we put the smaller arm first. Both are correct. |
| `return auVar2 << 0x40` | Guess. A 16-byte value with `param_3` in the low half, shifted into the high half, is how the decompiler expresses `%rax = 0` with `%rdx` unknown. |

修复的版本在`code/fixtures/ghidra/ledger-x86_64-O1.ghidra-repaired.c`测试只编辑这个函数,并将其与81个输入对的源进行比较.

### 级别

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
static const int lookup_14[5] = {1, 2, 3, 4, 4}; | uint grade(int param_1)
int grade(int param_1)                         | {
{                                              |   uint uVar1;
  int eax_1;                                   |   uint uVar2;
  int ecx_1;                                   |   
                                               |   uVar2 = param_1 / 10 - 6;
  ecx_1 = param_1 / 10 - 6;                    |   uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);
  eax_1 = 0;                                   |   if (uVar2 < 5) {
  if ((unsigned int)ecx_1 <= 4) {              |     uVar1 = *(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4);
    eax_1 = lookup_14[(unsigned long)(unsigned int)ecx_1]; |   }
  }                                            |   return uVar1;
  return eax_1;                                | }
}                                              |
```

| Difference | Instruction or guess |
|---|---|
| `param_1 / 10 - 6` in both | Instruction. Both recognized the `0x66666667` idiom. |
| `uVar1 = -(param_1 >> 0x1f) ^ (uint)(... >> 0x3f)` against `eax_1 = 0` | Instruction `xorl %eax, %eax` at `grade+0x1a`. The register is zero. Ghidra expressed the `xor` as the sign of `param_1` exclusive-or the sign bit of the product, two spellings of the same bit, and did not cancel them. The value is 0 for every `int`, and the test proves it over 141 scores. A reader who trusts the text believes the default depends on the score. |
| `uVar2 < 5` against `(unsigned int)ecx_1 <= 4` | Instruction `cmpl $0x4, %ecx` then `ja`. Same condition. Ghidra typed `uVar2` as `uint`, we cast at the use. |
| `*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)` against `lookup_14[...]` | Instruction plus relocation. Ghidra loaded the object with `.rodata` at `0x001000a0`, so `0x001000b4` is `.rodata + 0x14`. It applied the relocation and kept a byte address. We named the table and read its five values. |
| `uint` return against `int` | Guess on both sides. The instructions only say 32 bits. |

### 总数

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
int total(int *param_1, int param_2)           | int total(long param_1,uint param_2)
{                                              | {
  int eax_1;                                   |   int iVar1;
  int ecx_1;                                   |   ulong uVar2;
  long rdx_1;                                  |   
                                               |   if (0 < (int)param_2) {
  if (param_2 <= 0) {                          |     uVar2 = 0;
    return 0;                                  |     iVar1 = 0;
  }                                            |     do {
  ecx_1 = param_2;                             |       iVar1 = iVar1 + *(int *)(param_1 + uVar2 * 4);
  rdx_1 = 0;                                   |       uVar2 = uVar2 + 1;
  eax_1 = 0;                                   |     } while (param_2 != uVar2);
  do {                                         |     return iVar1;
    eax_1 = eax_1 + param_1[rdx_1];            |   }
    rdx_1 = rdx_1 + 1;                         |   return 0;
  } while ((unsigned long)(unsigned int)ecx_1 != rdx_1); | }
  return eax_1;                                |
}                                              |
```

| Difference | Instruction or guess |
|---|---|
| `int *param_1` against `long param_1` | Inference on our side from `addl (%rdi,%rdx,4), %eax`. Ghidra kept the width. |
| `uint param_2` and `(int)param_2` against `int param_2` | Ghidra chose unsigned because `cmpq %rdx, %rcx` compares the zero-extended count. We kept `int` and cast at the compare. |
| `do { } while` in both | Instruction `jne 0x50` from the block to itself. The compiler rotated the loop. The `-O0` object keeps the `while`. |
| `ecx_1 = param_2` kept against `param_2 != uVar2` | Our substitution rule stays inside one block. The copy `movl %esi, %ecx` is in the setup block and the compare is in the loop, so the variable stays. Ghidra propagated it. |
| `(unsigned long)(unsigned int)ecx_1` | Instruction. `movl %esi, %ecx` zero-extends, and the 64-bit compare reads the full register. |

行为测试编译了Ghidra文本`grade`其他`total`没有变化,增加了两个`typedef`线条和定义`DAT_001000b4`在20个字节中`.rodata+0x14`两个函数都与网格上的源相匹配.

## 研究对阅读伪代码的说法

根据"人价解"的测量,Ghidra仅可实现20.12%. 根据标准,每五个函数中就有一个重新编译并通过了与基准相关的声明. 通过使用6.7B模型来完善Ghidra的产量,将其提高到52.74%. 根据"2025年ACL发现"的规定,该公司运行了6个工业解译器和6个LLM方法,从130个程序中运行了23,400个功能. 士课程的读数更好,功能准确度也低了52%.

,拉夫和米辛斯基 (arXiv 2609.05370,2026年9月) 解释了这一差距. 他们使用了300个GitHub库功能和287个基于CVE的功能. 强大的精炼模式将Ghidra的构建率从75%提高到90%,而行为匹配率则从74%降至62%. 通过基准标准的每项测试的候选人仍然对模糊输入不同. 这在8个系统中的4.9%的病例中发生, 据了解,在此之前,

它们的原因很简单. 发明的字段,类型,字符号和守护者取代了解译器在文本中留下的可见未知.

这就是升降器印记的原因.`(unsigned long)(unsigned int)ecx_1`它们会打印一张纸条,而不是一张纸条.`switch`对于未解决的跳跃,`lookup_14`每一行都有一个可以读取的衍生值.`code/main.py`吉德拉的`undefined1 [16]`炼模型的清洁.`int percent_change(int, int)`同样的模型将在下一个函数上产生一个纯错误的签名.

## 建立它

运行演示:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
```

演示程序解码了该系统的四个功能`-O1`根据这些指令,`llvm-objdump`为了`percent_change`现在`grade`其他`total`它打印了SSA前的IR,SSA列表与phi函数,以及Ghidra文本旁边的伪码.`grade`由于分区识别已关闭,`-O0`版本`percent_change`现在`total`其他`grade_label`总结:

```json
{
  "functions": ["grade", "grade_label", "percent_change", "total"],
  "instructions": 56,
  "blocks": {"percent_change": 3, "grade": 3, "total": 5, "grade_label": 4},
  "phis": {"percent_change": 0, "grade": 1, "total": 2, "grade_label": 1},
  "objdumpDifferences": 0
}
```

阅读`code/main.py`在此顺序:

1. `read_object()`阅读部分标题,符号表,以及`.rela`转移可用的ELF64对象的部分.
2. `decode_one()`其他`render_att()`解码一个指示,然后打印它.`llvm-objdump`没有.
3. `build_blocks()`现在`immediate_dominators()`其他`dominance_frontiers()`建立图形的数据.
4. `lift_insn()`将一个指示转化为IR声明并解决迁移.
5. `insert_phis()`其他`rename()`建立SSA并插入宽度.
6. `make_simplifier()`现在`substitute_once()`其他`remove_dead_code()`走进里面`optimize()`现在,我们要去.
7. `collect_params()`现在`coalesce_webs()`其他`infer_types()`给网格一个名称,一个字符.
8. `structure()`随着`post_dominators()`其他`natural_loops()`发射C体.
9. `lift_function()`结合各个阶段`lift_all()`提升一个整体的物体,用电话签名.

## 用它

为了从你自己的对象上提升一个函数,`clang -target x86_64-unknown-linux-gnu -c`打电话`lift_all(read_object(path))`当解码器升级时`DecodeError`读取信息中的opcode字节,在SDM的opcode地图中找到它.然后在文件顶部的表中添加一个输入.当升降器在mnemonic上升起时,添加一个规则`lift_insn()`结构者发射了`goto`打印图表`IrPrinter`并且决定哪个形状规则缺失. 每个加值都是小的,每个则是对比的.`llvm-objdump`现在的测试也没有.

在Ghidra安装后,运行rea在同一函数上并将两个文本放在一边.在相信任何文本之前填写区别表.背后有指示的区别是演示选择.背后没有指示的区别是猜测,猜测可以在任何一方.

## 互动实验室

### 实验室A:一个函数通过五个阶段

关于`code`开放字符串:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode/code
python3 -q
```

运行:

```python
from main import *
obj = read_object(FIXTURES / "ledger-x86_64-O1.o")
lifted = lift_function(obj, "percent_change")
for insn in lifted.decoded: print(f"0x{insn.address:x}: {insn.raw.hex(' ')}  {insn.att()}")
print(lifted.ir_text)
print(lifted.ssa_text)
print(lifted.pseudocode)
```

追随`%esi`其他`subl %edi, %esi`为了`esi_1 = esi_0 - edi_0`为了`(param_2 - param_1)`字节`fe`是 电路的ModR/M`subl`随着`mod = 11`现在`reg = edi`其他`r/m = esi`现在,我们要去.

### 实验室B:分类语法

```python
print(lift_function(obj, "grade", recognize_division=False).pseudocode)
print(magic_divisor(0x66666667, 34), magic_divisor(0x66666667, 33), magic_divisor(0x66666667, 31))
```

首先打印原始乘法和转移.`10 5 None`找出为什么同一乘法被5分,一个转变少,为什么转变31失败错误界限.

### 实验室C:一个新的对象停止了解码器

设置在`-O2`然后抬起它.

```bash
clang -target x86_64-unknown-linux-gnu -O2 -fno-asynchronous-unwind-tables -c fixtures/ledger.c -o "$TMPDIR/ledger-O2.o"
python3 -c 'from main import *; from pathlib import Path; import os; lift_all(read_object(Path(os.environ["TMPDIR"]) / "ledger-O2.o"))'
```

解码器停在`0x56`随着`opcode 81 is outside the fixture subset`号码`81`命令是:`andl $0x7ffffff8, %edx`在向循环中使用了它`total`加入它旁边`0x83`在`decode_one()`单个字符串的输入量,然后再运行,并注意下一个字符号是什么.`llvm-objdump`现在,我们要去.

### 实验室D:Ghidra的标准值

```python
print(ghidra_pseudocode("grade"))
for score in (-25, 0, 59, 60, 100, 115):
    sign = -(score >> 31)
    product_sign = ((score * 0x66666667) & ((1 << 64) - 1)) >> 63
    print(score, sign ^ product_sign)
```

每一行都印出了0个字符.`uVar1`字体是正确的,不可读的,`xorl %eax, %eax`两字节中写出来.

## 实践实验室

加入`switch`恢复`grade`在`-O0`现在,我们要去.

要求:

- 阅读五个搬迁情况`.rela.rodata`其他`ledger-x86_64-O0.o`其他`R_X86_64_PC32.text+offset`根据表计算五个案例目标`.rodata+0x0`现在,我们要去.
- 添加一个`Switch`标签表,目标列表和默认的 IR `ja`在餐桌装载之前.
- 给开关块每一个目标一个继任者,以便主导者,后主导者和SSA继续工作.
- 发射`switch (local_10) { case 0:... }`只有一个`case`目标和结合后的共享尾.
- 留下任何间接跳跃的笔记,
- 通过语音编译结果,并与测试中分数格式的源相比较.

成功意味着`grade`在`-O0`其他`grade`在`-O1`电网上产出相同的输出,`switch`没有`goto`现在,我们要去.

## 艺术品

这一课产生了`outputs/skill-lifting-to-pseudocode.md`通过它将每个行的衍生函数升高,并将差异表与分解器的文本填写.该文件需要对每个不同行进行参考拆除,移动,调用公约,编译检查,行为比较和判决.

## 检查

运行演示和确定性套件:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

套房有25个测试.

- 解码器同意承诺`llvm-objdump`通过直播,`llvm-objdump`装机时
-  REX,ModR/M,SIB,以及移动场的抵消`leaq (%rip), %rcx`按SDM描述的方式进行解码
- 在子集之外的 opcode 增加`DecodeError`没有猜测
- 电脑电脑系统的读者列出了四个函数和三个函数.`.text`迁移
- `total`五个街区,`0x44`占据主导地位`0x50`其他`0x50`是一个循环头,它本身是身体
- `%rdi`其他`%rsi`成为`param_1`其他`param_2`其他`param_1`在`total`是`int *`
- `test`其他`cmp`随着`je`现在`jle`现在`ja`其他`jge`成为`==`现在`<=`没有签名`>`其他`<`
-  SSA 将两个直播fi函数放在 循环头上`total`没有任何东西.`percent_change`
- 编号码`percent_change`现在`grade`其他`total`相当于预期的文本
- `grade`读书`param_1 / 10 - 6`含有识别和含有识别的原料
- `0x66666667`换34分为10,换33分为5,身份保持在网格上
- 上堆积的插槽`-O0`成为`local_4`现在`local_8`现在`local_c`图指针从来没有出现
- 其他问题`.rodata+0x14`读书`1, 2, 3, 4, 4`通过迁移
- 直接跳进`grade`在`-O0`报告的四个未提起的块
- 达灯具包含16字节的返回,`param_3`没有简单的`xor`其他`long param_1`
- 六个发射单位编译与`clang -std=c11 -Wall -Werror`为了`x86_64-unknown-linux-gnu`
- 提升的函数,修复的Ghidra文本和源头在输入网上打印相同的232行

## 生产失败模式

| Failure | What the weak reading reports | What the derivation must show |
|---|---|---|
| Decompiler widens the return value | "returns a 16-byte struct" | `idivl` writes `%rdx`, and no caller fixes the width |
| Decompiler adds a parameter | "takes three arguments" | The register is read on one path only because the return width was guessed |
| Idiom goes unrecognized | "multiplies by a magic constant" | The multiplier is the ceiling of `2^s / d` and the error bound holds |
| Idiom is recognized and a `xor` is lost | "default depends on the input" | `xorl %eax, %eax` writes zero, two spellings of one bit cancel |
| Pointer typed as integer | "takes a long" | A scaled 4-byte load from the register is pointer evidence |
| Table length unknown | "reads past the table" | The next referenced offset or the section end bounds it, and the range check agrees |
| Loop shape misread | "while loop" | The header has no exit, the latch jumps back, so it is `do while` |
| Indirect jump guessed | "switch with five cases" | The targets come from `.rela.rodata`, or the jump is reported as unresolved |
| Substitution across a store | "one expression" | A loaded value moved past a store or a call changes meaning |
| Refinement model edits the text | "compiles and looks right" | Build rate up, behaviour match down, invented guards replaced visible unknowns |
| Decoder guesses an unknown opcode | "disassembly looks fine" | `DecodeError` with the address and the byte, then one table entry checked against `llvm-objdump` |

## 运营规则

按照你没有看到的衍生法来处理伪代码.用可以阅读的升降器将其从说明中复制.将两条文本放在一边,并在每条不同行后写说明或猜测.

## 进一步阅读

- [Intel SDM Volume 2, chapter 2, Instruction Format](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [System V AMD64 psABI, section 3.2, Function Calling Sequence](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Cytron et al., Efficiently Computing Static Single Assignment Form and the Control Dependence Graph, TOPLAS 1991](https://doi.org/10.1145/115372.115320)
- [Cooper, Harvey, Kennedy, A Simple, Fast Dominance Algorithm, 2001](https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/dom14.pdf)
- [Cifuentes, Reverse Compilation Techniques, PhD thesis, QUT 1994](https://eprints.qut.edu.au/36820/)
- [Yakdan et al., No More Gotos, NDSS 2015](https://www.ndss-symposium.org/ndss2015/ndss-2015-programme/no-more-gotos-decompilation-using-pattern-independent-control-flow-structuring-and-semantics/)
- [Granlund, Montgomery, Division by Invariant Integers using Multiplication, PLDI 1994](https://doi.org/10.1145/178243.178249)
- [Lee, Avgerinos, Brumley, TIE, NDSS 2011](https://www.ndss-symposium.org/ndss2011/tie-principled-reverse-engineering-of-types-in-binary-programs/)
- [Noonan, Loginov, Cok, Polymorphic Type Inference for Machine Code, PLDI 2016](https://arxiv.org/abs/1603.05495)
- [LLM4Decompile, EMNLP 2024](https://arxiv.org/abs/2403.05286)
- [DecompileBench, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1194/)
- [Liu, Raff, Micinski, When LLM Decompilers Recompile More and Preserve Less, 2026](https://arxiv.org/abs/2609.05370)
- [rea 6.5.0 Ghidra provider notes](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/ghidra-provider.md)
- 标签 `3b79af2`这一课描述了:`src/domain/functionComparisonDimensions.ts`现在`website/evidence/dx-ball-sound-pan.md`现在,我们要去.

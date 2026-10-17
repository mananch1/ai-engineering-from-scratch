# رفع إلى رمزية مزيفة: IR، SSA، والهيكل من الصفر

> إن إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصدار إصاد إصاد إصاد إصاد إصاد إصاد إصاد إصاد إصاد إصاد إصاد إصاد إصاد إصاد إصاد إ إ إصاد إصاد إ إصاد إصاد إ إصاد إصاد إ إ إصاد إ إ إصاد إصاد إ إصاد إ إ

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow)
**Time:** ~150 minutes

## أهداف التعلم

- قام لفيت بتشفير التعليمات x86-64 إلى إير صغير مع العلامات الصريحة والحملات والمخازن والفروع والمكالمات والرد.
- حول فتحات الإطار إلى متغيرات محلية و سجلات الحجج في النظام V إلى ملامح.
- بناء SSA الحد الأدنى مع وظائف phi على حدود الهيمنة، ثم استبدال، تبسيط، وإزالة الرمز الميت.
- استعادة`if`،`if`مع`else`،`while`و`do while`من الرسم البياني للتدفق التحكم مع المسيطرة و الحواف الخلفية.
- اشرح الارتفاع بالـ `0x66666667`و إثبات الهوية
- اقرأ رمز غيدرا للعمل نفسه خط بعد خط وسمي التعليمات أو التخمينات وراء كل اختلاف

## المشكلة

لقد طلبت من (ريا) رمزية مزيفة`percent_change`في`ledger-x86_64-O1.o`جيدرا 12.1.4 أجاب في حوالي عشر ثواني:

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

المصدر أربع خطوط طويلة.`int`القيم والعائدات واحد `int`. غيدرا تقرير قيمة 16 بايت العودة وبرامج ثالثة التي لا يمتلكها المصدر. السادسة عشر بايت من رمز الآلة صحيحة، وغيدرا فك كل منهم بشكل صحيح. التوقيع هو تخمين، ولا شيء في النص يُعَلِّم ذلك كواحدة.

يعيد rea هذا النص على أنه `procedure_pseudo_code`نتيجة ويربط مع قيود واحد إلى السجل. يقرأ القيود "Pseudocode هو Ghidra إصدار مفكّر، وليس المصدر الأصلي ولا نص-متساوية إلى إصدار هوبر".`compare_functions`الوسيلة تختلف عن الرمز السريع كخطوط من النص. سجل الأبعاد الخاص بها يقول "التجميع هو نص مزود غير شفاف" وأن التطبيعات الإعادة التنظيمية غير متوفرة. لا يمكن للشخصية أن تنظر داخل النص، ولا يمكن للعامل الذي يقرأه فقط.

سجل "دي إكس بال" الذي أصدرته "ريا" يظهر التكلفة`FUN_00406400(void)`اتصل`__ftol()`. عرض التعليمات أظهر مدخل كومة و تعبير x87. الوكيل استمر فقط لأن المشاهدين لا يوافقون.

هذه الدروس تبني المشتق الذي يخفي المتحلل. تبدأ من 56 تعليمات التي فك الدروس 33، رفعها إلى IR، إعادة تسمية إلى SSA، تبسيط، والهيكل. ثم تضع رمزك السوديو بجوار Ghidra ل`percent_change`،`grade`و`total`الاختبارات تجمع الخروج الخاص بك مع كلانغ وتشغيله ضد المصدر على شبكة المدخل.

```figure
lifting-stages-percent-change
```

## المُثبت والكلمة

`code/fixtures/ledger.c`هو المكون المشترك لهذا الجزء من الدورة:

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

`grade_label`و هو أيضا في الملف و في الكائن و الظهور يرفع ذلك أيضا.

لقد تم بناء هذا الموضوع`clang -target x86_64-unknown-linux-gnu -O1 -fno-asynchronous-unwind-tables -c ledger.c`- المُؤمنون`-O0`الجسم يجلس بجانبه لقطاع حفرة الدرج. `code/fixtures/README.md`يذكر كل من الأوامر، SHA-256 لكل جسم، والإرشاد rea الذي أنتج مصالح Ghidra.

الـ (أوراكول) لـ (الـ (ديكودر) هو`llvm-objdump -d -r`. نصها لكلتا الجانبين ملتزم`ledger-x86_64-O1.objdump.txt`و`ledger-x86_64-O0.objdump.txt`. يقوم اختبار بمحاكمة تلك الملفات ويقارن كل تعليمات عن طريق العنوان، عن طريق البايتات، و عن طريق نص AT&T. وهذا يشمل جميع الـ 56 تعليمات في `-O1`وكل تعليمات في`-O0`متى`llvm-objdump`يتم تركيبها على المسار المثبت، ثم يتم إجراء اختبار ثان على الهواء مباشرة ويتم مقارنة مرة أخرى.

ها هي`percent_change`كما يطبعها الوسيلة:

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

## المرحلة الأولى: البايتات إلى التعليمات

دروس 33 بنيت المفكّر. هذا الدروس يحتفظ بنسخة صغيرة حتى يتم تشغيل الشفرة بمفردها.`ledger.c`في`-O0`و`-O1`لأي شيء آخر يثير`DecodeError`مع العنوان و البايت

يتبع المفكّر ترتيب البايتات من نظام إدارة التكنولوجيا الفعلية Intel SDM، الجدول 2، القسم 2.1. تأتي المواعيد المسبقة أولاً، ثم رمز الخيار، ثم ModR/M، ثم SIB، ثم التنقل، ثم الفورية. القسم 2.2.1 يضيف مقدمة REX، بيع واحد من `0x40`إلى`0x4F`و يجب أن يجلس مباشرة قبل رمز الاختيار. REX.W يختار أوبراندا 64 بت، REX.R يمتد `reg`في هذه الحالة، تمدد REX.X مؤشر SIB، و REX.B تمدد `r/m`أو قاعدة SIB. في القسم 2.1.3 يتم تعريف الحقول الثلاثة ModR/M `mod`،`reg`و`r/m`، و 3 حقل SIB`scale`،`index`و`base`في وضع 64 بت`mod = 00`مع`r/m = 101`يعني تحرك 32 بت بالنسبة إلى التعليمات التالية (الفقرة 2.2.1.6).

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

- نعم`leaq (%rip), %rcx`في`grade+0x23`هو سبعة بايتات مع صفر التحرك.`R_X86_64_PC32.rodata+0x10`في التكافؤ`0x36`،حقل التنقل، يحلّه المصعد في المرحلة الثالثة

## المرحلة الثانية: الحدود والسيطرة

كتلة أساسية هي سلسلة من البيانات مع إدخال واحد في الأعلى ومخرج واحد في الأسفل. هذا هو التعريف من آلن (1970) ، وكل ورقة تحويل يستخدمها مرة أخرى. `build_blocks()`تجمع القادة: إدخال الوظيفة، كل هدف قفزة داخل الوظيفة، وكل عنوان بعد قفزة أو عودة. كل كتلة تحصل على خلفيين من آخر تعليماتها. قفزة مشروطة لديها اثنين من خلفيين و`jmp`لديه واحد.`ret`لا يوجد أي منها، وكل حجر ينتهي في تعليم عادي يمر إلى القائد التالي.

`total`في`-O1`يحتوي على خمسة كتب:

| Block | Instructions | Successors |
|---|---|---|
| `0x40` | `testl %esi, %esi` then `jle 0x5c` | `0x5c`, `0x44` |
| `0x44` | `movl %esi, %ecx`, `xorl %edx, %edx`, `xorl %eax, %eax`, `nopw` | `0x50` |
| `0x50` | `addl (%rdi,%rdx,4), %eax`, `incq %rdx`, `cmpq %rdx, %rcx`, `jne 0x50` | `0x50`, `0x5b` |
| `0x5b` | `retq` | exit |
| `0x5c` | `xorl %eax, %eax` then `retq` | exit |

`immediate_dominators()`هو خوارزمية التكرار من كوبر، هارفي، وكينيدي (2001). يمر الكتل في الترتيب الخلفي المعاكس ويقطع زعماء السلفين المعالجين حتى لا يتغير أي شيء. على الرسوم البيانية هذه الصغيرة أنهى في اثنين من الممرات.`total`يقول ذلك`0x44`يهيمن`0x50`كما يقول أيضاً`0x50`يهيمن على نفسه من خلال الحافة الخلفية، وهذا هو كيفية مرحلة 8 يجد الحلقة.

نفس الوظيفة تعمل مرتين أكثر`dominance_frontiers()`يستخدمها في SSA في المرحلة 5. `post_dominators()`يستخدم الرسم البياني المعاكس مع عقدة خروج افتراضية للعثور على الجمعيات في المرحلة 8.

## المرحلة الثالثة: رفع إلى جهاز إشعاعي صغير

الـ IR لديها ستة أنواع من البيانات: `Assign`،`Store`،`Branch`،`Jump`،`IndirectJump`و`Return`. تعبيرات هي ثابتات، متغيرات، عمليات ثنائية مع عرض،`sext`،`zext`،`trunc`والحملة. هناك خمسة أشكال أخرى فقط لتبسيطها في وقت لاحق:`Flags`،`Cond`،`Concat`،`SignFill`و`Symbolic`. . .

السجلات هي المتغيرات التي تسميها عائلة 64 بت.`%eax`هو`Assign`إلى`rax`مع عرض 32، وقراءة `%eax`هو قراءة من `rax`في وضع 64 بتة كتابة 32 بتة يمتد صفر في السجل الكامل، لذلك قراءة لاحقة 64 بتة من نفس القيمة هي `zext`. لا يعرف المرفع تعريف الوصول لذلك يسجل عرض القراءة ويسمح لـ SSA بإدخال الصبغ في المرحلة 5.

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

### العلامات هي متغيرة

كل تعليمات تحدد رموز الحالة تكتب متغير واحد يسمى `flags`وكل قفزة مشروطة تقرأها`cmp a, b`يكتب`cmp(a, b)`، أ`test a, b`يكتب`test(a, b)`و`sub`يكتب نفس الشيء`cmp`أ.`jcc`يصبح`Branch(Cond(cc, flags), target, fallthrough)`. رمز الشروط يقرر عامل المقارنة عندما يلتقي الاثنان في المرحلة 6:

| Code | After `cmp(a, b)` | After `test(a, a)` |
|---|---|---|
| `je` | `a == b` | `a == 0` |
| `jne` | `a != b` | `a != 0` |
| `jl`, `jge`, `jle`, `jg` | signed `<`, `>=`, `<=`, `>` | signed compare with 0 |
| `jb`, `jae`, `jbe`, `ja` | unsigned `<`, `>=`, `<=`, `>` | unsigned compare with 0 |

`xorl %eax, %eax`يرفع مباشرة إلى`eax = 0`. العلامات التي يضعها مكتوبة`add`العلامات التي لا يقال أي قفزة في العرض`xor`الـ " إيقوم "`grade`حيث فقدت (غيدرا)

### المجموعة والإدب

`cltd`يملأ`%edx`مع علامة صغيرة من`%eax`. .`idivl %edi`يُقسم زوج 64 بت `%edx:%eax`من`%edi`, يضع النسبة في `%eax`، والباقي في`%edx`. المرفع يكتب الزوج في اسم مؤقت`dividend`أولاً، البقية من البيان بعد ذلك تقرأ القديمة`%eax`وليس النسبة الطازجة في SSA النصف العلوي يظهر ك`signfill`من النصف المنخفض. يقلل من السهل زوج إلى النصف المنخفض، والقسم يصبح 32 بت `/`. . .

### فتحات الدرج عند -0

في`-O0`كلانغ) يبقي كل محلي في الإطار)`percent_change`المتاجر`%edi`إلى`-0x8(%rbp)`و`%esi`إلى`-0xc(%rbp)`ثم يقرأها مرة أخرى لكل استخدام.`pushq %rbp`يليها `movq %rsp, %rbp`بعد ذلك المنتهية، عمل ذاكرة مع قاعدة`%rbp`، لا مؤشر، والتحرك يصبح متغير يسمى`local_<hex>`. .`-0x8(%rbp)`هو`local_8`،`-0xc(%rbp)`هو`local_c`. . .

تعليمات الإطار نفسها لا تنتج أي IR، ولا تفعل`subq`و`addq`على`%rsp`أي عمل آخر في الذاكرة يبقى`Load`أو`Store`. . .

### عمليات نقل

- نعم`leaq (%rip), %rcx`في`grade`لا يوجد أيّة نقل في الملف.`0x36`يقول`R_X86_64_PC32.rodata+0x10`. الرابط سيكتب`S + A - P`إلى الميدان`S`هو عنوان `.rodata`،`A`هو الإضافة`0x10`و`P`هو عنوان الحقل. ثم التعليمات الحسابات `next_ip + disp = (P + 4) + S + A - P = S + A + 4`. . .

يطبق المرفع نفس الحسابات دون رابط. تعويض الهدف هو الإضافة زائد المسافة من الحقل إلى نهاية التعليمات.`0x10 + 4 = 0x14`- الإشارة تقول`rcx = &.rodata+0x14`و الطاولة هناك تحتوي على الخمسة`int`القيم`1, 2, 3, 4, 4`. . .

## المرحلة الرابعة: أجزاء من اتفاقية الدعوة

النظام V AMD64 psABI، القسم 3.2.3، يمر حجج فئة INTEGER في `%rdi`،`%rsi`،`%rdx`،`%rcx`،`%r8`،`%r9`النتيجة تعود في`%rax`ثم`%rdx`القسم 3.2.1 يقول أن`%rbp`،`%rbx`و`%r12`من خلال`%r15`ينتمي إلى الدعوة ويجب أن يتم الحفاظ عليها.

`collect_params()`يطبق قاعدة واحدة: سجل الحجج الذي يتم قراءته قبل أي كتابة في الوظيفة هو معايير. أعلى مثل هذا السجل يصلح العد، لذلك وظيفة التي تقرأ `%rsi`و أبداً`%rdi`لا يزال لديهما معايير`percent_change`يقرأ`%edi`و`%esi`، لذلك هو`param_1`و`param_2`. .`grade`يقرأ فقط`%edi`. .`total`يقرأ`%rdi`كمقاعدة 64 بت و `%esi`كمعدل 32 بت

الآن (غيدرا)`param_3`لديه تفسير`idivl`يكتب`%rdx`، و السابي يسمح بعودة إنتيغر 128 بت في `%rax:%rdx`في ملف كائن لا يوجد مكالمة لتقرير حجم القيمة العائدة.

على مسار الصفر`xorl %eax, %eax`يكتب فقط`%eax`غيدرا قراءة`%rdx`هناك كمدخول للعمل، وهذا المدخل أصبح `param_3`التوجيهات صحيحة التوقيع هو تخمين حول الأجزاء التي تركها التوجيهات مفتوحة

## المرحلة 5: SSA، اسم واحد لكل تعريف

بعد رفعها`grade`يكتب`rax`أربع مرات و`rcx`ثلاث مرات سيترون، فيرانتيه، روزن، ويغمان، و زاديك (1991) تحدد شكل التعيين الواحد ثابت. يتم تعيين كل متغير مرة واحدة بالضبط، وتقوم وظائف phi في نقاط الاندماج باختيار من بين التعريفات التي تصل إليها. بناؤهم له خطوتين يضع القسم 5.1 وظائف phi باستخدام حدود الهيمنة، ويقوم القسم 5.2 بإعادة تسمية كل تعريف واستخدام مع المشي فوق شجرة الهيمنة.`insert_phis()`و`rename()`اتبع هذين القسمين

قائمة الـ SSA`total`يظهر ما ينتجته البناء:

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

النسخة 0 من السجل هو قيمته عند الدخول. `esi_0`و`rdi_0`هذه هي المعلمات.`0x50`لديه اثنين من السلفين، كتلة الإعداد والنفس، لذلك كل متغير مع تعريف في الحلقة يحصل على phi هناك.`flags_2`فاي ميت عند وصوله، وذلك لأن "مينيمل إس إس إيه"`flags`لديه تعريفات في كلا السلفين، والمرحلة 6 إزالتها لأن لا شيء يقرأها.

الخطوة التي تُعيد تسميتها تحدد أيضاً الواسعة. `edx_1 = 0`هو تعريف 32 بت و `rdx_3 = rdx_2 + 1`هو 64 بت، لذلك فاي`rdx_2`يُستغرق الواسعة الأوسع.`%rcx`في`cmpq %rdx, %rcx`هو 64 بت بينما تعريفها `ecx_1 = esi_0`هو 32 بت، لذلك الاستخدام يصبح `zext64(ecx_1)`كل ممثلة في الرمز السودوي النهائي تأتي من هذه القاعدة الواحدة

الجزء "اللئ" هو الطريق الخروج من SSA. `coalesce_webs()`يجمع كل phi مع حججاته في متغير واحد ، يسمى شبكة الإنترنت ، ويعطي شبكة الإنترنت اسمًا واحدًا. يعمل هذا للأشكال التي ينبعثها المجمع لهذه الوظائف. يمكن أن ينتهي الأمر على نسخ متوازية ، حيث يجمع اثنان من الفي في واحد قيم التبادل ، ولا يتعامل الدروس مع هذه الحالة. يصف قسم سيترون 7 الخروج العام من خلال التلوين.

## المرحلة 6: تبسيط، استبدال، إزالة الرمز الميت

`optimize()`يكرر ثلاث مرّات حتى لا يتغير أي شيء. يكرر البسط المُبسيط تعبيرات أسفل إلى الأعلى مع هذه القواعد:

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

تحويل المستخدم إلى المستخدم الوحيد. القاعدة لديها ثلاثة شروط. الاستخدام في نفس الكتلة. الاستخدام هو بيان وليس phi. عندما يحتوي التعريف على تحميل، لا يوجد أي مخزن أو دعوة بين التعريف والاستخدام. يتم استبدال عنوان ثابت أو قسم في أي مكان باستثناء phi، لأن حجة phi يجب أن تبقى متغيرة للشبكة.

إزالة الرمز الميت يزيل مهمة أو phi لا تستخدم وجهتها والتي لا تُستخدم تعبيرها.`percent_change`الباقي`edx_2 = dividend_1 % edi_0`هو ميت، لذا يذهب أولاً.`edx_1 = signfill(eax_1)`مع استخدام واحد، لذلك يتم استبدالها في `dividend_1 = edx_1:eax_1`. . .

قاعدة الزوجات ثم تقلل `signfill(eax_1):eax_1`إلى`eax_1`، والانقسام يصبح`eax_1 / edi_0`الآن`eax_1 = esi_1 * 100`يستخدمها فقط مرة واحدة، و`esi_1 = esi_0 - edi_0`يستخدم واحد، وكلاهما ينتقل إلى بيان العودة.`flags`تم استبدال التعريفات في ظروف الفرع وتختفي.

```c
return (param_2 - param_1) * 100 / param_1;
```

### ضرب 0x66666667

`grade`يُقسم النتيجة بـ10 و "كلانغ" يُصدّر رقم لا`idiv`.إنها تنبعث`imulq $0x66666667, %rax, %rax`، ثم`sarq $0x22, %rcx`على نسخة، ثم `shrq $0x3f, %rax`ويقوم غرانلاند ومونتغومري، "التقسيم بواسطة عدد كامل غير متغير باستخدام الارتفاع" (PLDI 1994) ، بتصف الطريقة. يختار المجمع مضاعفا `m`ودورة`s`لكل شخص`x`في النطاق`(x * m) >> s`يجب أن يكون متساويا`x / d`. . .

ها هي`m = 0x66666667 = 1717986919`و`s = 34`التحقق من الحساب:`2^34 / 10 = 1717986918.4`، والسقف هو`1717986919`الخطأ هو`m * 10 - 2^34 = 6`لتحفظ الهوية على كل 32 بت`x`، أوقات الخطأ`2^31`يجب أن تبقى أسفل`2^34`، لذلك يجب أن يكون الخطأ على الأكثر من 8 ستة هو على الأكثر من 8، لذلك `(x * m) >> 34`متساوية`floor(x / 10)`لكل شخص غير سلبي`int`. . .

للخسارة`x`الأرضية تدور بعيدا عن الصفر بينما C تدور نحو الصفر. `shrq $0x3f`يستخرج جزء علامة من المنتج، وهو 1 بالضبط عندما `x`هو سلبي. إضافة ذلك تصحيح التجويب.

`magic_divisor(m, s)`يختبر زوج من المرشحين.`d`كـ (الـ)`2^s`من`m`ويحاول`d`و`d + 1`. إنه يقبل`d`فقط عندما`m`يساوي السقف`2^s / d`و الخطأ المرتبط يظل. نفس المضاعف مع `s = 33`يعطي`d = 5`، وهو صحيح، لأن `0x66666667`هو أيضا السقف من `2^33 / 5`اختبار يُحقق من الهوية ضد`c_div(x, 10)`على 2406 قيم بما في ذلك كلا الطرفين من`int`المدى

مع الاعتراف على،`grade`يقرأ`ecx_1 = param_1 / 10 - 6;`مع إيقاف التعرف ، الطباعة الديمو النموذج الخام:

```c
rax_1 = (long)param_1 * 0x66666667;
ecx_1 = (int)(rax_1 >> 34) + (int)((unsigned long)rax_1 >> 63) - 6;
```

كل منهما يجمع وكلا الحساب نفس القيم. احتفظ بالتحول. عندما تقرأ الكود من مُجمّع لم تدرس، فإن النسخة الخام هي الصادقة.

## المرحلة السابعة: أنواع الاستخدام

رمز الآلة لديه عرض وليس هناك طرق`infer_types()`يُحوّل ثلاثة أنواع من الأدلة إلى أنواع "C". شبكة تعريفها 32 بت على عريض`int`، و شبكة 64 بتة هي`long`شبكة يُعد أساس وصول عنصر`base[idx]`مع عناصر 4 بايت هي `int *`. نوع المؤشر ينتشر على طول النسخ والفيس، لذلك `local_8 = param_1`في`-O0`يجعل كلاهما`int *`. نوع العودة هو عرض القيمة في `return`البيانات

النتيجة`total`هو`int total(int *param_1, int param_2)`كتب (غيدرا)`int total(long param_1,uint param_2)`كلاهما متوافق مع التعليمات، قام المرفع باستخدام الحمل المقياسي كدليل مؤشر، ويحفظ غيدرا عرض السجل وكتب الوصول على النص:`*(int *)(param_1 + uVar2 * 4)`لا يمكن أن ترى`const`ولا أحد يستطيع أن يقول`int *`من`unsigned *`. . .

لي، أفجيرينوس، وبروملي (TIE، NDSS 2011) استرداد نوع الإطار كحل القيود على شبكة. النتيجة هي الحدود العليا والأسفل بدلا من تخمين واحد. نونان، لوجينوف، وكوك (Retypd، PLDI 2016) إضافة البوليمورفيزم والتكريض. يبلغون عن 98 في المائة من التذكر على `const`هذه الدروس تستخدم قواعد الأرخص من هذا الخط من العمل وتسمى ما لا يمكن رؤيته.

يتم حفظ الحساب غير الموقّع في التعابير وليس في أنواع المتغيرات. `ja`بعد`cmpl $0x4, %ecx`بصمات مثل`(unsigned int)ecx_1 > 4`و`shrq`بصمات مثل`(unsigned long)rax_1 >> 63`. تبقى المتغيرات موقعة بحيث يرى القارئ قيمة السجل والمكان الوحيد الذي تعامله التعليمات كغير موقعة.

## المرحلة الثامنة: التهيئة من الرسم البياني

تشيفونيتس (1994) ، الفصل 6، يطابق مناطق الرسم البياني مع أشكال لغة عالية المستوى. تأتي الحلقات أولاً (6.6.1) ، ثم الشروط الثنائية (6.6.2) ، ثم الشروط الثنائية (6.6.3). يقتبس ياكدان، إشويلر، جيراردز-باديلا، وسميث (NDSS 2015) الطريق الآخر. ويتم استنباط الحالة التي تنفذ تحتها كل عقدة ويتم بناء الهيكل من تلك الحالات، دون أشكال ثابتة. الدافع لديهم هو قياس النقاط الماسحة على عينة واحدة`goto`البيانات في 49514 سطر، واحد لكل 32 سطر، وتهيكلها لم ينتج أي من 1530 وظيفة أساسية.

`structure()`هو عضو صغير في العائلة الأولى. يحتاج إلى اثنين من الحقائق من المرحلة 2.`Branch`الحلقة الطبيعية هي مجموعة من الكتل التي تصل إلى المصدر الحافة الخلفية دون عبور الرأس.

لأجل`Branch`مع المشترك `j`:

- إذا`j`هو الهدف القفز، إصدار`if (!cond) { fallthrough region }`واصلي`j`. . .
- إذا`j`هو الانهيار، إصدار`if (cond) { target region }`واصلي`j`. . .
- إذا`j`هو الخروج من الوظيفة، كلتا الذراعين تعود. إبعاد الذراع الأصغر ك`if (cond) {... }`والذراع الأخرى بعدها، بدون`else`هذا هو العودة المبكرة في`percent_change`و`total`. . .
- وإلا إصدار`if (cond) { target } else { fallthrough }`واصلي`j`هذا هو`percent_change`في`-O0`، حيث تخزن كلتا الأسلحة`local_4`و نلتقي في المنتدى

لقب الحلقة:

- إذا كان العنوان ينتهي في `Branch`مع خلف واحد خارج الحلقة، إصدار `while (cond) { body }`. .`total`في`-O0`يحتوي على هذا الشكل: العنوان يقارن `local_14`مع`local_c`و يترك`jge`. . .
- إذا كان الرأس هو أيضا الكتلة مع الحافة الخلفية، أو الخروج من القفل بدلا من ذلك، إرسال `do { body } while (cond);`. .`total`في`-O1`يحتوي على هذا الشكل:`0x50`يضيف، يزيد، يُقارن، ويقفز إلى نفسه.

- نعم`IndirectJump`ينتهي حيًا بدون خلفاء`grade`في`-O0`هو طاولة قفزة حقيقية`jmpq *%rax`من خلال خمسة تعويضات نسبية في `.rodata+0x0`و الدروس 33 تحل تلك الأهداف هنا يقوم الرقيب بطبع مذكرة بدلاً من ذلك`0x76`،`0x7f`،`0x88`و`0x91`يصلون فقط من خلال القفز غير المباشر ولم يتم رفعهم.`switch`. . .

### ما الذي فعله -O1 بالمفتاح

في`-O1`لم يحتفظ (كلانغ) بمكتب القفز`grade`لقد حاسبت`score / 10 - 6`ثم قام بتحويل القيمة المرجعية من جدول 5 إدخالات من `int`في`.rodata+0x14`. المقارنة غير الموقعة هي تعليمة واحدة تغطي كلتا`score / 10 < 6`و`score / 10 > 10`الفرق السلبي هو قيمة كبيرة غير مؤكدة، لذلك فإنه يفشل نفس الشيك.

الطاولة تحمل`1, 2, 3, 4, 4`للنسب من 6 إلى 10.`switch`مع خمسة `return`تصبح البيانات`if`و قراءة صف، و هذا ما يُصدره المرفع و يسميه الطاولة`lookup_14`بعد تعويض قطعه

طول الجدول هو هيرستيكي`table_values()`القراءة من الجدول الاول الى القادم`.rodata`تعويض أي `.text`إشارات إعادة التأهيل، أو إلى نهاية القسم. في هذا الكائن التوجه المرجح التالي هو `0x28`، جدول `grade_label`، لذا`lookup_14`- نعم`(0x28 - 0x14) / 4 = 5`الإدخالات. المقارنة غير الموقعة مقابل 4 توافق. غيدرا لا يعرف طول أيضا.`*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)`، عنوان بايت زائد تعويض، وترك الخمسة القيم للقارئ للبحث.

## جنبا إلى جنب مع غيدرا

الـ (ديمو) طبعت كل زوج من النص من (غيدرا)`code/fixtures/ghidra/*.json`، سجلات الأدلة التي عادت في 2026-10-14.`npx -y rea-agents@6.5.0 decompile ledger-x86_64-O1.o <function> --provider ghidra --format json`كل سجل يحمل`confidence: observed`و`authority: shipped-artifact`النص يلاحظ، والادعاءات داخل النص مشتقة، والسجلات لا تقول أي منها.

### %_change

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

النسخة المصحوبة في`code/fixtures/ghidra/ledger-x86_64-O1.ghidra-repaired.c`يعدل هذا الوظيفة فقط، ويقارن الاختبار بها مع المصدر على أكثر من 81 زوج مدخل.

### الدرجة

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

### إجمالي

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

اختبار السلوك يجمع نص Ghidra ل `grade`و`total`غير متغيرة، يضيف اثنين`typedef`الخطوط و تعريف`DAT_001000b4`من العشرين بايت في `.rodata+0x14`. كلتا الوظائف تتطابق مع المصدر عبر الشبكة

## ما يقوله البحوث عن قراءة الرمز السودوكود

وقاس LLM4Decompile (EMNLP 2024) Ghidra وحدها عند إعادة تنفيذية 20.12 في المئة على HumanEval-Decompile. حوالي واحد من كل خمس وظائف أعادت وتجاوزت التأكيدات التي تأتي مع المعيار المرجعي. إصلاح إنتاج غيدرا مع طراز 6.7B رفع ذلك إلى 52.74 في المئة. أدار DecompileBench (Findings of ACL 2025) ستة محركات تحلل صناعية وستة مقاربات لدرجة الماجستير في 23،400 وظيفة من 130 برنامجًا. نتائج الماجستير في الدراسة كانت أفضل وكانت دقة الوظائف أقل بنسبة 52.2 بالمئة.

تشرح ليو وراف وميسينسكي (arXiv 2609.05370, سبتمبر 2026) الفجوة. استخدموا 300 وظيفة مكتبة GitHub و 287 وظيفة CVE. أشد نموذج التحسين رفع معدل بناء غيدرا من 75 إلى 90 في المائة بينما انخفض معدل مطابقة السلوك من 74 إلى 62 في المائة. لا يزال المرشحون الذين اجتازوا كل اختبار في المعيار المرجعي يختلفون عن المدخلات المضطربة. حدث ذلك في 4.9 في المائة من الحالات عبر ثمانية أنظمة، و في ما يصل إلى 13 في المائة من أنظمة واحدة. حتى واحد عشر من نقاط الضعف المكتشفة اختفت من الرمز المكرر.

السبب الذي يلاحظونه بسيط. استبدل الحقول والأنواع والحراس والحراس المختلفة المعلومة التي تركها المُفكّر في النص.

هذا هو السبب في أن المصعد يطبع`(unsigned long)(unsigned int)ecx_1`بدلاً من اسم جميل، فإنه يطبع ملاحظة بدلاً من`switch`للقفز غير المحل، و `lookup_14`بدلاً من اللقب ما قد يختلق النموذج. كل سطر لديه مشتق يمكنك قراءته`code/main.py`-جيدرا`undefined1 [16]`إنه قبيح و صادق، النموذج النقي نظيف`int percent_change(int, int)`نفس النموذج سوف ينتج توقيعًا خاطئًا على الوظيفة التالية.

## بناءها

أطلق التجربة:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
```

الاكتشاف الاكتشافي جميع الوظائف الأربعة من الـ `-O1`ويقارن كل تعليم مع المشاركة`llvm-objdump`النص`percent_change`،`grade`و`total`يطبخ الإشارة الإير قبل SSA، وإدراج SSA مع وظائف phi، والرمز السوداوي بجوار نص Ghidra. ثم يطبخ `grade`مع إيقاف إدراك التقسيم،`-O0`نسخ `percent_change`،`total`و`grade_label`وملخص:

```json
{
  "functions": ["grade", "grade_label", "percent_change", "total"],
  "instructions": 56,
  "blocks": {"percent_change": 3, "grade": 3, "total": 5, "grade_label": 4},
  "phis": {"percent_change": 0, "grade": 1, "total": 2, "grade_label": 1},
  "objdumpDifferences": 0
}
```

اقرأ`code/main.py`في هذا الترتيب:

1. `read_object()`يقرأ عناوين القسم، والجدول الرمزي، وال `.rela`أجزاء من كائن ELF64 قابل للتنقل.
2. `decode_one()`و`render_att()`فكّر إرشاد واحد وطبّقها على الطريق`llvm-objdump`-لا، لا
3. `build_blocks()`،`immediate_dominators()`و`dominance_frontiers()`بناء الحقائق الرسمية.
4. `lift_insn()`تحويل تعليمات واحدة إلى بيانات إضافية وتحل عمليات نقل.
5. `insert_phis()`و`rename()`بناء SSA وإدخال ألقى عرض.
6. `make_simplifier()`،`substitute_once()`و`remove_dead_code()`أركض إلى الداخل`optimize()`. . .
7. `collect_params()`،`coalesce_webs()`و`infer_types()`اسم وتصفيف الشبكات.
8. `structure()`مع`post_dominators()`و`natural_loops()`يُبعث الجسم C.
9. `lift_function()`يربط المراحل معاً و`lift_all()`يرفع كائن كامل مع توقيعات من "كالي".

## استخدمها

لرفع وظيفة من جسمك، قم ببناءها`clang -target x86_64-unknown-linux-gnu -c`و اتصل`lift_all(read_object(path))`عندما يرفع المُشفّر`DecodeError`، قراءة البايت رمز الخيار من الرسالة والعثور عليه في خريطة SDM رمز الخيار. ثم إضافة إدخال واحد إلى الجداول في أعلى الملف. عندما يرفع المرفع على الممنيونيك، إضافة قاعدة واحدة إلى `lift_insn()`عندما يُصدّر المُصنّع`goto`، طبع الرسم البياني مع `IrPrinter`ويقرر أي قاعدة الشكل مفقودة. كل إضافة صغيرة و كل واحد يتم التحقق من`llvm-objdump`أو الصوت من خلال الاختبارات القائمة

عندما يتم تثبيت Ghidra، قم بتشغيل rea على نفس الوظيفة ووضع النصين جانبا إلى جانبه. املأ جدول الفرق قبل أن تصدق أي من النصين. الفرق مع تعليمات وراءه هو اختيار العرض. الفرق بدون تعليمات وراءه هو تخمين، ويمكن أن يكون تخمين على أي من الجانبين.

## المختبر التفاعلي

### المختبر (أ): وظيفة واحدة عبر خمس مراحل

من`code`الإرشادات، مفتوحة Python:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode/code
python3 -q
```

أركض

```python
from main import *
obj = read_object(FIXTURES / "ledger-x86_64-O1.o")
lifted = lift_function(obj, "percent_change")
for insn in lifted.decoded: print(f"0x{insn.address:x}: {insn.raw.hex(' ')}  {insn.att()}")
print(lifted.ir_text)
print(lifted.ssa_text)
print(lifted.pseudocode)
```

اتبع`%esi`من`subl %edi, %esi`إلى`esi_1 = esi_0 - edi_0`إلى`(param_2 - param_1)`. البايت`fe`هو ModR/M من `subl`، مع`mod = 11`،`reg = edi`و`r/m = esi`. . .

### المختبر ب: لغة القسم

```python
print(lift_function(obj, "grade", recognize_division=False).pseudocode)
print(magic_divisor(0x66666667, 34), magic_divisor(0x66666667, 33), magic_divisor(0x66666667, 31))
```

الخط الأول يطبع المضاعفة الخامة والتحركات. الخط الثاني يطبع`10 5 None`. اكتشف لماذا يُقسم نفس المضاعف بـ 5 مع خفض واحد من التحول، ولماذا التحول 31 يفشل في حد الخطأ.

### المختبر C: كائن جديد يوقف المفكّر

قم ببناء الجهاز في `-O2`ورفعها

```bash
clang -target x86_64-unknown-linux-gnu -O2 -fno-asynchronous-unwind-tables -c fixtures/ledger.c -o "$TMPDIR/ledger-O2.o"
python3 -c 'from main import *; from pathlib import Path; import os; lift_all(read_object(Path(os.environ["TMPDIR"]) / "ledger-O2.o"))'
```

القيادة توقف عند `0x56`مع`opcode 81 is outside the fixture subset`- رمز البيانات`81`هو النموذج المجموعة 1 مع 32 بتة مباشرة.`andl $0x7ffffff8, %edx`، و استخدمه (كلانغ) في الحلقة المتجهة للخلفية`total`إضافة إلى الجانب`0x83`في`decode_one()`، إعادة تشغيل، ونلاحظ أي رمز opcode يوقفك بعد ذلك. يزداد الجمع الفرعي إدخال واحد في وقت واحد، ويتم التحقق من كل إدخال مقابل `llvm-objdump`. . .

### المختبر D: قيمة افتراضية Ghidra في الدرجة

```python
print(ghidra_pseudocode("grade"))
for score in (-25, 0, 59, 60, 100, 115):
    sign = -(score >> 31)
    product_sign = ((score * 0x66666667) & ((1 << 64) - 1)) >> 63
    print(score, sign ^ product_sign)
```

كل سطر يطبع 0.`uVar1`تعبير صحيح وغير قراءة، والإرشاد `xorl %eax, %eax`يقول ذلك في بايتين

## مختبر التدريب

إضافة`switch`التعافي`grade`في`-O0`. . .

المتطلبات:

- اقرأ خمسة عمليات نقل في`.rela.rodata`من`ledger-x86_64-O0.o`كل واحد`R_X86_64_PC32.text+offset`و قم بحساب الأهداف الخمسة من الجدول`.rodata+0x0`. . .
- إضافة`Switch`المنتهي إلى IR مع تعبير المؤشر، قائمة المستهدفة، والمتخلفة من `ja`قبل تحميل الطاولة
- أعطوا حظر المفتاح خلفاً واحداً لكل هدف حتى يستمر المهيمنون و المهيمنون و المتحكمون في السيطرة
- إصدار`switch (local_10) { case 0:... }`مع واحد`case`لكل هدف والإيقاع المشترك بعد الانضمام
- احتفظي بالخطاب لأي قفزة غير مباشرة لا يمكنك ربط طاولة
- قم بتجميع النتيجة مع الكلانغ ومقارنةها مع المصدر على شبكة النتيجة في الاختبارات.

النجاح يعني`grade`في`-O0`و`grade`في`-O1`إنتاج نفس الناتج على الشبكة، مع واحد `switch`و لا`goto`. . .

## القطع الأثرية

هذا الدرس يُنتج`outputs/skill-lifting-to-pseudocode.md`. استخدمها لرفع وظيفة مع مشتق لكل سطر وملأ جدول الاختلاف ضد نص المتحلل. يتطلب الفن الإجمالي التفكيك المرجعي، والتحويلات، والاتفاقية الدعوة، والتحقق من التجميع، ومقارنة السلوك، وحكم لكل سطر مختلف.

## تحقق من ذلك

إشغال التجربة والسلسلة التحديدية:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

الفندق يحتوي على 25 اختبار، التحقق يثبت:

- المُشفّر يوافق على المُتّخذ بالالتزام`llvm-objdump`نص على كل تعليمات كل من الأشياء، ومع حية `llvm-objdump`عندما يتم تثبيته
- REX، ModR/M، SIB، ومقاطع حقل التحرك `leaq (%rip), %rcx`يتم فك رموزها كما يصف MDD
- رمز عكس خارج مجموعة الفرعية يرفع `DecodeError`بدلاً من التخمين
- قراءة ELF يصف الأدوات الأربعة والثلاثة `.text`عمليات نقل
- `total`لديه خمسة كتلة`0x44`يهيمن`0x50`و`0x50`هو رأس حلقة مع نفسها كالجسم
- `%rdi`و`%rsi`أصبح`param_1`و`param_2`و`param_1`في`total`هو`int *`
- `test`و`cmp`مع`je`،`jle`،`ja`و`jge`أصبح`==`،`<=`غير موقعة`>`و`<`
- يضع SSA وظائف Phi مباشرة في رأس الحلقة من `total`ولا أحد في`percent_change`
- الرمز الرسمي لـ`percent_change`،`grade`و`total`يساوي النص المتوقع
- `grade`يقرأ`param_1 / 10 - 6`مع الاعتراف على و مضاعفة الخام مع الاعتراف خارج
- `0x66666667`مع التحول 34 مقسمة بـ 10 ، مع التحول 33 مقسمة بـ 5 ، والهوية تحتفظ على الشبكة
- فتحات التجميع في`-O0`أصبح`local_4`،`local_8`،`local_c`، و لا يظهر مؤشر الإطار أبدا
- الطاولة في `.rodata+0x14`يقرأ`1, 2, 3, 4, 4`من خلال الإنتقال
- القفز غير المباشر في`grade`في`-O0`يتم الإبلاغ عن أربعة كتلة غير مرفوعة
- أجهزة "غيدرا" تحتوي على العودة البايت 16،`param_3`، غير مبسط`xor`، و `long param_1`
- الـ6 وحدات المنبعثة تقوم بتجميعها مع`clang -std=c11 -Wall -Werror`لـ`x86_64-unknown-linux-gnu`
- الوظائف المرفوعة، نص Ghidra المصلح، والصادر طباعة نفس 232 خط على شبكة المدخل

## أساليب فشل الإنتاج

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

## قاعدة العمليات

اعتبر الرمز السوداني مشتقًا لم ترينه. قم بتكرارها من التعليمات باستخدام جهاز رفع القراءة. ضع النصين جانباً إلى جانبه، واكتب التعليمات أو التخمينات وراء كل سطر يختلف. قم بتجميع كليهما وارتكبهما ضد المصدر عندما يكون لديك. لا يهم كم يقرأ خطًا بدون تعليمات وراءه يبقى تخمينًا.

## المزيد من القراءة

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
- 6.5.0 مصدر في التج `3b79af2`أن هذا الدروس يصف: `src/domain/functionComparisonDimensions.ts`،`website/evidence/dx-ball-sound-pan.md`. . .

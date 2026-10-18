# Tăng lên Pseudocode: IR, SSA, và cấu trúc từ đầu

> Khi bạn tự tạo ra dẫn xuất, mỗi dòng của đầu ra đó trở thành một thứ bạn có thể kiểm tra.

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow)
**Time:** ~150 minutes

## Mục tiêu học tập

- Lift giải mã các hướng dẫn x86-64 vào một IR nhỏ với các cờ rõ ràng, tải, cửa hàng, chi nhánh, cuộc gọi và trả lời.
- Chuyển các khe khung thành biến địa phương và các hồ sơ lập luận của Hệ thống V thành tham số.
- Xây dựng SSA tối thiểu với các chức năng phi trên biên giới thống trị, sau đó thay thế, đơn giản hóa và loại bỏ mã chết.
- Tái phục hồi `if`- `if`với `else`- `while`, và`do while`từ biểu đồ lưu lượng điều khiển với các bộ thống trị sau và cạnh sau.
- Giải thích nhân bằng `0x66666667`như là chia bằng 10 và chứng minh danh tính.
- Đọc mã giả Ghidra cho cùng một hàm hàng xét và đặt tên hướng dẫn, hoặc đoán, đằng sau mọi sự khác biệt.

## Vấn đề

Anh đã hỏi rea về mã giả của `percent_change`trong `ledger-x86_64-O1.o`Ghidra 12.1.4 trả lời trong khoảng 10 giây:

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

Nguồn là bốn dòng dài.`int`giá trị và trả lại một `int`Ghidra báo cáo giá trị trả lại 16 byte và một tham số thứ ba mà nguồn không có. 16 byte mã máy là chính xác, và Ghidra giải mã đúng mỗi một trong số họ.

rea trả lại văn bản này như `procedure_pseudo_code`kết quả và gắn một giới hạn cho bản ghi. giới hạn đọc "Pseudocode là đầu ra Ghidra decompiler, không phải nguồn gốc và không phải văn bản tương đương với đầu ra Hopper".`compare_functions`công cụ khác biệt pseudocode như các dòng văn bản. hồ sơ kích thước của nó nói "Sự tập hợp là văn bản cung cấp không minh bạch" và việc chuẩn hóa di chuyển không có sẵn. rea không thể nhìn vào nội dung văn bản, và cũng không thể một đại lý chỉ đọc nó.

Phiên bản DX-Ball của rea cho thấy chi phí.`FUN_00406400(void)`gọi `__ftol()`. Khả năng hiển thị một đầu vào hàng và một biểu thức x87.

Bài học này xây dựng nguồn gốc mà máy giải mã ẩn. Bạn bắt đầu từ 56 hướng dẫn mà bài học 33 giải mã, nâng chúng lên một IR, đổi tên thành SSA, đơn giản hóa, và cấu trúc. Sau đó bạn đặt mã giả bên cạnh Ghidra của`percent_change`- `grade`, và`total`Các thử nghiệm biên soạn đầu ra của bạn với clang và chạy nó với nguồn trên một lưới đầu vào.

```figure
lifting-stages-percent-change
```

## Đường và Kinh

`code/fixtures/ledger.c`là sự cố chung của phần này của khóa học:

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

`grade_label`và trong các mục khác, và các trình bày cũng nâng nó lên. ba chức năng trên là những gì bài học này kiểm tra.

Vật vật được xây dựng với `clang -target x86_64-unknown-linux-gnu -O1 -fno-asynchronous-unwind-tables -c ledger.c`- `-O0`đối tượng ngồi bên cạnh nó cho phần khe xếp chồng. `code/fixtures/README.md`liệt kê cả hai lệnh, SHA-256 của mỗi đối tượng, và lệnh rea tạo ra các thiết bị Ghidra.

Lời tiên tri cho máy giải mã là `llvm-objdump -d -r`. Văn bản của nó cho cả hai đối tượng được cam kết như `ledger-x86_64-O1.objdump.txt`và `ledger-x86_64-O0.objdump.txt`Một thử nghiệm phân tích các tệp đó và so sánh từng hướng dẫn theo địa chỉ, theo byte, và theo văn bản AT&T.`-O1`và mọi hướng dẫn ở `-O0`Khi nào?`llvm-objdump`được cài đặt ở đường đi bị gắn, một thử nghiệm thứ hai chạy nó trực tiếp và so sánh lại.

Đây rồi.`percent_change`Như lời tiên tri đã in:

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

## Giai đoạn 1: Bytes đến hướng dẫn

Bài học 33 xây dựng bộ giải mã. Bài học này giữ một bản sao nhỏ để mã chạy tự động. Bản sao bao gồm chính xác các mã opcode mà âm thanh phát ra cho.`ledger.c``-O0`và `-O1`- Đối với bất cứ thứ gì khác mà nó làm tăng .`DecodeError`với địa chỉ và byte opcode.

Bộ giải mã theo thứ tự byte của Intel SDM, Volume 2, phần 2.1. Các tiền đề đi trước, sau đó là opcode, sau đó là ModR/M, sau đó là SIB, sau đó là di chuyển, sau đó là ngay lập tức. Phần 2.2.1 thêm tiền đề REX, một byte từ `0x40`đến`0x4F`, phải nằm ngay trước opcode. REX.W chọn một operand 64-bit, REX.R mở rộng `reg`trường, REX.X mở rộng chỉ số SIB, và REX.B mở rộng `r/m`Phần 2.1.3 xác định ba trường ModR/M `mod`- `reg`, và`r/m`, và ba trường SIB `scale`- `index`, và`base`Trong chế độ 64 bit,`mod = 00`với `r/m = 101`nghĩa là một sự di chuyển 32 bit so với hướng dẫn tiếp theo (phần 2.2.1.6).

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

- `leaq (%rip), %rcx``grade+0x23`là 7 byte với một sự di chuyển bằng không.`R_X86_64_PC32.rodata+0x10`trong sự bù đắp`0x36`, trường di chuyển. người nâng giải quyết nó trong giai đoạn 3.

## Giai đoạn 2: Các khối và các nhà thống trị

Một khối cơ bản là một chuỗi các câu nói với một mục vào phía trên và một lối ra ở phía dưới. Đó là định nghĩa từ Allen (1970), và mỗi giấy giải biên dịch sử dụng lại nó. `build_blocks()`thu thập các nhà lãnh đạo: mục nhập hàm, mỗi mục tiêu nhảy bên trong hàm, và mỗi địa chỉ sau một nhảy hoặc trở lại. Mỗi khối nhận được người kế nhiệm từ hướng dẫn cuối cùng của nó.`jmp`có một.`ret`không có, và một khối kết thúc trong một hướng dẫn thông thường rơi qua đến người lãnh đạo tiếp theo.

`total``-O1`có năm khối:

| Block | Instructions | Successors |
|---|---|---|
| `0x40` | `testl %esi, %esi` then `jle 0x5c` | `0x5c`, `0x44` |
| `0x44` | `movl %esi, %ecx`, `xorl %edx, %edx`, `xorl %eax, %eax`, `nopw` | `0x50` |
| `0x50` | `addl (%rdi,%rdx,4), %eax`, `incq %rdx`, `cmpq %rdx, %rcx`, `jne 0x50` | `0x50`, `0x5b` |
| `0x5b` | `retq` | exit |
| `0x5c` | `xorl %eax, %eax` then `retq` | exit |

`immediate_dominators()`là thuật toán lặp lại của Cooper, Harvey, và Kennedy (2001). Nó đi bộ các khối theo thứ tự hậu ngược và cắt ngang các chủ trương của các tiền nhiệm được xử lý cho đến khi không có gì thay đổi.`total`Nó nói rằng`0x44`thống trị `0x50`Nó cũng nói rằng`0x50`tự thống trị qua cạnh sau, đó là cách giai đoạn 8 tìm thấy vòng lặp.

Tương tự chức năng chạy gấp đôi. `dominance_frontiers()`sử dụng nó cho SSA ở giai đoạn 5. `post_dominators()`chạy nó trên biểu đồ đảo ngược với một nút thoát ảo để tìm các liên kết trong giai đoạn 8.

## Giai đoạn 3: Tăng lên một IR nhỏ

IR có sáu loại tuyên bố: `Assign`- `Store`- `Branch`- `Jump`- `IndirectJump`, và`Return`. Các biểu thức là các định vị, biến, các hoạt động nhị phân với chiều rộng, các loại (`sext`- `zext`- `trunc`5 hình dạng khác tồn tại chỉ để sau đó được đơn giản hóa:`Flags`- `Cond`- `Concat`- `SignFill`, và`Symbolic`- Tôi không biết.

Các sổ đăng ký là các biến được đặt tên theo gia đình 64-bit của chúng.`%eax`là một `Assign`đến`rax`với chiều rộng 32, và một đọc của `%eax`là một bài đọc của `rax`Trong chế độ 64-bit, một 32 bit viết không mở rộng vào toàn bộ sổ đăng ký, do đó, một đọc 64 bit sau cùng giá trị là `zext`. Máy nâng không biết định nghĩa tiếp cận, vì vậy nó ghi lại chiều rộng đọc và cho phép SSA đưa giáp vào giai đoạn 5.

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

### Các cờ là một biến

Mỗi lệnh đặt mã điều kiện viết một biến có tên `flags`, và mỗi nhảy có điều kiện đọc nó.`cmp a, b`viết `cmp(a, b)`, một `test a, b`viết `test(a, b)`, và`sub`viết như vậy `cmp`hình dạng như một tác dụng phụ.`jcc`trở thành `Branch(Cond(cc, flags), target, fallthrough)`Mã điều kiện quyết định nhà khai thác so sánh khi hai gặp nhau trong giai đoạn 6:

| Code | After `cmp(a, b)` | After `test(a, a)` |
|---|---|---|
| `je` | `a == b` | `a == 0` |
| `jne` | `a != b` | `a != 0` |
| `jl`, `jge`, `jle`, `jg` | signed `<`, `>=`, `<=`, `>` | signed compare with 0 |
| `jb`, `jae`, `jbe`, `ja` | unsigned `<`, `>=`, `<=`, `>` | unsigned compare with 0 |

`xorl %eax, %eax`được nâng thẳng lên `eax = 0`Những lá cờ nó đặt được viết là`add`Những lá cờ kiểu mà không có nhảy trong thiết bị đọc.`xor`ngôn ngữ quan trọng trong `grade`, nơi Ghidra mất đi nó.

### CLtd và IDV

`cltd``%edx`với dấu hiệu bit của `%eax`- `idivl %edi`chia cặp 64 bit `%edx:%eax`bởi `%edi`, đưa số lượng vào `%eax`, và phần còn lại trong `%edx`Người nâng máy viết cặp vào một tên tạm thời`dividend`Đầu tiên, phần còn lại của câu nói sau đó đọc câu cũ`%eax`Trong SSA, nửa cao xuất hiện như`signfill`Các đơn giản hóa giảm cặp xuống nửa thấp, và chia trở thành 32 bit `/`- Tôi không biết.

### Các khe xếp ở -O0

Tại `-O0`Clang giữ mọi người trong khung hình.`percent_change`cửa hàng `%edi`đến`-0x8(%rbp)`và `%esi`đến`-0xc(%rbp)`, sau đó đọc lại cho mỗi lần sử dụng.`pushq %rbp`tiếp theo là `movq %rsp, %rbp`Sau đoạn đầu, một bộ nhớ có cơ sở.`%rbp`, không có chỉ số, và một sự di chuyển trở thành một biến có tên `local_<hex>`- `-0x8(%rbp)`là `local_8`- `-0xc(%rbp)`là `local_c`- Tôi không biết.

Các hướng dẫn khung tự tạo ra không IR, và cũng không làm `subq`và `addq``%rsp`Bất kỳ bộ nhớ nào khác vẫn là một bộ nhớ .`Load`hoặc `Store`- Tôi không biết.

### Việc di chuyển

- `leaq (%rip), %rcx`trong `grade`có chuyển động không trong hồ sơ.`0x36`nói `R_X86_64_PC32.rodata+0x10`- Người liên kết sẽ viết`S + A - P`vào sân. `S`là địa chỉ của `.rodata`- `A`là bổ sung `0x10`, và`P`là địa chỉ của trường. hướng dẫn sau đó tính toán `next_ip + disp = (P + 4) + S + A - P = S + A + 4`- Tôi không biết.

Người nâng áp dụng toán học tương tự mà không cần một liên kết. mục tiêu bù đắp là phụ gia cộng với khoảng cách từ trường đến cuối hướng dẫn, `0x10 + 4 = 0x14`- Bộ tư vấn nói`rcx = &.rodata+0x14`, và bàn ở đó chứa 5 người`int`giá trị `1, 2, 3, 4, 4`- Tôi không biết.

## Giai đoạn 4: Các tham số từ hội nghị mời

Hệ thống V AMD64 psABI, phần 3.2.3, vượt qua các lập luận lớp INTEGER trong `%rdi`- `%rsi`- `%rdx`- `%rcx`- `%r8`- `%r9`Kết quả sẽ trở lại trong`%rax`Và sau đó`%rdx`Phần 3.2.1 nói rằng `%rbp`- `%rbx`, và`%r12`qua `%r15`thuộc về người gọi và phải được bảo vệ.

`collect_params()`áp dụng một quy tắc: một sổ đăng ký lập luận được đọc trước khi bất kỳ viết nào trong hàm là một tham số.`%rsi`Và không bao giờ`%rdi`vẫn có hai tham số. `percent_change`đọc `%edi`và `%esi`, nên nó đã`param_1`và `param_2`- `grade`chỉ đọc thôi`%edi`- `total`đọc `%rdi`như một cơ sở 64-bit và `%esi`như là một con số 32 bit.

Giờ là Ghidra `param_3`Có một lời giải thích.`idivl`viết `%rdx`, và psABI cho phép một 128-bit INTEGER trở lại trong `%rax:%rdx`Trong một tệp đối tượng không có người gọi để nói giá trị trả lại rộng bao nhiêu. Ghidra giữ cả hai sổ đăng ký như giá trị trả lại.

Trên đường đi không,`xorl %eax, %eax`chỉ viết thôi `%eax`Ghidra đọc `%rdx`là một đầu vào của hàm, và đầu vào đó trở thành `param_3`Các chỉ dẫn đúng, chữ ký là một đoán về những phần mà chỉ dẫn để lại mở.

## Giai đoạn 5: SSA, Một tên cho mỗi định nghĩa

Sau khi nâng, `grade`viết `rax`4 lần và `rcx`ba lần. Cytron, Ferrante, Rosen, Wegman và Zadeck (1991) xác định hình thức giao dịch đơn tĩnh. Mỗi biến được gán chính xác một lần, và các hàm phi tại điểm kết nối chọn từ các định nghĩa đạt đến chúng. Việc xây dựng của chúng có hai bước. Phần 5.1 đặt phi hàm sử dụng biên giới thống trị, và phần 5.2 đổi tên mọi định nghĩa và sử dụng bằng cách đi bộ trên cây thống trị.`insert_phis()`và `rename()`Hãy theo hai phần đó.

Danh sách SSA của `total`cho thấy những gì xây dựng tạo ra:

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

Phiên bản 0 của một sổ đăng ký là giá trị của nó khi nhập. `esi_0`và `rdi_0`là các tham số.`0x50`có hai tiền nhiệm, khối thiết lập và chính nó, vì vậy mỗi biến có một định nghĩa trong vòng lặp nhận được phi ở đó.`flags_2`Phi đã chết khi đến.`flags`có định nghĩa trong cả hai phiên bản trước, và giai đoạn 6 loại bỏ nó vì không có gì đọc nó.

Bước đổi tên cũng sửa đổi chiều rộng. `edx_1 = 0`là một định nghĩa 32-bit và `rdx_3 = rdx_2 + 1`là 64 bit, vì vậy phi `rdx_2`- Đọc của `%rcx`trong `cmpq %rdx, %rcx`là 64 bit trong khi định nghĩa của nó `ecx_1 = esi_0`là 32 bit, do đó việc sử dụng trở thành `zext64(ecx_1)`Mỗi bộ phim trong mã giả cuối cùng đều được tạo ra bởi một quy tắc.

Phần "lời nhẹ" là cách thoát khỏi SSA. `coalesce_webs()`kết hợp mỗi phi với các lập luận của nó thành một biến, được gọi là web, và cho web một tên. Điều này hoạt động cho các hình dạng một trình biên dịch phát ra cho các chức năng này. Nó có thể phá vỡ trên các bản sao song song, nơi hai phis tại một kết hợp các giá trị trao đổi, và bài học không xử lý trường hợp đó. Phần 7 của Cytron mô tả lối ra chung thông qua màu sắc.

## Giai đoạn 6: Làm đơn giản, thay thế, loại bỏ mã chết

`optimize()`lặp lại ba lần cho đến khi không có gì thay đổi.

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

Việc thay thế chuyển một định nghĩa thành một cách sử dụng duy nhất. Quy tắc có ba điều kiện. Việc sử dụng là trong cùng một khối. Việc sử dụng là một tuyên bố chứ không phải là phi. Khi định nghĩa chứa tải, không có cửa hàng hoặc cuộc gọi nằm giữa định nghĩa và việc sử dụng. Một liên tục hoặc một địa chỉ phần được thay thế ở bất cứ nơi nào ngoại trừ phi, bởi vì một lập luận phi phải giữ một biến cho web.

Xóa mã chết xóa một nhiệm vụ hoặc phi mà không có mục đích và không có biểu hiện có cuộc gọi.`percent_change`phần còn lại`edx_2 = dividend_1 % edi_0`là chết, nên nó đi trước.`edx_1 = signfill(eax_1)`với một sử dụng, do đó nó được thay thế thành `dividend_1 = edx_1:eax_1`- Tôi không biết.

Quy tắc cặp sau đó giảm `signfill(eax_1):eax_1`đến`eax_1`, và sự chia rẽ trở thành `eax_1 / edi_0`Giờ thì`eax_1 = esi_1 * 100`có một sử dụng, và `esi_1 = esi_0 - edi_0`có một sử dụng, và cả hai chuyển sang tuyên bố trả lại.`flags`Các định nghĩa đã được thay thế vào các điều kiện nhánh và biến mất.

```c
return (param_2 - param_1) * 100 / param_1;
```

### Số nhân 0x66666667

`grade`chia điểm bằng 10, và clang phát không `idiv`Nó phát ra .`imulq $0x66666667, %rax, %rax`, sau đó `sarq $0x22, %rcx`- Vậy thì `shrq $0x3f, %rax`, sau đó thêm hai. Granlund và Montgomery, "Division by Invariant Integers using Multiplication" (PLDI 1994), mô tả phương pháp.`m`và một thay `s`. Đối với mỗi `x`trong phạm vi, `(x * m) >> s`phải bằng `x / d`- Tôi không biết.

Đây .`m = 0x66666667 = 1717986919`và `s = 34`- Số lượng toán học xác minh:`2^34 / 10 = 1717986918.4`, và trần nhà là `1717986919`Thầm là`m * 10 - 2^34 = 6`Để nhận dạng được giữ lại trên mỗi 32-bit`x`, thời gian lỗi `2^31`phải ở dưới `2^34`, vì vậy lỗi phải là tối đa 8. 6 là tối đa 8.`(x * m) >> 34`=`floor(x / 10)`cho mỗi không âm `int`- Tôi không biết.

Đối với âm`x`sàn tròn xa xíu trong khi C tròn hướng tới xíu. `shrq $0x3f`lấy bit dấu hiệu của sản phẩm, đó là 1 chính xác khi `x`là âm. Thêm nó sửa chữa vòng tròn.

`magic_divisor(m, s)`kiểm tra một cặp ứng cử viên. Nó tính toán`d`như số nguyên nhân của `2^s`bởi `m`và cố gắng`d`và `d + 1`Nó chấp nhận một`d`Chỉ khi`m`bằng với trần của `2^s / d`và lỗi liên kết giữ. cùng nhân với `s = 33`cho `d = 5`, điều đó là đúng, bởi vì `0x66666667`cũng là trần của `2^33 / 5`Một xét nghiệm kiểm tra danh tính với`c_div(x, 10)`trên 2.406 giá trị bao gồm cả cả hai đầu của `int`tầm bắn.

Với sự công nhận của tôi,`grade`đọc `ecx_1 = param_1 / 10 - 6;`Khi không nhận dạng, bản demo in dạng nguyên liệu:

```c
rax_1 = (long)param_1 * 0x66666667;
ecx_1 = (int)(rax_1 >> 34) + (int)((unsigned long)rax_1 >> 63) - 6;
```

Khi bạn đọc mã từ một trình biên dịch mà bạn chưa nghiên cứu, dạng nguyên liệu là dạng trung thực.

## Giai đoạn 7: Các loại từ sử dụng

Mã máy có chiều rộng và không có kiểu. `infer_types()`chuyển 3 loại bằng chứng thành loại C. Một mạng có định nghĩa rộng 32 bit là`int`, và một mạng 64 bit là `long`Một mạng lưới là cơ sở của một access element`base[idx]`với các yếu tố 4 byte là `int *`. kiểu chỉ dẫn lan rộng dọc theo các bản sao và phis, vì vậy `local_8 = param_1``-O0`làm cho cả hai `int *`. Phép trả lại là chiều rộng của giá trị trong `return`Các tuyên bố.

Kết quả của `total`là `int total(int *param_1, int param_2)`Ghidra đã viết`int total(long param_1,uint param_2)`Cả hai đều phù hợp với hướng dẫn. người nâng sử dụng tải trọng quy mô như là bằng chứng chỉ. Ghidra giữ chiều rộng sổ đăng ký và viết truy cập như`*(int *)(param_1 + uVar2 * 4)`- Không thấy cả .`const`, và không ai biết được`int *`từ `unsigned *`- Tôi không biết.

Lee, Avgerinos và Brumley (TIE, NDSS 2011) lấy lại kiểu khung như giải quyết hạn chế trên một lưới. Kết quả là ranh giới trên và dưới thay vì một đoán. Noonan, Loginov và Cok (Retypd, PLDI 2016) thêm polymorphism và tái tạo.`const`Bài học này sử dụng hai quy tắc rẻ nhất từ dòng công việc đó và đặt tên những gì họ không thể thấy.

Số học không ký được lưu giữ trong các biểu thức chứ không phải trong các loại biến. `ja`sau đó`cmpl $0x4, %ecx`Nhận dạng như `(unsigned int)ecx_1 > 4`, và`shrq`Nhận dạng như `(unsigned long)rax_1 >> 63`Các biến vẫn được ký để người đọc thấy giá trị đăng ký và nơi duy nhất mà hướng dẫn xử lý nó như không ký.

## Giai đoạn 8: Xây dựng theo biểu đồ

Cifuentes (1994), chương 6, phù hợp với các vùng của biểu đồ đối với các hình dạng của một ngôn ngữ cấp cao. Các vòng lặp đi đầu (6.6.1), sau đó là các điều kiện hai chiều (6.6.2), sau đó là các điều kiện n chiều (6.6.3). Yakdan, Eschweiler, Gerhards-Padilla, và Smith (NDSS 2015) đi theo con đường khác. Chúng bắt nguồn từ điều kiện mà mỗi nút thực hiện và xây dựng cấu trúc từ những điều kiện đó, không có hình dạng cố định. Động lực của họ là một thước đo. Hex-Rays trên một mẫu được sản xuất 1,571 `goto`các câu trong 49.514 dòng, một trong 32 dòng, và cấu trúc của họ không tạo ra bất kỳ trong 1.530 chức năng cốt lõi.

`structure()`là một thành viên nhỏ của gia đình đầu tiên. Nó cần hai sự thật từ giai đoạn 2.`Branch`là kết hợp của điều kiện. Một cạnh sau là một cạnh của một khối thống trị nguồn của nó. vòng tự nhiên là tập hợp các khối đạt đến nguồn cạnh sau mà không vượt qua tiêu đề.

Để một `Branch`với join `j`- Có thể là:

- Nếu`j`là mục tiêu nhảy, phát `if (!cond) { fallthrough region }`và tiếp tục tại `j`- Tôi không biết.
- Nếu`j`là sự vượt qua, phát ra `if (cond) { target region }`và tiếp tục tại `j`- Tôi không biết.
- Nếu`j`là hàm thoát, cả hai cánh tay trở lại.`if (cond) {... }`Và cánh tay kia sau nó, mà không có một `else`Đây là sự trở lại sớm của`percent_change`và `total`- Tôi không biết.
- Nếu không phát `if (cond) { target } else { fallthrough }`và tiếp tục tại `j`Đây là`percent_change``-O0`, nơi cả hai vũ khí lưu trữ `local_4`và gặp nhau tại buổi diễn văn.

Đối với một loop header:

- Nếu tiêu đề kết thúc trong một `Branch`với một người kế nhiệm bên ngoài vòng lặp, phát `while (cond) { body }`- `total``-O0`có hình dạng như sau: tiêu đề so sánh `local_14`với `local_c`và nó còn tiếp tục `jge`- Tôi không biết.
- Nếu tiêu đề cũng là khối với cạnh phía sau, hoặc ổ khóa ra thay vào đó, phát `do { body } while (cond);`- `total``-O1`có hình dạng như: khối`0x50`thêm, tăng, so sánh, và nhảy lên chính nó.

Một `IndirectJump`kết thúc một khối mà không có người kế nhiệm. `grade``-O0`là một bàn nhảy thực sự,`jmpq *%rax`thông qua 5 khoản bù đắp tương đối tại `.rodata+0x0`Và bài học 33 giải quyết những mục tiêu đó.`0x76`- `0x7f`- `0x88`, và`0x91`được đạt được chỉ thông qua nhảy gián tiếp và không được nâng.`switch`- Tôi không biết.

### -O1 đã làm gì với bộ chuyển đổi

Tại `-O1`Clang không giữ bàn nhảy cho `grade`Nó tính toán.`score / 10 - 6`và so sánh kết quả với 4 như một số không ký kết.`int``.rodata+0x14`. Đơn so sánh không ký là một hướng dẫn bao gồm cả hai .`score / 10 < 6`và `score / 10 > 10`Một sự khác biệt âm là một giá trị lớn không được ký kết, vì vậy nó không kiểm tra tương tự.

Bàn có thể giữ được.`1, 2, 3, 4, 4`cho các nhân tố 6 đến 10.`switch`với năm `return`Các tuyên bố đã trở thành một `if`Và một mảng đọc. Người nâng phát ra chính xác điều đó, và đặt tên cho bàn.`lookup_14`sau khi phần của nó bị bù đắp.

Độ dài của bảng là một heuristic. `table_values()`đọc từ bảng bắt đầu đến tiếp theo `.rodata`bồi thường bất kỳ `.text`Các tham chiếu chuyển chỗ, hoặc đến cuối phần. Trong đối tượng này, các tham chiếu sau là `0x28`, bảng của `grade_label`, vậy `lookup_14`- Có `(0x28 - 0x14) / 4 = 5`Ghidra cũng không biết chiều dài.`*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)`, một địa chỉ byte cộng với một ofset, và để lại năm giá trị cho người đọc để tìm kiếm.

## Cùng với Ghidra

Các bản demo in mỗi cặp.`code/fixtures/ghidra/*.json`, các hồ sơ bằng chứng rằng rea trở lại vào ngày 2026-10-14.`npx -y rea-agents@6.5.0 decompile ledger-x86_64-O1.o <function> --provider ghidra --format json`Mỗi bản ghi đều có`confidence: observed`và `authority: shipped-artifact`Các tuyên bố bên trong văn bản được bắt nguồn, và các hồ sơ không nói về những gì.

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

Phiên bản sửa chữa trong `code/fixtures/ghidra/ledger-x86_64-O1.ghidra-repaired.c`chỉ chỉnh sửa hàm này, và thử nghiệm so sánh nó với nguồn trên 81 cặp đầu vào.

### cấp

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

### tổng số

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

Thử nghiệm hành vi biên soạn văn bản Ghidra cho `grade`và `total`Không thay đổi. Nó thêm hai `typedef`Các dòng và một định nghĩa của `DAT_001000b4`từ 20 byte ở `.rodata+0x14`Cả hai hàm đều phù hợp với nguồn trên lưới.

## Những gì nghiên cứu nói về việc đọc mã giả

LLM4Decompile (EMNLP 2024) đo Ghidra một mình với khả năng thực hiện lại 20,12 phần trăm trên HumanEval-Decompile. Khoảng một trong năm chức năng đã biên soạn lại và vượt qua các tuyên bố đi kèm với chỉ số chuẩn. Việc tinh chỉnh sản lượng của Ghidra với mô hình 6.7B của họ đã nâng cao đến 52,74%. DecompileBench (Findings of ACL 2025) đã chạy sáu máy phân tích công nghiệp và sáu cách tiếp cận LLM trên 23,400 chức năng từ 130 chương trình. Các kết quả của LLM đọc tốt hơn và có độ chính xác chức năng thấp hơn 52,2%.

Liu, Raff và Micinski (arXiv 2609.05370, tháng 9 năm 2026) giải thích khoảng cách. Họ sử dụng 300 chức năng thư viện GitHub và 287 chức năng dựa trên CVE. Mô hình tinh chế mạnh nhất đã nâng tỷ lệ xây dựng của Ghidra từ 75 đến 90 phần trăm trong khi tỷ lệ phù hợp hành vi giảm từ 74 đến 62 phần trăm. Các ứng viên đã vượt qua mọi bài kiểm tra trong chỉ số chuẩn vẫn khác nhau về các đầu vào bị mờ. Điều đó xảy ra trong 4,9% các trường hợp trên 8 hệ thống, và lên đến 13% cho một hệ thống. Có tới một phần mười các lỗ hổng được tiết lộ đã biến mất từ mã tinh tế.

Nguyên nhân mà họ tìm ra là đơn giản: các trường hợp, kiểu, tên và người bảo vệ đã thay thế những thứ không rõ ràng mà người phân tích đã để lại trong văn bản.

Đó là lý do mà máy nâng có thể ghi lại.`(unsigned long)(unsigned int)ecx_1`Nó in một ghi chú thay vì một cái tên đẹp hơn.`switch`cho nhảy chưa được giải quyết, và `lookup_14`thay vì một nhãn mô hình sẽ phát minh ra.`code/main.py`- Ghidra.`undefined1 [16]`Một mô hình tinh tế là sạch sẽ.`int percent_change(int, int)`cùng mô hình sẽ tạo ra một chữ ký sai lầm rõ ràng trên hàm tiếp theo.

## Hãy xây dựng nó

Chạy demo:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
```

Dòng demo giải mã tất cả bốn chức năng của `-O1`đối tượng và so sánh mọi hướng dẫn với những gì đã thực hiện `llvm-objdump`- Thư.`percent_change`- `grade`, và`total`nó in IR trước SSA, danh sách SSA với các hàm phi, và mã giả bên cạnh văn bản Ghidra.`grade`với việc nhận dạng phân chia bị tắt,`-O0`phiên bản của `percent_change`- `total`, và`grade_label`, và một tóm tắt:

```json
{
  "functions": ["grade", "grade_label", "percent_change", "total"],
  "instructions": 56,
  "blocks": {"percent_change": 3, "grade": 3, "total": 5, "grade_label": 4},
  "phis": {"percent_change": 0, "grade": 1, "total": 2, "grade_label": 1},
  "objdumpDifferences": 0
}
```

Đọc `code/main.py`theo thứ tự này:

1. `read_object()`đọc tiêu đề phần, bảng biểu tượng, và `.rela`Các phần của một vật thể ELF64 có thể di chuyển.
2. `decode_one()`và `render_att()`- Đánh mã một hướng dẫn và in nó theo cách này.`llvm-objdump`- Không.
3. `build_blocks()`- `immediate_dominators()`, và`dominance_frontiers()`xây dựng các dữ liệu biểu đồ.
4. `lift_insn()`chuyển một hướng dẫn thành các tuyên bố IR và giải quyết việc di chuyển.
5. `insert_phis()`và `rename()`xây dựng SSA và chèn các loại hình rộng.
6. `make_simplifier()`- `substitute_once()`, và`remove_dead_code()`chạy vào trong`optimize()`- Tôi không biết.
7. `collect_params()`- `coalesce_webs()`, và`infer_types()`tên và nhập các mạng lưới.
8. `structure()`với `post_dominators()`và `natural_loops()`phát ra cơ thể C.
9. `lift_function()`kết nối các giai đoạn với nhau và `lift_all()`nâng một vật thể toàn bộ với chữ ký của Callee.

## Sử dụng nó

Để nâng một chức năng từ đối tượng của riêng bạn, xây dựng nó với `clang -target x86_64-unknown-linux-gnu -c`và gọi`lift_all(read_object(path))`Khi máy giải mã tăng lên`DecodeError`, đọc byte opcode từ tin nhắn và tìm nó trong bản đồ opcode SDM. Sau đó thêm một mục vào các bảng ở phía trên cùng của tệp. Khi nâng lên trên một mnemonic, thêm một quy tắc cho `lift_insn()`Khi nhà xây dựng phát ra một`goto`, in biểu đồ với `IrPrinter`và quyết định quy tắc hình dạng nào thiếu. Mỗi cộng nhỏ và mỗi một được kiểm tra với`llvm-objdump`hoặc clang bởi các thử nghiệm hiện tại.

Khi Ghidra được cài đặt, chạy rea trên cùng một hàm và đặt hai văn bản cạnh nhau. Sẵn đi vào bảng khác biệt trước khi bạn tin vào bất kỳ văn bản nào. Một khác biệt với một hướng dẫn đằng sau nó là một lựa chọn trình bày. Một khác biệt mà không có hướng dẫn đằng sau nó là một đoán, và đoán có thể là ở cả hai phía.

## Phòng thí nghiệm tương tác

### Phòng thí nghiệm A: một chức năng thông qua năm giai đoạn

Từ `code`thư mục, mở Python:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode/code
python3 -q
```

Đi chạy:

```python
from main import *
obj = read_object(FIXTURES / "ledger-x86_64-O1.o")
lifted = lift_function(obj, "percent_change")
for insn in lifted.decoded: print(f"0x{insn.address:x}: {insn.raw.hex(' ')}  {insn.att()}")
print(lifted.ir_text)
print(lifted.ssa_text)
print(lifted.pseudocode)
```

Theo dõi`%esi`từ `subl %edi, %esi`đến`esi_1 = esi_0 - edi_0`đến`(param_2 - param_1)`- Byte`fe`là ModR/M của `subl`, với `mod = 11`- `reg = edi`, và`r/m = esi`- Tôi không biết.

### Phòng thí nghiệm B: ngôn ngữ phân chia

```python
print(lift_function(obj, "grade", recognize_division=False).pseudocode)
print(magic_divisor(0x66666667, 34), magic_divisor(0x66666667, 33), magic_divisor(0x66666667, 31))
```

Dòng đầu tiên in số nhân nguyên liệu và thay đổi.`10 5 None`Tìm hiểu tại sao cùng một nhân nhân chia bằng 5 với một chuyển giảm, và tại sao chuyển 31 không đạt được giới hạn lỗi.

### Phòng thí nghiệm C: một đối tượng mới dừng máy giải mã

Xây dựng thiết bị tại `-O2`và nâng nó lên:

```bash
clang -target x86_64-unknown-linux-gnu -O2 -fno-asynchronous-unwind-tables -c fixtures/ledger.c -o "$TMPDIR/ledger-O2.o"
python3 -c 'from main import *; from pathlib import Path; import os; lift_all(read_object(Path(os.environ["TMPDIR"]) / "ledger-O2.o"))'
```

Máy giải mã dừng ở `0x56`với `opcode 81 is outside the fixture subset`- Opcode `81`là biểu mẫu nhóm 1 với 32 bit ngay lập tức.`andl $0x7ffffff8, %edx`, và clang sử dụng nó trong vòng tròn vectorized của `total`Thêm vào bên cạnh.`0x83`trong `decode_one()`, lặp lại, và ghi lại mã opcode nào dừng bạn tiếp theo.`llvm-objdump`- Tôi không biết.

### Phòng thí nghiệm D: Giá trị mặc định của Ghidra trong lớp

```python
print(ghidra_pseudocode("grade"))
for score in (-25, 0, 59, 60, 100, 115):
    sign = -(score >> 31)
    product_sign = ((score * 0x66666667) & ((1 << 64) - 1)) >> 63
    print(score, sign ^ product_sign)
```

Mỗi hàng đều có 0, Ghidra `uVar1`biểu hiện là đúng và không thể đọc được, và hướng dẫn `xorl %eax, %eax`nói trong hai byte.

## Phòng thí nghiệm thực hành

Thêm `switch`thu hồi cho `grade``-O0`- Tôi không biết.

Yêu cầu:

- Hãy đọc 5 vụ di chuyển trong `.rela.rodata`của `ledger-x86_64-O0.o`, mỗi `R_X86_64_PC32.text+offset`, và tính toán năm mục tiêu trường hợp từ bảng tại `.rodata+0x0`- Tôi không biết.
- Thêm một `Switch`kết thúc đến IR với biểu thức chỉ số, danh sách mục tiêu và mặc định từ `ja`trước khi tải bàn.
- Cho khối chuyển đổi một người kế nhiệm cho mỗi mục tiêu để các kẻ thống trị, kẻ hậu thống trị và SSA tiếp tục hoạt động.
- Thả `switch (local_10) { case 0:... }`với một người`case`cho mỗi mục tiêu và các bài viết chung sau khi kết hợp.
- Hãy ghi lại cho bất kỳ nhảy gián tiếp nào mà bạn không thể buộc bàn.
- Sắp xếp kết quả bằng clang và so sánh nó với nguồn trên lưới điểm trong các bài kiểm tra.

Thành công là nghĩa của`grade``-O0`và `grade``-O1`tạo ra cùng một đầu ra trên lưới, với một `switch`Và không`goto`- Tôi không biết.

## Các tác phẩm

Bài học này sẽ mang lại kết quả `outputs/skill-lifting-to-pseudocode.md`Sử dụng nó để nâng một hàm với một dẫn xuất cho mỗi dòng và để lấp đầy bảng khác biệt với văn bản của một decompiler.

## Hãy kiểm tra

Chạy demo và bộ xác định:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Phòng có 25 thử nghiệm.

- decoder đồng ý với các cam kết `llvm-objdump`văn bản trên mỗi hướng dẫn của cả hai đối tượng, và với một trực tiếp `llvm-objdump`khi nó được cài đặt
- REX, ModR/M, SIB, và sự bù đắp trường di chuyển của `leaq (%rip), %rcx`được giải mã theo cách mô tả trong SDM
- một opcode bên ngoài bộ phụ tăng `DecodeError`thay vì đoán
- trình đọc ELF liệt kê bốn chức năng và ba chức năng `.text`di chuyển
- `total`có 5 khối,`0x44`thống trị `0x50`, và`0x50`là một đầu vòng với chính nó như cơ thể
- `%rdi`và `%rsi`trở thành`param_1`và `param_2`, và`param_1`trong `total`là `int *`
- `test`và `cmp`với `je`- `jle`- `ja`, và`jge`trở thành`==`- `<=`, không ký `>`, và`<`
- SSA đặt hai chức năng phi trực tiếp vào đầu vòng của `total`và không có ai trong `percent_change`
- mã giả của `percent_change`- `grade`, và`total`bằng với văn bản dự kiến
- `grade`đọc `param_1 / 10 - 6`với công nhận và sự nhân nguyên liệu với công nhận off
- `0x66666667`với shift 34 chia bằng 10, với shift 33 chia bằng 5, và danh tính giữ trên lưới
- `-O0`trở thành`local_4`- `local_8`- `local_c`, và chỉ dẫn khung không bao giờ xuất hiện
- bàn ở `.rodata+0x14`đọc `1, 2, 3, 4, 4`thông qua việc di chuyển
- nhảy không gián tiếp vào `grade``-O0`được báo cáo với bốn khối không được nâng
- Các thiết bị Ghidra chứa 16 byte trở lại,`param_3`, không đơn giản hóa`xor`, và `long param_1`
- 6 đơn vị phát hành được biên soạn với `clang -std=c11 -Wall -Werror`cho `x86_64-unknown-linux-gnu`
- các hàm được nâng, văn bản Ghidra được sửa chữa và nguồn in cùng 232 dòng trên lưới đầu vào

## Các chế độ sản xuất thất bại

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

## Quy tắc hoạt động

Hãy coi pseudocode là một dẫn xuất mà bạn chưa thấy. Hãy tái tạo nó từ các hướng dẫn bằng một máy nâng mà bạn có thể đọc. Hãy đặt hai văn bản bên cạnh nhau, và viết hướng dẫn hoặc đoán sau mỗi dòng khác nhau.

## Đọc thêm

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
- rea 6.5.0 nguồn tại tag `3b79af2`bài học này mô tả: `src/domain/functionComparisonDimensions.ts`- `website/evidence/dx-ball-sound-pan.md`- Tôi không biết.

# Binary Diffing và thực hiện biểu tượng: Tích hợp các chức năng và giải quyết các đầu vào

> Một sự phù hợp giữa hai hàm là một tuyên bố được thực hiện bởi một chữ ký với một chất lượng, và một đầu vào được giải quyết là một mô hình của một điều kiện đường dẫn; bài học này xây dựng cả hai và kiểm tra chúng so với radiff2, Ghidra, angr và z3.

**Type:** Build
**Languages:** Python, C, x86-64 assembly, Java (one Ghidra script)
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), 34 (lifting to pseudocode), 36 (rebuild and prove equal)
**Time:** ~150 minutes

## Mục tiêu học tập

- Xây dựng năm chữ ký chức năng từ một đối tượng và danh sách của nó: byte chính xác, byte với các trường di chuyển được che giấu, chuỗi mnemonic, hình dạng đồ thị dòng chảy và tên biểu tượng.
- Hãy chạy một trận đấu tham lam được dàn dựng theo phong cách BinDiff, nơi một chữ ký chỉ phù hợp khi nó độc đáo ở cả hai bên, và ghi lại giai đoạn và chất lượng đằng sau mỗi trận đấu.
- Tái tạo trường hợp khi hai hàm bằng byte giống nhau không có trình tương thích byte trên hình ảnh tải và vượt qua nó sau khi che giấu, với toán học di chuyển giải thích bốn byte thay đổi.
- Tái tạo lại tỷ lệ tương đồng của radiff2 từ khoảng cách Levenshtein và giải thích tại sao một tỷ lệ chỉ xuất hiện khi so sánh sử dụng 82 byte radar2 đạt được.
- Đếm số các con đường của một hàm trên một IR nhỏ, mang theo một điều kiện cho mỗi con đường, và giải quyết mỗi lá cho đầu vào với một tìm kiếm giới hạn.
- Đo sự bùng nổ đường trên một vòng lặp với ranh giới biểu tượng và so sánh các đếm, lớp và mô hình với angr 10.0.3 và z3 5.1.0.

## Vấn đề

Anh xây dựng lại`ledger.c``-O2`và hỏi một đại lý liệu có gì thay đổi kể từ khi `-O1`Cảnh sát đã chạy một cuộc kiểm tra và báo cáo rằng`grade`đã được sửa đổi, bởi vì các byte của nó khác nhau trong hai chương trình tải. sau đó nó đã chi tiêu ngân sách của mình giải thích một thay đổi mà không tồn tại.`grade`là giống nhau trong hai tệp đối tượng. Các hình ảnh so sánh sau khi di chuyển khác nhau, và `.rodata`nằm 0x70 byte hơn trong `-O2`Chương trình, vì vậy một sự di chuyển trong `grade`tăng lên 0x70.

Cùng buổi chiều đó anh hỏi điểm nào là gì`grade`trả lời 4. đại lý đọc mã giả và trả lời "90". Đó là một câu trả lời. hàm trả lại 4 cho mỗi điểm từ 90 đến 109, và câu hỏi không có ràng buộc, vì vậy câu trả lời của đại lý là một mẫu nơi bạn cần một bộ.

Cả hai thất bại đều xuất phát từ khoảng cách tương tự. Một sự phù hợp và một đầu vào được giải quyết là mỗi yêu cầu với một phương pháp đằng sau nó. Bài học này xây dựng hai phương pháp trong thư viện Python tiêu chuẩn và chạy các công cụ sản xuất bên cạnh chúng.

```figure
re-44-matches-and-paths
```

## Một chữ ký chức năng là một tuyên bố có chất lượng

Flake khung phân biệt nhị phân vào năm 2004 như là xây dựng một sự đồng dạng giữa các tập hợp chức năng của hai trình thực tương tự. Dullien và Rolles đã mở rộng nó vào năm 2005 cho các khối và hướng dẫn cơ bản. Trang khái niệm BinDiff mô tả chiến lược xuất phát từ công việc đó. Xét một thuộc tính cho mỗi hàm ở cả hai bên. Một giá trị xuất hiện chính xác một lần trên mỗi bên là phù hợp. Một giá trị xuất hiện nhiều hơn một lần là không rõ ràng và được đào xuống với thuộc tính tiếp theo trong lớp tương đương đó. Một giá trị mà không có đối tác vẫn không có đối thủ. Sau mỗi lần vượt qua, các người gọi và gọi của các trận đấu mới tạo thành các tập hợp nhỏ nơi thuộc tính tiếp theo có thể là độc đáo, và quá trình lặp lại với thuộc tính tiếp theo.

Mỗi thuật toán phù hợp mang theo một chất lượng. Các trang đánh giá sự phù hợp hash của các byte hàm nguyên thô và sự phù hợp hash tên là rất tốt, chỉ số MD và chữ ký chính (mnemonic mỗi một số chính nhỏ, sản phẩm trên hàm) là tốt, tham chiếu chuỗi là trung bình, và số lượng vòng hoặc chuỗi địa chỉ là kém. Sự tự tin của một trận đấu là chất lượng trung bình của các thuật toán đã tạo ra nó, được cân bằng bằng một sigmoid. Sự tương đồng của chức năng là khoảng 25% cạnh biểu đồ dòng chảy phù hợp, 15% khối cơ bản phù hợp, 10% hướng dẫn phù hợp, và 50% sự khác biệt trong chỉ số MD biểu đồ dòng chảy, nhân bằng độ tin cậy.

Bài học này giữ được năm thuộc tính và một quy tắc quan trọng: độc đáo ở cả hai bên hoặc không phù hợp. Nó loại bỏ chỉ số MD, bởi vì các trang khái niệm đặt tên cho nó mà không có công thức, và sử dụng một chuỗi độ đơn giản thay thế. Các nhãn chất lượng vẫn còn, bởi vì một sự phù hợp chỉ tốt như thực tế yếu nhất đã tạo ra nó.

## Năm chữ ký từ một bản ghi chức năng

Diff đọc hai thứ cho mỗi đối tượng. ELF cho các byte, kích thước biểu tượng, và `.rela.*`Các hồ sơ.`llvm-objdump -d`danh sách cho giới hạn hướng dẫn và mnemonics, và bài học 33 chứng minh là một decoder bằng với danh sách trên mỗi hướng dẫn của các đối tượng này. `function_record()`kiểm tra rằng các byte của mỗi lệnh được liệt kê bằng các byte của đối tượng tại địa chỉ đó và tăng khác.

Nhận đi`grade`trong `ledger-x86_64-O1.o`. Biểu tượng nói bù đắp 0x10 và kích thước 46, do đó, hồ sơ bao gồm 0x10 đến 0x3e và loại trừ hai-byte `nop`Điều này sắp xếp các hàm tiếp theo.

Các chữ ký của các byte chính xác là SHA-256 của 46 byte. chữ ký của các byte che giấu đánh giá bằng không trước tiên. Có một trường: ghi`R_X86_64_PC32 .rodata+0x10`ở trục trục 0x36, bốn byte rộng, sự di chuyển của `leaq (%rip), %rcx`Trong một đối tượng RELA, bốn byte đó đã giữ 0 rồi, vì vậy hai chữ ký đồng ý về mọi hàm của mọi đối tượng ở đây.

Các chữ ký mnemonic hashs chuỗi của 14 mnemonics:

```text
movslq imulq movq sarq shrq addl addl xorl cmpl ja movl leaq movl retq
```

Các chữ ký hình dạng đồ thị dòng chảy đến từ các khối cơ bản và các cạnh đánh dấu của bài học 33, giới hạn ở các khối mà mục có thể đạt được. `grade``-O1`có ba khối và ba cạnh. khối 0x10 kết thúc ở`ja 0x3d`Với một cạnh được lấy và một cạnh rơi, khối 0x31 rơi qua đến 0x3d, và 0x3d trả lại.`3b 3e [0/2,1/1,2/0]`- Vì `grade``-O0`hồ sơ giải quyết sự nhảy gián tiếp qua năm `.rela.rodata`các mục chính xác như bài học 33 đã làm, cho tám khối và mười một cạnh với bốn `table`Các cạnh.

Tên chữ ký là tên biểu tượng, nó là sự thật mạnh nhất khi nó tồn tại và không có gì khi xây dựng được tước đi.

Một trường hơn đi cùng cho các trường hợp vá sau đó.`lea`di chuyển vào `.rodata`, kiểm tra giới hạn, và một tải indexed,`referenced_table()`đọc bảng theo cách bài học 36 đã làm: `cmpl $0x4`trước đây`ja`đưa ra năm mục và tải `movl (%rcx,%rax,4), %eax`cho mỗi 4 byte.`grade``-O1`bàn ở `.rodata+0x14`đọc `1, 2, 3, 4, 4`- Tôi không biết.

## Người tham lam

`match_functions()`cho mỗi giai đoạn nó nhóm các chức năng vẫn chưa được sánh ngang ở mỗi bên bằng chữ ký của giai đoạn đó và chấp nhận mọi giá trị có chính xác một chức năng ở mỗi bên.

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

Trình tự bài học chạy bốn giai đoạn cấu trúc trước và tên cuối cùng, vì vậy các giai đoạn cấu trúc được kiểm tra và tên đánh giá chúng sau đó.`-O1`chống lại`-O2`- Có thể là:

| Function | Stage | Agreeing signatures | Similarity |
|---|---|---|---|
| `percent_change` | exact bytes | all five | 1.000 |
| `grade` | exact bytes | all five | 1.000 |
| `total` | name | name only | 0.171 |
| `grade_label` | exact bytes | all five | 1.000 |

`total`Clang đã tăng từ 31 byte và 13 hướng dẫn lên 140 byte và 41 hướng dẫn, bởi vì clang đã vectorized vòng lặp với SSE2 và giữ một vòng lặp còn lại scalar. hình dạng của nó đã đi từ năm khối và năm cạnh lên mười khối và mười ba cạnh. Không có gì cấu trúc tồn tại, và chỉ có biểu tượng liên kết hai.

Vì `-O0`chống lại`-O1`cùng một thứ tự tạo ra một sự phù hợp sai:

| Function at -O0 | Matched at -O1 | Stage | Names agree |
|---|---|---|---|
| `percent_change` | `grade_label` | flow-graph shape | no |
| `grade` | `grade` | name | yes |
| `total` | `total` | name | yes |
| `grade_label` | unmatched | | |

`percent_change``-O0`là một viên kim cương bốn khối: một mục có thể phân nhánh trên `before == 0`, hai cánh tay, và một liên kết mà trở lại.`0/2, 1/1, 1/1, 2/0`- `grade_label``-O1`- Có `grade`hình dạng là duy nhất ở cả hai bên và quy tắc chấp nhận nó. nhãn chất lượng trung bình là cảnh báo, và giai đoạn tên, chạy sau đó, mâu thuẫn với nó.`grade_label`bên phải, vậy là thực `percent_change`và `grade_label`Những cặp vẫn không thể sánh được.

Trật tự BinDiff chạy tên phù hợp như giai đoạn thứ hai khi tên thực tồn tại, và `BINDIFF_ORDER`Trong mã cũng như vậy.`-O0`chức năng phù hợp với họ `-O1`Các phiên bản theo tên, đó là kết quả Ghidra báo cáo bên dưới. Bài học giữ cả hai lệnh trong demo để thất bại là hiển thị.`if`và `else`- Thậm chí là trước khi trở lại.

## Các bit giống nhau, chuyển đổi và hoạt động

Ghidra áp dụng các chuyển nhượng khi nhập khẩu một đối tượng.`load_image()`Nó đi bộ các phần được phân bổ theo thứ tự tiêu đề, phù hợp với mỗi phần của nó `sh_addralign`, và đặt chúng từ 0x100000.`S + A - P`được viết dưới dạng giá trị 32 bit ký kết cho `R_X86_64_PC32`và `R_X86_64_PLT32`- `S + A`như 64 bit cho `R_X86_64_64`- Tôi không biết.

Layout phù hợp với Ghidra 12.1.4 trên mỗi phần khởi động trong cả hai chương trình, mà bộ thử nghiệm kiểm tra với đầu ra theo dõi phiên bản được chụp:

| Section | `-O1` program | `-O2` program |
|---|---|---|
| `.text` | 0x100000, 159 bytes | 0x100000, 271 bytes |
| `.rodata` | 0x1000a0 | 0x100110 |
| `.rodata.str1.1` | 0x1000f0 | 0x100160 |

`.text`là 112 byte dài hơn ở `-O2`, sự tăng trưởng của `total`, vậy `.rodata`bắt đầu 0x70 byte hơn nữa. `grade`ở 0x100010 trong cả hai chương trình.`leaq`tại ofset 0x33 có trường di chuyển của nó ở 0x36 và kết thúc ở 0x3a.`.rodata+0x14`- Tôi không biết.

Trong `-O1`hình ảnh: `0x1000a0 + 0x14 - 0x10003a = 0x7a = 122`Trong `-O2`hình ảnh: `0x100110 + 0x14 - 0x10003a = 0xea = 234`Các byte trường là `7a 00 00 00`trong một hình ảnh và `ea 00 00 00`Một byte của 46 khác nhau, tỷ lệ Levenshtein là 0,978, và giai đoạn-byte chính xác thất bại. giai đoạn-byte che giấu đánh bại trường và đi qua. `grade_label`chuyển động bằng cùng 0x70 như bảng của nó, do đó sự di chuyển của nó vẫn giữ nguyên và nó chuyển qua các byte chính xác trong cả hai quan điểm.

Ghidra báo cáo về sự chia rẽ tương tự.`-O1`chống lại`-O2`, Exact Function Bytes Match được tìm thấy `percent_change`và `grade_label`, Đáp ứng hướng dẫn chức năng chính xác tìm thấy hai cộng với`grade`, và Exact Function Mnemonics Match tìm thấy cùng ba. Bộ tương ứng hướng dẫn che giấu các hoạt động, bao gồm sự di chuyển.

6,5.0 rea `compare_functions`Các dòng dòng difs cung cấp văn bản. trên chiều bộ sưu tập nó gắn hạn chế "định lệ hóa di chuyển không có sẵn".

## Những biện pháp nào

radar2 6.2.4 tàu `radiff2`- Với `-AC`Nó phân tích cả hai tập tin và in một hàng cho mỗi chức năng: kích thước bên trái và địa chỉ, MATCH hoặc UNMATCH, tỷ lệ, và kích thước và địa chỉ phải. Các hàng được chụp cho các đối tượng của chúng tôi:

| Pair | `percent_change` | `grade` | `total` | `grade_label` |
|---|---|---|---|---|
| `-O0` vs `-O1` | UNMATCH 0.130435 | UNMATCH 0.426829 | UNMATCH 0.106061 | UNMATCH 0.269841 |
| `-O1` vs `-O2` | MATCH 1.000000 | MATCH 1.000000 | UNMATCH 0.171429 | MATCH 1.000000 |
| `-O1` vs `-O1` | MATCH 1.000000 | MATCH 1.000000 | MATCH 1.000000 | MATCH 1.000000 |

Tỷ lệ trong nguồn radar2 là `1 - distance / max(len_a, len_b)`, nơi khoảng cách là khoảng cách Levenshtein giữa hai dấu vân tay hàm và một dấu vân tay là kết nối các khối cơ bản của hàm theo thứ tự địa chỉ.`radare2_ratio()`trong `code/main.py`thực hiện công thức tương tự trên các byte hàm nguyên liệu và tái tạo 0.130435, 0.106061, 0.269841, và 0.171429 đến sáu thập phân.

Nó không tái tạo 0.426829 cho `grade``-O0`. Trên 118 byte của hàm tỷ lệ là 0,296610.`118`Không có chuyển đổi được áp dụng nó không thể giải quyết bảng nhảy, do đó phân tích của nó đạt đến khối nhập, khối kết thúc trong `jmpq *%rax`, trường hợp mặc định, và trả lại: 82 byte, cùng 82 mà bài học 33 của sự hạ thấp tái tạo đơn giản tuyên bố.`-O1`là 47, và`1 - 47/82 = 0.426829`- Với `-e bin.relocs.apply=true`radar2 tìm thấy tất cả 118 byte và in 0.288136, tương đương với tỷ lệ bài học trên các hình ảnh được di chuyển, nơi bốn byte di chuyển khác nhau từ các trường không của đối tượng. Hai công cụ chỉ đồng ý khi so sánh cùng một byte, và cột kích thước không nói bao nhiêu byte đã vào.

`-O`giới hạn dấu vân tay cho opcode byte và đánh giá 0 các operand.`-O0`Vâng`-O1`tỷ lệ tăng lên 0,239130, 0,512195, 0,303030, và 0,523810, và `total``-O1`Vâng`-O2`tăng lên 0.192857. khoảng cách toàn bộ tập tin cũng tồn tại: `-s`(Myers) báo cáo sự tương đồng 0,956 và khoảng cách 162 giữa hai đối tượng,`-ss`(Levenshtein) 0,927 và 137. Những con số đó bao gồm tiêu đề, bảng biểu tượng, và các phần di chuyển và không nói gì về các chức năng.

## Ghidra Version Tracking Correlators

Công cụ theo dõi phiên bản của Ghidra phù hợp với chương trình nguồn với chương trình đích với các mối tương quan. Các hệ tương quan chức năng là Exact Function Bytes Match, Exact Function Instructions Match (operende che giấu), Exact Function Mnemonics Match và Duplicate Function Instructions Match (1:N và N:M). Các tương quan dữ liệu là chính xác, trùng lặp và tương tự dữ liệu. Các mối tương quan biểu tượng phù hợp với tên chính xác, như là trùng lặp, hoặc bởi sự tương đồng. Các hệ tương quan tham chiếu sử dụng các sự phù hợp được chấp nhận để tìm thêm. Các chức năng phải được xác định trong chương trình đích, và chiều dài tối thiểu của chức năng mặc định là 10 byte. Auto Version Tracking chạy Tên biểu tượng chính xác, Dữ liệu chính xác, Bytes chức năng chính xác, hướng dẫn chức năng chính xác, Mnemonics chức năng chính xác, hướng dẫn chức năng trùng lặp, sau đó là một trình tương thích tham chiếu, và tạo ra sự phù hợp ngầm cho các chức năng được chấp nhận. Một bộ tương quan BSim đã được thêm vào trong 11.2.

Bài học đã bắt được các mối tương quan vô đầu. `capture_ghidra_vt.sh`nhập cả ba đối tượng vào một dự án và chạy `ListVersionTrackingMatches.java`, một kịch bản ngắn kéo dài `AbstractGhidraVersionTrackingScript`, mở các chương trình nguồn và đích theo đường dẫn, chạy sáu liên kết không tham chiếu trong tự động phiên bản theo dõi, và in một dòng cho mỗi khối, chức năng, liên kết và phù hợp.

| Correlator | `-O0` to `-O1` | `-O1` to `-O2` |
|---|---|---|
| Exact Symbol Name Match | 4 functions | 4 functions, 5 data items (`labels.rel`, `.L.str.1` to `.L.str.4`) |
| Exact Data Match | 0 | 0 |
| Exact Function Bytes Match | 0 | 2 (`percent_change`, `grade_label`) |
| Exact Function Instructions Match | 0 | 3 (adds `grade`) |
| Exact Function Mnemonics Match | 0 | 3 |
| Duplicate Function Instructions Match | 0 | 0 |

Mỗi hàm thay đổi giữa `-O0`và `-O1`, vì vậy chỉ có tên phù hợp, đó là kết quả mà người phù hợp của bài học đưa ra trong BinDiff. Mỗi trận đấu in tương đồng 1.000 và sự tin cậy 10.000. Trang giúp mô tả các tương quan chính xác như sự tương đồng 1.0 và sự tin cậy 1.0, vì vậy quy mô của sự tin tưởng in là một câu hỏi mở bài học này ghi lại và không giải quyết.

Program Differences, trong Tool, là một điều khác. Nó so sánh hai chương trình với không gian địa chỉ tương tự trên các byte, nhãn, đơn vị mã, tham chiếu, bình luận và chức năng ở địa chỉ tương đương.

## Diaphora và Binary Ninja, Từ Tài liệu của Chúng

Diaphora là một plugin IDA Pro, lần đầu tiên được phát hành vào năm 2015 và phiên bản 3.4.2 từ năm 2026-08-18. Nó là AGPL-3.0 kể từ 2.0 và cần IDAPython, vì vậy nó chạy trong IDA Home hoặc IDA Pro và không phải trong IDA Free. Nó xuất khẩu mỗi cơ sở dữ liệu sang SQLite và chạy các tính toán của nó như các truy vấn SQL. Mỗi heuristic có một tên, một loại Best, Partial, hoặc Unreliable, một loại tỷ lệ, và các cờ tùy chọn như SAME_CPU hoặc SLOW. Tên trong nguồn hiện tại đọc như các giai đoạn của bài học này với nhiều hơn: Bytes hash, Địa chỉ và mnemonics tương tự, Phong hợp được làm sạch tương tự, Mnemonics nhỏ-primes- sản phẩm, Các nút tương tự, cạnh, vòng lặp và các thành phần kết nối chặt chẽ, Topological sort hash, và mã giả mạo fuzzy hash cần Hex-Rays decompiler. Các ghi chú phát hành của nó cho thấy lịch sử điều chỉnh: các heuristik chậm chỉ chạy dưới khoảng 2.000 hàm, và "Đồng đồ đồ" chuyển sang danh mục không đáng tin cậy. Chuyển động đó là viên kim cương bốn khối của bài học này, được nhìn thấy ở quy mô.

Binary Ninja 6.0, phát hành năm 2026-09-03, đã thêm Binary Similarity như một framework với các nhà cung cấp và giải quyết. Nó gửi BinDiff như một nhà cung cấp cấu trúc và WARP cho các kết hợp chức năng chính xác dung nạp các operand có thể di chuyển, và hướng dẫn của nó nói nên sử dụng WARP khi các chức năng phù hợp ngoài hướng dẫn có thể di chuyển và BinDiff khi chúng vẫn có liên quan cấu trúc sau khi tái biên soạn. Tính năng này chỉ có trong phiên bản Ultimate. Binary Ninja không có trình quản lý gói được cài đặt ở đây và không chạy, vì vậy đoạn văn này đến từ tài liệu của nó và không có gì khác.

BinDiff tự nó đã được mở nguồn từ năm 2023-09-25 dưới Apache-2.0, với v8 phiên bản mới nhất. Nó cần một trình giải thể và plugin BinExport, hỗ trợ IDA Pro 9.4 hoặc sau, Binary Ninja và Ghidra.

## Mã giống nhau, không đổi

Bài học 36 giữ một ứng cử viên xây dựng lại của `grade`bảng có 3 như mục cuối cùng, trong khi bản gốc có 4. `grade_case10_wrong.o`có cùng 46 byte mã như `grade``-O1`. Các cặp phù hợp chúng trên các byte chính xác với mỗi chữ ký đồng ý.`1, 2, 3, 4, 4`thông qua một chuyển nhượng và`1, 2, 3, 4, 3`thông qua các báo cáo và các báo cáo khác `dataEqual: false`Ba người khác.`-O1`các hàm không có gì ở bên trái, bởi vì đối tượng ứng cử viên giữ một hàm.

Đây là hình dạng phân tích váy. Một hàm khác nhau dừng ở byte nói không có gì thay đổi. Hình dung dữ liệu chính xác của Ghidra và sự phù hợp tham chiếu chuỗi của BinDiff tồn tại vì cùng một lý do. Phiên bản của bài học nhỏ: cho mỗi cặp phù hợp, so sánh các byte điểm di chuyển của mỗi hàm, với số lượng từ biên giới kiểm tra và chiều rộng từ tải.

## Việc thực hiện biểu tượng: Các con đường, điều kiện, giải pháp

King mô tả hành động biểu tượng vào năm 1976. Các đầu vào trở thành biểu tượng. Mỗi câu nói cập nhật một kho tượng trưng. Mỗi nhánh thêm một điều kiện vào con đường, và một người giải quyết liệu con đường có thể thực hiện được hay không và tạo ra một đầu vào cụ thể theo nó. Một hàm với nhánh trở thành một cây của con đường, và một lá là một trở lại với một điều kiện gắn liền.

Baldoni, Coppa, D'Elia, Demetrescu và Finocchi đã khảo sát lĩnh vực này vào năm 2018. Phần 5 của họ đặt tên là vụ nổ đường dẫn như là chi phí trung tâm: một người thực thi có thể chia rẽ tại mỗi nhánh, vì vậy số lượng trạng thái có thể tăng theo tỉ lệ thoáng trong số các nhánh, và một vòng lặp có đường biên phụ thuộc vào một đầu vào biểu tượng cung cấp một bộ đường dẫn không giới hạn. Các giảm thiểu họ liệt kê là cắt các con đường không thể thực hiện bằng cách giải quyết tại mỗi chi nhánh, tóm tắt cho các chức năng và vòng lặp, kết hợp trạng thái và quá gần gũi. Việc thực hiện concolic chạy chương trình một cách cụ thể và biểu tượng cùng một lúc, ghi lại các điều kiện dọc theo chạy cụ thể, phủ nhận một, và yêu cầu người giải quyết cho đầu vào tiếp theo. DART (2005), CUTE (2005), và SAGE (2008) là các hệ thống được đặt tên.

KLEE (Cadar, Dunbar, Engler, OSDI 2008) là một máy ảo biểu tượng trên mã bit LLVM, vì vậy đầu vào của nó là `clang -emit-llvm`KLEE 3.2 được phát hành năm 2025-12-23 theo giấy phép NCSA, với LLVM 16 được khuyến cáo.

Angr làm việc trên các binary. `angr.Project`có một bộ tải CLE mà lập bản đồ đối tượng và áp dụng di chuyển, một máy nâng mà biến một khối thành VEX IR (`block.vex`) với mã ống thay thế, và một nhà máy cho các tiểu bang.`SimState`chứa các sổ đăng ký, bộ nhớ và một máy giải quyết.`SimulationManager`bước ẩn của các tiểu bang được đặt tên `active`- `deadended`- `found`- `avoided`, và`errored`, và`explore(find=..., avoid=...)`Mỗi bước các động cơ chạy theo thứ tự: thất bại, syscall, hook, unicorn cho dữ liệu bê tông, và VEX như là sự vứt bỏ. claripy là đầu trước của giải pháp trên Z3, và trong Angr 10.0 nó là một ứng dụng Rust tự động nhập khẩu như `from angr import claripy`. angr 10.0.3 được tải lên vào 2026-10-10, là BSD-2-Clause, và cần Python 3.12 hoặc mới hơn. Z3 5.1.0, phát hành năm 2026-08-16 dưới MIT, là giải pháp đằng sau angr, Triton và backend tùy chọn của KLEE.

## Một người hành xử nhỏ trên bài học 34 IR

Người hành quyết trong `code/main.py`sử dụng một IR nhỏ với các hình thức tuyên bố của bài học 34: `Assign`- `Branch`- `Jump`- `Return`, cộng với `Switch`- Trình bày là:`Const`- `Var`- `Bin`với chiều rộng, `Cast`- `Load`từ một bàn bê tông có tên, và`Not`. Bốn chức năng cố định được nâng bằng tay tại mức học tập 34 đạt sau khi đơn giản hóa, nơi `0x66666667`nhân hóa đã trở thành `score / 10`, và mỗi khối IR liệt kê các địa chỉ hướng dẫn nó đến từ. `check_ir_against_listing()`khẳng định rằng các địa chỉ bao gồm danh sách của hàm chính xác một lần.

`grade``-O0`trong IR đó:

```text
0x30: local_8 = score; local_10 = score / 10 - 6; if (local_10 >u 4) goto 0x9a else 0x62
0x62: switch local_10 -> 0x91, 0x88, 0x7f, 0x76, 0x76
0x76: local_4 = 4; goto 0xa1        0x7f: local_4 = 3; goto 0xa1
0x88: local_4 = 2; goto 0xa1        0x91: local_4 = 1; goto 0xa1
0x9a: local_4 = 0                    0xa1: return local_4
```

Các mục tiêu chuyển đổi đến từ `jump_table_targets()`, cùng số toán di chuyển như bài học 33, vì vậy IR đọc các thiết bị.

`execute()`giữ một danh sách làm việc của các trạng thái đường dẫn. Một trạng thái có khối hiện tại, dấu vết của khối, danh sách điều kiện và một môi trường mà lập bản đồ mỗi biến thành một biểu hiện trên các đầu vào. `Assign`thay thế môi trường vào nguồn và lưu trữ kết quả. `Branch`thay thế vào điều kiện, sau đó hỏi giải pháp hai lần: một lần với điều kiện được thêm vào, một lần với âm tính của nó. Một mặt mà giải pháp không thể thỏa mãn được thả, đó là việc cắt đứt mong muốn khảo sát mô tả. `Switch`nhóm các mục nhập bảng theo mục tiêu và các sợi một lần cho mỗi mục tiêu riêng biệt với sự phân chia các bình đẳng nhập cảnh, đó là cách Angr xử lý một mục tiêu nhảy biểu tượng cũng vậy. `Return`Một dấu vết dài hơn giới hạn bước đánh dấu con đường bị cắt đứt và ngăn chặn nó.

Hãy đi theo con đường dẫn đến `return 4`bằng tay.`local_10 = score / 10 - 6`- Tình trạng của nhánh`local_10 >u 4`là sai trên con đường này, vì vậy danh sách điều kiện là đúng `!(score / 10 - 6 >u 4)`. Các mục chuyển đổi 3 và 4 cả hai chỉ số tại 0x76, do đó, gao lên 0x76 thêm `score / 10 - 6 == 3 || score / 10 - 6 == 4`. Nhóm 0x76 được phân bổ `local_4 = 4`, nhảy lên 0xa1, và thay thế trở lại cho `4`- Bảng là cặp của điều kiện đó và giá trị đó.

- Thử giải là`BoundedSearch`. Nó có một miền hữu hạn cho mỗi đầu vào, liệt kê sản phẩm của các miền, và đánh giá mỗi hạn chế bằng ngữ nghĩa C: 32 bit bao quanh, chia cắt, so sánh ký và không ký, và chuyển đổi toán học hoặc logic.`score`trong -200 đến 1200 năm con đường của `grade`Giá trị đánh giá 44,460 và vài millisecond. Giải quyết mỗi lá cho bộ đầu vào cho ra:

| Leaf | Returns | Condition | Scores in the bound |
|---|---|---|---|
| 0x76 | 4 | `!(q >u 4) && (q == 3 \|\| q == 4)` | 90 to 109, 20 inputs |
| 0x7f | 3 | `!(q >u 4) && q == 2` | 80 to 89, 10 inputs |
| 0x88 | 2 | `!(q >u 4) && q == 1` | 70 to 79, 10 inputs |
| 0x91 | 1 | `!(q >u 4) && q == 0` | 60 to 69, 10 inputs |
| 0x9a | 0 | `q >u 4` | 1,351 inputs, the rest of the bound |

`q`nghĩa là`score / 10 - 6`- Các lớp học tương tự xuất hiện từ`grade``-O1`, nơi IR có hai con đường, con đường bảng trở lại `lookup_14[q]`và đường mặc định trả về 0, và câu hỏi trở thành một hạn chế đối với giá trị trả lại, bởi vì không có khối trường hợp để đạt được.`lookup_14`được đọc từ đối tượng `.rodata`ở 0x14.

Các điểm số được xác định là số điểm của các phương pháp. Một tìm kiếm có giới hạn hoàn thành bên trong phạm vi của nó và im lặng bên ngoài nó. Trong -200 đến 1200 không có điểm số trong 60 đến 109 trả về 0 và không có điểm số 110 hoặc hơn trả về một giá trị không bằng 0. z3 chứng minh cả hai cho tất cả các điểm số 2 ^ 32, và phần tiếp theo cho thấy nơi hai phương pháp trả về các nhân chứng khác nhau.

## Đường độ nổ được đo

`total``-O1`là vòng lặp của bài học 33: một bài kiểm tra trên `count`, một cơ thể vòng tròn thêm `values[rdx]`và tăng trưởng`rdx`, và một `jne`quay lại cơ thể trong khi `rdx != rcx`Trong IR`rcx`là `zext64(count)`và `rdx`bắt đầu từ 0. Mỗi bước qua các vách phía sau `rdx != count`với `rdx`một liên tục, vì vậy con đường cho `count = c`- Đánh bộ cơ thể.`c`Thời gian và lối ra.`count`giới hạn đến 0 đến 2, người thực thi tìm thấy 3 đường. Với 0 đến 4, nó tìm thấy 5, và với 0 đến 8, nó tìm thấy 9. Mỗi đường trả lại tổng tam giác của số lượng của nó: (0, 0), (1, 1), (2, 3), lên đến (8, 36). angr báo cáo 3, 5 và 9 trạng thái bị kết thúc và cùng chín cặp số và số lượng trên tất cả ba cấp độ tối ưu hóa, bao gồm `-O2`, vòng vector cần một con số ít nhất 8 và do đó không bao giờ thay đổi con số đường trong phạm vi này.

Không bị ràng buộc, trình thực không bao giờ hoàn thành, bởi vì mỗi lần lặp lại là một con đường thực hiện hơn.`count`cho đến 64, trả lại 22 đường, cắt giảm 2, và không bao giờ bắt đầu 41 số còn lại trong số 65 hiện có. báo cáo cho ba số đó. chi phí giải quyết cũng tăng lên: 33 đánh giá ở đường 2, 115 ở 4, 531 ở 8, và 6,900 cho chạy giới hạn.

## Angr và z3 về các câu hỏi tương tự

`capture_angr.py`chạy độ 10.0.3 trên ba đối tượng. CLE tải mỗi đối tượng được di chuyển với backend ELF tại 0x400000, áp dụng 13 di chuyển cho `-O0`và 8 cho những người khác, và giải quyết bốn biểu tượng.`grade`với một lập luận 64 bit biểu tượng, `run()`kết thúc với 5 quốc gia bị kết thúc vào năm`-O0`và 2 ở `-O1`và `-O2`, trong 0.475, 0.408, và 0.318 giây.`eax == v`cho mỗi nhà nước bị đóng cửa và yêu cầu nhà giải quyết cho mức tối thiểu và tối đa đã ký kết cho 60 đến 69, 70 đến 79, 80 đến 89, và 90 đến 109 cho `v`từ 1 đến 4 ở mỗi cấp độ, và -2147483648 đến 2147483647 cho 0. Cả hai kiểm tra chéo đều không được đáp ứng.

Các biểu mẫu Reach-a-block đã sử dụng CFGFast trên `grade``-O0`, trong đó liệt kê 8 khối bài học 33 được tìm thấy, sau đó `explore(find=0x400076)`cho khối lưu trữ 4. Nó tìm thấy một trạng thái với ba vẫn hoạt động trong 0,47 giây, với tối thiểu 90 và tối đa 109 trên lập luận và các mẫu 91, 93, 94, 97, và 101.

Vì `percent_change`, hai quốc gia bị ngã ngã với những hạn chế `before[31:0] != 0`và `before[31:0] == 0`- Đàn hỏi trạng thái không bằng 0`eax == 50`cho `before = 21055744, after = 332315210``-O0`- `40566804, 834296877``-O1`, và`-24484867, 993934694``-O2`- Tìm kiếm giới hạn trong số từ -16 đến 16 câu trả lời`before = -10, after = -15`. z3, đặt ra cùng một câu hỏi trên bài học 34 danh sách, câu trả lời `before = -25092599, after = 305712160`. Phòng thử đánh giá tất cả năm cặp với IR đánh giá và mỗi số trả 50. Một mô hình chứng minh rằng một đầu vào tồn tại.

`capture_z3.py`mô hình của `-O1`danh sách các `grade`trong 32-bit bit vector với nhân, hai thay đổi, và bảng, và hỏi bốn câu hỏi nữa.`switch`cho một số điểm 32 bit: không được đặt trong 0,068 giây, do đó hàm được biên soạn bằng nguồn trên tất cả các đầu vào 2 ^ 32.`score / 10`là 70634653 và chuyển đổi rơi vào mặc định trong khi ngưỡng trở lại 4. Sự khác biệt tương tự bị giới hạn ở 0 đến 109: không được đáp ứng, đó là lý do tại sao lưới bài học 36 không thể thấy nó.`percent_change`- `before == 3`với kết quả -133 được ngồi với `after = 1073741823`, tế bào tràn của bài học 36, và cuộc tìm kiếm hạn chế tìm thấy một nhân chứng thứ hai trong phạm vi của nó,`after = -1`, nơi `(-1 - 3) * 100 / 3`Truncates đến -133 mà không có quá tải.

## Hãy xây dựng nó

Chạy demo:

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution
python3 code/main.py
```

Kết quả là một báo cáo JSON. `diffs`giữ các tệp đối tượng và các hình ảnh tải phù hợp với `-O0`chống lại`-O1`và `-O1`chống lại`-O2`, cộng với `-O0`cặp trong BinDiff. `toolComparison`đặt giai đoạn bài học bên cạnh phán quyết và tỷ lệ radiff2 và các tương quan Ghidra cho mỗi hàm. `patch`là `grade_case10_wrong.o`- Vụ án.`radare2Ratios`Tái tạo lại các số radiff2, `gradeRelocationCase`cho thấy bố cục và hai di chuyển, và `symbolic`giữ các con đường, lớp học, và số vụ nổ.

Đọc `code/main.py`theo thứ tự này:

1. `ElfObject`đọc các phần, biểu tượng, và `.rela.*`- Các hồ sơ.`parse_listing()`đọc `llvm-objdump`- Thư.
2. `jump_table_targets()`và `build_cfg()`cho các khối có thể tiếp cận và các cạnh được đánh dấu. `FunctionRecord.signatures()`xây dựng 5 chữ ký.
3. `load_image()`Định nghĩa các phần được phân bổ và áp dụng `S + A - P`và `S + A`- Tôi không biết.
4. `match_functions()`là người chơi được sắp xếp. `levenshtein()`và `radare2_ratio()`cho sự tương đồng.
5. `parse_radiff2()`và `parse_ghidra_vt()`đọc các kết quả công cụ được chụp để so sánh.
6. `evaluate()`- `substitute()`, và`BoundedSearch`là ngữ nghĩa biểu hiện và giải pháp.
7. `execute()`liệt kê các con đường. `solve_returns()`yêu cầu mỗi lá một bộ đầu vào.
8. `grade_ir_o0()`- `grade_ir_o1()`- `percent_change_ir_o1()`, và`total_ir_o1()`là những chiếc nâng tay với địa chỉ hướng dẫn của chúng.

Có ba lệnh nữa.`python3 code/main.py diff A.o B.o`khác biệt hai đối tượng có danh sách nằm bên cạnh chúng, và `--image`khác nhau các hình ảnh tải của họ. `python3 code/main.py paths grade O0`in các đường và lớp của một hàm, `paths total 8`vòng lặp với một đường dây. `python3 code/main.py figure-data`in dữ liệu mà hình ảnh tương tác nhúng.

## Sử dụng nó

Sử dụng differ và executor trong thứ tự này trên hai build của chương trình của riêng bạn.

1. Tận ra một danh sách cho mỗi đối tượng và để `function_record()`Hãy kiểm tra nó với các byte, sửa nó trước bất cứ thứ gì khác.
2. Đánh phân các tệp đối tượng trước tiên. Đọc giai đoạn của mỗi trận đấu và xem các trận đấu hình dạng đồ thị dòng chảy như là gợi ý cho đến khi một chữ ký thứ hai đồng ý.
3. Differ các hình ảnh tải. Một hàm giảm từ các byte chính xác đến các byte che giấu di chuyển so với dữ liệu của nó, và số học di chuyển đặt tên cho phần di chuyển.
4. Đối với mỗi trận đấu, so sánh các bảng tham chiếu. cùng một mã với các định vị khác nhau là một hàm được sửa đổi.
5. Động hành các tên giai đoạn và so sánh. Khi xây dựng có biểu tượng, chạy BinDiff thứ tự và sử dụng kết quả cấu trúc chỉ để xếp hạng các giai đoạn.
6. Khi một công cụ báo cáo một tỷ lệ, tìm ra bao nhiêu byte đã vào nó trước khi bạn so sánh nó với của bạn.
7. Đối với hàm mà bạn muốn nhập, viết IR với một địa chỉ cho mỗi lệnh, chạy `execute()`với một đường dây trên mỗi con số vòng biểu tượng, và giải quyết mỗi lá.
8. Thực hiện cùng một truy vấn qua angr hoặc z3 khi chúng có sẵn và so sánh các giới hạn lớp.

## Phòng thí nghiệm tương tác

### Phòng thí nghiệm A: 5 chữ ký bằng tay

Từ `code`thư mục, mở Python:

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution/code
python3 -q
```

Đi chạy:

```python
from main import *
grade = {r.name: r for r in fixture_records("ledger-x86_64-O1.o")}["grade"]
grade.raw.hex(" ")
grade.masked_bytes() == grade.raw
grade.mnemonics
grade.cfg_shape()
grade.signatures()
```

Viết ra bốn byte tại offset 0x26 của hàm trước khi bạn in `grade.raw`Rồi chạy cùng một dòng.`grade`từ `ledger-x86_64-O0.o`và đếm số`table``grade.edges`- Tôi không biết.

### Phòng thí nghiệm B: xem việc di chuyển phá vỡ hash byte

```python
case = grade_relocation_case()
case["layout"]
case["displacements"], case["rodataShift"]
case["objectBytesEqual"], case["imageBytesEqual"], case["maskedImageBytesEqual"]
```

Lưu ý`0x100110 + 0x14 - 0x10003a`trên giấy trước khi đọc `displacements`Rồi thay căn cứ vào `load_image()`để 0x400000 và xác nhận rằng các chuyển động không thay đổi, bởi vì cả hai operand di chuyển cùng nhau.

### Phòng thí nghiệm C: sự phù hợp sai và thứ tự tránh nó

```python
report = diff_fixtures("ledger-x86_64-O1.o", "ledger-x86_64-O2.o")
[(m.left.name, m.right.name, m.stage) for m in report.matches]
report = diff_fixtures("ledger-x86_64-O0.o", "ledger-x86_64-O1.o")
[(m.left.name, m.right.name, m.stage, m.names_agree) for m in report.matches]
report = diff_fixtures("ledger-x86_64-O0.o", "ledger-x86_64-O1.o", stages=BINDIFF_ORDER)
[(m.left.name, m.right.name, m.stage) for m in report.matches]
```

Bác in `records(...)["percent_change"].cfg_shape()`cho `-O0`và `grade_label`cho `-O1`và đọc 2 chuỗi độ. sau đó viết một giai đoạn thứ sáu mà hash mnemonics sắp xếp, để thứ tự không còn quan trọng, và xem nó tách chúng ra hay không.

### Phòng thí nghiệm D: đường và đường biên

```python
report = grade_paths("O0")
[(p["trace"], p["condition"], p["returns"]) for p in report["paths"]]
report["returns"]["4"]
total_paths(2), total_paths(4), total_paths(8)
total_paths(64, step_limit=24)["truncated"]
```

Thay đổi`SCORE_DOMAIN`đến`range(0, 110)`và tái diễn`grade_paths("O0")`. lớp cho 0 giảm xuống 60 đầu vào và hai kiểm tra chéo vẫn giữ.`range(90, 100)`và đọc ra những lá nào không có mô hình.

## Phòng thí nghiệm thực hành

Hãy chạy BinDiff trên các thiết bị và so sánh nó với ba kết quả mà bạn đã có.

Yêu cầu:

- Lắp đặt phần mở rộng BinExport trong Ghidra 12.1.4 và xuất xuất ba chương trình sổ cái.
- Lắp đặt dòng lệnh BinDiff 8 và tạo ra `-O0`Vâng`-O1`và `-O1`Vâng`-O2`kết quả với tùy chọn chỉ tương tự.
- Chụp sự tương đồng và sự tin tưởng cho mỗi chức năng trong `code/fixtures/`với các lệnh chính xác trong `build_fixtures.sh`- Tôi không biết.
- Tăng `tool_comparison()`với một `bindiff`cột và thêm một bài kiểm tra so sánh BinDiff's phù hợp tập hợp trên `-O1`Vâng`-O2`với bộ máy đeo mặt nạ của bài học.
- Viết một đoạn văn có tên thuật toán BinDiff được sử dụng cho `grade`và liệu nó có so sánh hình ảnh tải.

Thành công có nghĩa là bản demo in bốn cột công cụ cho mỗi hàm và thử nghiệm mới cho biết một sự bình đẳng hoặc một sự khác biệt giải thích.

## Bài học: Cây tạo vật

Bài học này bao gồm:`outputs/skill-binary-diffing-and-symbolic-execution.md`Sử dụng nó để biến hai build và một hàm thành một sự khác biệt giai đoạn với một chất lượng được xác định cho mỗi trận đấu, một lời giải thích về mọi hàm không sánh hoặc thay đổi, và một báo cáo đường đi với một input được giải quyết cho mỗi lá và giới hạn tạo ra nó.

## Hãy kiểm tra

Chạy demo và bộ:

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Việc kiểm tra phải chứng minh:

- bảng biểu tượng cho bốn hàm với kích thước 46, 118, 66, 39 tại `-O0`, 16, 46, 31, 63 tại `-O1`, và 16, 46, 140, 63 tại `-O2`
- `.rela.text`ghi chép của `-O1`đối tượng mô tả như `llvm-objdump -r`in chúng
- mỗi lệnh được liệt kê nằm trong phạm vi biểu tượng của nó và các byte của nó bằng với đối tượng, và một thay đổi byte tăng
- `total``-O1`có năm khối, năm cạnh, và một vòng tự, và chuỗi hình dạng của nó là xác định
- `grade``-O0`phục hồi 5 mục tiêu nhảy bàn từ `.rela.rodata`Và có tám khối và mười một cạnh
- Không có bàn,`grade``-O0`đạt đến 82 byte trong bốn khối
- `percent_change``-O0`và `grade_label``-O1`có hình dạng bốn khối tương tự
- Hashes chính xác và ẩn đồng ý về mọi chức năng của một đối tượng có thể di chuyển
- `-O1`và `-O2`chia sẻ byte cho ba hàm, `-O0`và `-O1`không chia sẻ
- bảng tham chiếu của `grade`đọc `1, 2, 3, 4, 4`và những gì ứng cử viên đọc `1, 2, 3, 4, 3`
- chỗ tải `.text`- `.rodata`, và`.rodata.str1.1`Ghidra đã đưa họ vào cả hai chương trình
- `R_X86_64_PC32`được viết như `S + A - P`và tạo ra sự di chuyển 122, và `R_X86_64_64`Các mục trong `.data`nhận địa chỉ chuỗi
- `grade`Byte đồng ý trong các đối tượng và khác nhau trong hình ảnh bằng một thay đổi 0x70, và che giấu khôi phục sự phù hợp
- `-O1`Vâng`-O2`phù hợp với ba hàm trên các byte chính xác và `total`tên; hình ảnh xem phù hợp `grade`bởi các byte ẩn với sự tương đồng 1 - 1/46
- thứ tự bài học tạo ra một sự phù hợp hình dạng của biểu đồ dòng chảy giả trên `-O0`Vâng`-O1`, và BinDiff sắp xếp phù hợp với tất cả bốn tên
- ứng cử viên vá phù hợp trên byte và không phù hợp với dữ liệu tham chiếu
- tỷ lệ Levenshtein tái tạo 0.130435, 0.106061, 0.269841, 0.171429, 0.426829 trên 82 byte, và tỷ lệ chuyển-byte được in radiff2 với các chuyển
- bộ MATCH của radiff2 bằng với giai đoạn Exact-Bytes, và các bộ tương quan byte và hướng dẫn của Ghidra bằng với các bộ chính xác và ẩn trong hình ảnh
- Ghidra chỉ tìm thấy tên trên `-O0`Vâng`-O1`, với độ tương đồng 1.0 và độ tin cậy 10.0 trên mỗi trận đấu
- IR nâng tay bao gồm mỗi hướng dẫn được liệt kê một lần, và người đánh giá theo ngữ nghĩa C
- `grade`có 5 con đường ở `-O0`với điều kiện chuyển đổi và hai ở `-O1`với cùng lớp
- Các lớp giới hạn 60 đến 69, 70 đến 79, 80 đến 89, 90 đến 109 bằng angr ở cả ba cấp độ và z3, với 5 và 2 trạng thái bị kết thúc và phạm vi tiếp cận một khối 90 đến 109
- hai kiểm tra chéo không có mô hình trong đường ranh giới, trong angr, hoặc trong z3, và z3 chứng minh danh sách bằng với chuyển đổi nguồn
- mỗi `percent_change`mô hình từ tìm kiếm giới hạn, angr, và z3 đánh giá đến giá trị yêu cầu
- `total`cho 3, 5, 9 đường với số lượng ba giác, giống như angr, và bước đệm trả lại 22 và cắt 2
- dữ liệu hình ảnh chứa cả hai lượt xem, số di chuyển, và năm lá với bộ đầu vào của chúng, và mô-đun trang web nhúng cùng một dữ liệu

## Các chế độ sản xuất thất bại

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

## Quy tắc hoạt động

Đáp nhập một chữ ký tại một thời điểm, chấp nhận chỉ các giá trị độc đáo ở cả hai bên, và giữ cho giai đoạn và chất lượng của nó với sự phù hợp. Bước khác nhau các đối tượng và các hình ảnh tải và giải thích mọi hàm thay đổi giai đoạn giữa hai. Đếm các con đường với một giải pháp ở mỗi nhánh, liên kết mỗi vòng lặp biểu tượng, và báo cáo một bộ đầu vào với giới hạn của nó. Một mô hình đơn lẻ không giới hạn là một mẫu.

## Đọc thêm

- [BinDiff concepts: matching strategy, algorithms, and quality ratings](https://github.com/google/bindiff/blob/main/docs/concepts.md)
- [BinDiff repository and the v8 open-source release](https://github.com/google/bindiff/releases/tag/v8)
- [BinExport README: IDA Pro, Binary Ninja, and Ghidra exporters](https://github.com/google/binexport)
- [Flake, "Structural Comparison of Executable Objects" (DIMVA 2004) and Dullien, Rolles, "Graph-based comparison of Executable Objects" (SSTIC 2005), shipped in the BinDiff repository](https://github.com/google/bindiff/tree/main/docs/papers)
- [Diaphora README and heuristics source](https://github.com/joxeankoret/diaphora)
- [Ghidra Version Tracking correlators help page](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/VersionTracking/src/main/help/help/topics/VersionTrackingPlugin/VT_Correlators.html)
- [Ghidra Auto Version Tracking help page](https://www.ghidradocs.com/11.0_PUBLIC/help/VersionTracking/help/topics/VersionTrackingPlugin/VT_AutoVT.html)
- [Ghidra 12.1.4 release](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Binary Ninja 6.0 "Krypton" release notes](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html)và [Binary Similarity user guide](https://docs.binary.ninja/guide/similarity.html)
- [radare2 book: radiff2](https://book.rada.re/tools/radiff2/intro.html)và [radare2 6.2.4 source](https://github.com/radareorg/radare2/releases/tag/6.2.4)
- [King, "Symbolic execution and program testing" (CACM 1976)](https://dl.acm.org/doi/10.1145/360248.360252)
- [Baldoni et al., "A Survey of Symbolic Execution Techniques" (ACM Computing Surveys 2018)](https://dl.acm.org/doi/10.1145/3182657)
- [Cadar, Sen, "Symbolic execution for software testing: three decades later" (CACM 2013)](https://dl.acm.org/doi/10.1145/2408776.2408795)
- [Cadar, Dunbar, Engler, "KLEE: Unassisted and Automatic Generation of High-Coverage Tests for Complex Systems Programs" (OSDI 2008)](https://www.usenix.org/legacy/events/osdi08/tech/full_papers/cadar/cadar.pdf)và [KLEE 3.2 release](https://github.com/klee/klee/releases/tag/v3.2)
- [Shoshitaishvili et al., "SoK: (State of) The Art of War: Offensive Techniques in Binary Analysis" (IEEE S&P 2016)](https://oaklandsok.github.io/papers/shoshitaishvili2016.pdf)
- [angr documentation: loading, simulation managers, and the simulation pipeline](https://docs.angr.io/en/latest/core-concepts/toplevel.html)và [angr 10.0.3 on PyPI](https://pypi.org/project/angr/)
- [Z3 5.1.0 release](https://github.com/Z3Prover/z3/releases/tag/z3-5.1.0)
- [Triton dynamic binary analysis library](https://github.com/JonathanSalwan/Triton)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, Table 4.9: Relocation Types](https://gitlab.com/x86-psABIs/x86-64-ABI)

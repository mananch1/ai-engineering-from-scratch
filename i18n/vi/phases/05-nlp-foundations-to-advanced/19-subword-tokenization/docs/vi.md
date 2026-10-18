# Đánh dấu từ phụ  BPE, WordPiece, Unigram, SentencePiece

> Các mã thông báo từ bị ngạt vào những từ không thể thấy, các mã thông báo ký tự làm tăng chiều dài chuỗi, các mã thông báo từ dưới chia khác biệt, mỗi chương trình đại học hiện đại đều được chuyển sang một.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## Vấn đề

Từ vựng của bạn có 50.000 từ. Người dùng gõ "không thể nhận dạng".`[UNK]`Mô hình hiện không có tín hiệu về từ. Tệ hơn: tài liệu phần trăm 90 trong cơ quan của bạn có 40 từ hiếm, nghĩa là 40 bit thông tin bị bỏ rơi cho mỗi tài liệu.

Các từ chung vẫn là các token đơn lẻ.`untokenizable`→ `un`- `token`- `izable`Dữ liệu đào tạo bao gồm tất cả bởi vì bất kỳ chuỗi nào là một chuỗi của các byte.

Mỗi LLM biên giới vào năm 2026 được gửi trên một trong ba thuật toán (BPE, Unigram, WordPiece), được gói trong một trong ba thư viện (tiktoken, SentencePiece, HF Tokenizers). Bạn không thể gửi một mô hình ngôn ngữ mà không chọn một.

## Khái niệm

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).**Bắt đầu với một từ vựng ở cấp độ ký tự. Đếm từng cặp lân cận. Thôi cặp thường xuyên nhất vào một mã thông báo mới. Lặp lại cho đến khi bạn đạt đến kích thước từ vựng mục tiêu.

**Byte-level BPE.**Cùng thuật toán nhưng trên các byte thô (256 mã thông báo cơ sở) thay vì các ký tự Unicode.`[UNK]`token  bất kỳ mã hóa chuỗi byte nào. GPT-2 sử dụng 50.257 token (256 byte + 50.000 hợp nhất + 1 đặc biệt).

**Unigram.**Bắt đầu với một từ vựng khổng lồ. Đề xuất cho mỗi token một xác suất unigram. Thử cắt đứt lặp đi lặp lại các token mà việc loại bỏ ít nhất làm tăng xác suất ghi chép corpus. Có thể xác định: có thể lấy mẫu token (có ích cho tăng dữ liệu thông qua quy định phụ từ). Được sử dụng bởi T5, mBART, ALBERT, XLNet, Gemma.

**WordPiece.**Các cặp hợp nhất để tối đa hóa khả năng tập hợp tập thể thay vì tần số nguyên liệu.

**SentencePiece vs tiktoken.**SentencePiece là thư viện * đào tạo * từ vựng (BPE hoặc Unigram) trực tiếp trên văn bản Unicode thô, mã hóa không gian trắng như `▁`. tiktoken là mã hóa nhanh * của OpenAI chống lại từ vựng được xây dựng sẵn; nó không đào tạo.

Quy tắc:

- **Training a new vocabulary:**SentencePiece (hiện ngữ đa ngôn ngữ, không có pre-tokenization) hoặc HF Tokenizers.
- **Fast inference against GPT vocab:**tiktoken (cl100k_base, o200k_base).
- **Both:**HF Tokenizers  một thư viện, đào tạo + phục vụ.

```figure
bpe-merge
```

## Hãy xây dựng nó

### Bước 1: BPE từ đầu

Nhìn xem`code/main.py`- Chuyện này:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Ba sự thật mà thuật toán mã hóa.`</w>`dấu chấm kết thúc từ để "hèn" (đối hậu) và "hèn" (đối hậu) vẫn khác nhau. trọng lượng tần số làm cho cặp tần số cao thắng sớm. Danh sách kết hợp được sắp xếp  suy luận áp dụng kết hợp theo thứ tự đào tạo.

### Bước 2: mã hóa với các sự hợp nhất được học

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

Thới thới $O(n \cdot \lvert \text{merges} \rvert)$. Các triển khai sản xuất (tiktoken, HF Tokenizers) sử dụng tìm kiếm xếp hạng hợp nhất với hàng xếp ưu tiên và chạy trong thời gian gần như tuyến tính.

### Bước 3: Câu nóiPiece thực tế

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

Lưu ý: không cần thiết cho việc pre-tokenization, không gian được mã hóa như `▁`- `character_coverage`kiểm soát cách các ký tự hiếm có được bảo tồn và được lập bản đồ đến `<unk>`- Tôi không biết.

### Bước 4: Tiktoken cho các từ ngữ tương thích với OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Chỉ có mã hóa. nhanh (Rust backend). chính xác phù hợp với GPT-4/5 token hóa cho tính toán byte, ước tính chi phí, ngân sách cửa sổ bối cảnh.

## Những bẫy vẫn còn tồn tại vào năm 2026

- **Tokenizer drift.**Đào tạo về từ A, triển khai chống lại từ B. Đồ nhận mã khác nhau; mô hình ra rác.`tokenizer.json`hash trong CI.
- **Whitespace ambiguity.**BPE "hello" vs "hello" tạo ra các mã thông báo khác nhau.`add_special_tokens`và `add_prefix_space`rõ ràng.
- **Multilingual undertraining.**Corpora nặng tiếng Anh sản xuất từ vựng chia các chữ không phải tiếng Latinh thành 5-10 lần nhiều mã thông báo. cùng một lời nhắc chi phí 5-10 lần nhiều hơn trong tiếng Nhật / Ả Rập trên GPT-3.5. o200k_base một phần sửa chữa điều này.
- **Emoji splits.**Một emoji có thể lấy 5 token.

## Sử dụng nó

Số 2026:

| Situation | Pick |
|-----------|------|
| Training a monolingual model from scratch | HF Tokenizers (BPE) |
| Training a multilingual model | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Serving an OpenAI-compatible API | tiktoken (`o200k_base` for GPT-4+) |
| Domain-specific vocab (code, math, protein) | Train custom BPE on domain corpus, merge with base vocab |
| Edge inference, small model | Unigram (smaller vocabularies work better) |

Kích thước từ vựng là một quyết định quy mô, không phải là một định vị.

## Chuyển nó

Cứ như `outputs/skill-bpe-vs-wordpiece.md`- Có thể là:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## Các bài tập

1. **Easy.**Đưa một BPE 500 hợp nhất vào `code/main.py`Có bao nhiêu từ đã tạo ra chính xác 1 token so với > 1 token?
2. **Medium.**So sánh số lượng token trên 100 câu Wikipedia tiếng Anh giữa `cl100k_base`- `o200k_base`, và một SentencePiece BPE bạn đào tạo với từ ngữ = 32k. báo cáo tỷ lệ nén của mỗi.
3. **Hard.**Tập cùng một bộ với BPE, Unigram và WordPiece. đo độ chính xác dòng chảy khi sử dụng mỗi bộ phân loại cảm xúc nhỏ.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Greedy merge of most-frequent character pairs until target vocab size hit. |
| Byte-level BPE | No unknown tokens ever | BPE over raw 256 bytes; GPT-2 / Llama use this. |
| Unigram | Probabilistic tokenizer | Prunes from a large candidate set using log-likelihood; used by T5, Gemma. |
| SentencePiece | The whitespace one | Library that trains BPE/Unigram on raw text; space encoded as `▁`. |
| tiktoken | The fast one | OpenAI's Rust-backed BPE encoder for pre-built vocabs. No training. |
| Merge list | The magic numbers | Ordered list of `(a, b) → ab` merges; inference applies in order. |
| Character coverage | How rare is too rare? | Fraction of characters in training corpus the tokenizer must cover; ~0.9995 typical. |

## Đọc thêm

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) giấy BPE.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) tờ Unigram.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) thư viện.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) Khảo sát ngắn gọn.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) sách nấu ăn + danh sách mã hóa.

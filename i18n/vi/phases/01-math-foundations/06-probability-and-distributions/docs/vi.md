# Thi suất và phân phối

> Khả năng là ngôn ngữ AI sử dụng để thể hiện sự không chắc chắn.

**Type:** Learn
**Language:**Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## Mục tiêu học tập

- Thực hiện PMF và PDF từ đầu cho phân phối Bernoulli, categorical, Poisson, đồng nhất và bình thường
- Xét giá trị dự kiến, sự khác biệt và sử dụng định lý giới hạn trung tâm để giải thích lý do tại sao người Gauss chiếm ưu thế
- Xây dựng các hàm softmax và log-softmax bằng thủ thuật ổn định số (từ logit max)
- Xét mất tích entropy chéo từ logits và kết nối nó với xác suất log âm

## Vấn đề

Các sản phẩm phân loại `[0.03, 0.91, 0.06]`Một mô hình ngôn ngữ chọn từ tiếp theo từ 50.000 ứng viên. mô hình phân tán tạo ra hình ảnh bằng cách lấy mẫu từ phân phối được học. Tất cả những điều này là xác suất trong hành động.

Mỗi dự đoán mà mô hình đưa ra là phân bố xác suất. Mỗi hàm mất mát đo lường sự phân bố dự đoán xa như thế nào so với sự phân bố thực. Mỗi bước đào tạo điều chỉnh các tham số để làm cho một phân bố trông giống như một phân bố khác hơn. Nếu không có xác suất, bạn không thể đọc một bài báo ML duy nhất, sửa lỗi một mô hình duy nhất, hoặc hiểu tại sao mất mát đào tạo của bạn là NaN.

## Khái niệm

### Sự kiện, không gian mẫu và khả năng

Không gian mẫu $S$là tập hợp tất cả các kết quả có thể. Một sự kiện là một bộ phận của không gian mẫu.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Ba định nghĩa định nghĩa xác suất:
1. $P(A) \ge 0$cho bất kỳ sự kiện nào $A$
2. $P(S) = 1$(một cái gì đó luôn xảy ra)
3. $P(A \text{ or } B) = P(A) + P(B)$Khi nào $A$và $B$không thể xảy ra cả hai

Mọi thứ khác (định lý Bayes, kỳ vọng, phân phối) đều theo sau ba quy tắc này.

### Có thể có điều kiện và độc lập

$P(A \mid B)$là xác suất của $A$Với điều đó $B$đã xảy ra.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Hai sự kiện độc lập khi biết một không nói gì về người khác:

```math
\begin{aligned}
\text{Independent:} \quad & P(A \mid B) = P(A) \\
\text{Equivalent to:} \quad & P(A \text{ and } B) = P(A) \cdot P(B)
\end{aligned}
```

Việc đánh tiền xu là độc lập, và việc rút thẻ mà không có người thay thế thì không.

### Vận động cơ lượng xác suất so với Vận động mật độ xác suất

Các biến ngẫu nhiên nhỏ có hàm khối lượng xác suất (PMF). Mỗi kết quả có xác suất cụ thể mà bạn có thể đọc trực tiếp.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Các biến ngẫu nhiên liên tục có hàm mật độ xác suất (PDF). mật độ tại một điểm không phải là xác suất.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

Sự phân biệt này quan trọng trong ML. Các sản phẩm phân loại là PMF (các lựa chọn riêng biệt).

### Phân phối chung

**Bernoulli:**Một thử nghiệm, hai kết quả.

```math
\begin{aligned}
P(X = 1) &= p \\
P(X = 0) &= 1 - p \\
\text{Mean} = p, &\quad \text{Variance} = p(1-p)
\end{aligned}
```

**Categorical:**Một phiên tòa,$k$Kết quả: Mô hình phân loại đa lớp (tối đa đầu ra mềm).

```math
\begin{aligned}
& P(X = i) = p_i, \quad \text{where } \textstyle\sum_i p_i = 1 \\
& \text{Example: } P(\text{cat}) = 0.7, \quad P(\text{dog}) = 0.2, \quad P(\text{bird}) = 0.1
\end{aligned}
```

**Uniform:**được sử dụng để khởi tạo ngẫu nhiên.

```math
\begin{aligned}
\text{Discrete:} \quad & P(X = k) = 1/n \text{ for } k \in \lbrace 1, \ldots, n \rbrace \\
\text{Continuous:} \quad & f(x) = 1/(b-a) \text{ for } x \in [a, b]
\end{aligned}
```

**Normal (Gaussian):**cong chuông. được định nghĩa bằng trung bình ($\mu$) và sự khác biệt ($\sigma^2$().

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:**Số lượng các sự kiện hiếm trong một khoảng thời gian cố định.

```math
\begin{aligned}
P(X = k) &= \frac{\lambda^k e^{-\lambda}}{k!} \\
\text{Mean} = \lambda, &\quad \text{Variance} = \lambda
\end{aligned}
```

### Giá trị dự kiến và sự khác biệt

Giá trị dự kiến là kết quả trung bình trọng lượng.

```math
\begin{aligned}
\text{Discrete:} \quad & \mathbb{E}[X] = \sum_i x_i \, P(X = x_i) \\
\text{Continuous:} \quad & \mathbb{E}[X] = \int x \, f(x) \, dx
\end{aligned}
```

Các biện pháp biến thể được trải rộng xung quanh trung bình.

```math
\begin{aligned}
\mathrm{Var}(X) &= \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \\
\text{Standard deviation} &= \sqrt{\mathrm{Var}(X)}
\end{aligned}
```

Trong ML, giá trị dự kiến xuất hiện như hàm mất (sự mất trung bình trên phân phối dữ liệu).

### Phân phối chung và biên giới

Một phân phối chung $P(X, Y)$mô tả hai biến ngẫu nhiên cùng nhau.

Ví dụ về PMF chung ($X$= thời tiết,$Y$= dù):

| | $Y=0$ (no umbrella) | $Y=1$ (umbrella) | Marginal $P(X)$ |
|---|---|---|---|
| $X=0$ (sun) | 0.40 | 0.10 | $P(X=0) = 0.50$ |
| $X=1$ (rain) | 0.05 | 0.45 | $P(X=1) = 0.50$ |
| **Marginal $P(Y)$** | $P(Y=0) = 0.45$ | $P(Y=1) = 0.55$ | 1.00 |

Phân bố biên cộng lại biến khác:

```math
P(X = x) = \sum_{y} P(X = x, Y = y)
```

Tổng hàng và cột trong bảng trên là biên giới.

### Tại sao sự phân phối bình thường xuất hiện khắp nơi

Định lý giới hạn trung tâm: tổng (hoặc trung bình) của nhiều biến ngẫu nhiên độc lập hội tụ với phân bố bình thường, bất kể phân bố ban đầu.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

Đây là lý do tại sao:
- Các lỗi đo lường là bình thường (nhiều nguồn độc lập nhỏ)
- Các khởi tạo trọng lượng trong mạng thần kinh sử dụng phân phối bình thường
- Âm thanh gradient trong SGD là bình thường (tổng số các gradient mẫu)
- Phân bố bình thường là phân bố entropy tối đa cho một trung bình và sự biến động nhất định

### Khoản log

Những xác suất nguyên liệu gây ra các vấn đề số.

```math
\begin{aligned}
P(\text{sentence}) &= P(\text{word}_1) \cdot P(\text{word}_2) \cdot \ldots \cdot P(\text{word}_n) \\
&= 0.01 \cdot 0.003 \cdot 0.02 \cdot \ldots \\
&\to 0.0 \quad \text{(underflow after } \sim 30 \text{ terms)}
\end{aligned}
```

Log xác suất sửa chữa điều này.

```math
\begin{aligned}
\log P(\text{sentence}) &= \log P(\text{word}_1) + \log P(\text{word}_2) + \ldots + \log P(\text{word}_n) \\
&= -4.6 + -5.8 + -3.9 + \ldots \\
&\to \text{finite number (no underflow)}
\end{aligned}
```

Quy tắc:
- $\log(a \cdot b) = \log(a) + \log(b)$
- log xác suất luôn luôn $\le 0$(từ khi $0 < P \le 1$(văn)
- Thêm âm tính = ít khả năng
- Thiệt hại giao hợp là xác suất log âm của lớp chính xác

### Softmax như một phân phối xác suất

Các mạng thần kinh phát ra điểm số thô (logits). Softmax chuyển đổi chúng thành phân bố xác suất hợp lệ.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Trù mềmmax: trừ logit tối đa trước khi tăng số để ngăn chặn quá tải.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax kết hợp softmax và log cho sự ổn định số. PyTorch sử dụng điều này bên trong để mất entropy chéo.

### Tiêu chuẩn

Phân tích mẫu có nghĩa là rút ra các giá trị ngẫu nhiên từ phân phối.
- Thả các mẫu ngẫu nhiên mà các tế bào thần kinh để nới
- Dữ liệu tăng cường mẫu biến đổi ngẫu nhiên
- Các mô hình ngôn ngữ lấy mẫu token tiếp theo từ phân phối dự đoán
- Các mô hình phân phối lấy mẫu tiếng ồn và dần dần tiêu diệt

Việc lấy mẫu từ phân phối tùy tiện đòi hỏi các kỹ thuật như lấy mẫu biến đổi ngược, lấy mẫu từ chối hoặc thủ thuật tái định đo (chỉ được sử dụng trong VAEs).

```figure
gaussian-pdf
```

## Hãy xây dựng nó

### Bước 1: Các cơ sở xác suất

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Bước 2: PMF và PDF từ đầu

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Bước 3: Giá trị dự kiến và sự khác biệt

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Bước 4: Tiêu mẫu từ phân phối

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Bước 5: Softmax và xác suất log

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Bước 6: Phương pháp giới hạn trung tâm

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Bước 7: Hình ảnh

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Các thực hiện đầy đủ với tất cả các hình ảnh hóa là trong `code/probability.py`- Tôi không biết.

## Sử dụng nó

Với NumPy và SciPy, tất cả trên đều là một dòng:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

Anh đã xây dựng những cái này từ đầu rồi, giờ anh biết những gì thư viện gọi đang làm.

## Các bài tập

1. Thực hiện lấy mẫu biến đổi ngược cho phân bố theo hàm số. Kiểm tra bằng cách lấy mẫu 10.000 giá trị và so sánh histogram với PDF thực.

2. Xây dựng một bảng phân phối chung cho hai con số đốm có tải, tính toán các phân phối biên và kiểm tra xem các con số đốm có độc lập hay không.

3. Xét mất lượng entropy chéo cho một phân loại 5 lớp xuất logits `[2.0, 0.5, -1.0, 3.0, 0.1]`khi lớp đúng là chỉ số 3. Sau đó xác minh câu trả lời của bạn với PyTorch `nn.CrossEntropyLoss`- Tôi không biết.

4. Viết một hàm lấy danh sách xác suất log và trả lại chuỗi có khả năng nhất, tổng xác suất log và xác suất nguyên thô tương đương.

### Những gợi ý

**Exercise 1.**Chụp`u`từ phân phối và trả lại đồng nhất $-\ln(1 - u) / \lambda$Kiểm tra xem trung bình mẫu gần như là $1/\lambda$và các hộp histogram theo dõi $\lambda e^{-\lambda x}$- Tôi không biết.

**Exercise 2.**Thấp đầy một bảng 6x6 nơi mỗi tế bào là một xác suất và tổng của tất cả các tế bào là 1, sau đó gọi `joint_to_marginals`- Sử dụng`check_independence`: các xúc xắc chỉ độc lập nếu mỗi tế bào bằng với sản phẩm của hai bên cạnh của nó.

**Exercise 3.**Đơn `softmax`đến các logits và lấy $-\ln$của xác suất tại chỉ số 3. Kết quả phải gần 0,421, đó là giá trị `nn.CrossEntropyLoss`cho cùng một mục tiêu và mục tiêu.

**Exercise 4.**Thêm các xác suất log; không nhân xác suất nguyên thô. Đối với 50 từ với xác suất 0.01, tổng số phải là $50 \ln(0.01) \approx -230.26$, và xác suất thô là $10^{-100}$- Tôi không biết.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sample space | "All the possibilities" | The set $S$ of every possible outcome of an experiment |
| PMF | "The probability function" | A function that gives the exact probability of each discrete outcome, summing to 1 |
| PDF | "The probability curve" | A density function for continuous variables. Integrate it over an interval to get probability |
| Conditional probability | "Probability given something" | $P(A \mid B) = P(A \text{ and } B) / P(B)$. The foundation of Bayesian thinking and Bayes' theorem |
| Independence | "They don't affect each other" | $P(A \text{ and } B) = P(A) \cdot P(B)$. Knowing one event tells you nothing about the other |
| Expected value | "The average" | The probability-weighted sum of all outcomes. The loss function is an expected value |
| Variance | "How spread out" | The expected squared deviation from the mean. High variance = noisy, unstable estimates |
| Normal distribution | "The bell curve" | $f(x) = (1/\sqrt{2\pi\sigma^2}) \cdot \exp(-(x-\mu)^2/(2\sigma^2))$. Appears everywhere due to the CLT |
| Central Limit Theorem | "Averages become normal" | The mean of many independent samples converges to a normal distribution regardless of the source |
| Joint distribution | "Two variables together" | $P(X, Y)$ describes the probability of every combination of $X$ and $Y$ outcomes |
| Marginal distribution | "Sum out the other variable" | $P(X) = \sum_y P(X, Y)$. Recovers one variable's distribution from the joint |
| Log probability | "Log of the probability" | $\log P(x)$. Turns products into sums, preventing numerical underflow in long sequences |
| Softmax | "Turn scores into probabilities" | $\text{softmax}(z_i) = \exp(z_i) / \sum_j \exp(z_j)$. Maps real-valued logits to a valid probability distribution |
| Cross-entropy | "The loss function" | $-\sum p_{\text{true}} \cdot \log(p_{\text{predicted}})$. Measures how different two distributions are. Lower is better |
| Logits | "Raw model outputs" | Unnormalized scores before softmax. Named after the logistic function |
| Sampling | "Drawing random values" | Generating values according to a probability distribution. How models generate output |

## Đọc thêm

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo)- bằng chứng trực quan về lý do tại sao trung bình trở nên bình thường
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf)- tham chiếu ngắn gọn bao gồm tất cả mọi thứ ở đây và nhiều hơn nữa
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)- tại sao sự ổn định số lượng quan trọng và làm thế nào để đạt được nó

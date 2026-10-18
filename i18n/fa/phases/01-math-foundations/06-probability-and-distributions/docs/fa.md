# احتمال و توزیع

> احتمال زبانی است که هوش مصنوعی برای بیان عدم اطمینان استفاده می کند.

**Type:** Learn
**Language:**پیتون
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## اهداف یادگیری

- پیاده سازی PMF ها و PDF ها از ابتدا برای توزیع های برنوولی، دسته بندی، پویسون، یکسره و عادی
- ارزش انتظار می رود را محاسبه کنید، تفاوت را محاسبه کنید و از نظریه محدودیت مرکزی برای توضیح اینکه چرا گاسیان ها بر جهان تسلط دارند استفاده کنید
- با استفاده از ترفند ثبات عددی، تابع های softmax و log-softmax را بسازید (معاینه max logit)
- از دست دادن آنترپی در حال محاسبه از logits و ارتباط آن با احتمال log منفی

## مشکل

محصولات طبقه بندی کننده`[0.03, 0.91, 0.06]`یک مدل زبان کلمه بعدی را از بین ۵۰ هزار نامزد انتخاب می کند. یک مدل انتشار تصاویر را با نمونه گیری از توزیع های آموخته تولید می کند. همه اینها احتمال در عمل است.

هر پیش بینی که یک مدل انجام می دهد توزیع احتمال است. هر عملکرد ضرر اندازه می گیرد که توزیع پیش بینی شده از واقعی چقدر فاصله دارد. هر مرحله آموزش پارامترها را تنظیم می کند تا یک توزیع بیشتر شبیه به دیگری باشد. بدون احتمال، شما نمی توانید یک مقاله ML را بخوانید، یک مدل را خراب کنید یا درک کنید که چرا از دست دادن آموزش شما NaN است.

## مفهوم

### رویدادها، فضاهای نمونه و احتمال

فضای نمونه$S$مجموعه تمام نتایج ممکن است. یک رویداد زیر مجموعه ی فضای نمونه است. احتمال نمایش رویدادها به اعداد بین 0 و 1 است.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

سه محور احتمال را تعریف می کند:
1. $P(A) \ge 0$برای هر رویداد$A$
2. $P(S) = 1$(همیشه اتفاق می افتد)
3. $P(A \text{ or } B) = P(A) + P(B)$چه وقت$A$و$B$نمیتونه هر دو اتفاق بیفتد

همه چیز دیگر (نظریه بایز، انتظارات، توزیع) از این سه قانون پیروی می کند.

### احتمال مشروط و استقلال

$P(A \mid B)$احتمالي است$A$با توجه به این$B$اتفاقي افتاد

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

دو اتفاق مستقل هستند وقتی که می دونید یکی از آنها چیزی درباره ی دیگری نمیگه:

```math
\begin{aligned}
\text{Independent:} \quad & P(A \mid B) = P(A) \\
\text{Equivalent to:} \quad & P(A \text{ and } B) = P(A) \cdot P(B)
\end{aligned}
```

فلپ کردن سکه مستقل است.

### احتمال عملکردهای جرم در مقابل احتمال عملکردهای تراکم

متغیرهای تصادفی متمایز دارای یک تابع احتمالی (PMF) هستند. هر نتیجه دارای یک احتمال خاص است که می توانید مستقیماً آن را بخوانید.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

متغیرهای تصادفی مداوم تابع تراکم احتمال (PDF) دارند. تراکم در یک نقطه احتمال نیست. احتمال از ادغام تراکم در یک فاصله ناشی می شود.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

این تفاوت در ML مهم است. خروجی طبقه بندی PMF (انتخابات متمایز) هستند. فضاهای پنهان VAE از PDF (مستقیم) استفاده می کنند.

### توزیع های مشترک

**Bernoulli:**یک آزمایش، دو نتیجه مدل طبقه بندی دوگانه

```math
\begin{aligned}
P(X = 1) &= p \\
P(X = 0) &= 1 - p \\
\text{Mean} = p, &\quad \text{Variance} = p(1-p)
\end{aligned}
```

**Categorical:**یه محاکمه$k$نتایج: مدل های طبقه بندی چند طبقه (خروج نرم حداکثر).

```math
\begin{aligned}
& P(X = i) = p_i, \quad \text{where } \textstyle\sum_i p_i = 1 \\
& \text{Example: } P(\text{cat}) = 0.7, \quad P(\text{dog}) = 0.2, \quad P(\text{bird}) = 0.1
\end{aligned}
```

**Uniform:**تمام نتایج به طور مساوی احتمال دارد. برای ابتدایی تصادفی استفاده می شود.

```math
\begin{aligned}
\text{Discrete:} \quad & P(X = k) = 1/n \text{ for } k \in \lbrace 1, \ldots, n \rbrace \\
\text{Continuous:} \quad & f(x) = 1/(b-a) \text{ for } x \in [a, b]
\end{aligned}
```

**Normal (Gaussian):**منحنی زنگ. به اندازه متوسط ($\mu$) و تفاوت ($\sigma^2$)

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:**تعداد رویدادهای نادر در یک فاصله ثابت.

```math
\begin{aligned}
P(X = k) &= \frac{\lambda^k e^{-\lambda}}{k!} \\
\text{Mean} = \lambda, &\quad \text{Variance} = \lambda
\end{aligned}
```

### ارزش و تنوع انتظار می رود

ارزش انتظار می رود، نتیجه متوسط وزن شده است.

```math
\begin{aligned}
\text{Discrete:} \quad & \mathbb{E}[X] = \sum_i x_i \, P(X = x_i) \\
\text{Continuous:} \quad & \mathbb{E}[X] = \int x \, f(x) \, dx
\end{aligned}
```

اندازه گیری های متغیر در اطراف متوسط پخش می شوند.

```math
\begin{aligned}
\mathrm{Var}(X) &= \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \\
\text{Standard deviation} &= \sqrt{\mathrm{Var}(X)}
\end{aligned}
```

در ML، ارزش انتظار می رود به عنوان تابع از دست دادن (توسط از دست دادن در توزیع داده ها) ظاهر شود. تغیر به شما در مورد ثبات مدل می گوید. تغیرات بالا در گرادیانت ها به معنای آموزش های سر و صدا است.

### توزیع مشترک و مرزی

توزیع مشترک$P(X, Y)$دو متغیر تصادفی را با هم توصیف می کند.

نمونه ی PMF مشترک ($X$= آب و هوا$Y$= چتر:

| | $Y=0$ (no umbrella) | $Y=1$ (umbrella) | Marginal $P(X)$ |
|---|---|---|---|
| $X=0$ (sun) | 0.40 | 0.10 | $P(X=0) = 0.50$ |
| $X=1$ (rain) | 0.05 | 0.45 | $P(X=1) = 0.50$ |
| **Marginal $P(Y)$** | $P(Y=0) = 0.45$ | $P(Y=1) = 0.55$ | 1.00 |

توزیع حاشیه ای متغیر دیگر را جمع می کند:

```math
P(X = x) = \sum_{y} P(X = x, Y = y)
```

مجموع خط و ستون در جدول بالا، حاشیه ها هستند.

### چرا توزیع عادی در همه جا ظاهر می شود

نظریه محدودیتی مرکزی: مجموع (یا متوسط) بسیاری از متغیرهای تصادفی مستقل به توزیع عادی متقابل می شود، بدون توجه به توزیع اصلی.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

به همین دلیل:
- اشتباهات اندازه گیری تقریبا طبیعی است (بسیاری از منابع مستقل کوچک)
- شروع کردن وزن در شبکه های عصبی از توزیع های عادی استفاده می کند
- صدا در SGD تقریبا طبیعی است (همۀ بسیاری از gradients نمونه)
- توزیع طبیعی، حداکثر توزیع انتروپی برای یک متوسط و انحراف داده شده است

### احتمالات ثبت

احتمالات خام باعث مشکلات عددی می شوند. ضرب بسیاری از احتمالات کوچک به سرعت به صفر می رسد.

```math
\begin{aligned}
P(\text{sentence}) &= P(\text{word}_1) \cdot P(\text{word}_2) \cdot \ldots \cdot P(\text{word}_n) \\
&= 0.01 \cdot 0.003 \cdot 0.02 \cdot \ldots \\
&\to 0.0 \quad \text{(underflow after } \sim 30 \text{ terms)}
\end{aligned}
```

احتمالات ثبت این مسئله را حل می کنند. ضربات به اضافه شدن تبدیل می شوند.

```math
\begin{aligned}
\log P(\text{sentence}) &= \log P(\text{word}_1) + \log P(\text{word}_2) + \ldots + \log P(\text{word}_n) \\
&= -4.6 + -5.8 + -3.9 + \ldots \\
&\to \text{finite number (no underflow)}
\end{aligned}
```

قوانین:
- $\log(a \cdot b) = \log(a) + \log(b)$
- احتمالات ثبت همیشه$\le 0$(از زمان$0 < P \le 1$)
- منفی تر = احتمال کمتری
- از دست دادن کراس انتروپی احتمال منفی ثبت کلاس درست است

### Softmax به عنوان توزیع احتمال

شبکه های عصبی نمره های خام (logits) را تولید می کنند. Softmax آنها را به توزیع احتمالی معتبر تبدیل می کند.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

ترفند نرم: قبل از نمایی کردن، حداکثر منطق را برای جلوگیری از پرش از دست بردارید.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax نرم و log را برای ثبات عددی ترکیب می کند. PyTorch این را برای از دست دادن انتروپی متقابل در داخل استفاده می کند.

### نمونه گیری

نمونه گیری یعنی کشیدن مقادیر تصادفی از توزیع.
- نمونه هاي تصادفي رو از دست بده که از سلول هاي عصبی صفر بشه
- نمونه های افزایشی داده ها تحولات تصادفی
- مدل های زبان نمونه ای از نماد بعدی از توزیع پیش بینی شده
- مدل های انتشار نمونه های سر و صدا و به تدریج از آنها استفاده می کنند

نمونه گیری از توزیع های تعسفی نیازمند تکنیک هایی مانند نمونه گیری ترفورمات برعکس، نمونه گیری رد یا ترفورمترسازی (که در VAEs استفاده می شود) است.

```figure
gaussian-pdf
```

## آن را بسازید

### مرحله ی اول: اصول احتمال

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

### مرحله دوم: PMF و PDF از ابتدا

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

### مرحله سوم: ارزش انتظار شده و تفاوت

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

### مرحله 4: نمونه گیری از توزیع

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

### مرحله 5: نرمترین و احتمالات ثبت

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

### مرحله 6: نشان دادن نظریه محدودیت مرکزی

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### مرحله هفتم: تصویرسازی

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

پیاده سازی کامل با تمام تصویرسازی ها در حال انجام است `code/probability.py`. .

## ازش استفاده کن

با NumPy و SciPy، همه چیز بالا یک خط است:

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

تو اينها رو از ابتدا ساختي حالا ميدوني که تماس هاي کتابخانه چيکار ميکنن

## تمرینات

1. نمونه گیری تبدیل معکوس را برای توزیع نمایی اجرا کنید. با نمونه گیری از 10،000 ارزش و مقایسه هیستogram با PDF واقعی، بررسی کنید.

2. یک جدول توزیع مشترک برای دو جواهر باردار بسازید. توزیع هاماری را محاسبه کنید و بررسی کنید که آیا جواهرات مستقل هستند یا خیر.

3. خسارت های متقابل انتروپی را برای طبقه بندی کننده 5 طبقه که logits را تولید می کند محاسبه کنید `[2.0, 0.5, -1.0, 3.0, 0.1]`وقتی کلاس درست شاخص 3 باشد، سپس پاسخ خود را با PyTorch تایید کنید.`nn.CrossEntropyLoss`. .

4. یک تابع بنویسید که یک لیست از احتمالات ثبت نام را بگیرد و احتمال احتمالی ترین ردیف، احتمال کل ثبت نام و احتمال خام معادل را بازگرداند. آن را با جمله 50 کلمه ای که هر کلمه احتمال 0.01 دارد، آزمایش کنید.

### اشاره ها

**Exercise 1.**رسم کن`u`از یک توزیع و بازپرداخت یکسانی$-\ln(1 - u) / \lambda$. چک کن که متوسط نمونه نزدیک به$1/\lambda$و که کاشی های هیستوگرافی دنبالش می کنند$\lambda e^{-\lambda x}$. .

**Exercise 2.**یک جدول 6×6 را پر کنید که هر سلول یک احتمال است و مجموع تمام سلول ها 1 است، سپس تماس بگیرید`joint_to_marginals`استفاده کن`check_independence`: این اعداد تنها اگر هر سلول برابر به ضرب دو حاشیه اش باشد مستقل هستند.

**Exercise 3.**درخواست کنید`softmax`به دستگاه ها و برن$-\ln$در این حالت، احتمال در شاخص 3، نتیجه باید نزدیک به 0.421 باشد، که مقدار `nn.CrossEntropyLoss`به خاطر همین هدف و هدف می دهد.

**Exercise 4.**احتمالات ثبت را اضافه کنید، احتمالات خام را ضرب نکنید. برای 50 کلمه با احتمال 0.01، کل باید باشد $50 \ln(0.01) \approx -230.26$، و احتمال خامش$10^{-100}$. .

## اصطلاحات کلیدی

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

## خواندن بیشتر

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo)- اثبات بصری که چرا متوسط ها به حالت عادی می رسند
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf)- مرجع خلاصه ای که همه چیز را در اینجا و بیشتر پوشش می دهد
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)- چرا ثبات عددی مهم است و چگونه به آن دست یابیم

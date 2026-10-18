# ماشین های ویکتور پشتیبانی

> .برترین خیابان بین دو طبقه رو پیدا کن

**Type:** Build
**Language:**پیتون
**Prerequisites:** Phase 1 (Lessons 08 Optimization, 14 Norms and Distances, 18 Convex Optimization)
**Time:** ~90 minutes

## اهداف یادگیری

- پیاده سازی یک SVM خطی از ابتدا با استفاده از از دست دادن بازنویسی و کاهش گرادینت در فرمول اولیه
- اصول حداکثر مارژ را توضیح دهید و متورهای پشتیبانی را از یک مدل آموزش دیده شناسایی کنید
- مقایسه هسته های خطی، چندگانه و RBF و توضیح دهید که چگونه ترفند هسته از نقشه برداری آشکار با ابعاد بالا جلوگیری می کند
- ارزیابی تعادل کنترل شده توسط پارامتر C بین عرض حاشیه و خطاهای طبقه بندی

## مشکل

شما دو دسته از نقاط داده دارید و باید یک خط (یا هائپرپلین) را که آنها را از هم جدا می کند، بکشید. خطوط بی نهایت زیادی می توانند کار کنند. کدام یک را انتخاب کنید؟

این که بیشترین مارجین را دارد. مارجین فاصله بین مرز تصمیم گیری و نزدیک ترین نقاط داده در هر طرف است. مارجین گسترده تر به این معنی است که طبقه بندی کننده مطمئن تر است و به اطلاعات ندیده شده بهتر کلی می کند.

این حس منجر به ماشین آلات ویکتور پشتیبانی می شود، یکی از الگوریتم های ریاضی بسیار زیبا در ML. SVM ها روش طبقه بندی غالب قبل از یادگیری عمیق بودند و بهترین انتخاب برای مجموعه داده های کوچک، داده های ابعاد بالا و مشکلات هستند که شما نیاز به یک مدل اصول، با وضوح نظری با تضمینات دارید.

SVM ها مستقیماً به مرحله ۱ متصل می شوند: بهینه سازی مخروط است (درس ۱۸) ، حاشیه با قوانین اندازه گیری می شود (درس ۱۴) و ترفند هسته ای از محصولات نقطه ای برای مدیریت مرزهای غیر خطی بدون هیچ وقت محاسبه در فضای بالا استفاده می کند.

## مفهوم

### طبقه بندی کننده حداکثر مارژین

داده های خطی جدا کننده با برچسب ها$y_i \in \lbrace -1, +1 \rbrace$و متریزهای ویژگی$x_i$، ما يه هواپيماي هايپر مي خوايم$w^T x + b = 0$که کلاس ها را جدا می کند.

فاصله از یک نقطه$x_i$به سطح بالا:

```math
\text{distance} = \frac{\lvert w^T x_i + b \rvert}{\lVert w \rVert}
```

برای یک نقطه صحیح طبقه بندی شده: $y_i (w^T x_i + b) > 0$.حاشیه دو برابر فاصله از سطح بالا تا نزدیک ترین نقطه در هر دو طرف است.

```mermaid
graph LR
    subgraph Margin
        direction TB
        A["w^T x + b = +1"] ~~~ B["w^T x + b = 0"] ~~~ C["w^T x + b = -1"]
    end
    D["+ class points"] --> A
    E["- class points"] --> C
    B --- F["Decision boundary"]
```

مشکل بهینه سازی:

```math
\begin{aligned}
\text{maximize} \quad & \frac{2}{\lVert w \rVert} \quad \text{(the margin width)} \\
\text{subject to} \quad & y_i (w^T x_i + b) \ge 1 \quad \text{for all } i
\end{aligned}
```

به طور معادل (به حداقل رساندن)$\lVert w \rVert^2$بهینه سازی آسان تر است):

```math
\begin{aligned}
\text{minimize} \quad & \frac{1}{2} \lVert w \rVert^2 \\
\text{subject to} \quad & y_i (w^T x_i + b) \ge 1 \quad \text{for all } i
\end{aligned}
```

این یک برنامه مربع مخلوط است. این یک راه حل جهانی منحصر به فرد دارد. نقاط داده که دقیقا در مرز حاشیه قرار دارند (که در آن$y_i (w^T x_i + b) = 1$) متری های پشتیبانی هستند. آنها تنها نقاطی هستند که مرز تصمیم را تعیین می کنند. هر نقطه از متری غیر پشتیبانی را حرکت دهید یا حذف کنید و مرز تغییر نمی کند.

### متورهای پشتیبانی: تعداد کمی مهم

```mermaid
graph TD
    subgraph Classification
        SV1["Support Vector (+ class)<br>y(w'x+b) = 1"] --- DB["Decision Boundary<br>w'x+b = 0"]
        DB --- SV2["Support Vector (- class)<br>y(w'x+b) = 1"]
    end
    O1["Other + points<br>(do not affect boundary)"] -.-> SV1
    O2["Other - points<br>(do not affect boundary)"] -.-> SV2
```

اکثر نقاط آموزش بی ربط هستند. تنها متری های پشتیبانی مهم هستند. به همین دلیل است که SVM ها در زمان پیش بینی حافظه موثر هستند: شما فقط نیاز به ذخیره متری های پشتیبانی دارید، نه کل مجموعه آموزش.

تعداد ویکتورهای پشتیبانی همچنین یک محدودیت در خطای عمومی سازی را فراهم می کند. ویکتورهای پشتیبانی کمتر نسبت به اندازه مجموعه داده ها به معنای عمومی سازی بهتر است.

### حاشیه نرم: کنترل صدا با پارامتر C

داده های واقعی به ندرت کاملاً قابل جدایی هستند. برخی نقاط ممکن است در سمت اشتباه مرز یا داخل حاشیه باشند. فرمولا حاشیه نرم اجازه نقض را با معرفی متغیرهای خستگی می دهد.

```math
\begin{aligned}
\text{minimize} \quad & \frac{1}{2} \lVert w \rVert^2 + C \sum_i \xi_i \\
\text{subject to} \quad & y_i (w^T x_i + b) \ge 1 - \xi_i \\
& \xi_i \ge 0 \quad \text{for all } i
\end{aligned}
```

متغیر لچک$\xi_i$اندازه گیری چقدر نقطه$i$. به حاشيه نقض ميکنه$C$کنترل معامله را انجام می دهد:

| $C$ value | Behavior |
|---------|----------|
| Large $C$ | Penalizes violations heavily. Narrow margin, fewer misclassifications. Overfits |
| Small $C$ | Allows more violations. Wide margin, more misclassifications. Underfits |

$C$قدرت تنظیم، برعکس.$C$= کمتر منظم شدن$C$= منظم شدن بیشتر

### از دست دادن گره: عملکرد از دست دادن SVM

SVM مارجین نرم می تواند به عنوان یک بهینه سازی بدون محدودیت نوشته شود:

```math
\text{minimize} \quad \frac{1}{2} \lVert w \rVert^2 + C \sum_i \max(0, 1 - y_i (w^T x_i + b))
```

اصطلاح$\max(0, 1 - y_i f(x_i))$این صفر است وقتی که نقطه به درستی طبقه بندی شده و فراتر از مارژین است. آن خطی است وقتی که نقطه در داخل مارژین است یا طبقه بندی اشتباه است.

```
Hinge loss for a single point:

loss
  |
  | \
  |  \
  |   \
  |    \
  |     \_______________
  |
  +-----|-----|-------->  y * f(x)
       0     1

Zero loss when y*f(x) >= 1 (correctly classified, outside margin).
Linear penalty when y*f(x) < 1.
```

با از دست دادن لجستیک (رجرجشن لجستیک) مقایسه کنید:

```math
\begin{aligned}
\text{Hinge:} \quad & \max(0, 1 - y f(x)) && \text{Hard cutoff at margin} \\
\text{Logistic:} \quad & \log(1 + \exp(-y f(x))) && \text{Smooth, never exactly zero}
\end{aligned}
```

از دست دادن هنج راه حل های کمیاب را تولید می کند (تنها ویکتورهای پشتیبانی سهم غیر صفر دارند). از دست دادن لجستیک از همه نقاط داده استفاده می کند. این باعث می شود SVM ها در زمان پیش بینی حافظه بیشتری داشته باشند.

### آموزش یک SVM خطی با کاهش گرادینت

شما می توانید یک SVM خطی را با استفاده از کاهش گرادینت در ضایعات کنجک و همچنین تنظیم L2 آموزش دهید، بدون حل QP محدود:

```math
\begin{aligned}
& L(w, b) = \frac{\lambda}{2} \lVert w \rVert^2 + \frac{1}{n} \sum_i \max(0, 1 - y_i (w^T x_i + b)) \\[6pt]
& \text{Gradient with respect to } w\text{:} \\
& \quad \text{If } y_i (w^T x_i + b) \ge 1\text{:} \quad \frac{\partial L}{\partial w} = \lambda w \\
& \quad \text{If } y_i (w^T x_i + b) < 1\text{:} \quad \frac{\partial L}{\partial w} = \lambda w - y_i x_i \\[6pt]
& \text{Gradient with respect to } b\text{:} \\
& \quad \text{If } y_i (w^T x_i + b) \ge 1\text{:} \quad \frac{\partial L}{\partial b} = 0 \\
& \quad \text{If } y_i (w^T x_i + b) < 1\text{:} \quad \frac{\partial L}{\partial b} = -y_i
\end{aligned}
```

این را فرمول اولیه می نامند.$O(n \cdot d)$در هر دوره، در کجا$n$تعداد نمونه ها و$d$برای داده های بزرگ، کمیاب، ابعاد بالا (تصفیه متن) ، این سریع است.

### دو شکل و ترفند هسته

دوگانه لاگارنجی مشکل SVM (از مرحله 1 درس 18، شرایط KKT) عبارت است از:

```math
\begin{aligned}
\text{maximize} \quad & \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) \\
\text{subject to} \quad & 0 \le \alpha_i \le C \\
& \sum_i \alpha_i y_i = 0
\end{aligned}
```

دوگانه فقط شامل محصولات نقطه ای می شود $x_i \cdot x_j$بین نقاط داده ها. این بینش کلیدی است. هر محصول نقطه ای را با یک تابع هسته ای جایگزین کنید$K(x_i, x_j)$و SVM می تواند مرز های غیر خطی را بدون محاسبه صریح تحول یاد بگیرد.

```math
\begin{aligned}
\text{Linear kernel:} \quad & K(x, z) = x \cdot z \\
\text{Polynomial kernel:} \quad & K(x, z) = (x \cdot z + c)^d \\
\text{RBF (Gaussian):} \quad & K(x, z) = \exp(-\gamma \lVert x - z \rVert^2)
\end{aligned}
```

هسته RBF داده ها را به یک فضای بی نهایت ابعاد نقشه می زند. نقاط نزدیک در فضای ورودی دارای ارزش هسته نزدیک به 1. نقاط دور از یکدیگر دارای ارزش هسته نزدیک به 0. می تواند هر مرز تصمیم گیری صاف را یاد بگیرد.

```mermaid
graph LR
    subgraph "Input Space (not separable)"
        A["Data points in 2D<br>circular boundary"]
    end
    subgraph "Feature Space (separable)"
        B["Data points in higher dim<br>linear boundary"]
    end
    A -->|"Kernel trick<br>K(x,z) = phi(x).phi(z)"| B
```

این ترفند هسته محصول نقطه را در فضای بالا بدون رفتن به آنجا محاسبه می کند. برای هسته چندگانه درجه$d$در$D$ابعاد، فضای مشخصی دارد$O(D^d)$اما$K(x, z)$محاسبه شده در $O(D)$وقت

### SVM برای بازپسین (SVR)

پشتيباني متري بازپسين در يک لوله ي عرض قرار مي گيرد$\epsilon$نقاط داخل لوله صفر ضرر دارند نقاط خارج از لوله خطی مجازات می شوند

```math
\begin{aligned}
\text{minimize} \quad & \frac{1}{2} \lVert w \rVert^2 + C \sum_i (\xi_i + \xi_i^*) \\
\text{subject to} \quad & y_i - (w^T x_i + b) \le \epsilon + \xi_i \\
& (w^T x_i + b) - y_i \le \epsilon + \xi_i^* \\
& \xi_i, \xi_i^* \ge 0
\end{aligned}
```

.$\epsilon$پارامتر عرض لوله را کنترل می کند. لوله گسترده تر = متری پشتیبانی کمتر = تناسب صاف تر. لوله باریکتر = متری پشتیبانی بیشتر = تناسب محکم تر.

### چرا SVM ها به یادگیری عمیق دست پیدا کردند (و هنوز هم برنده می شوند)

SVM ها از اواخر دهه 1990 تا اوایل دهه 2010 بر ML تسلط داشتند. یادگیری عمیق به دلایل مختلفی از آنها فراتر رفت:

| Factor | SVMs | Deep learning |
|--------|------|---------------|
| Feature engineering | Requires it | Learns features |
| Scalability | $O(n^2)$ to $O(n^3)$ for kernel | $O(n)$ per epoch with SGD |
| Image/text/audio | Needs handcrafted features | Learns from raw data |
| Large datasets (>100k) | Slow | Scales well |
| GPU acceleration | Limited benefit | Massive speedup |

در این شرایط هم SVM برنده می شوند:
- مجموعه داده های کوچک (صد تا هزاران نمونه)
- داده های کمیاب با ابعاد بالا (متن با ویژگی های TF-IDF)
- وقتی به تضمین های ریاضی نیاز دارید (حدود های مارجن)
- زمانی که زمان آموزش باید کم باشد (SVM خطی بسیار سریع است)
- طبقه بندی دوگانه با ساختار واضح حاشیه
- تشخیص غیر معمول (SVM یک کلاس)

```figure
svm-margin
```

## آن را بسازید

### مرحله ی اول: از دست دادن لبه و گرادین

پایه، از دست دادن لوله ها و تراشها محاسبه کنید.

```python
def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] * (dot(w, X[i]) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n
```

### مرحله دوم: SVM خطی از طریق کاهش گرادینت

با حداقل کردن از دست دادن بازي هاي بازي شده، آموزش بده.

```python
class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                margin = y[i] * (dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    self.w = [wj - self.lr * self.lambda_param * wj
                              for wj in self.w]
                else:
                    self.w = [wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j])
                              for j, wj in enumerate(self.w)]
                    self.b -= self.lr * (-y[i])

    def predict(self, X):
        return [1 if dot(self.w, x) + self.b >= 0 else -1 for x in X]
```

### مرحله 3: عملکردهای هسته

هسته های خطی، چندگانه و RBF را پیاده سازی کنید.

```python
def linear_kernel(x, z):
    return dot(x, z)

def polynomial_kernel(x, z, degree=3, c=1.0):
    return (dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma=0.5):
    diff = [xi - zi for xi, zi in zip(x, z)]
    return math.exp(-gamma * dot(diff, diff))
```

### مرحله 4: شناسایی مرزهای و متورهای پشتیبانی

پس از آموزش، مشخص کنید که کدام نقاط متری پشتیبانی هستند و عرض حاشیه را محاسبه کنید.

```python
def find_support_vectors(X, y, w, b, tol=1e-3):
    support_vectors = []
    for i in range(len(X)):
        margin = y[i] * (dot(w, X[i]) + b)
        if abs(margin - 1.0) < tol:
            support_vectors.append(i)
    return support_vectors
```

ببین`code/svm.py`برای اجرای کامل با تمام نمایش ها.

## ازش استفاده کن

با سکیت-علم:

```python
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
])
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
print(f"Support vectors: {clf['svm'].n_support_}")
```

مهم: همیشه قبل از آموزش یک SVM ویژگی های خود را مقیاس بندی کنید. SVM ها نسبت به شدت ویژگی ها حساس هستند زیرا حاشیه بستگی به $\lVert w \rVert$، و ویژگی های غیر مقیاس دار هندسه را تحریف می کنند.

برای مجموعه داده های بزرگ، استفاده کنید `LinearSVC`(فرومول اولیه)$O(n)$در عوض از `SVC`(دو شکل)$O(n^2)$به$O(n^3){}$):

```python
from sklearn.svm import LinearSVC

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=1.0, max_iter=10000)),
])
```

## تمرینات

1. یک مجموعه داده های 2D خطی جدا کننده را تولید کنید. LinearSVM خود را تمرین کنید و متری پشتیبانی را شناسایی کنید. بررسی کنید که متری پشتیبانی نزدیک ترین نقطه به مرز تصمیم است.

2. مختلف$C$از 0.001 تا 1000 در مجموعه داده های سر و صدا.$C$ارزش: انتقال از مارجین وسیع (کم مناسب) به مارجین باریک (مبالغ بیش از حد) را مشاهده کنید.

3. مجموعه داده ای ایجاد کنید که مرزها در آن کلاس دایره ای (نه خطی) هستند. نشان دهید که یک SVM خطی شکست خورده است. ماتریس هسته RBF را محاسبه کنید و نشان دهید که کلاس ها در فضای ویژگی های ناشی از هسته قابل جدایی می شوند.

4. مقایسه از دست دادن بازنواز با از دست دادن لجستیک در یک مجموعه داده. یک SVM خطی و بازپسین لجستیک را تمرین کنید. شمارش کنید که چند نقطه آموزشی به مرز تصمیم گیری هر مدل کمک می کند (وکتورهای پشتیبانی در مقابل همه نقاط).

5. استفاده از SVR (خسارت حساسیت از اپسئلون) را انجام دهید.$y = \sin(x) + \text{noise}$. لوله ایپسیلون را در اطراف پیش بینی ها نشان دهید و متری های پشتیبانی (نقطه های خارج از لوله) را برجسته کنید.

## اصطلاحات کلیدی

| Term | What it actually means |
|------|----------------------|
| Support vectors | The training points closest to the decision boundary. The only points that determine the hyperplane |
| Margin | The distance between the decision boundary and the nearest support vectors. SVMs maximize this |
| Hinge loss | $\max(0, 1 - y f(x))$. Zero when correctly classified and outside the margin. Linear penalty otherwise |
| C parameter | Trade-off between margin width and classification errors. Large $C$ = narrow margin, small $C$ = wide margin |
| Soft margin | SVM formulation that allows margin violations via slack variables. Handles non-separable data |
| Kernel trick | Computing dot products in a high-dimensional feature space without explicitly mapping to that space |
| Linear kernel | $K(x, z) = x \cdot z$. Equivalent to standard dot product. For linearly separable data |
| RBF kernel | $K(x, z) = \exp(-\gamma \Vert x-z \Vert^2)$. Maps to infinite dimensions. Learns any smooth boundary |
| Polynomial kernel | $K(x, z) = (x \cdot z + c)^d$. Maps to a feature space of polynomial combinations |
| Dual formulation | Reformulation of the SVM problem that depends only on dot products between data points. Enables kernels |
| SVR | Support Vector Regression. Fits an epsilon-tube around the data. Points inside the tube have zero loss |
| Slack variables | $\xi_i$: measures how much a point violates the margin. Zero for correctly classified points outside margin |
| Maximum margin | The principle of choosing the hyperplane that maximizes the distance to the nearest points of each class |

## خواندن بیشتر

- [Vapnik: The Nature of Statistical Learning Theory (1995)](https://link.springer.com/book/10.1007/978-1-4757-3264-1)- متن اساسی در مورد SVM و یادگیری آماری
- [Cortes & Vapnik: Support-vector networks (1995)](https://link.springer.com/article/10.1007/BF00994018)- ورق اصلی SVM
- [Platt: Sequential Minimal Optimization (1998)](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)- الگوریتم SMO که آموزش SVM را عملی کرد
- [scikit-learn SVM documentation](https://scikit-learn.org/stable/modules/svm.html)- راهنمای عملی با جزئیات اجرای
- [LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)- کتابخانه C++ پشت اکثر پیاده سازی های SVM

# متریسه ها و عملیات

> هر شبکه عصبی فقط ضرب ماتریکس با مراحل اضافی است.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## اهداف یادگیری

- یک کلاس ماتریکس با عملیات های هوشمند عنصر، ضرب ماتریکس، انتقال، تعیین کننده و معکوس بسازید
- ضربات بر اساس عناصر را از ضربات ماتریکس تشخیص دهید و توضیح دهید که هر کدام چه زمانی اعمال می شود
- پیاده سازی یک لایه شبکه عصبی ضخیم (`relu(W @ x + b)`) تنها با استفاده از کلاس ماتریکس از ابتدا
- قوانین پخش و نحوه عملکرد اضافه تعصب در چارچوب های شبکه عصبی را توضیح دهید

## مشکل

شما می خواهید یک شبکه عصبی بسازید. شما کد را می خوانید و این را می بینید:

```
output = activation(weights @ input + bias)
```

اون`@`ضرب ماتریکس است.`weights`. یک ماتریکس هستند`input`اگر نمی دانید که این عملیات چه کار می کنند، این خط جادویی است. اگر می دانید، این تمام عبور جلو یک لایه در سه عملیات است.

هر تصویر که مدل شما پردازش می کند یک ماتریس از ارزش های پیکسل است. هر کلمه ای که در آن گنجانده می شود یک ویکتور است. هر لایه ای از هر شبکه عصبی یک تحول ماتریس است. شما نمی توانید سیستم های هوش مصنوعی را بدون تسلط در عملیات ماتریس بسازید، به همان شیوه که نمی توانید کد را بدون درک متغیرها بنویسید.

این درس این روانی را از ابتدا می سازد.

## مفهوم

### متری: لیست های ترتیب شده از اعداد

ویکتور یک لیست از اعداد با جهت و بزرگی است. در AI، ویکتورها نقاط داده، ویژگی ها یا پارامترها را نشان می دهند.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

یک متری دو بعدی`[3, 4]`نقاط مربوط به نقاط هماهنگی (3, 4) در یک خط است. طول آن (بخش) 5 (ت مثلث 3-4-5) است.

### ماتریس: شبکه های اعداد

یک ماتریکس یک شبکه دو بعدی است. ردیفها و ستون ها.$m \times n$ماتریکس داره$m$صف ها و$n$ستون ها

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \text{2x3 matrix (2 rows, 3 columns)}
```

در شبکه های عصبی، ماتریس های وزن متریزه های ورودی را به متریزهای خروجی تبدیل می کنند. یک لایه با 784 ورودی و 128 خروجی از ماتریس وزن 128x784 استفاده می کند.

### چرا شکل ها مهم هستند

ضرب ماتریکس یک قانون سختگیرانه دارد:`(m x n) @ (n x p) = (m x p)`ابعاد داخلي بايد با هم مطابقت داشته باشه

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

اگه در PyTorch خطا عدم مطابقت شکل رو پيدا کني، اين دليلش هست.

### نقشه عملیات

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | Element-wise combine | Adding bias to output |
| Scalar multiply | Scale every element | Learning rate * gradients |
| Matrix multiply | Transform vectors | Layer forward pass |
| Transpose | Flip rows and columns | Backpropagation |
| Determinant | Single number summary | Checking invertibility |
| Inverse | Undo a transformation | Solving linear systems |
| Identity | Do-nothing matrix | Initialization, residual connections |

### ضرب عنصر با ماتریس

این تفاوت شروع به شروع کردن را به طور مداوم به عقب می اندازد.

از نظر عنصر: موقعیت های مشابه را چند برابر کنید. هر دو ماتریس باید شکل مشابهی داشته باشند.

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} * \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}
```

ضرب ماتریکس: محصولات نقطه ای از ردیف ها و ستون ها. ابعاد داخلی باید مطابق باشد.

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} @ \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
```

عملیات های مختلف، نتایج مختلف، قوانین مختلف.

### پخش

وقتی یک ویکتور تعصب را به یک ماتریس از خروجی اضافه می کنید، اشکال با هم مطابقت نمی کند. پخش، آرایه کوچکتر را برای تناسب می کند.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

هر چارچوب مدرن این کار را به طور خودکار انجام می دهد. درک آن باعث می شود در هنگام شکل های اشتباه اما کد اجرا شود، از سردرگمی جلوگیری شود.

```figure
vector-projection
```

## آن را بسازید

### مرحله اول: کلاس ویکتور

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### مرحله 2: کلاس ماتریکس با عملیات هسته ای

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### مرحله سوم: تا کارش رو ببین

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### مرحله 4: اتصال به شبکه های عصبی

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

این یک لایه گنده است:`output = relu(W @ x + b)`هر لایه کثیف در هر شبکه عصبی دقیقاً این کار را می کند.

## ازش استفاده کن

NumPy همه چیز رو در خط های کمتر و در دستورات بزرگتر سریع تر انجام میده

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

.`@`آپراتور در تماس های پایتون`__matmul__`NumPy با روتین های بهینه شده BLAS نوشته شده در C و Fortran اجرا می کند. همان ریاضیات، 100 برابر سریعتر.

پخش در NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy به طور خودکار تعصب یک بعدی را در هر دو ردیف پخش می کند. این نحوه کار اضافه تعصب در هر چارچوب شبکه عصبی است.

## -باده

این درس به آموزش عملیات ماتریس از طریق حس هندسی کمک می کند.`outputs/prompt-matrix-operations.md`. .

کلاس ماتریکس که اینجا ساخته شده پایه ی شبکه عصبی کوچک است که در مرحله سوم، درس ۱۰ ساخته ایم.

## تمرینات

1. **Verify the inverse.**چندانش`A @ A.inverse_2x2()`و تایید کنید که شما میتریس هویت را بدست آورده اید. با سه ماتریس 2x2 مختلف امتحان کنید. چه اتفاقی می افتد وقتی تعیین کننده صفر است؟

2. **Implement 3x3 inverse.**کلاس ماتریکس را برای محاسبه معکوس برای ماتریکس 3x3 با استفاده از روش جوب کنید. آن را با NumPy آزمایش کنید `np.linalg.inv`. .

3. **Build a two-layer network.**با استفاده از فقط کلاس ماتریکس خود (هیچ NumPy) ، یک شبکه عصبی دو لایه ایجاد کنید: ورودی (3) -> پنهان (4) -> خروجی (2). وزنه های تصادفی را آغاز کنید، یک گذرگاه پیش رو اجرا کنید و تمام اشکال را درست کنید.

### اشاره ها

**Exercise 1.**چندانش`A @ A.inverse_2x2()`برای سه ماتریس و مقایسه هر سلول با هویت در `1e-9`. برای ماتریس با تعیین کننده 0 مانند [[1, 2], [2, 4]], `inverse_2x2`یک`ValueError`، و این نتیجه انتظار می رود.

**Exercise 2.**ماتریس 3x3 کوفاکتور را محاسبه کنید، آن را برای دریافت adjugate منتقل کنید، سپس هر سلول را با  تقسیم کنید`determinant()`- اینو چک کن`A @ A_inv`بهشون ميگه که هر سلول همديگه`np.linalg.inv(A)`در داخل`1e-9`. .

**Exercise 3.**قبل از چندان شکل هر ماتریکس را بنویسید: `W1`4x3 و `W2`2x4 است، مثل `demo_neural_network_layer`. یک ورودی 3×1 باید یک ویکتور 4×1 پنهان و یک ورودی 2×1 را بدهد.

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Vector | "An arrow" | An ordered list of numbers. In AI: a point in high-dimensional space. |
| Matrix | "A table of numbers" | A linear transformation. It maps vectors from one space to another. |
| Matrix multiply | "Just multiply the numbers" | Dot products between every row of the first matrix and every column of the second. Order matters. |
| Transpose | "Flip it" | Swap rows and columns. Turns an $m \times n$ matrix into $n \times m$. Critical in backpropagation. |
| Determinant | "Some number from the matrix" | Measures how much the matrix scales area (2D) or volume (3D). Zero means the transformation crushes a dimension. |
| Inverse | "Undo the matrix" | The matrix that reverses the transformation. Only exists when the determinant is not zero. |
| Identity matrix | "The boring matrix" | The matrix equivalent of multiplying by 1. Used in residual connections (ResNets). |
| Broadcasting | "Magic shape fixing" | Stretching a smaller array to match a larger one by repeating along missing dimensions. |
| Element-wise | "Regular multiplication" | Multiply matching positions. Both arrays must have the same shape (or be broadcastable). |

## خواندن بیشتر

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)- حس بصری برای هر عملیاتی که در اینجا پوشش داده شده
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)- قوانین دقیق NumPy
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)- مرجع خلاصه ای برای الجبر خطی خاص ML

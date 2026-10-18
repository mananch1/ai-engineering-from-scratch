# Các vector, matrix & Operations

> Mỗi mạng thần kinh chỉ là sự nhân số của một số matrix với các bước bổ sung.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## Mục tiêu học tập

- Xây dựng một lớp Matrix với các hoạt động thông minh về các yếu tố, nhân số matrix, chuyển giao, xác định và ngược lại
- Hóa ra sự nhân nhân bằng các yếu tố và sự nhân tử và giải thích khi nào mỗi ứng dụng
- Thực hiện một lớp mạng thần kinh dày đặc duy nhất (`relu(W @ x + b)`) chỉ sử dụng lớp Matrix từ đầu
- Giải thích các quy tắc phát sóng và cách việc bổ sung thiên vị hoạt động trong các khung mạng thần kinh

## Vấn đề

Bạn muốn xây dựng một mạng lưới thần kinh. Bạn đọc mã và thấy điều này:

```
output = activation(weights @ input + bias)
```

Đó là`@`là nhân tử liệu.`weights`là một matrix.`input`nếu bạn không biết các hoạt động đó làm gì, đường này là phép thuật. nếu bạn biết, nó là toàn bộ chuyển tiếp về phía trước của một lớp trong ba hoạt động.

Mỗi hình ảnh mà mô hình của bạn xử lý là một matrix của các giá trị pixel. Mỗi từ nhúng là một vector. Mỗi lớp của mỗi mạng thần kinh là một sự chuyển đổi matrix. Bạn không thể xây dựng hệ thống AI mà không biết matrix một cách thông thạo, giống như bạn không thể viết mã mà không hiểu các biến.

Bài học này giúp bạn có sự thịnh vượng từ đầu.

## Khái niệm

### Các vector: danh sách số được sắp xếp

Một vector là một danh sách số với một hướng và quy mô. Trong AI, vector đại diện cho các điểm dữ liệu, tính năng hoặc tham số.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

Một vector 2D `[3, 4]`chỉ ra các phối hợp (3, 4) trên một máy bay. chiều dài (lượng lớn) của nó là 5 (lối ba - 3-4 - 5).

### Matrix: lưới số

Một matrix là một lưới 2D. Dòng và cột.$m \times n$Matrix có$m$hàng và $n$Các cột.

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \text{2x3 matrix (2 rows, 3 columns)}
```

Trong mạng thần kinh, các matrices trọng lượng chuyển đổi các vector đầu vào thành vector đầu ra. Một lớp có 784 đầu vào và 128 đầu ra sử dụng một matrices trọng lượng 128x784.

### Tại sao hình dạng quan trọng

Sự nhân số của matrix có một quy tắc nghiêm ngặt:`(m x n) @ (n x p) = (m x p)`- Các kích thước bên trong phải phù hợp.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

Nếu bạn nhận được một sai lầm không phù hợp hình dạng trong PyTorch, đây là lý do tại sao.

### Bản đồ hoạt động

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | Element-wise combine | Adding bias to output |
| Scalar multiply | Scale every element | Learning rate * gradients |
| Matrix multiply | Transform vectors | Layer forward pass |
| Transpose | Flip rows and columns | Backpropagation |
| Determinant | Single number summary | Checking invertibility |
| Inverse | Undo a transformation | Solving linear systems |
| Identity | Do-nothing matrix | Initialization, residual connections |

### Sự nhân từ các yếu tố so với số tử

Sự phân biệt này thường khiến người mới bắt đầu gặp khó khăn.

Điểm yếu tố: nhân các vị trí phù hợp. Cả hai matrices phải có cùng hình dạng.

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} * \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}
```

Tần số tử liệu: sản phẩm điểm của các hàng và cột.

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} @ \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
```

Các hoạt động khác nhau, kết quả khác nhau, quy tắc khác nhau.

### Truyền thông

Khi bạn thêm một vector thiên vị vào một số lượng đầu ra, các hình dạng không phù hợp.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Mỗi khung hiện đại tự động làm điều này. hiểu nó ngăn ngừa sự nhầm lẫn khi hình dạng dường như sai nhưng mã chạy.

```figure
vector-projection
```

## Hãy xây dựng nó

### Bước 1: lớp vector

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

### Bước 2: lớp matrix với các hoạt động cốt lõi

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

### Bước 3: Hãy xem nó hoạt động

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

### Bước 4: Kết nối với mạng lưới thần kinh

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

Đây là một lớp dày đặc đơn:`output = relu(W @ x + b)`Mỗi lớp dày đặc trong mỗi mạng thần kinh đều làm điều này.

## Sử dụng nó

NumPy làm mọi thứ ở trên trong ít đường và các thứ tự lớn hơn nhanh hơn.

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

- `@`operator trong Python call `__matmul__`NumPy thực hiện nó với các thói quen BLAS tối ưu được viết bằng C và Fortran.

Truyền thông trong NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy tự động phát sóng sự thiên vị 1D qua cả hai hàng. Đây là cách việc bổ sung thiên vị hoạt động trong mọi khung mạng thần kinh.

## Chuyển nó

Bài học này tạo ra một lời nhắc để dạy các hoạt động matrix thông qua trực giác hình học.`outputs/prompt-matrix-operations.md`- Tôi không biết.

Các lớp Matrix được xây dựng ở đây là nền tảng cho các hệ thống mạng lưới thần kinh nhỏ chúng tôi xây dựng trong giai đoạn 3, Bài học 10.

## Các bài tập

1. **Verify the inverse.**Tăng nhiều`A @ A.inverse_2x2()`và xác nhận bạn có được các mã số danh tính. thử với ba mã số 2x2 khác nhau.

2. **Implement 3x3 inverse.**Lớn thêm lớp Matrix để tính toán ngược cho các matrices 3x3 bằng cách sử dụng phương pháp adjugate.`np.linalg.inv`- Tôi không biết.

3. **Build a two-layer network.**Sử dụng chỉ lớp Matrix của bạn (không có NumPy), tạo một mạng thần kinh hai lớp: đầu vào (3) -> ẩn (4) -> đầu ra (2). Tạo ra các trọng lượng ngẫu nhiên, chạy một bước đi về phía trước và xác minh tất cả các hình dạng đều chính xác.

### Những gợi ý

**Exercise 1.**Tăng nhiều`A @ A.inverse_2x2()`cho ba matrices và so sánh mỗi tế bào với danh tính trong `1e-9`. Đối với một số liệu với số đo 0, chẳng hạn như [[1, 2], [2, 4]], `inverse_2x2`làm tăng một `ValueError`, và đây là kết quả mong đợi.

**Exercise 2.**Xét toán các matrix cofactor 3x3, chuyển nó để có được adjugate, sau đó chia từng tế bào bằng `determinant()`Nhìn xem.`A @ A_inv`cho nhận dạng và mỗi tế bào phù hợp `np.linalg.inv(A)`trong `1e-9`- Tôi không biết.

**Exercise 3.**Viết hình dạng của mỗi matrix trước khi bạn nhân: `W1`là 4x3 và `W2`là 2x4, như trong `demo_neural_network_layer`. Một đầu vào 3x1 phải cung cấp một vector ẩn 4x1 và một đầu ra 2x1.

## Các điều khoản chính

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

## Đọc thêm

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)- trực giác thị giác cho mỗi hoạt động được đề cập ở đây
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)- các quy tắc chính xác của NumPy
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)- tham chiếu ngắn gọn cho toán toán tuyến tính cụ thể ML

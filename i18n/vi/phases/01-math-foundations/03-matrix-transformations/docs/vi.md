# Các biến đổi matrix

> Một matrix là một máy tạo ra không gian, học được nó làm gì cho mọi điểm, và bạn hiểu được toàn bộ sự biến đổi.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lessons 01-02 (Linear Algebra Intuition, Vectors & Matrices Operations)
**Time:** ~75 minutes

## Mục tiêu học tập

- Xây dựng các matrices xoay, quy mô, cắt và phản xạ và áp dụng chúng cho các điểm 2D và 3D
- Sắp xếp nhiều biến đổi bằng cách nhân tử và xác minh rằng thứ tự quan trọng
- Xét các giá trị riêng và vector riêng của các matrices 2x2 từ phương trình đặc trưng
- Giải thích lý do tại sao các giá trị riêng xác định hướng PCA, ổn định RNN và hành vi nhóm phân loại quang phổ

## Vấn đề

Bạn đọc về PCA và thấy "đ tìm các vector chính của matrix covariance". Bạn đọc về sự ổn định của mô hình và thấy "đánh giá xem tất cả các giá trị chính có kích thước nhỏ hơn 1 không". Bạn đọc về tăng dữ liệu và thấy "được áp dụng một xoay ngẫu nhiên". Không có điều gì trong những điều này có ý nghĩa cho đến khi bạn hiểu những gì các matrix làm cho không gian theo hình học.

Các matrix không chỉ là lưới số. Chúng là máy không gian. Một matrix quay quay điểm. Một matrix quy mô kéo dài chúng. Một matrix cắt nghiêng chúng. Mỗi biến đổi một mạng thần kinh áp dụng cho dữ liệu là một trong những hoạt động này hoặc một thành phần của chúng. Bài học này làm cho các hoạt động đó cụ thể.

## Khái niệm

### Các biến đổi như các matrices

Mỗi biến đổi tuyến tính trong 2D có thể được viết như một matrix 2x2. Matrix cho bạn biết chính xác các vector cơ sở [1, 0] và [0, 1] kết thúc ở đâu. Mọi thứ khác tiếp theo.

```mermaid
graph LR
    subgraph Before["Standard Basis"]
        e1["e1 = [1, 0] (along x)"]
        e2["e2 = [0, 1] (along y)"]
    end
    subgraph Transform["Matrix M"]
        M["M = columns are new basis vectors"]
    end
    subgraph After["After Transformation M"]
        e1p["e1' = new x-basis"]
        e2p["e2' = new y-basis"]
    end
    e1 --> M --> e1p
    e2 --> M --> e2p
```

### Chuyển đổi

Một vòng quay 2D theo góc $\theta$Nó giữ khoảng cách và góc nguyên vẹn. Nó di chuyển mọi điểm dọc theo một vòng cung tròn.

```mermaid
graph LR
    subgraph Before["Before Rotation"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Rot["Rotate 45 degrees"]
        R["R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]"]
    end
    subgraph After["After Rotation"]
        Ap["A'(0.71, 2.12)"]
        Bp["B'(-1.41, 1.41)"]
    end
    A --> R --> Ap
    B --> R --> Bp
```

Trong 3D, bạn xoay quanh một trục. Mỗi trục có trục xoay riêng của nó:

```math
\begin{aligned}
R_z(\theta) &= \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} \quad \text{Rotate around z-axis (x-y plane spins, z stays)} \\[6pt]
R_x(\theta) &= \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix} \quad \text{Rotate around x-axis (y-z plane spins, x stays)} \\[6pt]
R_y(\theta) &= \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix} \quad \text{Rotate around y-axis (x-z plane spins, y stays)}
\end{aligned}
```

### Tăng quy mô

Scaling kéo dài hoặc nén dọc theo mỗi trục độc lập.

```mermaid
graph LR
    subgraph Before["Before Scaling"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Scale["Scale sx=2, sy=0.5"]
        S["S = [[2, 0], [0, 0.5]]"]
    end
    subgraph After["After Scaling"]
        Ap["A'(4, 0.5)"]
        Bp["B'(0, 1)"]
    end
    A --> S --> Ap
    B --> S --> Bp
```

### Trẻ

Việc cắt nghiêng một trục trong khi giữ trục kia cố định. Nó biến hình chữ nhật thành song song.

```mermaid
graph LR
    subgraph Before["Before Shear"]
        A["A(1, 0)"]
        B["B(0, 1)"]
    end
    subgraph Shear["Shear in x, k=1"]
        Sh["Shx = [[1, k], [0, 1]]"]
    end
    subgraph After["After Shear"]
        Ap["A(1, 0) unchanged"]
        Bp["B'(1, 1) shifted"]
    end
    A --> Sh --> Ap
    B --> Sh --> Bp
```

Các matrices cắt:
- `Shx = [[1, k], [0, 1]]`thay đổi$x$bởi $k \cdot y$
- `Shy = [[1, 0], [k, 1]]`thay đổi$y$bởi $k \cdot x$

### Nhận xét

Nhấp phản chiếu các điểm qua một trục hoặc đường.

```mermaid
graph LR
    subgraph Before["Before Reflection"]
        A["A(2, 1)"]
    end
    subgraph Reflect["Reflect across y-axis"]
        R["[[-1, 0], [0, 1]]"]
    end
    subgraph After["After Reflection"]
        Ap["A'(-2, 1)"]
    end
    A --> R --> Ap
```

Các matrix phản xạ:
- Nhấp vào trục y: `[[-1, 0], [0, 1]]`
- Nhấp vào trục x: `[[1, 0], [0, -1]]`

### Thành phần: chuyển đổi chuỗi

Sử dụng biến đổi A sau đó là B giống như nhân các matrices của chúng: `result = B @ A @ point`Trò quay sau đó quy mô sẽ đưa ra kết quả khác với quy mô sau đó quay.

```mermaid
graph LR
    subgraph Path1["Rotate 90 then Scale (2, 0.5)"]
        P1["(1, 0)"] -->|"Rotate 90"| P2["(0, 1)"] -->|"Scale"| P3["(0, 0.5)"]
    end
```

Thành phần: `S @ R = [[0, -2], [0.5, 0]]`

```mermaid
graph LR
    subgraph Path2["Scale (2, 0.5) then Rotate 90"]
        Q1["(1, 0)"] -->|"Scale"| Q2["(2, 0)"] -->|"Rotate 90"| Q3["(0, 2)"]
    end
```

Thành phần: `R @ S = [[0, -0.5], [2, 0]]`

Kết quả khác nhau. nhân số tử hình không phải là chuyển đổi.

### Giá trị và vector riêng

Hầu hết các vector thay đổi hướng khi một matrix chạm vào chúng. Eigenvectors là đặc biệt: matrix chỉ quy mô chúng, không bao giờ xoay chúng.

```
A @ v = lambda * v

v is the eigenvector (direction that survives)
lambda is the eigenvalue (how much it stretches)

Example: A = | 2  1 |
             | 1  2 |

Eigenvector [1, 1] with eigenvalue 3:
  A @ [1,1] = [3, 3] = 3 * [1, 1]     (same direction, scaled by 3)

Eigenvector [1, -1] with eigenvalue 1:
  A @ [1,-1] = [1, -1] = 1 * [1, -1]  (same direction, unchanged)
```

Các hình tử liệu kéo dài không gian bằng 3x dọc theo [1, 1] và giữ cho [1, -1] không thay đổi.

### Thành phần của nó

Nếu một matrix có$n$các vector tự độc lập tuyến tính, nó có thể bị phân hủy:

```
A = V @ D @ V^(-1)

V = matrix whose columns are eigenvectors
D = diagonal matrix of eigenvalues
V^(-1) = inverse of V

This says: rotate into eigenvector coordinates, scale along each axis, rotate back.
```

### Tại sao giá trị bản chất quan trọng

**PCA.**Các vector chính của matrix covariance là các thành phần chính. các giá trị chính cho bạn biết mỗi thành phần nắm bắt bao nhiêu sự biến động.$k$, và bạn có sự giảm chiều kích.

**Stability.**Trong các mạng tái phát và hệ thống động, các giá trị tự với độ lớn > 1 gây ra các đầu ra nổ. độ lớn < 1 gây ra chúng biến mất. Đây là vấn đề biến mất / bùng nổ gradient được nêu trong một câu.

**Spectral methods.**Các mạng lưới thần kinh đồ họa sử dụng giá trị riêng của ma trận lân cận. Nhóm phân tử quang phổ sử dụng giá trị riêng của Laplacian. Các vector riêng cho thấy cấu trúc của đồ họa.

### Định nghĩa là yếu tố quy mô khối lượng

Các định đo của một số liệu chuyển hóa cho bạn biết nó quy mô bao nhiêu diện tích (2D) hoặc khối lượng (3D).

```
det = 1:   area preserved (rotation)
det = 2:   area doubled
det = 0:   space crushed to lower dimension (singular)
det = -1:  area preserved but orientation flipped (reflection)

| det(Rotation) | = 1        (always)
| det(Scale sx, sy) | = sx * sy
| det(Shear) | = 1           (area preserved)
| det(Reflection) | = -1     (orientation flipped)
```

```figure
matrix-transform
```

## Hãy xây dựng nó

### Bước 1: Các matrices chuyển đổi từ đầu (Python)

```python
import math

def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

def reflection_x():
    return [[1, 0], [0, -1]]

def reflection_y():
    return [[-1, 0], [0, 1]]

def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def mat_mul(a, b):
    rows_a, cols_b = len(a), len(b[0])
    cols_a = len(a[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]

point = [1.0, 0.0]
angle = math.pi / 4

rotated = mat_vec_mul(rotation_2d(angle), point)
print(f"Rotate (1,0) by 45 deg: ({rotated[0]:.4f}, {rotated[1]:.4f})")

scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
print(f"Scale (1,1) by (2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")

sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
print(f"Shear (1,1) kx=1: ({sheared[0]:.1f}, {sheared[1]:.1f})")

reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
print(f"Reflect (2,1) across y: ({reflected[0]:.1f}, {reflected[1]:.1f})")
```

### Bước 2: Thành phần của các biến đổi

```python
R = rotation_2d(math.pi / 2)
S = scaling_2d(2, 0.5)

rotate_then_scale = mat_mul(S, R)
scale_then_rotate = mat_mul(R, S)

point = [1.0, 0.0]
result1 = mat_vec_mul(rotate_then_scale, point)
result2 = mat_vec_mul(scale_then_rotate, point)

print(f"Rotate 90 then scale: ({result1[0]:.2f}, {result1[1]:.2f})")
print(f"Scale then rotate 90: ({result2[0]:.2f}, {result2[1]:.2f})")
print(f"Same? {result1 == result2}")
```

### Bước 3: Giá trị riêng từ đầu (2x2)

Đối với một matrix 2x2 `[[a, b], [c, d]]`, giá trị riêng giải quyết phương trình đặc trưng: $\lambda^2 - (a+d)\lambda + (ad - bc) = 0$- Tôi không biết.

```python
def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]

A = [[2, 1], [1, 2]]
vals = eigenvalues_2x2(A)
print(f"Matrix: {A}")
print(f"Eigenvalues: {vals[0]:.4f}, {vals[1]:.4f}")

for val in vals:
    vec = eigenvector_2x2(A, val)
    result = mat_vec_mul(A, vec)
    scaled = [val * vec[0], val * vec[1]]
    print(f"  lambda={val:.1f}, v={[round(x,4) for x in vec]}")
    print(f"    A@v = {[round(x,4) for x in result]}")
    print(f"    l*v = {[round(x,4) for x in scaled]}")
```

### Bước 4: Định nghĩa là yếu tố quy mô khối lượng

```python
def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

print(f"det(rotation 45) = {det_2x2(rotation_2d(math.pi/4)):.4f}")
print(f"det(scale 2,3)   = {det_2x2(scaling_2d(2, 3)):.1f}")
print(f"det(shear kx=1)  = {det_2x2(shearing_2d(1, 0)):.1f}")
print(f"det(reflect y)   = {det_2x2(reflection_y()):.1f}")

singular = [[1, 2], [2, 4]]
print(f"det(singular)     = {det_2x2(singular):.1f}")
print("Singular: columns are proportional, space collapses to a line.")
```

## Sử dụng nó

NumPy xử lý tất cả những điều này với các thói quen tối ưu hóa.

```python
import numpy as np

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

point = np.array([1.0, 0.0])
print(f"Rotate (1,0) by 45 deg: {R @ point}")

S = np.diag([2.0, 3.0])
composed = S @ R
print(f"Scale(2,3) after Rotate(45): {composed @ point}")

A = np.array([[2, 1], [1, 2]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"  A @ v{i} = {A @ v}, lambda * v{i} = {lam * v}")

print(f"\ndet(R) = {np.linalg.det(R):.4f}")
print(f"det(S) = {np.linalg.det(S):.1f}")

B = np.array([[3, 1], [0, 2]], dtype=float)
vals, vecs = np.linalg.eig(B)
D = np.diag(vals)
V = vecs
reconstructed = V @ D @ np.linalg.inv(V)
print(f"\nEigendecomposition A = V @ D @ V^-1:")
print(f"Original:\n{B}")
print(f"Reconstructed:\n{reconstructed}")
```

### Chuyển đổi 3D với NumPy

```python
def rotation_3d_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotation_3d_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

point_3d = np.array([1.0, 0.0, 0.0])
rotated_z = rotation_3d_z(np.pi / 2) @ point_3d
rotated_x = rotation_3d_x(np.pi / 2) @ point_3d

print(f"\n3D point: {point_3d}")
print(f"Rotate 90 around z: {np.round(rotated_z, 4)}")
print(f"Rotate 90 around x: {np.round(rotated_x, 4)}")
```

## Chuyển nó

Bài học này xây dựng nền tảng hình học cho PCA (Phase 2) và phân tích trọng lượng mạng thần kinh. Mã eigenvalue / eigenvector được xây dựng ở đây là cùng một thuật toán hỗ trợ giảm chiều kích, phân nhóm quang phổ và phân tích ổn định trong các hệ thống ML sản xuất.

## Các bài tập

1. Lấy xoay, quy mô và cắt lên một hình vuông đơn vị (vùng góc ở [0,0], [1,0], [1,1], [0,1]). Bác in các góc biến đổi cho mỗi góc.

2. Tìm các giá trị riêng của các mã số [[4, 2], [1, 3]] bằng tay bằng cách sử dụng phương trình đặc trưng. Sau đó xác minh bằng hàm từ đầu và với NumPy.

3. Tạo một cấu trúc gồm ba biến đổi (đối tròn 30 độ, đo bằng [1,5, 0,8], cắt bằng kx=0,3) và áp dụng nó cho 8 điểm được sắp xếp trong một vòng tròn. Bác trước và sau các phối hợp. Xét định số của các matrix được tạo thành và xác minh nó bằng với sản phẩm của các định số riêng lẻ.

### Những gợi ý

**Exercise 1.**Đơn `rotation_2d`- `scaling_2d`, và`shearing_2d`đến mỗi góc với `mat_vec_mul`. Đối với sự xoay quanh, tính toán 4 chiều dài bên và 2 đường chọc trước và sau; mỗi phải giữ nguyên.

**Exercise 2.**Viết phương trình đặc trưng $\lambda^2 - (\text{trace})\lambda + \det = 0$Hãy kiểm tra hai giá trị của bạn cộng với dấu vết (7) và nhân lên số xác định (10), sau đó so sánh với `eigenvalues_2x2`và `np.linalg.eigvals`- Tôi không biết.

**Exercise 3.**Xây dựng ba bộ tử, với `shearing_2d(0.3, 0)`cho việc cắt, nhân chúng với `mat_mul`theo thứ tự ứng dụng, và áp dụng kết quả cho 8 điểm $(\cos\theta, \sin\theta)$. Chuyển quay có một định đo của 1 và cắt có một định đo của 1, vì vậy `det_2x2`của thành phần phải bằng $1.5 \times 0.8 = 1.2$- Tôi không biết.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Rotation matrix | "Spins things" | An orthogonal matrix that moves points along circular arcs while preserving distances and angles. Determinant is always 1. |
| Scaling matrix | "Makes things bigger" | A diagonal matrix that stretches or compresses independently along each axis. Determinant is the product of scale factors. |
| Shearing matrix | "Slants things" | A matrix that shifts one coordinate proportionally to another, turning rectangles into parallelograms. Determinant is 1. |
| Reflection | "Mirrors things" | A matrix that flips space across an axis or plane. Determinant is -1. |
| Composition | "Do two things" | Multiplying transformation matrices to chain operations. Order matters: B @ A means apply A first, then B. |
| Eigenvector | "Special direction" | A direction that the matrix only scales, never rotates. The transformation's fingerprint. |
| Eigenvalue | "How much it stretches" | The scalar factor by which the matrix scales its eigenvector. Can be negative (flip) or complex (rotation). |
| Eigendecomposition | "Break the matrix apart" | Writing a matrix as $V D V^{-1}$, separating it into its fundamental scaling directions and magnitudes. |
| Determinant | "A single number from a matrix" | The factor by which the transformation scales area (2D) or volume (3D). Zero means the transformation is irreversible. |
| Characteristic equation | "Where eigenvalues come from" | $\det(A - \lambda I) = 0$. The polynomial whose roots are the eigenvalues. |

## Đọc thêm

- [3Blue1Brown: Linear Transformations](https://www.3blue1brown.com/lessons/linear-transformations)-- trực giác thị giác về cách các trền tạo định hình lại không gian
- [3Blue1Brown: Eigenvectors and Eigenvalues](https://www.3blue1brown.com/lessons/eigenvalues)-- giải thích trực quan tốt nhất về ý nghĩa của các vector tự
- [MIT 18.06 Lecture 21: Eigenvalues and Eigenvectors](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)- Điều trị cổ điển của Gilbert Strang

# Quy tắc chuỗi & phân biệt tự động

> Quy tắc chuỗi là động cơ đằng sau mỗi mạng thần kinh học được.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 1, Lesson 04 (Derivatives & Gradients)
**Time:** ~90 minutes

## Mục tiêu học tập

- Xây dựng một động cơ tự động cấp thấp nhất (kiểu giá trị) ghi lại hoạt động và tính toán độ sốc thông qua chế độ tự động ngược
- Thực hiện các đường đi về phía trước và ngược qua biểu đồ tính toán bằng cách sử dụng phân loại topological
- XOR xây dựng và huấn luyện một perceptron đa lớp chỉ sử dụng động cơ tự động từ đầu
- Kiểm tra độ chính xác tự động bằng cách kiểm tra gradient đối với sự khác biệt số hữu hạn

## Vấn đề

Bạn có thể tính toán các phái sinh của các hàm đơn giản. Nhưng một mạng thần kinh không phải là một hàm đơn giản. Nó là hàng trăm hàm được tạo thành với nhau: tử liệu nhân, thêm thiên vị, áp dụng kích hoạt, tử liệu nhân lần nữa, softmax, mất entropy chéo.

Để đào tạo mạng, bạn cần gradient của sự mất liên quan đến mỗi trọng lượng. Làm điều này bằng tay là không thể cho hàng triệu tham số. Làm nó về số (các khác biệt hữu hạn) là quá chậm.

Quy tắc chuỗi cho bạn toán học. phân biệt tự động cho bạn thuật toán. Cùng với nhau chúng cho phép bạn tính toán gradient chính xác thông qua các thành phần tùy tiện của các hàm trong thời gian tương xứng với một lần đi trước duy nhất.

Đây là cách PyTorch, TensorFlow và JAX hoạt động. Bạn sẽ xây dựng một phiên bản miniatur từ đầu.

## Khái niệm

### Quy tắc chuỗi

Nếu$y = f(g(x))$, dẫn xuất của $y$đối với $x$là:

```math
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)
```

Bội số các phái sinh dọc theo chuỗi. Mỗi liên kết đóng góp phái sinh địa phương của nó.

Ví dụ: $y = \sin(x^2)$

```math
\begin{aligned}
g(x) &= x^2 \qquad & g'(x) &= 2x \\
f(g) &= \sin(g) \qquad & f'(g) &= \cos(g) \\[4pt]
\frac{dy}{dx} &= \cos(x^2) \cdot 2x
\end{aligned}
```

Đối với các thành phần sâu hơn, chuỗi mở rộng:

```math
\begin{aligned}
y &= f(g(h(x))) \\[4pt]
\frac{dy}{dx} &= f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)
\end{aligned}
```

Mỗi lớp trong một mạng thần kinh là một liên kết trong chuỗi này.

### Hình đồ tính toán

Một biểu đồ tính toán làm cho quy tắc chuỗi hình ảnh. Mỗi hoạt động trở thành một nút. Dữ liệu chảy về phía trước thông qua biểu đồ.

**Forward pass (compute values):**

```mermaid
graph TD
    x1["x1 = 2"] --> mul["* (multiply)"]
    x2["x2 = 3"] --> mul
    mul -->|"a = 6"| add["+ (add)"]
    b["b = 1"] --> add
    add -->|"c = 7"| relu["relu"]
    relu -->|"y = 7"| y["output y"]
```

**Backward pass (compute gradients):**

```mermaid
graph TD
    dy["dy/dy = 1"] -->|"relu'(c)=1 since c>0"| dc["dy/dc = 1"]
    dc -->|"dc/da = 1"| da["dy/da = 1"]
    dc -->|"dc/db = 1"| db["dy/db = 1"]
    da -->|"da/dx1 = x2 = 3"| dx1["dy/dx1 = 3"]
    da -->|"da/dx2 = x1 = 2"| dx2["dy/dx2 = 2"]
```

Việc đi ngược áp dụng quy tắc chuỗi tại mỗi nút, lan truyền gradient từ đầu ra vào đầu vào.

### Phương thức tiến ngược

Có hai cách để áp dụng quy tắc chuỗi thông qua biểu đồ.

**Forward mode**bắt đầu từ các đầu vào và đẩy các phái sinh về phía trước.$dx/dx = 1$và lây lan thông qua mỗi hoạt động. tốt khi bạn có ít đầu vào và nhiều đầu ra.

```
Forward mode: seed dx/dx = 1, propagate forward

  x = 2       (dx/dx = 1)
  a = x^2     (da/dx = 2x = 4)
  y = sin(a)  (dy/dx = cos(a) * da/dx = cos(4) * 4 = -2.615)
```

**Reverse mode**bắt đầu ở đầu ra và kéo gradient ngược lại.$dy/dy = 1$và lan truyền qua mỗi hoạt động ngược lại. tốt khi bạn có nhiều đầu vào và ít đầu ra.

```
Reverse mode: seed dy/dy = 1, propagate backward

  y = sin(a)  (dy/dy = 1)
  a = x^2     (dy/da = cos(a) = cos(4) = -0.654)
  x = 2       (dy/dx = dy/da * da/dx = -0.654 * 4 = -2.615)
```

Các mạng thần kinh có hàng triệu đầu vào (năng lượng) và một đầu ra (sự mất mát). chế độ ngược tính toán tất cả các gradient trong một lần đi ngược.

| Mode | Seed | Direction | Best when |
|------|------|-----------|-----------|
| Forward | $dx_i/dx_i = 1$ | Input to output | Few inputs, many outputs |
| Reverse | $dy/dy = 1$ | Output to input | Many inputs, few outputs (neural nets) |

### Số hai cho chế độ đi trước

Phương thức đi trước có thể được thực hiện đẹp đẽ với số hai.$a + b\epsilon$nơi $\epsilon^2 = 0$- Tôi không biết.

```
Dual number: (value, derivative)

(2, 1) means: value is 2, derivative w.r.t. x is 1

Arithmetic rules:
  (a, a') + (b, b') = (a+b, a'+b')
  (a, a') * (b, b') = (a*b, a'*b + a*b')
  sin(a, a')         = (sin(a), cos(a)*a')
```

Cây biến đầu vào với dẫn xuất 1. dẫn xuất tự động lan truyền thông qua mỗi hoạt động.

### Xây dựng một động cơ Autograd

Một động cơ tự động cần ba thứ:

1. **Value wrapping.**Bị gói mọi số trong một đối tượng lưu trữ giá trị và độ nghiêng của nó.
2. **Graph recording.**Mỗi hoạt động ghi lại đầu vào của nó và chức năng gradient địa phương.
3. **Backward pass.**Topological sắp xếp biểu đồ, sau đó đi nó ngược lại, áp dụng quy tắc chuỗi tại mỗi nút.

Đây chính xác là PyTorch.`autograd`- Có.`torch.Tensor`lớp gói các giá trị, ghi lại các hoạt động khi `requires_grad=True`, và tính toán gradient khi bạn gọi`.backward()`- Tôi không biết.

### Làm thế nào PyTorch Autograd hoạt động dưới nắp

Khi bạn viết mã PyTorch:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # 7.0 = 2*x + 3 = 2*2 + 3
```

PyTorch bên trong:

1. Tạo ra một `Tensor`nút cho `x`với `requires_grad=True`
2. Mỗi hoạt động (`**`- `*`- `+`) tạo ra một nút mới và ghi lại hàm ngược
3. `y.backward()`kích hoạt chế độ ngược tự động thông qua biểu đồ ghi
4. Mỗi nút của nó`grad_fn`tính toán gradient địa phương và chuyển chúng sang các nút bậc cha
5. Các gradient tích lũy trong `.grad`thuộc tính thông qua bổ sung (không thay thế)

Chữ đồ họa là động (định nghĩa theo chạy). Chữ đồ họa mới được xây dựng trên mỗi bước đi về phía trước.

```figure
chain-rule
```

## Hãy xây dựng nó

### Bước 1: Kiểu giá trị

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

Mỗi người`Value`lưu trữ dữ liệu số của nó, độ nghiêng của nó (trước đầu là không), một hàm ngược, và chỉ dẫn đến các nút trẻ em đã tạo ra nó.

### Bước 2: Các hoạt động toán học với theo dõi gradient

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
```

Mỗi hoạt động tạo ra một kết thúc biết cách tính toán gradient địa phương và nhân bằng gradient trên dòng (`out.grad`).`+=`xử lý trường hợp một giá trị được sử dụng trong nhiều hoạt động.

### Bước 3: Đi ngược

```python
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

Topological sort đảm bảo gradient của mỗi nút được tính toán đầy đủ trước khi nó lây lan cho con cái của nó.$dy/dy = 1$().

### Bước 4: Nhiều hoạt động hơn cho một động cơ hoàn chỉnh

Các lớp giá trị cơ bản xử lý cộng, nhân và relu. Một động cơ tự động thực sự cần nhiều hơn. Dưới đây là các hoạt động bạn cần để xây dựng mạng thần kinh:

```python
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
```

**Why each operation matters:**

| Operation | Backward rule | Used in |
|-----------|--------------|---------|
| `__sub__` | Reuses add + neg | Loss computation (pred - target) |
| `__pow__` | $n \cdot x^{n-1}$ | Polynomial activations, MSE ($\text{error}^2$) |
| `__truediv__` | Reuses mul + pow(-1) | Normalization, learning rate scaling |
| `exp` | $\exp(x) \cdot \text{upstream}$ | Softmax, log-likelihood |
| `log` | $(1/x) \cdot \text{upstream}$ | Cross-entropy loss, log probabilities |
| `tanh` | $(1 - \tanh^2) \cdot \text{upstream}$ | Classic activation function |

Phần thông minh:`__sub__`và `__truediv__`được định nghĩa theo các hoạt động hiện có. Họ nhận được gradient chính xác miễn phí vì quy tắc chuỗi được tạo ra thông qua các hoạt động cộng/mul/pow cơ bản.

### Bước 5: Mini MLP từ đầu

Với lớp giá trị hoàn chỉnh, bạn có thể xây dựng một mạng lưới thần kinh không PyTorch, không NumPy, chỉ có giá trị và quy tắc chuỗi.

```python
import random

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

A `Neuron`tính toán`tanh(w1*x1 + w2*x2 + ... + b)`. A `Layer`là một danh sách các tế bào thần kinh.`MLP`Mỗi trọng lượng là một`Value`, vì vậy gọi `loss.backward()`truyền gradient đến mọi tham số.

**Training on XOR:**

```python
random.seed(42)
model = MLP([2, 4, 1])  # 2 inputs, 4 hidden neurons, 1 output

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR pattern (using -1/1 for tanh)

for step in range(100):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 20 == 0:
        print(f"step {step:3d}  loss = {loss.data:.4f}")

print("\nPredictions after training:")
for x, y in zip(xs, ys):
    print(f"  input={x}  target={y:2d}  pred={model(x).data:6.3f}")
```

Đây là micrograd. Một vòng đào tạo mạng thần kinh hoàn chỉnh trong Python tinh khiết với sự phân biệt tự động.

### Bước 6: Kiểm tra độ

Làm sao bạn biết tự định nghĩa của mình là đúng? So sánh nó với các phái sinh số. Đây là kiểm tra gradient.

```python
def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff
```

Hãy thử nó bằng một biểu hiện phức tạp:

```python
def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_check(expr, 0.5)
print(f"Autodiff:  {ad:.8f}")
print(f"Numerical: {num:.8f}")
print(f"Difference: {diff:.2e}")
# Difference should be < 1e-5
```

Kiểm tra độ cao là điều cần thiết khi thực hiện các hoạt động mới. Nếu thẻ ngược của bạn có lỗi, kiểm tra số sẽ bắt được nó. Mỗi thực hiện sâu học nghiêm trọng chạy kiểm tra độ cao trong quá trình phát triển.

**When to use gradient checking:**

| Situation | Do gradient check? |
|-----------|-------------------|
| Adding a new operation to your autograd | Yes, always |
| Debugging a training loop that won't converge | Yes, check gradients first |
| Production training | No, too slow (2x forward passes per parameter) |
| Unit tests for autograd code | Yes, automate it |

### Bước 7: Kiểm tra tính toán thủ công

```python
x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2          # a = 6.0
b = a + Value(1.0)    # b = 7.0
y = b.relu()          # y = 7.0

y.backward()

print(f"y = {y.data}")          # 7.0
print(f"dy/dx1 = {x1.grad}")   # 3.0 (= x2)
print(f"dy/dx2 = {x2.grad}")   # 2.0 (= x1)
```

Kiểm tra thủ công: `y = relu(x1*x2 + 1)`Từ đó`x1*x2 + 1 = 7 > 0`, relu là danh tính.
`dy/dx1 = x2 = 3`- `dy/dx2 = x1 = 2`- Động cơ phù hợp.

## Sử dụng nó

### Kiểm tra với PyTorch

```python
import torch

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()

print(f"PyTorch dy/dx1 = {x1.grad.item()}")  # 3.0
print(f"PyTorch dy/dx2 = {x2.grad.item()}")  # 2.0
```

Động cơ của bạn tính toán kết quả tương tự như PyTorch bởi vì toán học là tương tự: tự động chuyển đổi theo chế độ ngược qua quy tắc chuỗi.

### Một biểu hiện phức tạp hơn

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = (a * b + c).relu()  # relu(2*(-3) + 10) = relu(4) = 4

f.backward()
print(f"df/da = {a.grad}")  # -3.0 (= b)
print(f"df/db = {b.grad}")  #  2.0 (= a)
print(f"df/dc = {c.grad}")  #  1.0
```

## Chuyển nó

Bài học này mang lại:
- `outputs/skill-autodiff.md`-- một kỹ năng xây dựng và debugging hệ thống autograd
- `code/autodiff.py`-- một động cơ tự động tối thiểu bạn có thể mở rộng

Các lớp giá trị được xây dựng ở đây là nền tảng cho vòng đào tạo mạng thần kinh trong giai đoạn 3.

## Các bài tập

1. Thêm `__pow__`để bạn có thể tính toán`x ** n`- Hãy kiểm tra.$\frac{d}{dx}(x^3)$$x=2$=`12.0`- Tôi không biết.

2. Thêm `tanh`làm chức năng kích hoạt.$\tanh'(0) = 1$và $\tanh'(2) = 0.0707$(các khoảng).

3. Xây dựng một biểu đồ tính toán cho một tế bào thần kinh duy nhất: `y = relu(w1*x1 + w2*x2 + b)`- Xét tất cả 5 gradient và xác minh với PyTorch.

4. Thực hiện tự động định dạng hướng trước sử dụng hai số.`Dual`lớp và xác minh nó cung cấp các phái sinh tương tự như động cơ chế ngược của bạn.

### Những gợi ý

**Exercise 1.** `Value`đã xác định `__pow__`- Đọc nó.`_backward`và xác nhận nó áp dụng $n x^{n-1}$đến độ nghiêng tiếp cận.`x = Value(2.0)`- `y = x ** 3`- `y.backward()``x.grad`phải là 12,0.

**Exercise 2.**Sử dụng $\tanh'(x) = 1 - \tanh(x)^2$trong `_backward`Hãy kiểm tra với sự khác biệt hữu hạn của chính mình:$\frac{\tanh(x + h) - \tanh(x - h)}{2h}$phải đồng ý với $x = 0$và $x = 2$- Tôi không biết.

**Exercise 3.**- Cây lên`w1*x1 + w2*x2 + b`từ `Value`các đối tượng, áp dụng `relu`, và gọi `backward`Trước khi so sánh với PyTorch, kiểm tra bằng tay: nếu đầu vào đến `relu`là tích cực, gradient cho `w1`=`x1`; nếu nó là âm, tất cả năm gradient là 0.

**Exercise 4.**Đặt một giá trị và một phái sinh trong mỗi `Dual`, và sử dụng quy tắc sản phẩm $(a, a')(b, b') = (ab, a'b + ab')$Để nhân. Thấy đầu vào với phái sinh 1 và so sánh kết quả với `x.grad`từ động cơ chế ngược cho cùng một chức năng.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Chain rule | "Multiply the derivatives" | The derivative of composed functions equals the product of each function's local derivative, evaluated at the right point |
| Computational graph | "The network diagram" | A directed acyclic graph where nodes are operations and edges carry values (forward) or gradients (backward) |
| Forward mode | "Push derivatives forward" | Autodiff that propagates derivatives from inputs to outputs. One pass per input variable. |
| Reverse mode | "Backpropagation" | Autodiff that propagates gradients from outputs to inputs. One pass per output variable. |
| Autograd | "Automatic gradients" | A system that records operations on values, builds a graph, and computes exact gradients via the chain rule |
| Dual numbers | "Value plus derivative" | Numbers of the form $a + b\epsilon$ ($\epsilon^2 = 0$) that carry derivative information through arithmetic |
| Topological sort | "Dependency order" | Ordering graph nodes so every node comes after all its dependencies. Required for correct gradient propagation. |
| Gradient accumulation | "Add, don't replace" | When a value feeds into multiple operations, its gradient is the sum of all incoming gradient contributions |
| Dynamic graph | "Define by run" | A computation graph rebuilt on every forward pass, allowing Python control flow inside models (PyTorch style) |
| Gradient checking | "Numerical verification" | Comparing autodiff gradients against numerical finite-difference gradients to verify correctness. Essential for debugging. |
| MLP | "Multi-layer perceptron" | A neural network with one or more hidden layers of neurons. Each neuron computes a weighted sum plus bias, then applies an activation function. |
| Neuron | "Weighted sum + activation" | The basic unit: $\text{output} = \text{activation}(w_1 x_1 + w_2 x_2 + \cdots + b)$. The weights and bias are learnable parameters. |

## Đọc thêm

- [3Blue1Brown: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)-- giải thích trực quan về quy tắc chuỗi trong mạng thần kinh
- [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)-- cách hệ thống thực sự hoạt động
- [Baydin et al., Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)-- tham chiếu toàn diện

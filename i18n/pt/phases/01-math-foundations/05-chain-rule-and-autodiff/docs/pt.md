# Regra da cadeia e diferenciação automática

> A regra da cadeia é o motor por trás de todas as redes neurais que aprendem.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 1, Lesson 04 (Derivatives & Gradients)
**Time:** ~90 minutes

## Objetivos de aprendizagem

- Construir um motor de autogrado mínimo (classe de valor) que registre operações e computa gradientes através de auto-difusão de modo inverso
- Implementar passagens para frente e para trás através de um gráfico de cálculo usando o tipo topológico
- Construir e treinar um perceptron de várias camadas no XOR usando apenas o motor de autograd do zero
- Verificar a correcção do auto-diff usando a verificação de gradientes contra diferenças finitas numéricas

## O problema

Você pode calcular derivadas de funções simples. Mas uma rede neural não é uma função simples. É centenas de funções compostas juntas: multiplicar matriz, adicionar viés, aplicar ativação, multiplicar matriz novamente, softmax, perda de entropia cruzada. A saída é uma função de uma função de uma função.

Para treinar a rede, é necessário o gradiente da perda em relação a cada peso individual. Fazer isso à mão é impossível para milhões de parâmetros. Fazer numérica (diferências finitas) é muito lento.

A regra da cadeia dá-lhe a matemática. A diferenciação automática dá-lhe o algoritmo. Juntos eles permitem-lhe calcular gradientes exatos através de composições arbitrárias de funções em tempo proporcionais a uma única passagem para a frente.

É assim que PyTorch, TensorFlow e JAX funcionam.

## O conceito

### A Regra da Cadeia

Se$y = f(g(x))$, a derivada de $y$Relativamente a$x$é:

```math
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)
```

Multiplicar os derivados ao longo da cadeia. Cada elo contribui com a sua derivada local.

Exemplo: $y = \sin(x^2)$

```math
\begin{aligned}
g(x) &= x^2 \qquad & g'(x) &= 2x \\
f(g) &= \sin(g) \qquad & f'(g) &= \cos(g) \\[4pt]
\frac{dy}{dx} &= \cos(x^2) \cdot 2x
\end{aligned}
```

Para composições mais profundas, a cadeia se estende:

```math
\begin{aligned}
y &= f(g(h(x))) \\[4pt]
\frac{dy}{dx} &= f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)
\end{aligned}
```

Cada camada numa rede neural é um elo nesta cadeia.

### Gráficos computacionais

Um gráfico computacional torna a regra da cadeia visual. Cada operação se torna um nó. Dados fluem para frente através do gráfico.

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

A passagem para trás aplica a regra da cadeia em cada nó, propagando gradientes da saída para as entradas.

### Modo Avançado vs Modo Reverso

Há duas maneiras de aplicar a regra da cadeia através de um gráfico.

**Forward mode**O sistema de cálculo de uma base de dados é um sistema de cálculo de dados.$dx/dx = 1$E se propaga através de cada operação. bom quando você tem poucas entradas e muitas saídas.

```
Forward mode: seed dx/dx = 1, propagate forward

  x = 2       (dx/dx = 1)
  a = x^2     (da/dx = 2x = 4)
  y = sin(a)  (dy/dx = cos(a) * da/dx = cos(4) * 4 = -2.615)
```

**Reverse mode**Começa na saída e puxa os gradientes para trás.$dy/dy = 1$E se propaga através de cada operação de forma inversa. bom quando você tem muitas entradas e poucas saídas.

```
Reverse mode: seed dy/dy = 1, propagate backward

  y = sin(a)  (dy/dy = 1)
  a = x^2     (dy/da = cos(a) = cos(4) = -0.654)
  x = 2       (dy/dx = dy/da * da/dx = -0.654 * 4 = -2.615)
```

As redes neurais têm milhões de entradas (pesos) e uma saída (perda). O modo inverso calcula todos os gradientes em um passo para trás. É por isso que a propagação para trás usa o modo inverso.

| Mode | Seed | Direction | Best when |
|------|------|-----------|-----------|
| Forward | $dx_i/dx_i = 1$ | Input to output | Few inputs, many outputs |
| Reverse | $dy/dy = 1$ | Output to input | Many inputs, few outputs (neural nets) |

### Dual Numbers para Modo Avançado

O modo de avanço pode ser implementado de forma elegante com números duplos.$a + b\epsilon$onde$\epsilon^2 = 0$- Não .

```
Dual number: (value, derivative)

(2, 1) means: value is 2, derivative w.r.t. x is 1

Arithmetic rules:
  (a, a') + (b, b') = (a+b, a'+b')
  (a, a') * (b, b') = (a*b, a'*b + a*b')
  sin(a, a')         = (sin(a), cos(a)*a')
```

Seem a variável de entrada com derivada 1. A derivada se propaga automaticamente através de cada operação.

### Construindo um motor Autograd

Um motor autograd precisa de três coisas:

1. **Value wrapping.**Envolva cada número num objeto que armazene seu valor e gradiente.
2. **Graph recording.**Cada operação registra as suas entradas e a função de gradiente local.
3. **Backward pass.**A classificação topológica do gráfico, depois caminhando-o ao contrário, aplicando a regra da cadeia em cada nó.

É exatamente isso que PyTorch é.`autograd`- Não, não.`torch.Tensor`classe envolve valores, registra operações quando `requires_grad=True`, e calcula gradientes quando você chama`.backward()`- Não .

### Como funciona a PyTorch Autograd sob o capuz

Quando escrever código PyTorch:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # 7.0 = 2*x + 3 = 2*2 + 3
```

PyTorch internamente:

1. Cria um `Tensor`nó para `x`com`requires_grad=True`
2. Cada operação (`**`- Não .`*`- Não .`+`) cria um novo nó e registra a função retrógrada
3. `y.backward()`desencadeia auto-desvio de modo inverso através do gráfico gravado
4. Cada nó é `grad_fn`computa gradientes locais e os passa para nós-mãe
5. Os gradientes se acumulam em `.grad`Atributos por adição (não substituição)

O gráfico é dinâmico (definido por execução). Um novo gráfico é construído em cada passagem avançada. É por isso que PyTorch suporta o fluxo de controle (se / outros, loop) dentro dos modelos.

```figure
chain-rule
```

## Construí-lo

### Passo 1: A classe de valor

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

Todos .`Value`armazena os dados numéricos, o seu gradiente (início zero), uma função retrógrada e aponta para os nós menores que o produziram.

### Passo 2: Operações aritméticas com rastreamento de gradientes

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

Cada operação cria um fechamento que sabe como calcular gradientes locais e multiplicar pelo gradiente ascendente (`out.grad`O `+=`Tratam o caso em que um valor é utilizado em múltiplas operações.

### Passo 3: Passagem para trás

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

A classificação topológica garante que o gradiente de cada nó seja totalmente calculado antes de se propagar aos seus filhos.$dy/dy = 1$)).

### Passo 4: Mais operações para um motor completo

A classe de valor básica lida com adição, multiplicação e relú. Um motor autograd real precisa de mais. Aqui estão as operações que você precisa para construir redes neurais:

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

A parte inteligente:`__sub__`E ...`__truediv__`Os gradientes corretos são obtidos gratuitamente porque a regra da cadeia se compõe através das operações subjacentes adição/mul/pow.

### Passo 5: Mini MLP a partir do zero

Com uma classe de Valores completa, podes construir uma rede neural sem PyTorch, sem NumPy, apenas valores e a regra da cadeia.

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

A.`Neuron`Computação`tanh(w1*x1 + w2*x2 + ... + b)`- A.`Layer`É uma lista de neurônios.`MLP`Cada peso é um`Value`, por isso chamando`loss.backward()`Propaga gradientes a todos os parâmetros.

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

Isto é microgrado. Um ciclo completo de treinamento de rede neural em Python puro com diferenciação automática.

### Passo 6: Verificação gradual

Como saber se o seu auto-difiguração é correta? Compare-o com derivadas numéricas.

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

Teste-o numa expressão complexa:

```python
def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_check(expr, 0.5)
print(f"Autodiff:  {ad:.8f}")
print(f"Numerical: {num:.8f}")
print(f"Difference: {diff:.2e}")
# Difference should be < 1e-5
```

A verificação de gradientes é essencial ao implementar novas operações. Se o seu pass para trás tiver um bug, a verificação numérica o apanha.

**When to use gradient checking:**

| Situation | Do gradient check? |
|-----------|-------------------|
| Adding a new operation to your autograd | Yes, always |
| Debugging a training loop that won't converge | Yes, check gradients first |
| Production training | No, too slow (2x forward passes per parameter) |
| Unit tests for autograd code | Yes, automate it |

### Passo 7: Verificar contra o cálculo manual

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

Verificação manual: `y = relu(x1*x2 + 1)`Desde que ...`x1*x2 + 1 = 7 > 0`Relu é identidade.
`dy/dx1 = x2 = 3`- Não .`dy/dx2 = x1 = 2`O motor coincide.

## Usá-lo

### Verificar contra PyTorch

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

O motor calcula o mesmo resultado que o PyTorch porque a matemática é a mesma: auto-difusão de modo inverso através da regra da cadeia.

### Uma expressão mais complexa

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

## Envia-o

Esta lição produz:
- `outputs/skill-autodiff.md`-- uma habilidade para construir e depurar sistemas autograd
- `code/autodiff.py`- um motor de auto-grada mínimo que você pode estender

A classe de valor construída aqui é a base para o ciclo de treinamento da rede neural na Fase 3.

## Exercícios

1. Adicionar`__pow__`para a classe Valor para que você possa calcular `x ** n`Verifica isso .$\frac{d}{dx}(x^3)$- Não .$x=2$- É igual .`12.0`- Não .

2. Adicionar`tanh`Como uma função de ativação.$\tanh'(0) = 1$E ...$\tanh'(2) = 0.0707$- Não.

3. Construa um gráfico de cálculo para um único neurônio: `y = relu(w1*x1 + w2*x2 + b)`Calcule os cinco gradientes e verifique contra a PyTorch.

4. Implementar o auto-difuso em modo avançado usando números duplos.`Dual`A classe e verificar que dá as mesmas derivadas que o motor de modo inverso.

### Indicios

**Exercise 1.** `Value`já define `__pow__`- Leia o seu .`_backward`e confirmar que se aplica.$n x^{n-1}$- Para o gradiente de entrada.`x = Value(2.0)`- Não .`y = x ** 3`- Não .`y.backward()`- O que é ?`x.grad`Deve ser 12,0.

**Exercise 2.**Utilização$\tanh'(x) = 1 - \tanh(x)^2$em `_backward`Verifique contra a sua própria diferença finita:$\frac{\tanh(x + h) - \tanh(x - h)}{2h}$Deve concordar em$x = 0$E ...$x = 2$- Não .

**Exercise 3.**Construir`w1*x1 + w2*x2 + b`de`Value`objetos, aplicar `relu`, e ligar`backward`Antes de comparar com PyTorch, verifique manualmente: se a entrada para `relu`é positivo, o gradiente para `w1`- É igual .`x1`Se for negativo, os cinco gradientes são 0.

**Exercise 4.**Armazenar um valor e um derivado em cada `Dual`, e usar a regra do produto $(a, a')(b, b') = (ab, a'b + ab')$Para multiplicação. semente a entrada com a derivada 1 e compare o resultado com `x.grad`do motor de modo inverso para a mesma função.

## Termos-chave

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

## Mais leitura

- [3Blue1Brown: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)-- explicação visual da regra da cadeia em redes neurais
- [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)- Como funciona o sistema real
- [Baydin et al., Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)-- referência abrangente

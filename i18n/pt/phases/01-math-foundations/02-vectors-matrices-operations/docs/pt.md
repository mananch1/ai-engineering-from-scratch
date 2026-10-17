# Vectores, Matriças e Operações

> Cada rede neural é apenas uma multiplicação de matriz com passos extras.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## Objetivos de aprendizagem

- Construir uma classe de Matrix com operações de elementos, multiplicação de matriz, transposição, determinante e inversos
- Distinguir a multiplicação por elemento da multiplicação por matriz e explicar quando cada uma se aplica
- Implementar uma única camada de rede neural densa (`relu(W @ x + b)`) utilizando apenas a classe Matrix do zero
- Explique as regras de radiodifusão e como a adição de preconceitos funciona em estruturas de rede neural

## O problema

Se quiser construir uma rede neural, lê o código e vê isto:

```
output = activation(weights @ input + bias)
```

Isso .`@`é a multiplicação de matriz.`weights`São uma matriz.`input`Se você não sabe o que essas operações fazem, esta linha é mágica. se você sabe, é toda a passagem para a frente de uma camada em três operações.

Cada imagem que o seu modelo processa é uma matriz de valores de píxeles. Cada palavra incorporada é um vetor. Cada camada de cada rede neural é uma transformação de matriz. Você não pode construir sistemas de IA sem ser fluente em operações de matriz da mesma forma que não pode escrever código sem entender variáveis.

Esta lição construiu essa fluência a partir do zero.

## O conceito

### Vectores: listas ordenadas de números

Um vetor é uma lista de números com uma direção e magnitude.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

Um vetor 2D `[3, 4]`A sua extensão (magnitude) é de 5 (o triângulo 3-4-5).

### Matrizes: grades de números

Uma matriz é uma grade 2D.$m \times n$A matriz tem $m$- e$n$Colunas.

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \text{2x3 matrix (2 rows, 3 columns)}
```

Em redes neurais, matrizes de peso transformam vetores de entrada em vetores de saída. Uma camada com 784 entradas e 128 saídas usa uma matriz de peso de 128x784.

### Por que as formas importam

A multiplicação de matriz tem uma regra estritamente regida:`(m x n) @ (n x p) = (m x p)`As dimensões internas devem corresponder.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

Se você receber um erro de desajuste de forma em PyTorch, é por isso.

### Mapa das operações

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | Element-wise combine | Adding bias to output |
| Scalar multiply | Scale every element | Learning rate * gradients |
| Matrix multiply | Transform vectors | Layer forward pass |
| Transpose | Flip rows and columns | Backpropagation |
| Determinant | Single number summary | Checking invertibility |
| Inverse | Undo a transformation | Solving linear systems |
| Identity | Do-nothing matrix | Initialization, residual connections |

### Multiplicação por elemento versus matriz

Esta distinção atrapalha os iniciantes constantemente.

Elementos-wise: multiplicar posições correspondentes. Ambas as matrizes devem ser da mesma forma.

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} * \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}
```

Multiplicação de matriz: produtos de pontos de linhas e colunas.

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} @ \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
```

Diferentes operações, resultados diferentes, regras diferentes.

### Transmissão

Quando adicionar um vetor de viés a uma matriz de saídas, as formas não coincidem.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Cada estrutura moderna faz isso automaticamente.

```figure
vector-projection
```

## Construí-lo

### Passo 1: Classe de vetores

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

### Passo 2: Classe de matriz com operações de núcleo

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

### Passo 3: Veja como funciona

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

### Passo 4: Conectar-se a redes neurais

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

Esta é uma única camada densa:`output = relu(W @ x + b)`Cada camada densa em cada rede neural faz exatamente isso.

## Usá-lo

NumPy faz tudo acima em menos linhas e ordens de magnitude mais rápido.

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

O `@`operador em chamadas Python `__matmul__`NumPy implementa com rotinas BLAS optimizadas escritas em C e Fortran.

Transmissão em NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy transmite automaticamente o preconceito 1D em ambas as linhas. É assim que a adição de preconceitos funciona em cada framework de rede neural.

## Envia-o

Esta lição produz um prompt para ensinar operações de matriz através da intuição geométrica.`outputs/prompt-matrix-operations.md`- Não .

A classe Matrix construída aqui é a base para a estrutura de rede neural mini que construímos na fase 3, lição 10.

## Exercícios

1. **Verify the inverse.**Multiplicar`A @ A.inverse_2x2()`e confirmar que você tem a matriz de identidade. Tente com três diferentes matriz 2x2. O que acontece quando o determinante é zero?

2. **Implement 3x3 inverse.**Extenda a classe Matrix para calcular inversos para matrizes 3x3 usando o método de adjugação.`np.linalg.inv`- Não .

3. **Build a two-layer network.**Usando apenas a sua classe Matrix (sem NumPy), crie uma rede neural de duas camadas: entrada (3) -> oculta (4) -> saída (2). Inicie pesos aleatórios, execute uma passagem para a frente e verifique se todas as formas são corretas.

### Indicios

**Exercise 1.**Multiplicar`A @ A.inverse_2x2()`para três matrizes e comparar cada célula com a identidade dentro `1e-9`. Para uma matriz com um determinante de 0, como [[1, 2], [2, 4]], `inverse_2x2`Levanta um `ValueError`, e este é o resultado esperado.

**Exercise 2.**Calcule a matriz de cofatores 3x3, transponha-a para obter o adjugado, e depois divide cada célula por `determinant()`- Vê isso .`A @ A_inv`dá a identidade e que cada célula coincide `np.linalg.inv(A)`dentro de`1e-9`- Não .

**Exercise 3.**Escreva a forma de cada matriz antes de multiplicar: `W1`é 4x3 e `W2`é 2x4, como em `demo_neural_network_layer`Uma entrada 3x1 deve dar um vetor oculto 4x1 e uma saída 2x1.

## Termos-chave

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

## Mais leitura

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)- Intuição visual para cada operação aqui coberta
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)- as regras exatas que o NumPy segue
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)- referência concisa para a álgebra linear específica do ML

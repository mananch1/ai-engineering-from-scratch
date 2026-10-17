# Intuição de álgebra linear

> Todos os modelos de IA são apenas matemáticas de matriz usando um chapéu elegante.

**Type:** Learn
**Languages:** Python, Julia
**Prerequisites:** Phase 0
**Time:** ~60 minutes

## Objetivos de aprendizagem

- Implementar operações de vetores e matrizes (adição, produto de pontos, multiplicação de matrizes) a partir do zero no Python
- Explique geometricamente o que o produto de pontos, a projeção e o processo de Gram-Schmidt fazem
- Determine a independência linear, a classificação e a base de um conjunto de vetores usando a redução de filas
- Conecte conceitos de álgebra linear às suas aplicações de IA: embalagens, pontuações de atenção e LoRA

## O problema

Abre qualquer documento de ML. Na primeira página, verá vetores, matrizes, produtos de pontos e transformações. Sem a intuição de álgebra linear, estes são apenas símbolos. Com ele, você pode ver o que uma rede neural está realmente fazendo - movendo pontos no espaço.

Não é preciso ser matemático, é preciso ver o que estas operações significam geométricamente, e depois codificá-las.

## O conceito

### Vectores são pontos (e direções)

Um vetor é apenas uma lista de números. Mas esses números significam algo - são coordenadas no espaço.

**2D vector [3, 2]:**

| x | y | Point |
|---|---|-------|
| 3 | 2 | The vector points from origin (0,0) to (3, 2) on the plane |

O vetor tem magnitude .$\sqrt{3^2 + 2^2} = \sqrt{13}$e aponta para cima e para a direita.

Na IA, vetores representam tudo:
- Uma palavra → um vetor de 768 números (seu "significado" em espaço de inserção)
- Uma imagem → um vetor de milhões de valores de pixels
- Um usuário → um vetor de preferências

### Matrizes são transformações

Uma matriz transforma um vetor em outro. Pode girar, escalar, esticar ou projetar.

```mermaid
graph LR
    subgraph Before
        A["Point A"]
        B["Point B"]
    end
    subgraph Matrix["Matrix Multiplication"]
        M["M (transformation)"]
    end
    subgraph After
        A2["Point A'"]
        B2["Point B'"]
    end
    A --> M
    B --> M
    M --> A2
    M --> B2
```

Na IA, as matrizes são o modelo:
- Pesos de rede neural → matrizes que transformam entrada em saída
- Pontos de atenção → matrizes que decidem no que se concentrar
- Embedings → matrizes que mapeam palavras para vetores

### A semelhança das medidas do produto

O produto de pontos de dois vetores diz-lhe o quão semelhantes são.

```math
\begin{aligned}
a \cdot b &= a_1 \times b_1 + a_2 \times b_2 + \cdots + a_n \times b_n \\[4pt]
\text{Same direction:} \quad a \cdot b &> 0 \quad \text{(similar)} \\
\text{Perpendicular:} \quad a \cdot b &= 0 \quad \text{(unrelated)} \\
\text{Opposite direction:} \quad a \cdot b &< 0 \quad \text{(dissimilar)}
\end{aligned}
```

É literalmente assim que os motores de busca, os sistemas de recomendação e o RAG funcionam: encontram vetores com produtos de pontos altos.

### Independência Linear

Os vetores são linearmente independentes se nenhum vetor no conjunto pode ser escrito como uma combinação dos outros.$v_1, v_2, v_3$Se um é uma combinação dos outros, eles apenas abrangem um plano.

Por que é importante para a IA: a matriz de características deve ter colunas linearmente independentes. Se duas características estão perfeitamente correlacionadas (linearmente dependentes), o modelo não pode distinguir os seus efeitos. Isso causa multicolinariedade na regressão - a matriz de peso se torna instável, e pequenas mudanças de entrada produz oscilações selvagens de saída.

**Concrete example:**

```
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 1, 0]   # v3 = 2*v1 + v2
```

$v_1$E ...$v_2$Não é um múltiplo escalar ou uma combinação do outro.$v_3 = 2 v_1 + v_2$- Então ...$\lbrace v_1, v_2, v_3 \rbrace$é um conjunto dependente. Estes três vetores estão todos no plano xy. Não importa como você os combina, você não pode chegar a [0, 0, 1]. Você tem três vetores, mas apenas duas dimensões de liberdade.

Em um conjunto de dados: se feature_3 = 2*feature_1 + feature_2, adicionar feature_3 dá ao modelo zero novas informações. Pior, torna as equações normais singulares - não há solução única para os pesos.

### Base e posição

Uma base é um conjunto mínimo de vetores linearmente independentes que abrangem todo o espaço.

A base padrão para o espaço 3D é {[1,0,0], [0,1,0], [0,0,1]}. Mas quaisquer três vetores independentes em 3D formam uma base válida.

Rango de uma matriz = número de colunas linearmente independentes = número de linhas linearmente independentes.$\text{rank} < \min(\text{rows}, \text{cols})$, a matriz é deficiente em grau.
- O sistema tem infinitamente muitas soluções (ou nenhuma)
- Informações perdidas na transformação
- A matriz não pode ser invertida

| Situation | Rank | What it means for ML |
|-----------|------|---------------------|
| Full rank $(\text{rank} = \min(m, n))$ | Maximum possible | Unique least-squares solution exists. Model is well-conditioned. |
| Rank deficient $(\text{rank} < \min(m, n))$ | Below maximum | Features are redundant. Infinitely many weight solutions. Regularization needed. |
| Rank 1 | 1 | Every column is a scaled copy of one vector. All data lies on a line. |
| Near rank-deficient (small singular values) | Numerically low | Matrix is ill-conditioned. Tiny input noise causes large output changes. Use SVD truncation or ridge regression. |

### Projeção

Vêctor de projecção **a**sobre o vetor **b**dá a componente de **a**na direcção de **b**- Não .

```math
\mathop{\mathrm{proj}}_b(a) = \frac{a \cdot b}{b \cdot b} \, b
```

O resíduo$(a - \mathop{\mathrm{proj}}_b(a))$é perpendicular a $b$Esta decomposição ortogonal é a base do ajuste de mínimos quadrados.

A projecção está em todo o ML:
- Regressão linear minimiza a distância das observações para o espaço coluna - a solução é uma projeção
- A PCA projeta dados nas direções da variância máxima
- A atenção em transformadores calcula projeções de consultas em chaves

```mermaid
graph LR
    subgraph Projection["Projection of a onto b"]
        direction TB
        O["Origin"] --> |"b (direction)"| B["b"]
        O --> |"a (original)"| A["a"]
        O --> |"proj_b(a)"| P["projection"]
        A -.-> |"residual (perpendicular)"| P
    end
```

**Example:** $a = [3, 4]$- Não .$b = [1, 0]$

$\mathop{\mathrm{proj}}_b(a) = (3 \cdot 1 + 4 \cdot 0) / (1 \cdot 1 + 0 \cdot 0) \cdot [1, 0] = 3 \cdot [1, 0] = [3, 0]$

A projeção deixa cair o componente y. Isto é a redução de dimensionalidade na sua forma mais simples - jogar fora as direções que não nos importam.

### Processo Gram-Schmidt

Converter qualquer conjunto de vetores independentes em uma base ortonormal. Ortonormal significa que cada vetor tem comprimento 1 e cada par é perpendicular.

O algoritmo:
1. Pegue o primeiro vetor, normalize-o
2. Pegue o segundo vetor, subtraga sua projeção para o primeiro, normalize
3. Tome o terceiro vetor, subtraia suas projeções para todos os vetores anteriores, normalize
4. Repita para vetores restantes

```
Input:  v1, v2, v3, ... (linearly independent)

u1 = v1 / |v1|

w2 = v2 - (v2 dot u1) * u1
u2 = w2 / |w2|

w3 = v3 - (v3 dot u1) * u1 - (v3 dot u2) * u2
u3 = w3 / |w3|

Output: u1, u2, u3, ... (orthonormal basis)
```

É assim que a decomposição QR funciona internamente. Q é a base ortonormal, R capta os coeficientes de projeção.
- Solução de sistemas lineares (mais estáveis do que a eliminação gaussiana)
- Calculação de valores próprios (algoritmo de RQ)
- Regressão dos mínimos quadrados (método numérico padrão)

```figure
eigen-directions
```

## Construí-lo

### Passo 1: Vectores a partir do zero (Python)

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
```

### Passo 2: Matrizes a partir do zero (Python)

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")
```

### Passo 3: Por que isso importa para a IA

```python
import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication.")
```

### Passo 4: versão Julia

```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a + b = ", a + b)
println("a · b = ", a ⋅ b)       # Julia supports unicode operators
println("|a| = ", √(a ⋅ a))
println("cosine = ", (a ⋅ b) / (√(a ⋅ a) * √(b ⋅ b)))

# Matrix-vector multiplication
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]
println("Wx = ", W * x)
println("This is a neural network layer.")
```

### Passo 5: Independência linear e projeção a partir do zero (Python)

```python
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])
for i, u in enumerate(basis):
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {u.magnitude():.6f}")

print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
```

## Usá-lo

Agora a mesma coisa com o NumPy -- o que realmente usará na prática:

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")
```

### Rank, Projeção e QR com NumPy

```python
import numpy as np

A = np.array([[1, 2], [2, 4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3))
print(f"Q is orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")
```

### PyTorch - Tensores são vetores com Autodiff

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
```

O gradiente do produto de pontos em relação a x é apenas y. PyTorch calculou isso automaticamente. Toda operação em uma rede neural é construída a partir de operações como esta - multiplicadores de matriz, produtos de pontos, projeções - e auto-difusão rastreia gradientes através de todos eles.

Construiste do zero o que o NumPy faz numa linha.

## Envia-o

Esta lição produz:
- `outputs/prompt-linear-algebra-tutor.md`-- um aviso para os assistentes de IA para ensinar álgebra linear através da intuição geométrica

## Relações

Tudo nesta lição está ligado a partes específicas da IA moderna:

| Concept | Where it shows up |
|---------|------------------|
| Dot product | Attention scores in transformers, cosine similarity in RAG |
| Matrix multiply | Every neural network layer, every linear transformation |
| Linear independence | Feature selection, avoiding multicollinearity |
| Rank | Determining if a system is solvable, LoRA (low-rank adaptation) |
| Projection | Linear regression (projecting onto column space), PCA |
| Gram-Schmidt / QR | Numerical solvers, eigenvalue computation |
| Orthonormal basis | Stable numerical computation, whitening transforms |

A LoRA merece uma menção especial. Ele sintoniza os modelos de linguagem grandes, decomponendo as atualizações de peso em matrizes de baixo nível. Em vez de atualizar uma matriz de peso 4096x4096 (16M parâmetros), a LoRA atualiza duas matrizes de tamanho 4096x16 e 16x4096 (131K parâmetros). A restrição de classificação 16 significa que o LoRA assume que a atualização de peso vive num subespaço 16 dimensiones do espaço completo de 4096 dimensiones. É álgebra linear a fazer trabalho real.

## Exercícios

1. Implementação `Vector.angle_between(other)`que retorna o ângulo em graus entre dois vetores
2. Criar uma matriz de escala 2D que dobra a coordenada x e triplica a coordenada y, em seguida, aplicá-la ao vetor [1, 1]
3. Dados 5 vetores aleatórios semelhantes a palavras (dimensão 50), encontrar os dois mais semelhantes usando similaridade cosínica
4. Verifique se a saída de Gram-Schmidt é verdadeiramente ortônormal: verifique se cada par tem produto ponto 0 e cada vetor tem magnitude 1
5. Crie uma matriz 3x3 com o rango 2. Verifique usando o `rank()`Então, explique que objeto geométrico as colunas abrangem.
6. Projete o vetor [1, 2, 3] para [1, 1, 1].

### Indicios

**Exercise 1.**Começa com `cosine_similarity`, manter o valor entre -1 e 1, então tomar o cosino de arco e converter radianos em graus.`Vector([1, 0])`E ...`Vector([0, 1])`dar 90 e que um vetor em si mesmo dá 0.

**Exercise 2.**Coloque os dois fatores de escala na diagonal de uma matriz 2x2 e coloque 0 nas outras duas células. O resultado para [1, 1] deve ser [2, 3].

**Exercise 3.**Calcule a semelhança cosínica para cada par dos 5 vetores (10 pares) e mantenha o maior valor. Verifique primeiro o seu ciclo com um vetor contra si mesmo, que deve dar 1.

**Exercise 4.**Utilização`gram_schmidt`Em três vetores independentes, então computa`dot`para cada par e `magnitude`Comparar com uma tolerância como `1e-9`, porque o ponto flutuante não dá zeros exatos.

**Exercise 5.**Faça a terceira linha ou coluna a soma das duas primeiras, para que uma direção se repita. `rank()`Deve retornar 2, e as colunas então atravessar um plano através da origem.

**Exercise 6.**Calcule a escala $\frac{a \cdot b}{b \cdot b}$e multiplicam .$b$ou chamem.`project_onto`Verifique o resto .$a$menos a projeção tem um produto de ponto de 0 com [1, 1, 1].

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Vector | "An arrow" | A list of numbers representing a point or direction in n-dimensional space |
| Matrix | "A table of numbers" | A transformation that maps vectors from one space to another |
| Dot product | "Multiply and sum" | A measure of how aligned two vectors are -- the core of similarity search |
| Embedding | "Some AI magic" | A vector that represents the meaning of something (word, image, user) |
| Linear independence | "They don't overlap" | No vector in the set can be written as a combination of the others |
| Rank | "How many dimensions" | The number of linearly independent columns (or rows) in a matrix |
| Projection | "The shadow" | The component of one vector in the direction of another |
| Basis | "The coordinate axes" | A minimal set of independent vectors that span the space |
| Orthonormal | "Perpendicular unit vectors" | Vectors that are mutually perpendicular and each have length 1 |

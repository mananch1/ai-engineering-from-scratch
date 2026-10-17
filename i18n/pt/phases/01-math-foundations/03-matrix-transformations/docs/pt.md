# Transformações de Matriz

> Uma matriz é uma máquina que remodela o espaço. Aprenda o que faz a cada ponto, e você entende toda a transformação.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lessons 01-02 (Linear Algebra Intuition, Vectors & Matrices Operations)
**Time:** ~75 minutes

## Objetivos de aprendizagem

- Construir matrizes de rotação, escalagem, corte e reflecção e aplicá-las a pontos 2D e 3D
- Compõem múltiplas transformações por multiplicação de matriz e verifiquem que a ordem importa
- Compute valores próprios e vetores próprios de matrizes 2x2 a partir da equação característica
- Explique por que os valores próprios determinam as direções do PCA, a estabilidade do RNN e o comportamento de agrupamento espectral

## O problema

Você lê sobre PCA e vê "encontrar os vetores próprios da matriz de covariância". Você lê sobre estabilidade do modelo e vê "verifique se todos os valores próprios têm magnitude menor que 1." Você lê sobre aumento de dados e vê "aplicar uma rotação aleatória". Nada disso faz sentido até entender o que as matrizes fazem ao espaço geométricamente.

Matriz não são apenas grades de números. São máquinas espaciais. Uma matriz de rotação gira pontos. Uma matriz de escalagem as estende. Uma matriz de corte as inclina. Cada transformação que uma rede neural aplica aos dados é uma dessas operações ou uma composição delas. Esta lição torna essas operações concretas.

## O conceito

### Transformações como matrizes

Cada transformação linear em 2D pode ser escrita como uma matriz 2x2. A matriz diz-lhe exatamente onde os vetores base [1, 0] e [0, 1] terminam. Tudo o resto segue.

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

### Rotatividade

Uma rotação 2D por ângulo $\theta$Mantém as distâncias e os ângulos intactos, e move todos os pontos ao longo de um arco circular.

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

Em 3D, você gira em torno de um eixo.

```math
\begin{aligned}
R_z(\theta) &= \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} \quad \text{Rotate around z-axis (x-y plane spins, z stays)} \\[6pt]
R_x(\theta) &= \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix} \quad \text{Rotate around x-axis (y-z plane spins, x stays)} \\[6pt]
R_y(\theta) &= \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix} \quad \text{Rotate around y-axis (x-z plane spins, y stays)}
\end{aligned}
```

### Escalada

O estiramento de escala é feito ou comprimido ao longo de cada eixo de forma independente.

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

### Tesouro

A corte inclinou um eixo mantendo o outro fixo, transformando retângulos em paralelogramas.

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

Matrizes de corte:
- `Shx = [[1, k], [0, 1]]`turnos$x$Por$k \cdot y$
- `Shy = [[1, 0], [k, 1]]`turnos$y$Por$k \cdot x$

### Reflexão

Os pontos de reflexo refletem pontos através de um eixo ou linha.

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

Matriz de reflecção:
- Refletir através do eixo y: `[[-1, 0], [0, 1]]`
- Refletir através do eixo x: `[[1, 0], [0, -1]]`

### Composição: transformações de cadeia

Aplicar a transformação A e depois B é o mesmo que multiplicar suas matrizes: `result = B @ A @ point`A ordem é importante. A rotação da escala dá resultados diferentes da escala da rotação.

```mermaid
graph LR
    subgraph Path1["Rotate 90 then Scale (2, 0.5)"]
        P1["(1, 0)"] -->|"Rotate 90"| P2["(0, 1)"] -->|"Scale"| P3["(0, 0.5)"]
    end
```

Composto: `S @ R = [[0, -2], [0.5, 0]]`

```mermaid
graph LR
    subgraph Path2["Scale (2, 0.5) then Rotate 90"]
        Q1["(1, 0)"] -->|"Scale"| Q2["(2, 0)"] -->|"Rotate 90"| Q3["(0, 2)"]
    end
```

Composto: `R @ S = [[0, -0.5], [2, 0]]`

Multiplicação de matriz não é commutativa.

### Valores próprios e vetores próprios

A maioria dos vetores muda de direção quando uma matriz os atinge. Os vetores próprios são especiais: a matriz apenas os escala, nunca os gira. O fator de escala é o valor próprio.

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

A matriz estende o espaço por 3x ao longo de [1, 1] e mantém [1, -1] inalterado.

### Composição própria

Se uma matriz tiver$n$Vêctores próprios linearmente independentes, podem ser decomposidos:

```
A = V @ D @ V^(-1)

V = matrix whose columns are eigenvectors
D = diagonal matrix of eigenvalues
V^(-1) = inverse of V

This says: rotate into eigenvector coordinates, scale along each axis, rotate back.
```

### Por que os valores próprios importam

**PCA.**Os vetores próprios da matriz de covariância são os componentes principais. Os valores próprios dizem-lhe quanta variância cada componente capta.$k$, e temos dimensional redução.

**Stability.**Em redes recorrentes e sistemas dinâmicos, os valores próprios com magnitude > 1 causam explosões de saídas.

**Spectral methods.**As redes neurais de gráficos usam os valores próprios da matriz adjacente. O agrupamento espectral usa os valores próprios do laplaciano. Os próprios vetores revelam a estrutura do gráfico.

### Determinante como fator de escalagem de volume

O determinante de uma matriz de transformação diz-lhe o quanto ele escala área (2D) ou volume (3D).

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

## Construí-lo

### Passo 1: Matriz de transformação a partir do zero (Python)

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

### Passo 2: Composição das transformações

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

### Passo 3: Valores próprios a partir do zero (2x2)

Para uma matriz 2x2 `[[a, b], [c, d]]`, os valores próprios resolvem a equação característica: $\lambda^2 - (a+d)\lambda + (ad - bc) = 0$- Não .

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

### Passo 4: Determinante como fator de escala de volume

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

## Usá-lo

O NumPy lida com tudo isto com rotinas otimizadas.

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

### Rotatividade 3D com NumPy

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

## Envia-o

Esta lição constrói a base geométrica para PCA (Fase 2) e análise de peso da rede neural. O código de valor próprio / eigenvector construído aqui é o mesmo algoritmo que alimenta a redução de dimensionalidade, agrupamento espectral e análise de estabilidade em sistemas de produção ML.

## Exercícios

1. Aplique rotação, escala e corte em uma unidade quadrada (curvas em [0,0], [1,0], [1,1], [0,1]). Imprima os cantos transformados para cada um. Verifique se a rotação preserva as distâncias entre os cantos.

2. Encontre os valores próprios da matriz [[4, 2], [1, 3]] manualmente usando a equação característica.

3. Crie uma composição de três transformações (rotação de 30 graus, escala por [1,5, 0,8], corte com kx=0,3) e aplique-a a 8 pontos dispostos em um círculo. Imprima antes e depois das coordenadas.

### Indicios

**Exercise 1.**Aplicar`rotation_2d`- Não .`scaling_2d`, e `shearing_2d`para cada canto com `mat_vec_mul`Para a rotação, calcule os 4 comprimentos laterais e os 2 diagonais antes e depois; cada um deve permanecer igual.

**Exercise 2.**Escreva a equação característica $\lambda^2 - (\text{trace})\lambda + \det = 0$Verifique se os seus dois valores somam-se ao rastro (7) e multiplicam-se ao determinante (10), então compare com `eigenvalues_2x2`E ...`np.linalg.eigvals`- Não .

**Exercise 3.**Construir as três matrizes, com `shearing_2d(0.3, 0)`Para o corte, multiplica-os por `mat_mul`na ordem de aplicação, e aplicar o resultado a 8 pontos $(\cos\theta, \sin\theta)$A rotação tem um determinante de 1 e o corte tem um determinante de 1, então `det_2x2`de composição deve ser igual $1.5 \times 0.8 = 1.2$- Não .

## Termos-chave

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

## Mais leitura

- [3Blue1Brown: Linear Transformations](https://www.3blue1brown.com/lessons/linear-transformations)-- intuição visual para como matrizes remodelar espaço
- [3Blue1Brown: Eigenvectors and Eigenvalues](https://www.3blue1brown.com/lessons/eigenvalues)- a melhor explicação visual do que os vetores próprios significam geométricamente
- [MIT 18.06 Lecture 21: Eigenvalues and Eigenvectors](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)- O tratamento clássico de Gilbert Strang.

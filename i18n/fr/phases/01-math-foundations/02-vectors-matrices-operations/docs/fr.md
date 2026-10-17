# Vecteurs, matrices et opérations

> Chaque réseau neural est juste une multiplication de matrice avec des étapes supplémentaires.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## Objectifs d'apprentissage

- Construire une classe de Matrix avec des opérations par élément, la multiplication de la matrice, la transposition, le déterminant et l'inverse
- Distinguer la multiplication par élément de la multiplication par matrice et expliquer quand chaque élément s'applique
- Implementer une seule couche de réseau neural dense (`relu(W @ x + b)`) utilisant uniquement la classe Matrix à partir de zéro
- Expliquer les règles de radiodiffusion et comment l'addition de biais fonctionne dans les cadres de réseaux neuronaux

## Le problème

Vous voulez construire un réseau neuronal, vous lisez le code et voyez ceci:

```
output = activation(weights @ input + bias)
```

Ça ...`@`est la multiplication de matrice.`weights`sont une matrice.`input`Si vous ne savez pas ce que ces opérations font, cette ligne est magique. si vous le savez, c'est l'ensemble de la passée vers l'avant d'une couche en trois opérations.

Chaque image que votre modèle traite est une matrice de valeurs de pixels. Chaque mot intégré est un vecteur. Chaque couche de chaque réseau neural est une transformation de matrice. Vous ne pouvez pas construire des systèmes d'IA sans être fluide dans les opérations de matrice de la même manière que vous ne pouvez pas écrire de code sans comprendre les variables.

Cette leçon construit cette fluidité à partir de zéro.

## Le concept

### Vecteurs: listes de nombres ordonnées

Un vecteur est une liste de nombres avec une direction et une magnitude.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

Un vecteur 2D `[3, 4]`Il est le point de départ de la ligne de référence de la ligne de référence.

### Matrices: grilles de nombres

Une matrice est une grille 2D.$m \times n$la matrice a$m$les lignes et $n$les colonnes.

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \text{2x3 matrix (2 rows, 3 columns)}
```

Dans les réseaux neuronaux, les matrices de poids transforment les vecteurs d'entrée en vecteurs de sortie.

### Pourquoi les formes comptent

La multiplication de matrice a une règle stricte:`(m x n) @ (n x p) = (m x p)`Les dimensions internes doivent être correspondantes.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

Si vous obtenez une erreur de déséquilibre de forme dans PyTorch, c'est pourquoi.

### Carte des opérations

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | Element-wise combine | Adding bias to output |
| Scalar multiply | Scale every element | Learning rate * gradients |
| Matrix multiply | Transform vectors | Layer forward pass |
| Transpose | Flip rows and columns | Backpropagation |
| Determinant | Single number summary | Checking invertibility |
| Inverse | Undo a transformation | Solving linear systems |
| Identity | Do-nothing matrix | Initialization, residual connections |

### Multiplication par élément par matrice

Cette distinction fait trébucher constamment les débutants.

Par élément: multipliez les positions correspondantes.

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} * \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}
```

Multiplication de matrice: produits de points de lignes et de colonnes.

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} @ \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
```

Des opérations différentes, des résultats différents, des règles différentes.

### La radiodiffusion

Quand on ajoute un vecteur de biais à une matrice de sorties, les formes ne correspondent pas.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Chaque cadre moderne le fait automatiquement, ce qui empêche la confusion lorsque les formes semblent erronées mais que le code fonctionne.

```figure
vector-projection
```

## Faites-le

### Étape 1: classe vectorielle

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

### Étape 2: classe de matrice avec les opérations de base

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

### Étape 3: voir fonctionner

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

### Étape 4: Connectez-vous aux réseaux neuronaux

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

C' est une seule couche dense:`output = relu(W @ x + b)`Chaque couche dense de chaque réseau neural fait exactement ça.

## Utilisez-le

NumPy fait tout ce qui est en haut en moins de lignes et des ordres de magnitude plus rapidement.

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

Le `@`opérateur dans les appels Python `__matmul__`NumPy le met en œuvre avec des routines BLAS optimisées écrites en C et Fortran.

La diffusion en numPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy diffuse automatiquement le biais 1D sur les deux rangées. C'est ainsi que l'addition de biais fonctionne dans chaque cadre de réseau neuronal.

## La faire partir

Cette leçon fournit un prompt pour enseigner les opérations de matrice par intuition géométrique.`outputs/prompt-matrix-operations.md`- Je suis désolé .

La classe Matrix construite ici est la base du framework de réseau mini-néural que nous construisons dans la phase 3, leçon 10.

## Exercices

1. **Verify the inverse.**Multipliez`A @ A.inverse_2x2()`et confirmez que vous obtenez la matrice d'identité. essayez avec trois matrices 2x2 différentes.

2. **Implement 3x3 inverse.**Élargir la classe Matrix pour calculer les inverses pour les matrices 3x3 en utilisant la méthode adjugée.`np.linalg.inv`- Je suis désolé .

3. **Build a two-layer network.**En utilisant uniquement votre classe Matrix (pas de NumPy), créez un réseau neural à deux couches: entrée (3) -> cachée (4) -> sortie (2). Initialize des poids aléatoires, exécutez un passage vers l'avant et vérifiez que toutes les formes sont correctes.

### Des indices

**Exercise 1.**Multipliez`A @ A.inverse_2x2()`pour trois matrices et comparer chaque cellule avec l'identité à l'intérieur `1e-9`. Pour une matrice avec un déterminant de 0, comme [1, 2], [2, 4], `inverse_2x2`- Il y a un`ValueError`, et c'est le résultat attendu.

**Exercise 2.**Compute la matrice de cofacteurs 3x3, transposez-la pour obtenir l'adjugé, puis divisez chaque cellule par `determinant()`Regarde ça .`A @ A_inv`donne l' identité et que chaque cellule correspond `np.linalg.inv(A)`dans le cadre de la`1e-9`- Je suis désolé .

**Exercise 3.**Écrivez la forme de chaque matrice avant de multiplier: `W1`est 4x3 et `W2`est 2x4, comme dans `demo_neural_network_layer`Une entrée 3x1 doit donner un vecteur caché 4x1 et une sortie 2x1.

## Les termes clés

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

## Pour en savoir plus

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)- l' intuition visuelle pour chaque opération couverte ici
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)- les règles exactes que suit NumPy
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)- référence concise pour l'algèbre linéaire spécifique à la méthode ML

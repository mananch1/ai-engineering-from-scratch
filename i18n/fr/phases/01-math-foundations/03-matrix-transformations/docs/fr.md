# Transformations de matrice

> Une matrice est une machine qui remodèle l'espace. Apprenez ce qu'elle fait à chaque point, et vous comprenez toute la transformation.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lessons 01-02 (Linear Algebra Intuition, Vectors & Matrices Operations)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Construire des matrices de rotation, d'échelle, de découpe et de réflexion et les appliquer aux points 2D et 3D
- Composer plusieurs transformations par multiplication de matrice et vérifier que l'ordre est important
- Compute les valeurs propres et les propres vecteurs des matrices 2x2 à partir de l'équation caractéristique
- Expliquer pourquoi les valeurs propres déterminent les directions PCA, la stabilité RNN et le comportement de regroupement spectrique

## Le problème

Vous lisez sur PCA et voyez " trouver les propres vecteurs de la matrice de covariance. " Vous lisez sur la stabilité du modèle et voyez " vérifier si toutes les valeurs propres ont une magnitude inférieure à 1 ". Vous lisez sur l'augmentation des données et voyez " appliquer une rotation aléatoire. " Rien de tout cela n'a de sens jusqu'à ce que vous compreniez ce que les matrices font à l'espace géométriquement.

Les matrices ne sont pas seulement des grilles de nombres. Ce sont des machines spatiales. Une matrice de rotation fait tourner des points. Une matrice d'échelle les étend. Une matrice de découpe les incline. Chaque transformation d'un réseau neuronal appliquée aux données est l'une de ces opérations ou une composition de celles-ci. Cette leçon rend ces opérations concrètes.

## Le concept

### Transformations en matrices

Chaque transformation linéaire en 2D peut être écrite comme une matrice 2x2. La matrice vous dit exactement où les vecteurs de base [1, 0] et [0, 1] finissent.

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

### Retour

Une rotation 2D par angle $\theta$Il maintient les distances et les angles intacts.

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

En 3D, vous tournez autour d'un axe.

```math
\begin{aligned}
R_z(\theta) &= \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} \quad \text{Rotate around z-axis (x-y plane spins, z stays)} \\[6pt]
R_x(\theta) &= \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix} \quad \text{Rotate around x-axis (y-z plane spins, x stays)} \\[6pt]
R_y(\theta) &= \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix} \quad \text{Rotate around y-axis (x-z plane spins, y stays)}
\end{aligned}
```

### Écalement

Les étirements ou les compressions de l'échelle se font indépendamment le long de chaque axe.

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

### Coupe de poing

Le découpeur incline un axe tout en maintenant l'autre fixe.

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

Matrices de découpe:
- `Shx = [[1, k], [0, 1]]`les changements $x$par $k \cdot y$
- `Shy = [[1, 0], [k, 1]]`les changements $y$par $k \cdot x$

### Réflexion

Les reflets reflètent les points sur un axe ou une ligne.

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

Matrices de réflexion:
- Réflexion à travers l' axe y: `[[-1, 0], [0, 1]]`
- Reflectez à travers l' axe x: `[[1, 0], [0, -1]]`

### Composition: transformations de chaîne

Appliquer la transformation A puis B est la même chose que de multiplier leurs matrices: `result = B @ A @ point`La rotation donne des résultats différents de la rotation.

```mermaid
graph LR
    subgraph Path1["Rotate 90 then Scale (2, 0.5)"]
        P1["(1, 0)"] -->|"Rotate 90"| P2["(0, 1)"] -->|"Scale"| P3["(0, 0.5)"]
    end
```

Composé: `S @ R = [[0, -2], [0.5, 0]]`

```mermaid
graph LR
    subgraph Path2["Scale (2, 0.5) then Rotate 90"]
        Q1["(1, 0)"] -->|"Scale"| Q2["(2, 0)"] -->|"Rotate 90"| Q3["(0, 2)"]
    end
```

Composé: `R @ S = [[0, -0.5], [2, 0]]`

La multiplication de matrice n'est pas commutative.

### Value propre et vecteurs propres

La plupart des vecteurs changent de direction lorsqu'une matrice les atteint. Les vecteurs propres sont spéciaux: la matrice ne les étalonne que, ne les rotate jamais. Le facteur d'échelle est la valeur propre.

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

La matrice étend l'espace de 3x le long de [1, 1] et garde [1, -1] inchangé.

### Composition propre

Si une matrice a$n$vecteurs propres indépendants linéairement, il peut être décomposé:

```
A = V @ D @ V^(-1)

V = matrix whose columns are eigenvectors
D = diagonal matrix of eigenvalues
V^(-1) = inverse of V

This says: rotate into eigenvector coordinates, scale along each axis, rotate back.
```

### Pourquoi les valeurs propres comptent

**PCA.**Les propres vecteurs de la matrice de covariance sont les composants principaux. Les valeurs propres vous indiquent combien de variance chaque composant capture.$k$, et vous avez une réduction de dimensionnalité.

**Stability.**Dans les réseaux récurrents et les systèmes dynamiques, les valeurs propres avec une magnitude > 1 provoquent une explosion des sorties. La magnitude < 1 les fait disparaître.

**Spectral methods.**Les réseaux neuraux du graphe utilisent les valeurs propres de la matrice adjacente. Le regroupement spectrique utilise les valeurs propres du Laplacien. Les propres vecteurs révèlent la structure du graphe.

### Déterminant en tant que facteur d'échelle de volume

Le déterminant d'une matrice de transformation vous indique à quel point il étalonne la surface (2D) ou le volume (3D).

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

## Faites-le

### Étape 1: Matrices de transformation à partir de zéro (Python)

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

### Étape 2: Composition des transformations

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

### Étape 3: Value propre à partir de zéro (2x2)

Pour une matrice 2x2 `[[a, b], [c, d]]`, les valeurs propres résolvent l'équation caractéristique: $\lambda^2 - (a+d)\lambda + (ad - bc) = 0$- Je suis désolé .

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

### Étape 4: Déterminant en tant que facteur d'échelle de volume

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

## Utilisez-le

NumPy gère tout cela avec des routines optimisées.

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

### Rotations 3D avec NumPy

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

## La faire partir

Cette leçon construit les bases géométriques de l'analyse de la PCA (phase 2) et du poids du réseau neuronal. Le code de valeur propre/eigenvecteur construit ici est le même algorithme qui alimente la réduction de dimensionnalité, le regroupement spectrique et l'analyse de stabilité dans les systèmes ML de production.

## Exercices

1. Appliquez la rotation, l'échelle et la découpe sur un carré unitaire (cornes à [0,0], [1,0], [1,1], [0,1]). Imprimez les coins transformés pour chacun. Vérifiez que la rotation préserve les distances entre les coins.

2. Trouvez les valeurs propres de la matrice [[4, 2], [1, 3]] à la main en utilisant l'équation caractéristique.

3. Créer une composition de trois transformations (rotation de 30 degrés, échelle par [1,5, 0,8], coupe avec kx=0,3) et appliquer à 8 points disposés dans un cercle. Imprimer avant et après les coordonnées. Compute le déterminant de la matrice composée et vérifier qu'il est égal au produit des déterminants individuels.

### Des indices

**Exercise 1.**Appliquer `rotation_2d`- Je suis là .`scaling_2d`, et `shearing_2d`à chaque coin avec `mat_vec_mul`Pour la rotation, calculer les 4 longueurs latérales et les 2 diagonales avant et après; chacune doit rester la même.

**Exercise 2.**Écrivez l' équation caractéristique $\lambda^2 - (\text{trace})\lambda + \det = 0$Vérifiez que vos deux valeurs s'ajoutent à la trace (7) et se multiplient par le déterminant (10), puis comparez avec `eigenvalues_2x2`et `np.linalg.eigvals`- Je suis désolé .

**Exercise 3.**Construisez les trois matrices, avec `shearing_2d(0.3, 0)`pour la découpe, multipliez-les par `mat_mul`dans l'ordre de la demande, et appliquer le résultat à 8 points $(\cos\theta, \sin\theta)$La rotation a un déterminant de 1 et la découpe a un déterminant de 1, donc `det_2x2`de la composition doit être égale $1.5 \times 0.8 = 1.2$- Je suis désolé .

## Les termes clés

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

## Pour en savoir plus

- [3Blue1Brown: Linear Transformations](https://www.3blue1brown.com/lessons/linear-transformations)-- l'intuition visuelle pour la façon dont les matrices remodelent l'espace
- [3Blue1Brown: Eigenvectors and Eigenvalues](https://www.3blue1brown.com/lessons/eigenvalues)-- la meilleure explication visuelle de ce que les propres vecteurs signifient géométriquement
- [MIT 18.06 Lecture 21: Eigenvalues and Eigenvectors](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)- Le traitement classique de Gilbert Strang

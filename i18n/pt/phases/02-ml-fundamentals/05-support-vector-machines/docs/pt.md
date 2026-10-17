# Máquinas de apoio de vetores

> Encontrar a rua mais larga entre duas classes.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 1 (Lessons 08 Optimization, 14 Norms and Distances, 18 Convex Optimization)
**Time:** ~90 minutes

## Objetivos de aprendizagem

- Implementar um SVM linear a partir do zero usando perda de biscoitos e descida de gradiente na formulação primária
- Explicar o princípio da margem máxima e identificar vetores de apoio de um modelo treinado
- Comparar kernels lineares, polinômios e RBF e explicar como o truque do kernel evita mapeamento explícito em alta dimensão
- Avaliação da compensação controlada pelo parâmetro C entre largura de margem e erros de classificação

## O problema

Você tem duas classes de pontos de dados e precisa desenhar uma linha (ou hiperplano) separando-os. infinitamente muitas linhas poderiam funcionar. Qual você deve escolher?

A margem é a distância entre o limite de decisão e os pontos de dados mais próximos em cada lado. Uma margem mais ampla significa que o classificador é mais confiante e generaliza melhor os dados invisíveis.

Esta intuição leva a Support Vector Machines, um dos algoritmos mais matematicamente elegantes do ML. Os SVM eram o método de classificação dominante antes da aprendizagem profunda e continuam sendo a melhor escolha para pequenos conjuntos de dados, dados de alta dimensão e problemas em que você precisa de um modelo de princípios bem compreendido com garantias teóricas.

Os SVM se conectam diretamente à Fase 1: a otimização é convexa (Lessão 18), a margem é medida com normas (Lessão 14), e o truque do kernel explora produtos de pontos para lidar com fronteiras não lineares sem nunca computação no espaço de alta dimensão.

## O conceito

### Classificador de margem máxima

Dados linearmente separáveis com rótulos $y_i \in \lbrace -1, +1 \rbrace$e vetores de características $x_i$Queremos um hiperplano .$w^T x + b = 0$que separa as classes.

A distância de um ponto$x_i$para o hiperplano é:

```math
\text{distance} = \frac{\lvert w^T x_i + b \rvert}{\lVert w \rVert}
```

Para um ponto corretamente classificado: $y_i (w^T x_i + b) > 0$A margem é o dobro da distância do hiperplano ao ponto mais próximo de cada lado.

```mermaid
graph LR
    subgraph Margin
        direction TB
        A["w^T x + b = +1"] ~~~ B["w^T x + b = 0"] ~~~ C["w^T x + b = -1"]
    end
    D["+ class points"] --> A
    E["- class points"] --> C
    B --- F["Decision boundary"]
```

O problema da otimização:

```math
\begin{aligned}
\text{maximize} \quad & \frac{2}{\lVert w \rVert} \quad \text{(the margin width)} \\
\text{subject to} \quad & y_i (w^T x_i + b) \ge 1 \quad \text{for all } i
\end{aligned}
```

Equivalentemente (minimizando $\lVert w \rVert^2$é mais fácil de otimizar):

```math
\begin{aligned}
\text{minimize} \quad & \frac{1}{2} \lVert w \rVert^2 \\
\text{subject to} \quad & y_i (w^T x_i + b) \ge 1 \quad \text{for all } i
\end{aligned}
```

Este é um programa quadrático convexo. Ele tem uma solução global única. Os pontos de dados que estão exatamente nos limites de margem (onde $y_i (w^T x_i + b) = 1$Os pontos de apoio são os únicos que determinam o limite de decisão.

### Vectores de apoio: os poucos críticos

```mermaid
graph TD
    subgraph Classification
        SV1["Support Vector (+ class)<br>y(w'x+b) = 1"] --- DB["Decision Boundary<br>w'x+b = 0"]
        DB --- SV2["Support Vector (- class)<br>y(w'x+b) = 1"]
    end
    O1["Other + points<br>(do not affect boundary)"] -.-> SV1
    O2["Other - points<br>(do not affect boundary)"] -.-> SV2
```

A maioria dos pontos de treinamento são irrelevantes. Somente os vetores de suporte são importantes. É por isso que os SVM são eficientes na memória no tempo de previsão: você só precisa armazenar os vetores de suporte, não todo o conjunto de treinamento.

O número de vetores de suporte também dá um limite no erro de generalização.

### Margem suave: ruído de manipulação com o parâmetro C

Os dados reais raramente são perfeitamente separaveis. Alguns pontos podem estar no lado errado da fronteira, ou dentro da margem. A formulação de margem macia permite violações introduzindo variáveis de folga.

```math
\begin{aligned}
\text{minimize} \quad & \frac{1}{2} \lVert w \rVert^2 + C \sum_i \xi_i \\
\text{subject to} \quad & y_i (w^T x_i + b) \ge 1 - \xi_i \\
& \xi_i \ge 0 \quad \text{for all } i
\end{aligned}
```

A variável de desconexão $\xi_i$Medem o valor de pontos$i$viola a margem. $C$Controlar a compensação:

| $C$ value | Behavior |
|---------|----------|
| Large $C$ | Penalizes violations heavily. Narrow margin, fewer misclassifications. Overfits |
| Small $C$ | Allows more violations. Wide margin, more misclassifications. Underfits |

$C$é a força de regularização, invertida.$C$- menos regularização.$C$= mais regularização.

### Perda de barriga: função de perda de SVM

O SVM de margem suave pode ser reescriturado como uma otimização sem restrições:

```math
\text{minimize} \quad \frac{1}{2} \lVert w \rVert^2 + C \sum_i \max(0, 1 - y_i (w^T x_i + b))
```

O termo $\max(0, 1 - y_i f(x_i))$É a perda de biscoito. É zero quando o ponto é corretamente classificado e além da margem. É linear quando o ponto está dentro da margem ou classificado erroneamente.

```
Hinge loss for a single point:

loss
  |
  | \
  |  \
  |   \
  |    \
  |     \_______________
  |
  +-----|-----|-------->  y * f(x)
       0     1

Zero loss when y*f(x) >= 1 (correctly classified, outside margin).
Linear penalty when y*f(x) < 1.
```

Comparar com a perda logística (regressão logística):

```math
\begin{aligned}
\text{Hinge:} \quad & \max(0, 1 - y f(x)) && \text{Hard cutoff at margin} \\
\text{Logistic:} \quad & \log(1 + \exp(-y f(x))) && \text{Smooth, never exactly zero}
\end{aligned}
```

A perda de engarrafamento produz soluções escassas (apenas os vetores de suporte têm contribuição não zero). A perda logística usa todos os pontos de dados.

### Treinamento de um SVM linear com descida de gradiente

Você pode treinar um SVM linear usando descida de gradiente na perda de cartilagem mais regularização L2, sem resolver o QP restrito:

```math
\begin{aligned}
& L(w, b) = \frac{\lambda}{2} \lVert w \rVert^2 + \frac{1}{n} \sum_i \max(0, 1 - y_i (w^T x_i + b)) \\[6pt]
& \text{Gradient with respect to } w\text{:} \\
& \quad \text{If } y_i (w^T x_i + b) \ge 1\text{:} \quad \frac{\partial L}{\partial w} = \lambda w \\
& \quad \text{If } y_i (w^T x_i + b) < 1\text{:} \quad \frac{\partial L}{\partial w} = \lambda w - y_i x_i \\[6pt]
& \text{Gradient with respect to } b\text{:} \\
& \quad \text{If } y_i (w^T x_i + b) \ge 1\text{:} \quad \frac{\partial L}{\partial b} = 0 \\
& \quad \text{If } y_i (w^T x_i + b) < 1\text{:} \quad \frac{\partial L}{\partial b} = -y_i
\end{aligned}
```

Isto é chamado de formulação primária.$O(n \cdot d)$por época, onde $n$é o número de amostras e $d$Para dados grandes, escassos e de alta dimensão (classificação de texto), isto é rápido.

### A dupla formulação e o truque do núcleo

O duplo Lagrangiano do problema SVM (a partir da lição de fase 1, condições KKT) é:

```math
\begin{aligned}
\text{maximize} \quad & \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) \\
\text{subject to} \quad & 0 \le \alpha_i \le C \\
& \sum_i \alpha_i y_i = 0
\end{aligned}
```

O duplo só envolve produtos dotados $x_i \cdot x_j$Esta é a chave para a visão. Substitua cada produto ponto com uma função do kernel$K(x_i, x_j)$E o SVM pode aprender limites não lineares sem nunca calcular a transformação explicitamente.

```math
\begin{aligned}
\text{Linear kernel:} \quad & K(x, z) = x \cdot z \\
\text{Polynomial kernel:} \quad & K(x, z) = (x \cdot z + c)^d \\
\text{RBF (Gaussian):} \quad & K(x, z) = \exp(-\gamma \lVert x - z \rVert^2)
\end{aligned}
```

O kernel RBF mapeia dados em um espaço de dimensões infinitas. Os pontos que estão próximos no espaço de entrada têm valor do kernel perto de 1.

```mermaid
graph LR
    subgraph "Input Space (not separable)"
        A["Data points in 2D<br>circular boundary"]
    end
    subgraph "Feature Space (separable)"
        B["Data points in higher dim<br>linear boundary"]
    end
    A -->|"Kernel trick<br>K(x,z) = phi(x).phi(z)"| B
```

O truque do núcleo calcula o produto de pontos no espaço de alta dimensão sem nunca ir lá.$d$em $D$dimensões, o espaço de características explícitas tem $O(D^d)$Mas...$K(x, z)$é calculado em $O(D)$- O tempo.

### MPS para regressão (MPS)

O VECTOR DE SUPORTA A Regressão se encaixa num tubo de largura $\epsilon$Os pontos dentro do tubo têm perda zero. Os pontos fora do tubo são penalizados linearmente.

```math
\begin{aligned}
\text{minimize} \quad & \frac{1}{2} \lVert w \rVert^2 + C \sum_i (\xi_i + \xi_i^*) \\
\text{subject to} \quad & y_i - (w^T x_i + b) \le \epsilon + \xi_i \\
& (w^T x_i + b) - y_i \le \epsilon + \xi_i^* \\
& \xi_i, \xi_i^* \ge 0
\end{aligned}
```

O $\epsilon$O parâmetro controla a largura do tubo. tubo mais amplo = menos vetores de apoio = mais suave. tubo mais estreito = mais vetores de apoio = mais apertado.

### Por que os SVM perderam para a aprendizagem profunda (e quando ainda ganham)

Os SVM dominaram a ML desde o final dos anos 1990 até o início dos anos 2010.

| Factor | SVMs | Deep learning |
|--------|------|---------------|
| Feature engineering | Requires it | Learns features |
| Scalability | $O(n^2)$ to $O(n^3)$ for kernel | $O(n)$ per epoch with SGD |
| Image/text/audio | Needs handcrafted features | Learns from raw data |
| Large datasets (>100k) | Slow | Scales well |
| GPU acceleration | Limited benefit | Massive speedup |

Os SVM ainda ganham nestas situações:
- Pequenas datas (centenas a milhares de amostras)
- Dados escassos de alta dimensão (texto com características TF-IDF)
- Quando precise de garantias matemáticas (limite de margem)
- Quando o tempo de formação deve ser mínimo (o SVM linear é muito rápido)
- Classificação binária com estrutura de margem clara
- Detecção de anomalias (SVM de uma classe)

```figure
svm-margin
```

## Construí-lo

### Passo 1: perda de barras e gradiente

Calcule a perda de biscoitos para um lote e a sua gradiência.

```python
def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] * (dot(w, X[i]) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n
```

### Passo 2: SVM linear através da descida de gradiente

Treinar minimizando a perda de biscoitos regulares.

```python
class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                margin = y[i] * (dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    self.w = [wj - self.lr * self.lambda_param * wj
                              for wj in self.w]
                else:
                    self.w = [wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j])
                              for j, wj in enumerate(self.w)]
                    self.b -= self.lr * (-y[i])

    def predict(self, X):
        return [1 if dot(self.w, x) + self.b >= 0 else -1 for x in X]
```

### Passo 3: Funções do núcleo

Implementar núcleos lineares, polinômios e RBF.

```python
def linear_kernel(x, z):
    return dot(x, z)

def polynomial_kernel(x, z, degree=3, c=1.0):
    return (dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma=0.5):
    diff = [xi - zi for xi, zi in zip(x, z)]
    return math.exp(-gamma * dot(diff, diff))
```

### Passo 4: Identificação de margens e vetores de apoio

Após o treino, identifique quais pontos são vetores de apoio e calcule a largura da margem.

```python
def find_support_vectors(X, y, w, b, tol=1e-3):
    support_vectors = []
    for i in range(len(X)):
        margin = y[i] * (dot(w, X[i]) + b)
        if abs(margin - 1.0) < tol:
            support_vectors.append(i)
    return support_vectors
```

Veja .`code/svm.py`para a implementação completa com todas as demonstrações.

## Usá-lo

Com a aprendizagem de escikit:

```python
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
])
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
print(f"Support vectors: {clf['svm'].n_support_}")
```

Importante: sempre escalando as suas características antes de treinar um SVM. Os SVM são sensíveis às magnitudes das características porque a margem depende de$\lVert w \rVert$, e as características não escaladas distorcem a geometria.

Para grandes conjuntos de dados, use `LinearSVC`(formulação primária, $O(n)$por época) em vez de `SVC`(formulação dupla, $O(n^2)$- Não .$O(n^3){}$):

```python
from sklearn.svm import LinearSVC

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=1.0, max_iter=10000)),
])
```

## Exercícios

1. Gerar um conjunto de dados linearmente separável em 2D. Treinar o seu LinearSVM e identificar os vetores de suporte. Verificar que os vetores de suporte são os pontos mais próximos do limite de decisão.

2. Vários .$C$Descrever o limite de decisão para cada um $C$Observe a transição da margem larga (sub-ajustamento) para a margem estreita (over-ajustamento).

3. Crie um conjunto de dados onde os limites das classes sejam circulares (não lineares). Mostre que um SVM linear falha. Compute a matriz do kernel RBF e mostre que as classes se tornam separáveis no espaço de recursos induzido pelo kernel.

4. Compare perda de biscoito vs perda logística no mesmo conjunto de dados. Treinar um SVM linear e regressão logística. Contar quantos pontos de treinamento contribuem para o limite de decisão de cada modelo (vectores de suporte vs todos os pontos).

5. Aplicar SVR (perda insensível à epsilon).$y = \sin(x) + \text{noise}$. Desenhar o tubo de epsilon em torno das previsões e destacar os vetores de apoio (pontos fora do tubo).

## Termos-chave

| Term | What it actually means |
|------|----------------------|
| Support vectors | The training points closest to the decision boundary. The only points that determine the hyperplane |
| Margin | The distance between the decision boundary and the nearest support vectors. SVMs maximize this |
| Hinge loss | $\max(0, 1 - y f(x))$. Zero when correctly classified and outside the margin. Linear penalty otherwise |
| C parameter | Trade-off between margin width and classification errors. Large $C$ = narrow margin, small $C$ = wide margin |
| Soft margin | SVM formulation that allows margin violations via slack variables. Handles non-separable data |
| Kernel trick | Computing dot products in a high-dimensional feature space without explicitly mapping to that space |
| Linear kernel | $K(x, z) = x \cdot z$. Equivalent to standard dot product. For linearly separable data |
| RBF kernel | $K(x, z) = \exp(-\gamma \Vert x-z \Vert^2)$. Maps to infinite dimensions. Learns any smooth boundary |
| Polynomial kernel | $K(x, z) = (x \cdot z + c)^d$. Maps to a feature space of polynomial combinations |
| Dual formulation | Reformulation of the SVM problem that depends only on dot products between data points. Enables kernels |
| SVR | Support Vector Regression. Fits an epsilon-tube around the data. Points inside the tube have zero loss |
| Slack variables | $\xi_i$: measures how much a point violates the margin. Zero for correctly classified points outside margin |
| Maximum margin | The principle of choosing the hyperplane that maximizes the distance to the nearest points of each class |

## Mais leitura

- [Vapnik: The Nature of Statistical Learning Theory (1995)](https://link.springer.com/book/10.1007/978-1-4757-3264-1)- o texto fundamental sobre os MSS e a aprendizagem estatística
- [Cortes & Vapnik: Support-vector networks (1995)](https://link.springer.com/article/10.1007/BF00994018)- o papel original do SVM
- [Platt: Sequential Minimal Optimization (1998)](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)- o algoritmo de gestão de dados que tornou prático o treinamento de dados de dados de dados
- [scikit-learn SVM documentation](https://scikit-learn.org/stable/modules/svm.html)- guia prático com detalhes de execução
- [LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)- a biblioteca C++ por trás da maioria das implementações SVM

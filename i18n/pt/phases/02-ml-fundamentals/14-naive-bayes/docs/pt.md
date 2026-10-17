# Bayes ingênuo

> A suposição "ingênua" é errada, e funciona de qualquer maneira.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## Objetivos de aprendizagem

- Implementar Bayes Naívo Multinômio a partir do zero com suavizamento Laplace para classificação de texto
- Explique por que a suposição ingênua de independência é matematicamente errada, mas produz classificações de classes corretas na prática
- Compare as variantes multinomial, bernoulli e gaussiana de Bayes ingênua e selecione a certa para um determinado tipo de característica
- Avaliação de Bayes Ingênuo contra a regressão logística em dados escassos de alta dimensão e explicação da compensação de variação de viés no trabalho

## O problema

Você precisa classificar o texto. E-mails em spam ou não-spam. avaliações de clientes em positivos ou negativos. bilhetes de suporte em categorias. Você tem milhares de recursos (um por palavra) e dados de treinamento limitados.

A maioria dos classificadores se esmaga aqui. A regressão logística precisa de amostras suficientes para estimar milhares de pesos de forma confiável. As árvores de decisão se dividem em uma palavra por vez e se encaixam de forma selvagem.

O Bayes ingênuo lidou com isto. Ele faz uma suposição matematicamente errada (que cada característica é independente de qualquer outra característica dada a classe), e ainda supera os modelos "mais inteligentes" na classificação de texto, especialmente com pequenos conjuntos de treinamento. Ele treina em uma única passagem através dos dados. Escala para milhões de características. Ele produz estimativas de probabilidade (embora muitas vezes mal calibrado devido à suposição de independência).

Entender por que uma suposição errada leva a boas previsões ensina-nos algo fundamental sobre aprendizado de máquina: o melhor modelo não é o mais correto, é aquele com a melhor compensação de variação de viés para os nossos dados.

## O conceito

### Teorema de Bayes (Revisão Rápida)

Teorema de Bayes inverte probabilidades condicionais:

```math
P(\text{class} \mid \text{features}) = \frac{P(\text{features} \mid \text{class}) \cdot P(\text{class})}{P(\text{features})}
```

Queremos .`P(class | features)`-- a probabilidade de que um documento pertença a uma classe dada as palavras nele.
- `P(features | class)`-- a probabilidade de ver estas palavras em documentos desta classe
- `P(class)`-- a probabilidade anterior da classe (como comum é o spam em geral?)
- `P(features)`- a evidência, igual para todas as classes, para que possamos ignorá-la quando comparamos

A classe com o mais alto .`P(class | features)`Ganha.

### A suposição ingênua de independência

Computação`P(features | class)`O que é preciso é estimar a probabilidade conjunta de todas as características juntas.$2^{10{,}000}$- Não é possível.

A suposição ingênua: cada característica é condicionalmente independente dada a classe.

```math
P(w_1, w_2, \ldots, w_n \mid \text{class}) = P(w_1 \mid \text{class}) \cdot P(w_2 \mid \text{class}) \cdots P(w_n \mid \text{class})
```

Em vez de uma distribuição conjunta impossível, estimam$n$distribuições simples por característica. Cada uma precisa apenas de uma contagem.

Esta suposição é obviamente errada. As palavras "máquina" e "aprendizagem" não são independentes em nenhum documento. Mas o classificador não precisa de estimativas de probabilidade corretas. Ele precisa de classificações corretas - qual classe tem a maior probabilidade. A suposição de independência introduz erros sistemáticos, mas esses erros afetam todas as classes de forma semelhante, então a classificação permanece correta.

### Por que ainda funciona

Três razões:

1. **Ranking over calibration.**A classificação só precisa da classe de topo para ser correta.$P(\text{spam}) = 0.99999$Quando a probabilidade real é 0,7, o classificador ainda escolhe o spam corretamente. Não precisamos de probabilidades corretas. Precisamos do vencedor correto.

2. **High bias, low variance.**A suposição de independência é um forte antecedente. Ele restringe fortemente o modelo, o que impede o excesso de encaixe. Com dados de treinamento limitados, um modelo que é ligeiramente errado, mas estável, supera um modelo que é teoricamente correto, mas muito instável.

3. **Feature redundancy cancels out.**As características correlacionadas fornecem evidências redundantes. O classificador contagia duas vezes essas evidências, mas também as contagia para a classe correta. Se "máquina" e "aprendizagem" aparecem sempre juntas, ambas fornecem evidências para a classe "tecnológica". NB as contagia duas vezes, mas as contagia duas vezes para a classe correta.

Uma quarta razão prática: Bayes ingênuo é extremamente rápido. O treinamento é uma única passagem através das frequências de contagem de dados. A previsão é uma multiplicação de matriz. Você pode treinar em um milhão de documentos em segundos. Esta velocidade significa que você pode iterar mais rápido, experimentar mais conjuntos de recursos e executar mais experimentos do que com modelos mais lentos.

### A matemática passo a passo

Vamos rastrear através de um exemplo concreto. Suponhamos que temos duas classes: spam e não-spam. O nosso vocabulário tem três palavras: "gratuito", "dinheiro", "reunião".

Dados de formação:
- Os e-mails de spam mencionam "gratuito" 80 vezes, "dinheiro" 60 vezes, "reunião" 10 vezes (150 palavras no total)
- E-mails não-spam mencionam "gratuito" 5 vezes, "dinheiro" 10 vezes, "reunião" 100 vezes (115 palavras no total)
- 40% dos e-mails são spam, 60% não são spam

Com suavizador Laplace ($\alpha=1$):

```math
\begin{aligned}
P(\text{free} \mid \text{spam}) &= (80 + 1) / (150 + 3) = 81/153 = 0.529 \\
P(\text{money} \mid \text{spam}) &= (60 + 1) / (150 + 3) = 61/153 = 0.399 \\
P(\text{meeting} \mid \text{spam}) &= (10 + 1) / (150 + 3) = 11/153 = 0.072 \\[6pt]
P(\text{free} \mid \text{not-spam}) &= (5 + 1) / (115 + 3) = 6/118 = 0.051 \\
P(\text{money} \mid \text{not-spam}) &= (10 + 1) / (115 + 3) = 11/118 = 0.093 \\
P(\text{meeting} \mid \text{not-spam}) &= (100 + 1) / (115 + 3) = 101/118 = 0.856
\end{aligned}
```

O novo e-mail contém: "gratuito" (2 vezes), "dinheiro" (1 vez), "reunião" (0 vezes).

```math
\begin{aligned}
\log P(\text{spam} \mid \text{email}) &= \log(0.4) + 2 \log(0.529) + 1 \log(0.399) + 0 \log(0.072) \\
&= -0.916 + 2(-0.637) + (-0.919) + 0 \\
&= -3.109 \\[6pt]
\log P(\text{not-spam} \mid \text{email}) &= \log(0.6) + 2 \log(0.051) + 1 \log(0.093) + 0 \log(0.856) \\
&= -0.511 + 2(-2.976) + (-2.375) + 0 \\
&= -8.838
\end{aligned}
```

O spam ganha por uma grande margem. A palavra "gratuito" aparecendo duas vezes é uma prova forte de spam.$0 \cdot \log P$O modelo de ausência de palavras é explicitamente o de Bernoulli NB.

### Três Variantes

O Bayes Ingénuo vem em três sabores.`P(feature | class)`- Não, não.

#### Bayes ingênuo multinômico

Modelos cada característica como uma contagem. Melhor para dados de texto onde as características são frequências de palavras ou valores TF-IDF.

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{count of } \text{word}_i \text{ in class} + \alpha}{\text{total words in class} + \alpha \cdot \text{vocab\_size}}
```

O `alpha`É o suavização Laplace (explicada abaixo). Esta variante é o cavalo de trabalho para classificação de texto.

#### Gaussian Naive Bayes

Modela cada característica como uma distribuição normal.

```math
P(x_i \mid \text{class}) = \frac{1}{\sqrt{2 \pi \cdot \text{var}}} \cdot \exp\left(-\frac{(x_i - \text{mean})^2}{2 \cdot \text{var}}\right)
```

Cada classe tem sua própria média e variação por característica. Isso funciona bem quando as características seguem genuinamente uma curva de sino dentro de cada classe.

#### Bernoulli Naive Bayes

Modela cada característica como binária (presente ou ausente). Melhor para texto curto ou vetores de características binárias.

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{docs in class containing } \text{word}_i + \alpha}{\text{total docs in class} + 2 \alpha}
```

Ao contrário do Multinomial, Bernoulli penaliza explicitamente a ausência de uma palavra. Se "livre" aparece tipicamente no spam, mas está ausente deste e-mail, Bernoulli conta que como evidência contra o spam.

### Quando usar cada variante

| Variant | Feature Type | Best For | Example |
|---------|-------------|----------|---------|
| Multinomial | Counts or frequencies | Text classification, bag-of-words | Email spam, topic classification |
| Gaussian | Continuous values | Tabular data with normal-ish features | Iris classification, sensor data |
| Bernoulli | Binary (0/1) | Short text, binary feature vectors | SMS spam, presence/absence features |

### Laplace Smoothing

O que acontece quando uma palavra aparece nos dados de teste mas nunca aparece nos dados de formação para uma determinada classe?

Sem suavizar: `P(word | class) = 0/N = 0`Um zero multiplicado por todo o produto faz`P(class | features) = 0`Uma única palavra invisível destrói toda a previsão, não importa quantos outros elementos a apoiem.

O suavizamento de laplace acrescenta uma pequena contagem .`alpha`(geralmente 1) para cada número de características:

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{count}(\text{word}_i, \text{class}) + \alpha}{\text{total\_words\_in\_class} + \alpha \cdot \text{vocab\_size}}
```

Com o$\alpha=1$A palavra "discombobulate" que aparece em um e-mail de teste não mata mais a probabilidade de spam.

O alfa superior significa suavização mais forte (distribuições mais uniformes).

Efeito do alfa:

| Alpha | Effect | When to use |
|-------|--------|-------------|
| 0.001 | Almost no smoothing, trust the data | Very large training set, no unseen features expected |
| 0.1 | Light smoothing | Large training set |
| 1.0 | Standard Laplace smoothing | Default starting point |
| 10.0 | Heavy smoothing, flattens distributions | Very small training set, many unseen features expected |

### Computação de log-espaço

Multiplicar centenas de probabilidades (cada uma menor que 1) causa o fluxo inferior de ponto flutuante. O produto se torna zero no ponto flutuante, mesmo que o valor verdadeiro seja um número positivo muito pequeno.

A solução: trabalhar no espaço de log. Em vez de multiplicar as probabilidades, adicionar os seus logaritmos:

```math
\log P(\text{class} \mid x_1, x_2, \ldots, x_n) = \log P(\text{class}) + \sum_i \log P(x_i \mid \text{class})
```

Isto transforma a previsão em um produto de pontos:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

Multiplicação de matriz. É por isso que a previsão de Bayes é tão rápida - é a mesma operação que um modelo linear de camada única.

### Bayes Ingênuo vs Regressão Logística

Ambos são classificadores lineares para texto. A diferença é no que eles modelam.

| Aspect | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| Type | Generative (models $P(X \mid Y){}$) | Discriminative (models $P(Y \mid X){}$) |
| Training | Count frequencies | Optimize loss function |
| Small data | Better (strong prior helps) | Worse (not enough to estimate weights) |
| Large data | Worse (wrong assumption hurts) | Better (flexible boundary) |
| Features | Assumes independence | Handles correlations |
| Speed | Single pass, very fast | Iterative optimization |
| Calibration | Poor probabilities | Better probabilities |

Regra geral: comece com Bayes Ingénuo. Se tiver dados suficientes e planícies NB, passe para regressão logística.

### Classificação de gasodutos

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

Na prática, trabalhamos no espaço log para evitar o fluxo inferior de pontos flutuantes. Em vez de multiplicar muitas pequenas probabilidades, adicionamos os seus logaritmos:

```math
\log P(\text{class} \mid \text{features}) = \log P(\text{class}) + \sum_i \log P(\text{feature}_i \mid \text{class})
```

```figure
naive-bayes
```

## Construí-lo

O código está em `code/naive_bayes.py`Implementa tanto o MultinomialNB como o GaussianNB a partir do zero.

### MultinômioNB

A implementação desde o zero:

1. **fit(X, y)**Para cada classe, conte a frequência de cada característica. Adicione suavização Laplace. Compute probabilidades de registro. Armazenar prioridades de classe (log de frequências de classe).

2. **predict_log_proba(X)**: Para cada amostra, calcular $\log P(\text{class})$+ soma de $\log P(\text{feature}_i \mid \text{class})$Para todas as classes. Esta é uma multiplicação de matriz: X @ log_probs.T + log_priors.

3. **predict(X)**Retorna a classe com maior probabilidade de registro.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

A ideia chave: depois de ser ajustado, a previsão é apenas uma multiplicação de matriz mais um viés.

### GaussianNB

Para características contínuas, estimamos a média e a variância por classe por característica:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

A previsão usa o PDF gaussiano por característica, multiplicado por características (adjunto no espaço de log).

### Demo: Classificação do texto

O código gera dados sintéticos de sacos de palavras que simulam duas classes (artículos tecnológicos vs artigos esportivos). Cada classe tem uma distribuição de frequência de palavras diferente.

Os dados sintéticos funcionam assim: criamos 200 "palavras" (colunas de características). As palavras 0-39 têm alta frequência em artigos de tecnologia e baixa frequência em esportes. As palavras 80-119 têm alta frequência em esportes e baixa frequência em tecnologia. As palavras 40-79 são de frequência média em ambos. Isso cria um cenário realista onde algumas palavras são fortes indicadores de classe e outras são ruídos.

### Demo: Características contínuas

O código gera dados similares a Iris (3 classes, 4 características, aglomerados gaussianos). GaussianNB classifica usando média e variância por classe. Cada classe tem um centro diferente (vector médio) e uma diferença de espalhamento (variância), imitando dados do mundo real onde as medições diferem sistematicamente entre categorias.

O código também demonstra:
- **Smoothing comparison:**Treinar a MultinomialNB com diferentes valores alfa para demonstrar o efeito da força de suavizagem na precisão.
- **Training size experiment:**Como a precisão da NB melhora à medida que os dados de treinamento crescem de 20 para 1600 amostras.
- **Confusion matrix:**Precisão por classe, recall e pontuação da F1 para mostrar onde o NB comete erros.

### Velocidade de previsão

A previsão de Bayes é uma multiplicação de matriz.$n$amostras com $d$características e $k$Classe:
- MultinomialNB: uma matriz multiplicar $(n \times d) \mathbin{@} (d \times k) = O(n \cdot d \cdot k)$
- GaussianNB: $n \cdot k$Avaliações Gaussian PDF, cada uma sobre $d$características = $O(n \cdot d \cdot k)$

Ambos são lineares em todas as dimensões. Compare isso com KNN (que requer computação de distância para todos os pontos de treinamento) ou SVM com kernel RBF (que requer avaliação do kernel contra todos os vetores de suporte). NB é mais rápido por ordens de magnitude no tempo de previsão.

## Usá-lo

Com sklearn, ambas as variantes são de linha única:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

Para classificação de textos com sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

O código está em `naive_bayes.py`Comparar implementações de zero com as de sklearn com base nos mesmos dados para verificar a correcção.

### TF-IDF com Naive Bayes

As contagens de palavras em bruto dão a cada palavra o mesmo peso por ocorrência. Mas palavras comuns como "o" e "é" aparecem com frequência em todas as classes - elas não contêm informações. TF-IDF (Term Frequency - Inverse Document Frequency) reduz o peso das palavras comuns e aumenta o peso das palavras raras e discriminatórias.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

Os valores TF-IDF são não negativos, por isso eles funcionam com MultinomialNB. A combinação de TF-IDF + MultinomialNB é uma das linhas de base mais fortes para classificação de texto. Frequentemente vence modelos mais complexos em conjuntos de dados com menos de 10.000 amostras de treinamento.

### BernoulliNB para texto curto

Para texto curto (tweets, SMS, mensagens de bate-papo), BernoulliNB pode superar MultinomialNB. textos curtos têm baixa contagem de palavras, então a informação de frequência que MultinomialNB depende é barulhenta. BernoulliNB só se importa com presença ou ausência, o que é mais confiável com texto curto.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

O `binary=True`A bandeira no CountVectorizer converte todas as contagens em 0/1. Sem ele, o BernoulliNB ainda funciona, mas está a ver contagens para as quais não foi projetado.

### Calibração NB Probabilidades

As probabilidades de NB são mal calibradas.$P(\text{spam}) = 0.95$Se precisar de estimativas de probabilidade fiáveis (por exemplo, para definir um limiar ou para combinar com outros modelos), use o CalibratedClassifierCV de sklearn:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

Isso corresponde a uma regressão logística em cima das pontuações brutas do NB usando validação cruzada.

### Gotchas comuns

1. **Negative feature values.**MultinomialNB requer recursos não negativos. Se você tem valores negativos (como TF-IDF com certas configurações ou recursos padronizados), use GaussianNB em vez disso, ou mude os recursos para ser positivo.

2. **Zero variance features.**O GaussianNB divide por variância. Se uma característica tem variância zero para uma classe (todos os valores são idênticos), a computação de probabilidade rompe. O código adiciona um pequeno termo de suavizamento (1e-9) a todas as variâncias para evitar isso.

3. **Class imbalance.**Se 99% dos e-mails não são spam, o anterior $P(\text{not-spam}) = 0.99$Pode definir os pré-requisitos de classe manualmente ou usar o parâmetro class_prior no sklearn.

4. **Feature scaling.**O MultinomialNB não precisa de escala (funciona com contagens). O GaussianNB também não precisa de escala (estima estatísticas por característica).

## Envia-o

Esta lição produz:
- `outputs/skill-naive-bayes-chooser.md`-- uma habilidade de decisão para escolher a variante NB certa
- `code/naive_bayes.py`-- MultinomialNB e GaussianNB a partir do zero, com comparação sklearn

### Quando Bayes, ingênuo, falha

A NB falha quando a suposição de independência causa classificações incorretas (não apenas probabilidades incorretas).

1. **Strong feature interactions.**Se a classe depende da combinação de duas características, mas não de qualquer um sozinho (patrões semelhantes a XOR), a NB irá perder completamente.

2. **Highly correlated features with opposing evidence.**Se a característica A diz "spam" e a característica B diz "não-spam", mas A e B estão perfeitamente correlacionadas (sempre concordam na realidade), NB verá evidências conflitantes onde não há nenhuma.

3. **Very large training sets.**Com dados suficientes, modelos discriminativos como a regressão logística aprendem o verdadeiro limite de decisão e superam a NB. A suposição de independência que ajudou com pequenos dados agora retém o modelo.

Na prática, esses modos de falha são raros para classificação de texto. As características do texto são numerosas, individualmente fracas, e os erros da suposição de independência tendem a cancelar. Para dados tabuleários com poucos recursos fortemente correlacionados, considere a regressão logística ou modelos baseados em árvores primeiro.

## Exercícios

1. **Smoothing experiment.**Treinar MultinomialNB em dados de texto com valores alfa de 0.01, 0.1, 1.0, 10.0 e 100.0.

2. **Feature independence test.**Tome um conjunto de dados de texto real. Escolha duas palavras que são obviamente correlacionadas ("máquina" e "aprendizagem").$P(\text{word}_1 \mid \text{class}) \cdot P(\text{word}_2 \mid \text{class})$e comparar com $P(\text{word}_1 \text{ AND } \text{word}_2 \mid \text{class})$O que é que a suposição de independência é errada?

3. **Bernoulli implementation.**Extenda o código com uma classe BernoulliNB. Converte sacos de palavras em binário (presente/ausente) e compare a precisão contra MultinomialNB em dados de texto. Quando Bernoulli ganha?

4. **NB vs Logistic Regression.**Treinar ambos com dados de texto. Comece com 100 amostras de treinamento e aumente para 10.000. A precisão da trama vs tamanho do conjunto de treinamento para ambos. Em que ponto a Regressão Logística ultrapassa o Bayes Ingénuo?

5. **Spam filter.**Construir um classificador de spam completo: tokenizar o texto de e-mail bruto, construir vocabulário, criar recursos de sacos de palavras, treinar MultinomialNB, avaliar com precisão e recordar (não apenas precisão - porquê?).

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Naive Bayes | "Simple probabilistic classifier" | A classifier that applies Bayes' theorem with the assumption that features are conditionally independent given the class |
| Conditional independence | "Features don't affect each other" | $P(A, B \mid C) = P(A \mid C) \cdot P(B \mid C)$ -- knowing $B$ tells you nothing new about $A$ once you know $C$ |
| Laplace smoothing | "Add-one smoothing" | Adding a small count to every feature to prevent zero probabilities from dominating the prediction |
| Prior | "What you believed before seeing data" | $P(\text{class})$ -- the probability of each class before observing any features |
| Likelihood | "How well the data fits" | $P(\text{features} \mid \text{class})$ -- the probability of observing these features if the class is known |
| Posterior | "What you believe after seeing data" | $P(\text{class} \mid \text{features})$ -- the updated probability of the class after observing the features |
| Generative model | "Models how data is generated" | A model that learns $P(X \mid Y)$ and $P(Y)$, then uses Bayes' theorem to get $P(Y \mid X)$ |
| Discriminative model | "Models the decision boundary" | A model that directly learns $P(Y \mid X)$ without modeling how $X$ is generated |
| Log probability | "Avoid underflow" | Working with $\log P$ instead of $P$ to prevent the product of many small numbers from becoming zero in floating point |

## Mais leitura

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html)- As três variantes com detalhes matemáticos
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)-- a comparação clássica de Multinomial vs Bernoulli para texto
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)-- Melhorias no texto
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)-- prova que a NB converge mais rapidamente do que a LR com menos dados

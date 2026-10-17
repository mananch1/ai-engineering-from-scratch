# Introdução ao JAX

> PyTorch muda tensores, TensorFlow cria gráficos, JAX compilou funções puras, e a última altera a forma como pensamos sobre aprendizagem profunda.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## Objetivos de aprendizagem

- Escreva código de rede neural de função pura usando a API funcional do JAX (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Explique a diferença de design fundamental entre a mutação ansiosa do PyTorch e o modelo de compilação funcional do JAX
- Aplicar compilação jit e vectorização vmap para acelerar os ciclos de treinamento em comparação com Python ingênuo
- Treinar uma rede simples no JAX e contrastar a gestão explícita de estado com a abordagem orientada a objeto do PyTorch

## O problema

Sabes como construir redes neurais em PyTorch.`nn.Module`- Não , não .`.backward()`Funciona, milhões de pessoas usam.

Mas a PyTorch tem uma limitação no seu DNA: ela rastreia as operações ansiosamente, uma por vez, no Python.`tensor + tensor`Cada etapa de treinamento reinterpreta o mesmo código Python. Isto funciona bem até que você precisa treinar um modelo de 540 bilhões de parâmetros em 2.048 TPUs.

O Google DeepMind treina Gemini no JAX. O Anthropic treinou Claude no JAX. Não são pequenas operações - são as maiores operações de treinamento de rede neural na Terra. Eles escolheram o JAX porque trata o seu ciclo de treinamento como um programa compilavel, não uma sequência de chamadas Python.

JAX é NumPy com três superpoderes: diferenciação automática, compilação JIT para XLA e vectorização automática. Você escreve uma função que processa um exemplo. JAX lhe dá uma função que processa um lote, calcula gradientes, compila para código de máquina e executa em vários dispositivos. Tudo sem alterar a função original.

## O conceito

### A Filosofia do JAX

O JAX é um quadro funcional, sem classes, sem estados mutáveis, sem`.backward()`- Em vez disso:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

Esta não é uma preferência de estilo. É uma restrição de compilador. A compilação JIT requer funções puras - as mesmas entradas sempre produzem as mesmas saídas, sem efeitos colaterais. Essa restrição é o que torna possível 100x velocidades.

### Jax.numpy: A Superfície Familiar

A JAX reimplementa a API NumPy em aceleradores:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Os mesmos nomes de funções, as mesmas regras de transmissão, a mesma semântica de corte, mas as matriz estão em GPU/TPU, e cada operação é rastreável pelo compilador.

Uma diferença crítica: as matrizes JAX são imutáveis.`a[0] = 5`Em vez disso:`a = a.at[0].set(5)`Isto parece estranho durante uma semana, e depois clique... a imutabilidade é o que faz as transformações como`grad`- Não .`jit`, e `vmap`- Compostabilidade.

### jax.grad: Autodiff funcional

A PyTorch liga gradientes a tensores (`.grad`O JAX liga gradientes às funções.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`O valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de um valor de valor de um valor de um valor de um valor de valor de um valor de um valor de valor de um valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de valor de`.backward()`Não há gráfico de cálculo armazenado em tensores. O gradiente é apenas outra função que você pode chamar, compor ou compilar JIT.

Isto compõe-se arbitrariamente:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Segundo derivativos, terceiro derivativos, jacobianos, hessianos, tudo por composição.`grad`PyTorch também pode fazer isto (`torch.autograd.functional.hessian`No JAX, é a base.

A restrição: `grad`Não há declarações impressas dentro (executa-las durante o rastreamento, não executam). Não há mutação do estado externo. Não há geração de números aleatórios sem gerenciamento de chaves explícito.

### JIT: Compile para XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

Na primeira chamada, o JAX rastreia a função - registra quais operações acontecem, sem executá-las. Depois entrega esse rastro para o XLA (Algebra Linear Acelerada), o compilador do Google para TPUs e GPUs. O XLA funde operações, elimina cópias redundantes de memória e gera código de máquina otimizado.

As chamadas subsequentes ignoram Python inteiramente. O código compilado é executado no acelerador à velocidade de C ++.

Quando o JIT ajuda:
- Passo de treinamento (a mesma computação repetida milhares de vezes)
- Inferência (mesmo modelo, entradas diferentes)
- Qualquer função chamada mais de uma vez com entradas de forma semelhante

Quando a JIT dói:
- Funções com fluxo de controlo Python que dependem de valores (`if x > 0`onde x é uma matriz rastreada)
- Computações de um só momento (compilação superior a tempo de execução)
- Debugging (tracing oculta a execução real)

A restrição de fluxo de controlo é real. `jax.lax.cond`Substitui`if/else`- Não .`jax.lax.scan`Substitui`for`Não são opcionais, são o preço da compilação.

### vmap: Vectorização automática

Você escreve uma função que processa um exemplo:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`Levanta-o para processar um lote:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`Mecanismo: não se recolectar em lote `params`(compartilhado), lote sobre o eixo 0 de `x`Não há manual .`for`Não há remodelação, não há threading de dimensão de lote, o JAX calcula a dimensão de lote e vectoriza toda a computação.

Não é açúcar sintáctico.`vmap`gera código vectorizado fundido que corre 10-100 vezes mais rápido do que um ciclo Python.`jit`E ...`grad`- Não .

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

É quase impossível em PyTorch sem hacks.

### pmap: Paralelismo de dados em dispositivos

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`Replica a função em todos os dispositivos disponíveis (GPUs/TPUs) e divide o lote.`jax.lax.pmean`E ...`jax.lax.psum`Sincronizar gradientes entre dispositivos.

O Google treina os Gémeos através de milhares de chips TPU v5e usando `pmap`(e seu sucessor `shard_map`O modelo de programação: escrever a versão de um único dispositivo, encerrar com `pmap`- Já está.

### Pytrees: A Estrutura Universal de Dados

O JAX opera em "pytrees" - combinações aninhadas de listas, tuples, dicts e matrizes.

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Cada transformação do JAX ...`grad`- Não .`jit`- Não .`vmap`- Sabe atravessar os pytrees.`jax.tree.map(f, tree)`aplica-se `f`É assim que os optimizadores atualizam todos os parâmetros de uma só vez:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Não , não .`.parameters()`Não há registro de parâmetros, a estrutura da árvore é o modelo.

### Funcional vs Orientado a Objetos

As lojas PyTorch afirmam dentro dos objetos:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX usa funções puras com estado explícito:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Os parâmetros são transmitidos. Nada é armazenado. Nada é mutado. Isso torna todas as funções testáveis, compostaveis e compiláveis. Também significa que você gerencia os parâmetros sozinho - ou usa uma biblioteca como o Flax ou Equinox.

### O ecossistema JAX

A JAX dá-te primitivos, as bibliotecas dão-te ergonomia.

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

O Optax é a biblioteca de otimização padrão. Ele separa a transformação de gradiente (Adam, SGD, clipping) da atualização de parâmetros, tornando trivial compor:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### Quando utilizar JAX vs PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class (Google built both) | Community-maintained (torch_xla) |
| GPU support | Good (CUDA via XLA) | Best-in-class (native CUDA) |
| Debugging | Hard (tracing + compilation) | Easy (eager, line-by-line) |
| Ecosystem | Research-focused (Flax, Equinox) | Massive (HuggingFace, torchvision, etc.) |
| Hiring | Niche (Google/DeepMind/Anthropic) | Mainstream (everywhere) |
| Large-scale training | Superior (XLA, pmap, mesh) | Good (FSDP, DeepSpeed) |
| Prototyping speed | Slower (functional overhead) | Faster (mutate and go) |
| Production inference | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Who uses it | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

A resposta honesta é: Use PyTorch a menos que tenha uma razão específica para usar JAX. Essas razões são: acesso a TPU, necessidade de gradientes por exemplo, treinamento em vários dispositivos em escala maciça, ou trabalhar no Google/DeepMind/Anthropic.

### Números aleatórios em JAX

JAX não tem um estado aleatório global.

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

Isto é irritante no início, mas garante reprodução em dispositivos e compilações - uma propriedade que PyTorch é`torch.manual_seed`Não pode garantir em configurações de GPUs múltiplos.

```figure
batchnorm-effect
```

## Construí-lo

### Passo 1: Configuração e dados

Vamos treinar um MLP de 3 camadas no MNIST usando JAX e Optax. 784 entradas, duas camadas ocultas de 256 e 128 neurônios, 10 classes de saída.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Passo 2: Iniciar Parâmetros

Não há classe, apenas uma função que retorna um pytree:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

Três chaves PRNG separadas de uma semente, cada peso é uma matriz imutável num dicto aninhado.

### Passo 3: Passagem para a frente

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Funções puras, param dentro, previsão fora.`self`Não há estado de armazenamento.`loss_fn`Computa a entropia cruzada a partir do zero. Softmax, log, média negativa.

### Passo 4: Passo de formação compilado no JIT

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad`Retorna tanto o valor de perda como os gradientes em uma passagem.`@jax.jit`O decorador compila ambas as funções para XLA. Após a primeira chamada, cada etapa de treinamento é executada sem tocar no Python.

### Passo 5: Loop de treinamento

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 épocas. ~ 97% de precisão de teste. A primeira é lenta (compilação JIT).

Observe o que falta: não .`.zero_grad()`Não , não .`.backward()`Não , não .`.step()`A atualização completa é uma chamada de função composta. Os gradientes são calculados, transformados por Adam e aplicados a parâmetros - todos dentro.`train_step`- Não .

## Usá-lo

### Linho: O padrão do Google

O Flax é a biblioteca mais comum da rede neural JAX.`nn.Module`- De volta, mas com gestão explícita do Estado:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

A mesma estrutura que a PyTorch, mas...`params`é separado do modelo. `model.init()`cria params. `model.apply(params, x)`O objeto modelo não tem estado.

### Equinoxo: A Alternativa Pitônica

Equinox (de Patrick Kidger) representa modelos como pytrees:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

O modelo em si é um pytree.`.apply()`Os parâmetros são apenas as folhas do modelo.

### Optax: Optimizadores compostos

O Optax descopla a transformação de gradiente da atualização:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

O corte de gradientes, o aquecimento da taxa de aprendizagem, a perda de peso, tudo composto por uma cadeia de transformações. Cada transformação vê os gradientes, os modifica e os passa para o próximo. Não há classe de optimizador monolitico.

## Envia-o

**Installation:**

```bash
pip install jax jaxlib optax flax
```

Para o suporte de GPU:

```bash
pip install jax[cuda12]
```

Para TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Performance gotchas:**

- A primeira chamada JIT é lenta (compilação).
- Evite os loops Python sobre matrizes JAX dentro do JIT.`jax.lax.scan`ou `jax.lax.fori_loop`- Não .
- `jax.debug.print()`Funciona dentro do JIT.`print()`Não é.
- Profil com `jax.profiler`A compilação XLA pode esconder gargalos de garrafa.
- JAX pré-aloca 75% da memória da GPU por padrão.`XLA_PYTHON_CLIENT_PREALLOCATE=false`para desativar.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**This lesson produces:**
- `outputs/prompt-jax-optimizer.md`-- um prompt para escolher a configuração certa JAX optimizador
- `outputs/skill-jax-patterns.md`-- uma habilidade que cobre padrões funcionais no JAX

## Exercícios

1. Adicione o desvio ao MLP. No JAX, o desvio requer uma chave PRNG - enrolar uma chave através do passante para a frente e dividir-a para cada camada de desvio. Compare a precisão do teste com e sem.

2. Utilização`jax.vmap`Para calcular gradientes por exemplo para um lote de 32 imagens MNIST. Calcule a norma de gradiente para cada exemplo. Que exemplos têm os maiores gradientes, e por quê?

3. Substitua a função manual para frente por uma genérica `mlp_forward(params, x)`que funciona para qualquer número de camadas.`jax.tree.leaves`para determinar automaticamente a profundidade.

4. Marque de referência o passo de formação com e sem `@jax.jit`Qual é a velocidade do hardware, qual é a taxa de compilação na primeira chamada?

5. Implementar cortes de gradiente através da composição `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`Treinar com e sem cortar, traçar a norma de gradiente sobre o treino para ver o efeito.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| XLA | "The thing that makes JAX fast" | Accelerated Linear Algebra -- a compiler that fuses operations and generates optimized GPU/TPU kernels from a computation graph |
| JIT | "Just-in-time compilation" | JAX traces the function on first call, compiles to XLA, then runs the compiled version on subsequent calls |
| Pure function | "No side effects" | A function where the output depends only on inputs -- no global state, no mutation, no randomness without explicit keys |
| vmap | "Auto-batching" | Transforms a function that processes one example into one that processes a batch, without rewriting |
| pmap | "Auto-parallelism" | Replicates a function across multiple devices and splits the input batch |
| Pytree | "Nested dict of arrays" | Any nested structure of lists, tuples, dicts, and arrays that JAX can traverse and transform |
| Tracing | "Recording the computation" | JAX executes the function with abstract values to build a computation graph, without computing real results |
| Functional autodiff | "grad of a function" | Computing derivatives by transforming functions, not by attaching gradient storage to tensors |
| Optax | "JAX's optimizer library" | A composable library of gradient transformations -- Adam, SGD, clipping, scheduling -- that chain together |
| Flax | "JAX's nn.Module" | Google's neural network library for JAX, adding layer abstractions while keeping state explicit |

## Mais leitura

- Documentação JAX: https://jax.readthedocs.io/- Os docentes oficiais, com excelentes tutoriais sobre grad, jit e vmap
- "JAX: transformações compostas de programas Python+NumPy" (Bradbury et al., 2018) -- o artigo original que explica a filosofia de design
- Documentação de linho: https://flax.readthedocs.io/-- A biblioteca de redes neurais do Google para JAX
- Patrick Kidger, "Equinox: redes neurais no JAX através de PyTrees chamáveis e transformações filtradas" (2021) -- a alternativa Pythonic ao Linho
- DeepMind, "Optax: transformação e otimização de gradientes compostos" -- a biblioteca padrão de otimização
- "You Don't Know JAX" (Colin Raffel, 2020) - um guia prático para as gotchas e padrões do JAX, de um dos autores do T5

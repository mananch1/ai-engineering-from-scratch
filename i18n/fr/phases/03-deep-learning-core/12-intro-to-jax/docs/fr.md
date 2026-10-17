# Introduction à JAX

> PyTorch mutera les tensors, TensorFlow construira des graphiques, JAX compilera des fonctions pures, ce dernier changera la façon dont vous pensez à l'apprentissage profond.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## Objectifs d'apprentissage

- Écrire le code de réseau neural pur fonctionnel à l'aide de l'API fonctionnelle de JAX (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Expliquez la différence de conception clé entre la mutation désireuse de PyTorch et le modèle de compilation fonctionnelle de JAX
- Appliquer la compilation jit et la vectorisation vmap pour accélérer les boucles d'entraînement par rapport à Python naïf
- Formez un réseau simple en JAX et comparez la gestion explicite de l'état avec l'approche orientée objet de PyTorch

## Le problème

Vous savez comment construire des réseaux neuronaux en PyTorch.`nn.Module`- Je vous appelle .`.backward()`Il fonctionne, des millions de personnes l'utilisent.

Mais PyTorch a une contrainte dans son ADN: il suit les opérations avec impatience, une à la fois, en Python.`tensor + tensor`Chaque étape d'entraînement réinterprète le même code Python. Cela fonctionne bien jusqu'à ce que vous ayez besoin d'entraîner un modèle de 540 milliards de paramètres sur 2 048 TPU.

Google DeepMind entraîne Gemini sur JAX. Anthropic a entraîné Claude sur JAX. Ce ne sont pas de petites opérations - ce sont les plus grandes opérations de formation de réseau neuronal sur Terre. Ils ont choisi JAX parce qu'il traite votre boucle de formation comme un programme compilable, pas une séquence d'appels Python.

JAX est NumPy avec trois superpuissances: différenciation automatique, compilation JIT à XLA et vectorisation automatique. Vous écrivez une fonction qui traite un exemple. JAX vous donne une fonction qui traite un lot, calcule les gradients, compile au code machine et fonctionne sur plusieurs appareils. Tout cela sans changer la fonction originale.

## Le concept

### La philosophie de JAX

JAX est un cadre fonctionnel.`.backward()`- Au lieu de cela:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

Ce n'est pas une préférence de style. C'est une contrainte de compilateur. La compilation JIT nécessite des fonctions pures - les mêmes entrées produisent toujours les mêmes sorties, pas d'effets secondaires. Cette restriction est ce qui rend possible des accélérations de 100 fois.

### Jax.numpy: La surface familière

JAX réimplemente l'API NumPy sur les accélérateurs:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Les mêmes noms de fonction, les mêmes règles de diffusion, la même sémantique de découpe, mais les matrices sont en GPU/TPU et chaque opération est traçable par le compilateur.

Une différence importante: les matrices JAX sont immuables.`a[0] = 5`Au lieu de ça:`a = a.at[0].set(5)`Ça fait mal pendant une semaine, puis ça clique -- l'immutabilité est ce qui fait que les transformations ressemblent`grad`- Je suis là .`jit`, et `vmap`- Il est compostable.

### jax.grad: Autodiff fonctionnel

PyTorch attache des gradients aux tensors (`.grad`Le JAX attache des gradients aux fonctions.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`prend une fonction et renvoie une nouvelle fonction qui compute le gradient.`.backward()`Le gradient est une autre fonction que vous pouvez appeler, composer ou compiler JIT.

Il s'agit de composer arbitrairement:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Deuxième dérivés, troisième dérivés, jacobins, hessiens, tous composés par`grad`PyTorch peut aussi faire ça.`torch.autograd.functional.hessian`Dans JAX, c'est la base.

La contrainte: `grad`Il ne fonctionne que sur des fonctions pures. Aucune déclaration d'impression à l'intérieur (elles s'exécutent pendant le suivi, pas l'exécution). Aucune mutation de l'état externe. Aucune génération de nombres aléatoires sans gestion de clés explicite.

### jit: Compile à XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

Au premier appel, JAX trace la fonction - il enregistre les opérations qui se produisent, sans les exécuter. Puis il remet cette trace à XLA (Algebra linéaire accélérée), le compilateur de Google pour les TPU et les GPU. XLA fusionne les opérations, élimine les copies de mémoire redondantes et génère un code machine optimisé.

Les appels suivants sautent complètement Python. Le code compilé fonctionne sur l'accélérateur à la vitesse C ++.

Lorsque le JIT aide:
- Pas de formation (les mêmes calculs répétés des milliers de fois)
- Inference (même modèle, différentes entrées)
- Tout fonction appelée plus d'une fois avec des entrées de forme similaire

Quand le JIT fait mal:
- Fonctions avec le flux de contrôle Python qui dépend des valeurs (`if x > 0`où x est un tableau tracé)
- Comptes à un seul coup (les frais de compilation dépassent le temps de fonctionnement)
- Déboguer (le suivi cache l'exécution réelle)

La restriction de flux de contrôle est réelle. `jax.lax.cond`remplace `if/else`- Je suis là .`jax.lax.scan`remplace `for`Ce ne sont pas facultatifs, ils sont le prix de la compilation.

### vmap: Vectorisation automatique

Vous écrivez une fonction qui traite un exemple:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`le lève pour traiter un lot:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`Les moyens: ne pas décharger `params`(partagé), lot sur l' axe 0 de `x`Pas de manuel .`for`JAX détermine la dimension du lot et vectorifie le calcul entier.

Ce n'est pas du sucre syntaxique.`vmap`génère un code vectorié fusionné qui fonctionne 10 à 100 fois plus vite qu'une boucle Python.`jit`et `grad`- Le numéro de la liste:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

C'est presque impossible dans PyTorch sans hacks.

### pmap: Parallélisme des données entre les appareils

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`La fonction est répliquée sur tous les appareils disponibles (GPU/TPU) et le lot est divisé.`jax.lax.pmean`et `jax.lax.psum`synchroniser les gradients entre les appareils.

Google entraîne les Gémeaux à travers des milliers de puces TPU v5e en utilisant `pmap`(et son successeur `shard_map`Le modèle de programmation: écrire la version à un seul appareil, envelopper avec `pmap`- C'est fait.

### Pytrees: la structure universelle des données

JAX fonctionne sur des "pytrees" - des combinaisons nichées de listes, de tuples, de dicts et d'arrangements.

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Chaque transformation de JAX...`grad`- Je suis là .`jit`- Je suis là .`vmap`- Il sait traverser les pytrees.`jax.tree.map(f, tree)`est appliquée `f`C'est ainsi que les optimisateurs mettent à jour tous les paramètres en même temps:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

- Je ne veux pas .`.parameters()`La structure de l'arbre est le modèle.

### Fonctionnel et orienté sur l'objet

Les magasins PyTorch indiquent à l'intérieur des objets:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX utilise des fonctions pures avec l'état explicite:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Les paramètres sont transmis. Rien n'est stocké. Rien n'est muté. Cela rend chaque fonction testable, comptable et compilable. Cela signifie également que vous gérez les paramètres vous-même - ou utilisez une bibliothèque comme Flax ou Equinox.

### L'écosystème JAX

JAX vous donne des primitifs.

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax est la bibliothèque d'optimisation standard. Elle sépare la transformation de gradient (Adam, SGD, clipping) de la mise à jour des paramètres, ce qui rend trivial la composition:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### Quand utiliser JAX contre PyTorch

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

La réponse honnête: utilisez PyTorch à moins d'avoir une raison spécifique d'utiliser JAX. Ces raisons sont: accès à TPU, besoin de gradients par exemple, formation multi-appareils à grande échelle, ou travailler chez Google/DeepMind/Anthropic.

### Numéros aléatoires dans JAX

JAX n'a pas d'état aléatoire global.

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

C'est ennuyeux au début, mais il garantit la reproductibilité sur tous les appareils et compilations, une propriété que PyTorch a développée.`torch.manual_seed`ne peut garantir dans les paramètres multi-GPU.

```figure
batchnorm-effect
```

## Faites-le

### Étape 1: Configuration et données

Nous allons former un MLP à 3 couches sur le MNIST en utilisant JAX et Optax. 784 entrées, deux couches cachées de 256 et 128 neurones, 10 classes de sortie.

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

### Étape 2: Initialement des paramètres

Pas de classe, juste une fonction qui renvoie un pytre:

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

Il est initialisé manuellement, trois clés PRNG séparées d'une seule graine, chaque poids est un ensemble immutable dans un dicton.

### Étape 3: Pass avant

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

Des paramètres, des prédictions.`self`, aucun état de stockage. `loss_fn`compute l'entropie croisée à partir de zéro - softmax, log, moyenne négative.

### Étape 4: Étape de formation compilée par le JIT

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

`jax.value_and_grad`Le taux de débit est le taux de débit de la valeur de la perte et de la valeur des gradients.`@jax.jit`Le décorateur compile les deux fonctions à XLA. Après le premier appel, chaque étape d'entraînement se déroule sans toucher Python.

### Étape 5: La formation

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

10 époques. ~ 97% de précision de test. La première époque est lente (compilation JIT).

Notez ce qui manque: non `.zero_grad()`- Non , pas du tout .`.backward()`- Non , pas du tout .`.step()`La mise à jour complète est une seule fonction composée appel. Gradients sont calculés, transformés par Adam, et appliqués aux paramètres - tous à l'intérieur`train_step`- Je suis désolé .

## Utilisez-le

### Le lin: la norme Google

Flax est la bibliothèque de réseau neural JAX la plus courante.`nn.Module`- mais avec une gestion explicite de l'État:

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

La même structure que PyTorch, mais `params`est séparé du modèle. `model.init()`crée des paramètres. `model.apply(params, x)`Le modèle n'a pas d'état.

### Équinoxe: l'alternative pythonique

L'équinoxe (de Patrick Kidger) représente les modèles en pytrees:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Le modèle lui-même est un pytre.`.apply()`Les paramètres sont juste les feuilles du modèle.

### Optax: Optimisateurs composables

Optax découple la transformation du gradient de la mise à jour:

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

Le coup de gradients, le réchauffement du rythme d'apprentissage, la perte de poids, tout cela est composé en une chaîne de transformations. Chaque transformation voit les gradients, les modifie et les passe à la suivante. Pas de classe d'optimisateur monolithique.

## La faire partir

**Installation:**

```bash
pip install jax jaxlib optax flax
```

Pour le support de GPU:

```bash
pip install jax[cuda12]
```

Pour le TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Performance gotchas:**

- Le premier appel JIT est lent (compilation).
- Évitez les boucles Python sur les matrices JAX à l'intérieur du JIT.`jax.lax.scan`ou `jax.lax.fori_loop`- Je suis désolé .
- `jax.debug.print()`fonctionne à l'intérieur du JIT.`print()`- Je ne sais pas.
- Profil avec `jax.profiler`La compilation XLA peut cacher des goulots d'étranglement.
- JAX préalloque 75% de la mémoire de la GPU par défaut.`XLA_PYTHON_CLIENT_PREALLOCATE=false`pour désactiver.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**This lesson produces:**
- `outputs/prompt-jax-optimizer.md`-- une demande pour choisir la bonne configuration de l'optimisateur JAX
- `outputs/skill-jax-patterns.md`-- une compétence couvrant les schémas fonctionnels dans JAX

## Exercices

1. Ajouter le dérapagement au MLP. Dans JAX, le dérapagement nécessite une clé PRNG - fil d'une clé à travers le passage avant et la diviser pour chaque couche de dérapagement. Comparer la précision du test avec et sans.

2. Utilisation `jax.vmap`Pour calculer les gradients par exemple pour un lot de 32 images MNIST.

3. Remplacez la fonction manuelle avant par une fonction générique `mlp_forward(params, x)`qui fonctionne pour un certain nombre de couches.`jax.tree.leaves`pour déterminer automatiquement la profondeur.

4. Résumé de l'étape de formation avec et sans`@jax.jit`Combien de temps pour accélérer votre matériel, combien de temps pour compiler le premier appel ?

5. Implémenter la découpe de gradient en composant `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`Trainer avec et sans coupe.

## Les termes clés

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

## Pour en savoir plus

- Documents JAX: https://jax.readthedocs.io/- les docteurs officiels, avec d'excellents tutoriels sur le diplôme, le jit et vmap
- "JAX: transformations composables des programmes Python+NumPy" (Bradbury et coll., 2018) -- le document original expliquant la philosophie de conception
- Documents en lin: https://flax.readthedocs.io/-- La bibliothèque de réseaux neuraux de Google pour JAX
- Patrick Kidger, "Equinox: réseaux neuronaux dans JAX via des PyTrees appelables et des transformations filtrées" (2021) -- l'alternative pythonic au lin
- DeepMind, "Optax: transformation et optimisation des gradients composables" -- la bibliothèque standard de l'optimisateur
- " Vous ne connaissez pas JAX " (Colin Raffel, 2020) - un guide pratique sur les goutchas et les modèles de JAX, de l'un des auteurs de T5

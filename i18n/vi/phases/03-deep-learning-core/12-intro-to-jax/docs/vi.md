# Khởi đầu với JAX

> PyTorch biến đổi các tensor. TensorFlow xây dựng đồ thị. JAX biên soạn các hàm tinh khiết.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## Mục tiêu học tập

- Viết mã mạng thần kinh chức năng thuần túy bằng cách sử dụng API chức năng của JAX (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Giải thích sự khác biệt thiết kế chính giữa đột biến nhiệt tình của PyTorch và mô hình biên soạn chức năng của JAX
- Sử dụng biên soạn jit và vmap vectorization để tăng tốc vòng đào tạo so với Python ngây thơ
- Trình luyện một mạng đơn giản trong JAX và so sánh quản lý trạng thái rõ ràng với cách tiếp cận định hướng đối tượng của PyTorch

## Vấn đề

Bạn biết cách xây dựng mạng thần kinh trong PyTorch.`nn.Module`, gọi `.backward()`Nó hoạt động, hàng triệu người sử dụng nó.

Nhưng PyTorch có một hạn chế trong DNA của nó: nó theo dõi các hoạt động với sự nhiệt tình, một lần một, trong Python.`tensor + tensor`mỗi bước đào tạo lại giải thích lại cùng một mã Python. Điều này hoạt động tốt cho đến khi bạn cần đào tạo một mô hình thông số 540 tỷ trên 2.048 TPU.

Google DeepMind đào tạo Gemini trên JAX. Anthropic đào tạo Claude trên JAX. Đây không phải là các hoạt động nhỏ - chúng là các hoạt động đào tạo mạng thần kinh lớn nhất trên Trái Đất. Họ chọn JAX vì nó xử lý vòng đào tạo của bạn như một chương trình có thể biên dịch, không phải là một chuỗi các cuộc gọi Python.

JAX là NumPy với ba siêu năng lực: phân biệt tự động, biên soạn JIT thành XLA và vectorization tự động. Bạn viết một hàm xử lý một ví dụ. JAX cho bạn một hàm xử lý một lô, tính toán gradient, biên soạn thành mã máy và chạy trên nhiều thiết bị. Tất cả mà không thay đổi chức năng ban đầu.

## Khái niệm

### Triết lý JAX

JAX là một hệ thống chức năng không có lớp, không có trạng thái thay đổi, không có`.backward()`Thay vào đó:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

Đây không phải là một sự ưu tiên về phong cách. Đó là một hạn chế biên dịch. Việc biên dịch JIT đòi hỏi các chức năng tinh khiết - cùng đầu vào luôn tạo ra cùng một kết quả, không có tác dụng phụ.

### jax.numpy: The Familiar Surface

JAX tái triển khai API NumPy trên các bộ đẩy:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

cùng tên chức năng, cùng quy tắc phát sóng, cùng ngữ nghĩa cắt, nhưng các mảng hoạt động trên GPU/TPU, và mọi hoạt động đều có thể theo dõi bởi trình biên dịch.

Một sự khác biệt quan trọng: các mảng JAX không thay đổi.`a[0] = 5`Thay vào đó:`a = a.at[0].set(5)`Điều này cảm thấy khó khăn trong một tuần, sau đó nó nhấp vào -- sự không thay đổi là điều làm cho những biến đổi như`grad`- `jit`, và`vmap`- Đơn vị.

### jax.grad: Functional Autodiff

PyTorch gắn gradient với các tensor (`.grad`JAX gắn gradient với các hàm.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`lấy một hàm và trả lại một hàm mới tính toán gradient.`.backward()`không có biểu đồ tính toán được lưu trữ trên các tensor. gradient chỉ là một chức năng khác bạn có thể gọi, soạn, hoặc JIT-compile.

Điều này tạo thành tùy tiện:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Các phái sinh thứ hai, phái sinh thứ ba, Jacobian, Hessian, tất cả bằng cách tạo ra`grad`PyTorch cũng có thể làm điều này (`torch.autograd.functional.hessian`Trong JAX, nó là nền tảng.

Sự hạn chế:`grad`Không có lệnh in bên trong (bạn chạy trong quá trình theo dõi, không thực hiện). Không có đột biến của trạng thái bên ngoài. Không tạo ra số ngẫu nhiên mà không có quản lý khóa rõ ràng.

### jit: Sẵn sàng để XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

Khi gọi đầu tiên, JAX theo dõi chức năng - nó ghi lại các hoạt động xảy ra mà không thực hiện chúng. Sau đó nó đưa manh mối đó đến XLA (Quá trình lập trình tuyến tính tăng tốc), bộ sưu tập của Google cho TPU và GPU. XLA hợp nhất các hoạt động, loại bỏ các bản sao bộ nhớ dư thừa, và tạo ra mã máy tối ưu hóa.

Các cuộc gọi sau đó bỏ qua Python hoàn toàn. Mã được biên soạn chạy trên bộ tăng tốc ở tốc độ C ++.

Khi JIT giúp:
- Các bước đào tạo (sự tính toán tương tự lặp lại hàng ngàn lần)
- Tự luận (một mô hình, đầu vào khác nhau)
- Bất kỳ hàm nào được gọi nhiều hơn một lần với các đầu vào hình dạng tương tự

Khi JIT đau:
- Các hàm với dòng kiểm soát Python phụ thuộc vào các giá trị (`if x > 0`nơi x là một mảng được theo dõi)
- Các tính toán một lần (giá tổng hợp vượt quá thời gian chạy)
- Debug (tracing che giấu thực tế thực hiện)

Sự hạn chế lưu lượng kiểm soát là thực. `jax.lax.cond`thay thế `if/else`- `jax.lax.scan`thay thế `for`Loops. Đây không phải là tùy chọn - đó là giá của việc biên soạn.

### vmap: Vektor hóa tự động

Bạn viết một hàm xử lý một ví dụ:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`nâng nó để xử lý một lô:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`phương tiện: không đợt đợt `params`(cùng), lô trên trục 0 của `x`Không có hướng dẫn.`for`không có hình dạng lại, không có chuỗi kích thước lô, JAX tính ra kích thước lô và vector hóa toàn bộ tính toán.

Đây không phải là đường tổng hợp.`vmap`tạo ra mã vector hóa hợp nhất chạy nhanh hơn 10-100 lần so với một vòng lặp Python.`jit`và `grad`- Có thể là:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Một đường, điều này gần như không thể trong PyTorch mà không có hack.

### pmap: Sự song song dữ liệu trên các thiết bị

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`Tái bản chức năng trên tất cả các thiết bị có sẵn (GPU / TPU) và chia các lô.`jax.lax.pmean`và `jax.lax.psum`đồng bộ hóa gradient trên các thiết bị.

Google đào tạo Gemini qua hàng ngàn chip TPU v5e sử dụng `pmap`(và người kế nhiệm của nó)`shard_map`). Mô hình lập trình: viết phiên bản đơn thiết bị, kết thúc với `pmap`- Được rồi.

### Pytrees: Cơ cấu dữ liệu phổ quát

JAX hoạt động trên "pytrees" - kết hợp tổ hợp của danh sách, tuples, dicts, và array.

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Mỗi sự biến đổi của JAX...`grad`- `jit`- `vmap`- biết cách vượt qua cây Pytrees.`jax.tree.map(f, tree)`áp dụng `f`Đây là cách mà các trình tối ưu hóa cập nhật tất cả các tham số cùng một lúc:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Không .`.parameters()`Không có ký hiệu tham số.

### Phục vụ đối với hướng đối tượng

Các cửa hàng PyTorch cho biết bên trong các vật thể:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX sử dụng các hàm thuần khiết với trạng thái rõ ràng:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Các param được truyền vào. Không có gì được lưu trữ. Không có gì được đột biến. Điều này làm cho mọi chức năng có thể kiểm tra, hợp tác và được biên soạn. Nó cũng có nghĩa là bạn tự quản lý các param - hoặc sử dụng thư viện như Flax hoặc Equinox.

### Hệ sinh thái JAX

JAX cho bạn những thứ nguyên thủy. Thư viện cho bạn những thứ ergonomic:

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax là thư viện tối ưu hóa tiêu chuẩn. Nó tách chuyển đổi gradient (Adam, SGD, cắt) khỏi bản cập nhật tham số, khiến nó trở nên tầm thường để soạn:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### Khi nào sử dụng JAX vs PyTorch

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

Câu trả lời trung thực: sử dụng PyTorch trừ khi bạn có lý do cụ thể để sử dụng JAX. Những lý do đó là: truy cập TPU, nhu cầu gradient mỗi ví dụ, đào tạo đa thiết bị quy mô lớn, hoặc làm việc tại Google/DeepMind/Anthropic.

### Số ngẫu nhiên trong JAX

JAX không có trạng thái ngẫu nhiên toàn cầu.

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

Điều này ban đầu khó chịu, nhưng nó đảm bảo khả năng tái tạo trên các thiết bị và các bộ sưu tập - một tính năng mà PyTorch đã tạo ra`torch.manual_seed`không thể đảm bảo trong cài đặt nhiều GPU.

```figure
batchnorm-effect
```

## Hãy xây dựng nó

### Bước 1: Thiết lập và dữ liệu

Chúng tôi sẽ đào tạo một MLP 3 tầng trên MNIST sử dụng JAX và Optax. 784 đầu vào, hai lớp ẩn của 256 và 128 tế bào thần kinh, 10 lớp đầu ra.

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

### Bước 2: Tạo ra các tham số

Không có lớp, chỉ là một hàm trả lại một cây:

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

- He-initialisation, làm bằng tay ba phím PRNG tách ra từ một hạt.

### Bước 3: Đi trước

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

- Các chức năng tinh khiết, các param vào, dự đoán ra.`self`, không lưu trữ trạng thái. `loss_fn`tính toán sự chuyển đổi từ đầu -- softmax, log, trung bình âm.

### Bước 4: Bước đào tạo được biên soạn bằng JIT

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

`jax.value_and_grad`trả lại cả giá trị mất và gradient trong một lần đi.`@jax.jit`khi thiết kế kết hợp cả hai chức năng cho XLA. Sau cuộc gọi đầu tiên, mỗi bước đào tạo chạy mà không chạm vào Python.

### Bước 5: Lòng huấn luyện

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

10 thời đại. ~ 97% độ chính xác thử nghiệm. thời đại đầu tiên chậm (sự biên soạn JIT).

Nhìn xem thiếu gì: không `.zero_grad()`Không .`.backward()`Không .`.step()`Toàn bộ bản cập nhật là một cuộc gọi hàm tổng hợp. Các gradient được tính toán, biến đổi bởi Adam, và áp dụng cho các tham số - tất cả bên trong`train_step`- Tôi không biết.

## Sử dụng nó

### Lựa: tiêu chuẩn Google

Flax là thư viện mạng thần kinh JAX phổ biến nhất.`nn.Module`trở lại, nhưng với quản lý nhà nước rõ ràng:

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

Tương tự như PyTorch, nhưng `params`được tách biệt với mô hình. `model.init()`tạo ra Params. `model.apply(params, x)`chạy đường đi trước. đối tượng mô hình không có trạng thái.

### Tương đương: Phương pháp thay thế Pythonic

Equinox (do Patrick Kidger) đại diện cho các mô hình như các cây pytrees:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Bản thân mô hình là một cây Pytree.`.apply()`Các thông số chỉ là lá của mô hình.

### Optax: Optimizers hợp tác

Optax tách chuyển đổi gradient từ bản cập nhật:

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

Giảm gradient, tăng tốc độ học tập, giảm cân - tất cả đều được tạo thành như một chuỗi chuyển đổi. Mỗi chuyển đổi nhìn thấy gradient, sửa đổi chúng, và chuyển chúng sang lớp tiếp theo. Không có lớp tối ưu hóa đơn phương.

## Chuyển nó

**Installation:**

```bash
pip install jax jaxlib optax flax
```

Đối với hỗ trợ GPU:

```bash
pip install jax[cuda12]
```

Đối với TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Performance gotchas:**

- Cuộc gọi đầu tiên của JIT là chậm (sự biên soạn).
- Tránh các vòng Python trên các mảng JAX bên trong JIT. Sử dụng `jax.lax.scan`hoặc `jax.lax.fori_loop`- Tôi không biết.
- `jax.debug.print()`làm việc trong JIT.`print()`Không.
- Hình ảnh với `jax.profiler`XLA có thể che giấu những lỗ hổng.
- JAX dự định phân bổ 75% bộ nhớ GPU theo mặc định.`XLA_PYTHON_CLIENT_PREALLOCATE=false`để vô hiệu hóa.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**This lesson produces:**
- `outputs/prompt-jax-optimizer.md`-- một lời nhắc cho việc chọn đúng cấu hình tối ưu hóa JAX
- `outputs/skill-jax-patterns.md`-- một kỹ năng bao gồm các mô hình chức năng trong JAX

## Các bài tập

1. Thêm dropup vào MLP. Trong JAX, dropup đòi hỏi một phím PRNG - đinh một phím qua các bước đi phía trước và chia nó cho mỗi lớp dropup. So sánh độ chính xác của thử nghiệm với và ngoài.

2. Sử dụng `jax.vmap`để tính toán gradient cho mỗi ví dụ cho một loạt 32 hình ảnh MNIST. tính toán chuẩn gradient cho mỗi ví dụ. ví dụ nào có gradient lớn nhất, và tại sao?

3. Thay thế hàm hướng về phía trước bằng một hàm chung `mlp_forward(params, x)`Nó có thể hoạt động cho bất kỳ lớp nào.`jax.tree.leaves`để xác định độ sâu tự động.

4. Đánh giá bước đào tạo với và không `@jax.jit`Thời gian 100 bước mỗi lần. tốc độ trên phần cứng của bạn là bao nhiêu?

5. Thực hiện cắt gradient bằng cách tạo `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`Tren với và không cắt, vẽ chuẩn gradient trên tập để xem hiệu quả.

## Các điều khoản chính

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

## Đọc thêm

- Tài liệu JAX: https://jax.readthedocs.io/- Các bác sĩ chính thức, với các hướng dẫn tuyệt vời về Graduate, jit, và vmap
- "JAX: những biến đổi hợp nhất của các chương trình Python+NumPy" (Bradbury et al., 2018) - bài báo ban đầu giải thích triết lý thiết kế
- Tài liệu bằng len: https://flax.readthedocs.io/-- Thư viện mạng thần kinh của Google cho JAX
- Patrick Kidger, "Equinox: mạng thần kinh trong JAX thông qua PyTrees có thể gọi và chuyển đổi lọc" (2021) - sự thay thế Pythonic cho Flax
- DeepMind, "Optax: biến đổi và tối ưu hóa gradient hợp nhất" -- thư viện tối ưu hóa tiêu chuẩn
- "You Don't Know JAX" (Colin Raffel, 2020) - một hướng dẫn thực tế về các trò chơi và mô hình JAX, từ một trong những tác giả của T5

# معرفی به JAX

> پایتورچ تنسورها را جهش می دهد. تنسور فلو نمودارها را ایجاد می کند. جاکس عملکردهای خالص را جمع آوری می کند. آخرین یکی نحوه تفکر شما در مورد یادگیری عمیق را تغییر می دهد.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## اهداف یادگیری

- کد شبکه عصبی عملکرد خالص را با استفاده از API عملکردی JAX بنویسید (jax.numpy، jax.grad، jax.jit، jax.vmap)
- تفاوت اصلی طراحی بین جهش مشتاق PyTorch و مدل کامپیلیشن عملکردی JAX را توضیح دهید
- استفاده از مجموعه jit و ویکتور سازی vmap برای سرعت بخشیدن به حلقه های آموزش در مقایسه با پایتون ساده
- آموزش یک شبکه ساده در JAX و مقایسه مدیریت دولت صریح با رویکرد هدفمند PyTorch

## مشکل

تو می دونی چطور شبکه های عصبی رو در PyTorch بسازی`nn.Module`، تماس بگیرید`.backward()`،و به سمت بهینه ساز فشار بده ، کار ميکنه . ميليون ها نفر ازش استفاده ميکنن

اما PyTorch یک محدودیت در DNA اش پخته شده است: آن را به دنبال عملیات مشتاقانه، یکی به یک زمان، در پایتون.`tensor + tensor`هر مرحله آموزش دوباره همان کد پایتون را تفسیر می کند. این کار خوب کار می کند تا زمانی که شما نیاز به آموزش یک مدل پارامتر 540 میلیارد در 2048 TPU دارید. سپس هزینه های بالای شما را می کشد.

گوگل DeepMind دوقلوها را بر روی JAX آموزش می دهد. آنترپیک کلود را بر روی JAX آموزش می دهد. این عملیات های کوچک نیستند - این بزرگترین عملیات آموزش شبکه عصبی در زمین هستند. آنها JAX را انتخاب کردند زیرا این چرخه آموزش شما را به عنوان یک برنامه قابل مرتب می کند، نه یک ردیف تماس های پایتون.

JAX با سه ابرقدرت: تفاوت خودکار، جمع آوری JIT به XLA و ویکتور سازی خودکار است. شما یک تابع را می نویسید که یک مثال را پردازش می کند. JAX به شما یک تابع را می دهد که یک دسته را پردازش می کند، گرادیانت ها را محاسبه می کند، به کد ماشین را مرتب می کند و در چندین دستگاه اجرا می کند. همه بدون تغییر عملکرد اصلی.

## مفهوم

### فلسفه جاکس

JAX يه چارچوبي فعاله بدون کلاس، بدون حالت متغير، نه`.backward()`روش.به جایش:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

این یک انتخاب سبک نیست. این یک محدودیت کامپایلر است. جمع آوری JIT نیازمند عملکردهای خالص است - ورودی های مشابه همیشه نتایج مشابهی را تولید می کنند، هیچ عوارض جانبی وجود ندارد. این محدودیت چیزی است که باعث می شود سرعت 100 برابر ممکن باشد.

### jax.numpy: سطح آشنا

JAX API NumPy را در تسریع کننده ها مجدداً پیاده سازی می کند:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

اسم هاي تابع همديگه، قوانين پخش همديگه، همديگه سيمانتیک برش دادن، اما آرایه ها روي GPU/TPU زنده هستند و هر عمل توسط کامپايلر قابل ردیابی است

يه تفاوت مهم: آرایه هاي JAX غير قابل تغيير هستن`a[0] = 5`. در عوض:`a = a.at[0].set(5)`این یک هفته عجیب و غریب است، سپس می کند -- تغییر ناپذیر بودن چیزی است که تحولات را شبیه به`grad`،`jit`و`vmap`قابل تدوین

### jax.grad: خودکشی عملکردی

پیتورچ گرادینت ها را به تنسورها متصل می کند (`.grad`جاکس گرادینت ها را به تابع ها متصل می کند.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`یک تابع را می گیرد و یک تابع جدید را که گرادینت را محاسبه می کند، باز می آورد.`.backward()`هیچ گراف محاسباتی در تنسورها ذخیره نشده است. گرادینت فقط یک تابع دیگر است که می توانید آن را فرا بگیرید، یا ترکیب کنید یا JIT-کمپیل کنید.

این به طور تعسفی تشکیل می شود:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

مشتقات دوم، مشتقات سوم، جاکوبیان، هسیان، همه با ترکیب`grad`.پایتورچ هم می تونه این کار رو بکنه`torch.autograd.functional.hessian`در جاکس، این پایه است.

محدودیت:`grad`هیچ گونه بیان نامه ای در داخل (آن ها در هنگام ردیابی اجرا می شوند، نه اجرا می شوند) هیچ جهش در حالت خارجی. هیچ تولید شماره تصادفی بدون مدیریت کلید صریح نیست.

### jit: به XLA کامپایل کنید

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

در اولین تماس، JAX عملکرد را ردیابی می کند - آن را ثبت می کند که کدام عملیات انجام می شود، بدون اجرای آنها. سپس آن را به XLA (الجهبری خطی سریع) ، کمپایلر Google برای TPU ها و GPU ها می رساند. XLA عملیات را ترکیب می کند، کپی های حافظه اضافی را از بین می برد و کد ماشین بهینه سازی شده تولید می کند.

تماس های بعدی به طور کامل از پایتون خارج می شوند. کد مرتب شده در سرعت افزونه C ++ اجرا می شود.

وقتی JIT کمک می کند:
- مراحل آموزش (همین محاسبه هزاران بار تکرار می شود)
- انفرنس (مثل مدل، ورودی های مختلف)
- هر تابع که بیش از یک بار با ورودی های مشابه نامیده شود

وقتی JIT درد میکنه:
- عملکردهای با جریان کنترل پایتون که به ارزش ها بستگی دارد (`if x > 0`جایی که x یک ردیف ردیابی است)
- محاسبه های یکبار (آموزش های مرتب شده بیش از زمان اجرا)
- بازیابی (تراسیج اجرای واقعی را پنهان می کند)

محدودیت جریان کنترل واقعیه`jax.lax.cond`جایگزین می شود`if/else`.`jax.lax.scan`جایگزین می شود`for`حلقه ها. این ها اختیاری نیستند. این ها قیمت جمع آوری هستند.

### vmap: متریزیشن خودکار

شما یک تابع را می نویدید که یک مثال را پردازش می کند:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`آن را برای پردازش یک دسته بلند می کند:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`روش: دسته بندی نکنید `params`(مشترکه) ، دسته ای از محور 0 از `x`. هيچ راهنمايي`for`.چاپ .بدون تغییر شکل .بدون رشته ابعاد دسته بندی .جاکس ابعاد دسته را مشخص می کند و کل محاسبات را متری می کند

اين قند سنتکسي نيست`vmap`کد متریزه ای مخلوط تولید می کند که 10-100 برابر سریعتر از یک حلقه پایتون اجرا می شود. و این ترکیب با `jit`و`grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

هر نمونه يه خط اين تقريباً بدون هک ها در PyTorch ناممکن است

### pmap: موازی داده ها در دستگاه ها

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`تکرار تابع در تمام دستگاه های موجود (GPU / TPU) و تقسیم دسته. در داخل تابع، `jax.lax.pmean`و`jax.lax.psum`هم وقت سازی گرادینت ها در دستگاه ها

گوگل با استفاده از هزاران تراشه TPU v5e، دوقلوها را آموزش می دهد`pmap`(و جانشینش)`shard_map`) مدل برنامه نویسی: نسخه ی دستگاه یگانه را بنویسید، با `pmap`، تموم شد

### پیترز: ساختار داده جهانی

جاکس روی "پایتری" کار می کند - ترکیبی از لیست ها، توپل ها، دیکت ها و آرایه ها. پارامترهای مدل شما یک پایتری هستند:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

هر تحول جاکس`grad`،`jit`،`vmap`-مي داند چطور از درختان پيتر عبور کنه`jax.tree.map(f, tree)`اعمال می شود`f`این روش است که بهینه سازی کننده ها تمام پارامترها را به یک بار به روز می کنند:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

نه`.parameters()`روش. بدون ثبت پارامتر. ساختار درخت مدل است.

### عملکردی در مقابل هدفمند

فروشگاه های PyTorch در داخل اشیاء می گویند:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX از تابع های خالص با حالت صریح استفاده می کند:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

پارام ها منتقل می شوند. هیچ چیز ذخیره نمی شود. هیچ چیز جهش نمی یابد. این باعث می شود که هر عملکرد قابل آزمایش، قابل ترکیب و قابل تجمع باشد. همچنین به این معنی است که شما خود پارام ها را مدیریت می کنید - یا از یک کتابخانه مانند فلان یا یکوینوکس استفاده کنید.

### اکوسیستم جی ای ایکس

جاکس به شما ابتدایی ها می دهد کتابخانه ها به شما ارگونومی می دهند:

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax کتابخانه بهینه سازی استاندارد است. این تغییر گرادینت (آدم، SGD، کپی) را از بروزرسانی پارامتر جدا می کند، بنابراین ترکیب آن معمولی است:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### چه زمانی باید JAX vs PyTorch را استفاده کنید

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

پاسخ صادقانه: از PyTorch استفاده کنید مگر اینکه دلیل خاصی برای استفاده از JAX داشته باشید. این دلایل این هستند: دسترسی به TPU، نیاز به هر مثال gradients، آموزش چند دستگاه در مقیاس گسترده، یا کار در گوگل / DeepMind / Anthropic.

### شماره های تصادفی در JAX

JAX دارای حالت تصادفی جهانی نیست. هر عملیات تصادفی نیازمند کلید PRNG صریح است:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

این در ابتدا ناراحت کننده است. اما این قابلیت بازیافت در دستگاه ها و مجموعه ها را تضمین می کند. یک ویژگی که PyTorch`torch.manual_seed`نمی تواند در تنظیمات چند GPU تضمین کند.

```figure
batchnorm-effect
```

## آن را بسازید

### مرحله اول: تنظیم و داده ها

ما با استفاده از JAX و Optax یک MLP سه لایه را در MNIST آموزش می دهیم. 784 ورودی، دو لایه مخفی از 256 و 128 نورون، 10 کلاس خروجی.

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

### مرحله دوم: شروع کردن پارامترها

هیچ کلاس نیست فقط یک تابع که یک پیتر را باز می گرداند:

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

اون شروع به کار ميکنه، دستي انجام ميده، سه تا کليد PRNG از يك دانه جدا شده هر وزن يه تشکيل غير قابل تغيير در يک فرماني هست

### مرحله سوم: عبور جلو

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

عملکرد خالص، پارامز وارد، پیش بینی خارج`self`، هیچ حالت ذخیره نشده`loss_fn`این در واقع یک مقدار بسیار زیاد است.

### مرحله چهارم: مرحله آموزش ی JIT

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

`jax.value_and_grad`هر دو ارزش از دست دادن و گرادینت ها را در یک گذر باز می کند.`@jax.jit`بعد از اولین تماس، هر مرحله آموزش بدون لمس پایتون اجرا می شود.

### مرحله پنجم: چرخه آموزش

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

10 دوره. ~ 97٪ دقت آزمون. اولین دوره آهسته است (توسعه JIT). 2-10 دوره سریع است.

توجه کن که چه چیزی از دست رفته: نه`.zero_grad()`نه`.backward()`نه`.step()`تمام بروزرسانی یک تماس تابع ترکیب است. درجه بندی ها محاسبه می شوند، توسط آدم تبدیل می شوند و به پارامترها اعمال می شوند - همه در داخل`train_step`. .

## ازش استفاده کن

### فله: استاندارد گوگل

فلانس رایج ترین کتابخانه شبکه عصبی JAX است.`nn.Module`پس از آن، اما با مدیریت صریح دولت:

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

ساختار مشابه با "پایتورچ" هست، اما`params`از مدل جدا شده است. `model.init()`پارامز رو ایجاد ميکنه`model.apply(params, x)`.مطابق جلو رو اجرا ميکنه .جسم مدل حالت نداره

### شبابر: جایگزین پیتون

یک شبه عصر (از طرف پاتریک کیجر) مدل ها را به عنوان پایتری نشان می دهد:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

خود مدل یک درخت است.`.apply()`پارامترها فقط برگ های مدل هستند این به طرز فکر جیاکس نزدیک تر است

### Optax: بهینه سازی سازنده

Optax تغییر گرادینت را از بروزرسانی جدا می کند:

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

کپی گریادیوتی، افزایش سرعت یادگیری، کاهش وزن همه اینها به عنوان یک زنجیره تحولات تشکیل شده است. هر تحول گرادیوتی را می بیند، آن ها را تغییر می دهد و آن ها را به بعدی منتقل می کند. هیچ کلاس بهینه سازی یکگانه ای وجود ندارد.

## -باده

**Installation:**

```bash
pip install jax jaxlib optax flax
```

برای پشتیبانی GPU:

```bash
pip install jax[cuda12]
```

برای TPU (گووگلب کلاو):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Performance gotchas:**

- اولین تماس JIT آهسته است (توسعه). قبل از مقایسه، گرم شوید.
- از حلقه هاي پايتون روي آرایه هاي JAX داخل JIT اجتناب کنيد.`jax.lax.scan`یا`jax.lax.fori_loop`. .
- `jax.debug.print()`کار در داخل JIT.`print()`نه، نه
- پروفایل با `jax.profiler`یا TensorBoard. مجموعه XLA می تواند گلو های بطری را پنهان کند.
- JAX 75 درصد حافظه GPU رو بطور پیش فرض اختصاص ميده`XLA_PYTHON_CLIENT_PREALLOCATE=false`برای غیرفعال کردن

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**This lesson produces:**
- `outputs/prompt-jax-optimizer.md`-- یک دستور برای انتخاب تنظیمات بهینه سازی JAX مناسب
- `outputs/skill-jax-patterns.md`-- يه مهارتي که الگوهای فعالي را در JAX پوشش ميده

## تمرینات

1. در JAX، ترک کردن نیاز به کلید PRNG دارد - یک کلید را از طریق گذرگاه جلو به هم بزنید و آن را برای هر لایه ترک کردن تقسیم کنید. دقت آزمون را با و بدون مقایسه کنید.

2. استفاده کنید`jax.vmap`برای محاسبه هر نمونه گرادینت برای یک دسته از 32 تصویر MNIST. برای هر مثال نرمال گرادینت را محاسبه کنید. کدام نمونه ها بیشترین گرادینت را دارند و چرا؟

3. عملکرد پیش رو دستی را با یک عمومی جایگزین کنید `mlp_forward(params, x)`که برای هر تعداد لایه ای کار می کند.`jax.tree.leaves`برای تعیین عمق به طور خودکار.

4. مرحله آموزش را با و بدون آن بنچ مارک کنید `@jax.jit`. زمان 100 مرحله هر کدام چقدر سرعت افزايش سخت افزاري شما بزرگ است؟

5. از طریق ترکیب کردن کتیج گرادینت را اجرا کنید `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. تمرین با و بدون برش. نماد گرادینت رو بر روی تمرین نشان بده تا تا اثر رو ببین

## اصطلاحات کلیدی

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

## خواندن بیشتر

- اسناد JAX: https://jax.readthedocs.io/-- دکترای رسمی، با آموزش های عالی در مورد Graduate، jit و vmap
- "JAX: تحولات قابل ترکیب برنامه های پایتون+نمپی" (برادبری و همکارانش، 2018) - مقاله اصلی که فلسفه طراحی را توضیح می دهد
- اسناد فلان: https://flax.readthedocs.io/-- کتابخانه شبکه عصبی گوگل برای JAX
- پاتریک کیجر، "اقیاس: شبکه های عصبی در JAX از طریق PyTrees قابل تماس و تحولات فیلتر شده" (2021) -- جایگزین پایتونیک برای فله
- DeepMind، "Optax: تبدیل و بهینه سازی گرادینت ترکیب شده" -- کتابخانه بهینه سازی استاندارد
- "تو نمی دانی که جاکس" (کولین رافل، 2020) - راهنمای عملی برای جاکس گات ها و الگوهای، از یکی از نویسندگان T5

# فرآیندهای استوکاستیک

> تصادف با ساختار، ریاضیات پشت پیاده روی های تصادفی، زنجیره های مارکوف و مدل های انتشار

**Type:** Learn
**Language:**پیتون
**Prerequisites:** Phase 1, Lessons 06-07 (probability, Bayes)
**Time:** ~75 minutes

## اهداف یادگیری

- حرکت تصادفی 1D و 2D را شبیه سازی کنید و بررسی کنید$\sqrt{n}$مقیاس بندی جابجایی
- ساخت یک شبیه ساز زنجیره مارکوف و محاسبه توزیع ثابت آن از طریق خودندکمپوزشن
- پیاده سازی دینامیک MCMC و Langevin Metropolis-Hastings برای نمونه گیری از توزیع هدف
- فرآیند انتشار جلو را به حرکت براون متصل کنید و توضیح دهید که چگونه فرآیند معکوس داده ها را تولید می کند

## مشکل

بسیاری از سیستم های هوش مصنوعی شامل تصادفی هستند که با گذشت زمان تکامل می یابد. تصادفی استاتیک نیست. تصادفی ساختار یافته و دنباله دار که هر مرحله بستگی به آنچه پیش از آن اتفاق افتاده است.

مدل های زبان یک به یک توکن تولید می کنند. هر توکن بستگی به زمینه قبلی دارد. مدل توزیع احتمال را تولید می کند، نمونه هایی از آن را تولید می کند و حرکت می کند. این یک فرآیند استوکاستیک است.

مدل های انتشار صدا را به یک تصویر به تدریج اضافه می کنند تا آن به حالت خالص ثابت تبدیل شود. سپس روند را معکوس می کنند و مرحله به مرحله تا یک تصویر جدید ظاهر می شود. فرآیند پیش رو زنجیره مارکوف است. فرآیند معکوس یک زنجیره مارکوف آموخته است که به عقب می رود.

عوامل یادگیری تقویت در یک محیط عمل می کنند. هر عمل به یک وضعیت جدید با احتمالاتی منجر می شود. عامل یک سیاست تصادفی را در یک جهان تصادفی دنبال می کند. همه چیز یک فرآیند تصمیم گیری مارکوف است.

نمونه گیری MCMC - ستون فقرات نتیجه گیری بیزیایی - یک زنجیره مارکوف را ایجاد می کند که توزیع ثابت آن قسمت عقب است که می خواهید از آن نمونه بگیرید.

همه اینها بر روی چهار ایده اساسی بنا شده اند:
1. پیاده روی های تصادفی ساده ترین فرآیند استوکاستیک
2. زنجیره های مارکوف -- تصادفی ساختار یافته با ماتریس انتقال
3. دینامیک لانژین - کاهش گرادینت با شور
4. متروپولیس-هستینگز - نمونه گیری از هر توزیع

## مفهوم

### پیاده روی های تصادفی

در هر مرحله یک سکه ی عادلانه را پر کنید. سرها: به سمت راست (+1) حرکت کنید. دمها: به سمت چپ (-1) حرکت کنید.

بعدش$n$قدم ها، موقعیت شما مجموع$n$تصادفی$\pm 1$ارزش ها. موقعیت انتظار می رود 0 (پیاده کردن غیر جانبدار است). اما فاصله انتظار شده از اصل به عنوان افزایش می یابد $\sqrt{n}$. .

این برعکس است. راه رفتن عادلانه است - هیچ انحراف در هر دو جهت. اما با گذشت زمان، از جایی که شروع کرد، دور و دور تر می رود. انحراف استاندارد پس از$n$قدم ها$\sqrt{n}$. .

```
Step 0:  Position = 0
Step 1:  Position = +1 or -1
Step 2:  Position = +2, 0, or -2
...
Step 100: Expected distance from origin ~ 10 (sqrt(100))
Step 10000: Expected distance from origin ~ 100 (sqrt(10000))
```

**In 2D**، راه رفتن به بالا، پایین، چپ، یا راست با احتمال برابر حرکت می کند.$\sqrt{n}$مقیاس گذاری به فاصله از اصل مربوط می شود. مسیر یک الگوی فراکتال را دنبال می کند.

**Why $\sqrt{n}$?**هر مرحله +1 یا -1 با احتمال برابر است.$n$قدم ها، موقعیت$S_n = X_1 + X_2 + \cdots + X_n$هر کدام$X_i$.$\pm 1$. تفاوت هر مرحله 1 است و مراحل مستقل هستند، بنابراین $\mathop{\mathrm{Var}}(S_n) = n$انحراف استاندارد = $\sqrt{n}$با تئورمه ي حد مرکزی$S_n / \sqrt{n}$به یک توزیع عادی استاندارد نزدیک می شود.

این$\sqrt{n}$مقیاس بندی در همه جا در ML ظاهر می شود. مقیاس های SGD صدا به عنوان 1/sqrt ((بچ_size)$\sqrt{d}$. ریشه مربع امضای افزونه های تصادفی مستقل است

**Connection to Brownian motion.**با اندازه پله ها يه قدم بيشتر بريم$1/\sqrt{n}$و$n$قدم ها در هر واحد زمان$n$به سمت بی نهایت می رود، راه رفتن به حرکت براون نزدیک می شود$B(t)$-- یک فرآیند مداوم که در آن$B(t)$معمولا با متوسط 0 و انحراف توزیع می شود$t$. .

حرکت براون پایه ریاضی انتشار است. این شکل گیری تصادفی ذرات در یک مایع، نوسانات قیمت سهام و - مهمتر از همه - روند سر و صدا در مدل های انتشار را می سازد.

**Gambler's ruin.**يه راهپيمايي تصادفي که از موقعيت شروع ميکنه$k$, با موانع جذب در 0 و $N$احتمال رسیدن به$N$قبل از صفر؟$P(\text{reach } N) = k/N$این به طرز شگفت انگیزی ساده و زیبا است. این به نظریه مارتینگال ها متصل می شود -- راه رفتن تصادفی عادلانه مارتینگال است (قیمت آینده انتظار می رود = ارزش فعلی).

### زنجیره های مارکوف

یک زنجیره مارکوف یک سیستم است که بر اساس احتمالات ثابت بین کشورها انتقال می یابد. ویژگی کلیدی: دولت بعدی فقط بر اساس وضعیت فعلی، نه تاریخ بستگی دارد.

```math
P(X_{t+1} = j \mid X_t = i, X_{t-1} = \ldots) = P(X_{t+1} = j \mid X_t = i)
```

این ویژگی مارکوف است. این بدان معنی است که شما می توانید تمام دینامیک را با یک ماتریس انتقال توصیف کنید.$P$:

```math
P[i][j] = \text{probability of going from state } i \text{ to state } j
```

هر ردیف از$P$به 1 (شما باید به جایی بروید)

**Example -- Weather:**

```
States: Sunny (0), Rainy (1), Cloudy (2)

P = [[0.7, 0.1, 0.2],    (if sunny: 70% sunny, 10% rainy, 20% cloudy)
     [0.3, 0.4, 0.3],    (if rainy: 30% sunny, 40% rainy, 30% cloudy)
     [0.4, 0.2, 0.4]]    (if cloudy: 40% sunny, 20% rainy, 40% cloudy)
```

بعد از چند انتقال، توزیع دولت ها به توزیع ثابت تبدیل می شود.$\pi$، کجا$\pi P = \pi$این است که متری خودروی چپ از$P$با ارزش خاص 1

برای زنجیره آب و هوا، توزیع ثابت [0.55, 0.18, 0.27] است -- در دراز مدت، 55 درصد از زمان بدون توجه به حالت آغاز، آفتاب است.

```mermaid
graph LR
    S["Sunny"] -->|0.7| S
    S -->|0.1| R["Rainy"]
    S -->|0.2| C["Cloudy"]
    R -->|0.3| S
    R -->|0.4| R
    R -->|0.3| C
    C -->|0.4| S
    C -->|0.2| R
    C -->|0.4| C
```

**Computing the stationary distribution.**دو روش وجود دارد:

1. **Power method**: هر توزیع اولیه را با  ضرب کنید$P$بعد از تکرار های کافی، همگام می شود.
2. **Eigenvalue method**: دنبال کننده ی خروجی چپ را پیدا کنید$P$با ارزش خاص 1 این ویکتور خاص $P^T$با ارزش خاص 1

هر دو رویکرد نیاز به زنجیره ای برای برآورده کردن شرایط تقارب دارند.

**Convergence conditions.**یک زنجیره مارکوف به یک توزیع ثابت منحصر به فرد متقابل می شود اگر:
- **Irreducible**: هر ایالت از هر ایالت دیگه قابل دسترسی است
- **Aperiodic**: زنجیره با دوره ای ثابت چرخه ای ندارد

اکثر زنجیره هایی که در ML پیدا می کنید هر دو شرط را برآورده می کنند.

**Absorbing states.**حالتى است كه جذب مي كند اگر كجا واردش شويد و هرگز بيرون نرويد$P[i][i] = 1$جذب زنجیره های مارکوف مدل فرآیندهای با حالت های نهایی -- یک بازی که پایان می یابد، یک مشتری که تند و تند می شود، یک توابع رمزنگاری که به نشانه پایان متن می رسد.

**Mixing time.**چند مرحله تا زنجیره "بزرگ" به توزیع ثابت باشد؟ به طور رسمی، تعداد مراحل تا فاصله ی تغییرات کل از ثابت بودن زیر یک حد پایین می آید. مخلوط سریع = چند مرحله مورد نیاز است. شکاف طیف ای از $P$(1 - دومین بزرگ ترین ارزش خونی) زمان مخلوط را کنترل می کند. فاصله بزرگتر = مخلوط سریعتر.

### ارتباط با مدل های زبان

تولید توکن در یک مدل زبان تقریباً یک فرآیند مارکوف است. با توجه به زمینه فعلی، مدل توزیع را در توکن بعدی تولید می کند. دمای کنترل تیز بودن:

```math
P(\text{token}_i) = \frac{\exp(\text{logit}_i / \text{temperature})}{\sum_j \exp(\text{logit}_j / \text{temperature})}
```

- دمای = 1.0: توزیع استاندارد
- دمای < 1,0: تیز تر (مقرر تر)
- دمای بیشتر از ۱.۰: مسطحتر (به طور تصادفی)
- درجه حرارت$\to 0$: آرگماکس (طمع)

نمونه گیری از بالا تا پایین تر$k$نمره گیری top-p ( هسته) به کوچکترین مجموعه از نمره ها که احتمال تجمعی آن بیشتر است، کوتاه می شود $p$هر دو احتمال انتقال مارکوف رو تغییر ميدن

### حرکت براون

محدوده زمان مستمر راه رفتن تصادفی$B(t)$سه ویژگی دارد:
1. $B(0) = 0$
2. $B(t) - B(s)$معمولا با متوسط 0 و انحراف توزیع می شود$t - s$(برای $t > s$)
3. افزایش در فواصل بدون تعادل مستقل است

حرکت براونین ثابت است اما هیچ جا قابل تشخیص نیست - در هر مقیاس می چرخد. مسیر دارای ابعاد فراکتال 2 در سطح است.

در شبیه سازی جداگانه، حرکت براون را با:

```math
B(t + dt) = B(t) + \sqrt{dt} \, z, \quad \text{where } z \sim \mathcal{N}(0, 1)
```

.$\sqrt{dt}$مقیاس بندی مهم است. این از نظریه محدودی مرکزی برای پیاده روی های تصادفی استفاده می شود.

### دینامیک لانژین

کاهش درجه ای حداقل یک تابع را پیدا می کند. دینامیک لانجین توزیع احتمال را متناسب با $\exp(-U(x)/T)$، کجا$U$یک تابع انرژی است و$T$درجه حرارت است.

```math
x_{t+1} = x_t - dt \, \nabla U(x_t) + \sqrt{2 T \, dt} \, z_t
```

دو نیروی روی ذره عمل می کنند:
1. **Gradient force**($-dt \cdot \nabla U$): به سمت انرژی پایین فشار می دهد (مانند کاهش گرادینتی)
2. **Random force**($\sqrt{2T dt} \cdot z$): در جهت های تصادفی فشار می دهد (کشف)

در درجه حرارت$T = 0$در درجه حرارت بالا، تقریباً یک پیاده روی تصادفی است. در درجه حرارت مناسب، ذره زمین انرژی را کشف می کند و زمان بیشتری را در مناطق کم انرژی صرف می کند.

**Connection to diffusion models.**فرآیند پیشروی یک مدل انتشار عبارت است از:

```math
x_t = \sqrt{\alpha_t} \, x_{t-1} + \sqrt{1 - \alpha_t} \cdot \text{noise}
```

این یک زنجیره مارکوف است که به تدریج داده ها را با صدا مخلوط می کند. پس از مراحل کافی،$x_T$صداي خالص گوسيه

فرآیند معکوس -- از صدا به داده ها باز می گردد -- همچنین یک زنجیره مارکوف است، اما احتمالات انتقال آن توسط یک شبکه عصبی آموخته می شود. شبکه یاد می گیرد تا صدای اضافه شده را در هر مرحله پیش بینی کند، سپس آن را حذف می کند.

```mermaid
graph LR
    subgraph "Forward Process (add noise)"
        X0["x_0 (data)"] -->|"+ noise"| X1["x_1"]
        X1 -->|"+ noise"| X2["x_2"]
        X2 -->|"..."| XT["x_T (pure noise)"]
    end
    subgraph "Reverse Process (denoise)"
        XT2["x_T (noise)"] -->|"neural net"| XR2["x_{T-1}"]
        XR2 -->|"neural net"| XR1["x_{T-2}"]
        XR1 -->|"..."| XR0["x_0 (generated data)"]
    end
```

### MCMC: زنجیره مارکوف مونت کارلو

گاهی اوقات باید از یک توزیع نمونه گیری کنید$p(x)$که می توانید ارزیابی کنید (تا یک ثابت) اما نمی توانید از طریق مستقیم نمونه بگیرید. پس زمینه های بیزیایی نمونه کلاسیک هستند -- شما احتمال را ضرب از قبل می دانید، اما ثابت عادی سازی غیر قابل حل است.

**Metropolis-Hastings**ساخت زنجیره مارکوف که توزیع ثابت آن$p(x)$:

1. از يه موقعیت شروع کن$x$
2. پیشنهاد یه موقعیت جدید بده$x'$از یک توزیع پیشنهادات$Q(x' \mid x)$
3. نسبت پذیرش محاسبه: $a = p(x') Q(x \mid x') / (p(x) Q(x' \mid x))$
4. قبول کن$x'$با احتمال$\min(1, a)$. در غیر اینصورت در$x$. .
5. تکرار کنم

اگه$Q$متقابل است (به عنوان مثال:$Q(x' \mid x) = Q(x \mid x') = \mathcal{N}(x, \sigma^2){}$) نسبت ساده می شود تا$a = p(x') / p(x)$شما فقط به نسبت احتمالات نیاز دارید -- ثابت عادی سازی لغو می شود.

تضمین شده که زنجیره به هم می پیوندد$p(x)$در شرایط نرم، اما اگر پیشنهاد خیلی کوچک (چراغ تصادفی) یا خیلی بزرگ (رفض بالا) باشد، کنورژن می تواند کند باشد.

**Why it works.**نسبت پذیرش اطمینان از تعادل دقیق: احتمال بودن در $x$و به سمت$x'$برابر با احتمالي بودن در $x'$و به سمت$x$تعادل مفصل به این معنی است که$p(x)$این توزیع ثابت زنجیره است. پس پس از مراحل کافی، نمونه ها از$p(x)$. .

**Practical considerations:**
- **Burn-in**: اولين رو رد کن$N$نمونه ها. زنجیره زمان لازم دارد تا از نقطه شروع خود به توزیع ثابت برسد.
- **Thinning**: همه رو نگه دار$k$- نمونه سوم براي کاهش خودتفاعل
- **Multiple chains**: چند زنجیره را از نقاط شروع مختلف اجرا کنید. اگر به یک توزیع یکسان همگام شوند، شواهد همگامگی وجود دارد.
- **Acceptance rate**: برای پیشنهادات گاس در سال$d$در طول طول طول، میزان پذیرش مطلوب حدود 23 درصد است (Roberts & Rosenthal، 2001). بیش از حد بالا به این معنی است که زنجیره به سختی حرکت می کند. بیش از حد پایین به این معنی است که همه چیز را رد می کند.

### فرآیندهای استوکاستیکی در هوش مصنوعی

| Process | AI Application |
|---------|---------------|
| Random walk | Exploration in RL, Node2Vec embeddings |
| Markov chain | Text generation, MCMC sampling |
| Brownian motion | Diffusion models (forward process) |
| Langevin dynamics | Score-based generative models, SGLD |
| Markov decision process | Reinforcement learning |
| Metropolis-Hastings | Bayesian inference, posterior sampling |

```figure
random-walk-diffusion
```

## آن را بسازید

### مرحله ی اول: شبیه ساز پیاده روی تصادفی

```python
import numpy as np

def random_walk_1d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions


def random_walk_2d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    directions = rng.choice(4, size=n_steps)
    dx = np.zeros(n_steps)
    dy = np.zeros(n_steps)
    dx[directions == 0] = 1   # right
    dx[directions == 1] = -1  # left
    dy[directions == 2] = 1   # up
    dy[directions == 3] = -1  # down
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    return x, y
```

قدم 1D جمع جمع را ذخیره می کند. هر مرحله +1 یا -1 است. پس از $n$در مرحله ای، موقعیت مجموع است.$n$، پس انحراف استاندارد به عنوان $\sqrt{n}$. .

### مرحله دوم: زنجیره مارکوف

```python
class MarkovChain:
    def __init__(self, transition_matrix, state_names=None):
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = len(self.P)
        self.state_names = state_names or [str(i) for i in range(self.n_states)]

    def step(self, current_state, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        probs = self.P[current_state]
        return rng.choice(self.n_states, p=probs)

    def simulate(self, start_state, n_steps, seed=None):
        rng = np.random.RandomState(seed)
        states = [start_state]
        current = start_state
        for _ in range(n_steps):
            current = self.step(current, rng)
            states.append(current)
        return states

    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)
```

توزیع ثابت، ویکتور خروجی چپ از $P$با ارزش خاص 1، ما آن را با محاسبه متریهای خاص از $P^T$(تجاوز وکتورهای خروجی چپ را به vektorهای خروجی راست تبدیل می کند).

### مرحله سوم: دینامیک لانژوین

```python
def langevin_dynamics(grad_U, x0, dt, temperature, n_steps, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        noise = rng.randn(*x.shape)
        x = x - dt * grad_U(x) + np.sqrt(2 * temperature * dt) * noise
        trajectory.append(x.copy())
    return np.array(trajectory)
```

گرادینت فشار میده$x$در حالت تعادل، توزیع نمونه ها متناسب با$\exp(-U(x)/\text{temperature})$. .

### مرحله چهارم: متروپولیس هاستینگ

```python
def metropolis_hastings(target_log_prob, proposal_std, x0, n_samples, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    accepted = 0
    for _ in range(n_samples - 1):
        x_proposed = x + rng.randn(*x.shape) * proposal_std
        log_ratio = target_log_prob(x_proposed) - target_log_prob(x)
        if np.log(rng.rand()) < log_ratio:
            x = x_proposed
            accepted += 1
        samples.append(x.copy())
    acceptance_rate = accepted / (n_samples - 1)
    return np.array(samples), acceptance_rate
```

الگوریتم یک نقطه جدید را پیشنهاد می کند، بررسی می کند که آیا احتمال بیشتری دارد (یا با احتمال متناسب با نسبت پذیرفته می شود) و تکرار می کند. نرخ پذیرش باید حدود 23-50٪ برای مخلوط خوب باشد.

## ازش استفاده کن

در عمل، شما از کتابخانه های موجود برای این الگوریتم ها استفاده می کنید اما درک مکانیک برای دیبگ و تنظیم مهم است.

```python
import numpy as np

rng = np.random.RandomState(42)
walk = np.cumsum(rng.choice([-1, 1], size=10000))
print(f"Final position: {walk[-1]}")
print(f"Expected distance: {np.sqrt(10000):.1f}")
print(f"Actual distance: {abs(walk[-1])}")
```

### نمپی برای ماتریس های انتقال

```python
import numpy as np

P = np.array([[0.7, 0.1, 0.2],
              [0.3, 0.4, 0.3],
              [0.4, 0.2, 0.4]])

distribution = np.array([1.0, 0.0, 0.0])
for _ in range(100):
    distribution = distribution @ P

print(f"Stationary distribution: {np.round(distribution, 4)}")
```

تقسیم اولیه را با $P$بعد از تکرار های کافی، این به توزیع ثابت تبدیل می شود، بدون توجه به جایی که شروع کردید. این روش قدرت برای پیدا کردن متور مالکیت چپ غالب است.

### ارتباطات با چارچوب های واقعی

- **PyTorch diffusion:**.`DDPMScheduler`در صورت بازداشت`diffusers`زنجیره های Markov را به جلو و عقب اجرا می کند
- **NumPyro / PyMC:**استفاده از MCMC (نمونهگر NUTS که در Metropolis-Hastings بهبود می یابد) برای نتیجه گیری بیزی
- **Gymnasium (RL):**تابع مرحله محیط زیست فرآیند تصمیم گیری مارکوف را تعریف می کند

### بررسی تراکنش زنجیره مارکوف

```python
import numpy as np

P = np.array([[0.9, 0.1], [0.3, 0.7]])

eigenvalues = np.linalg.eigvals(P)
spectral_gap = 1 - sorted(np.abs(eigenvalues))[-2]
print(f"Eigenvalues: {eigenvalues}")
print(f"Spectral gap: {spectral_gap:.4f}")
print(f"Approximate mixing time: {1/spectral_gap:.1f} steps")
```

شکاف طیف به شما می گوید که زنجیره چقدر سریع وضعیت اولیه خود را فراموش می کند. شکاف 0.2 به معنای حدود 5 مرحله برای مخلوط کردن است. شکاف 0.01 به معنای حدود 100 مرحله است. همیشه قبل از اجرای شبیه سازی های طولانی این را بررسی کنید - یک محاسبه مخلوط کننده آهسته زنجیره ی زباله.

## -باده

این درس نتیجه می دهد:
- `outputs/prompt-stochastic-process-advisor.md`-- یک پیامک که به شناسایی چارچوب فرآیند استوکاستیک برای یک مشکل خاص کمک می کند

## ارتباطات

| Concept | Where it shows up |
|---------|------------------|
| Random walk | Node2Vec graph embeddings, exploration in RL |
| Markov chain | Token generation in LLMs, MCMC sampling |
| Brownian motion | Forward diffusion process in DDPM, SDE-based models |
| Langevin dynamics | Score-based generative models, stochastic gradient Langevin dynamics (SGLD) |
| Stationary distribution | MCMC convergence target, PageRank |
| Metropolis-Hastings | Bayesian posterior sampling, simulated annealing |
| Temperature | LLM sampling, Boltzmann exploration in RL, simulated annealing |
| Mixing time | Convergence speed of MCMC, spectral gap analysis |
| Absorbing state | End-of-sequence token, terminal states in RL |
| Detailed balance | Correctness guarantee for MCMC samplers |

مدل های انتشار توجه ویژه ای را مستحق می باشند. DDPM (Ho et al., 2020) یک زنجیره مارکوف پیشرو را تعریف می کند:

```math
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I\right)
```

کجا$\beta_t$برنامه هاي شوريه بعد از$T$قدم ها$x_T$تقریباً$\mathcal{N}(0, I)$. پروسه معکوس توسط شبکه عصبی که از صدا پیش بینی می کند ، پارامتر شده است:

```math
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I\right)
```

هر مرحله ای از تولید یک مرحله در یک زنجیره مارکوف آموخته است. درک زنجیره های مارکوف به معنای درک چگونگی و چرا مدل های انتشار داده ها را تولید می کنند.

SGLD (داینامکس لنجین گرادین استوکاستیک) کاهش گرادینتی دسته کوچک را با صدای لنجین ترکیب می کند. به جای محاسبه تمام گرادینت، از یک تخمین استوکاستیک استفاده می کنید و صداهای کالیبر شده را اضافه می کنید. با کاهش سرعت یادگیری، SGLD از بهینه سازی به نمونه گیری می گذرد -- شما نمونه های پسروی بیزیایی را رایگان می گیرید. این یکی از ساده ترین راه ها برای دریافت تخمین های عدم اطمینان از یک شبکه عصبی است.

نکته کلیدی در تمام این ارتباطات: فرآیندهای استوکاستیک فقط ابزار نظری نیستند. آنها مکانیسم های محاسباتی در سیستم های مدرن هوش مصنوعی هستند. وقتی دمای مدرک تحصیلی را تنظیم می کنی، یک زنجیره مارکوف را تنظیم می کنی. وقتی یک مدل انتشار را آموزش می دهید، یاد می گیرید یک فرآیند حرکتی شبیه به حرکت براون را معکوس کنید. وقتی نتیجه گیری بیزیایی را اجرا می کنید، یک زنجیره ای را ایجاد می کنید که به عقب هم می رسد.

## تمرینات

1. **Simulate 1000 random walks of 10000 steps.**توزیع موقعیت های نهایی را نقشه برداری کنید. بررسی کنید که تقریباً Gaussian با میانگین 0 و انحراف استاندارد است.$\sqrt{10000} = 100$. .

2. **Build a text generator using a Markov chain.**آموزش روی یک کورپوس کوچک: برای هر کلمه، انتقال ها را به کلمه بعدی شمارش کنید. ماتریس انتقال را بسازید. با نمونه گیری از زنجیره جملات جدید تولید کنید.

3. **Implement simulated annealing**با استفاده از Metropolis-Hastings. شروع در دمای بالا (تقریباً همه چیز را قبول کنید) و به تدریج خنک شوید (تنها پیشرفت را قبول کنید). از آن برای پیدا کردن حداقل یک تابع با بسیاری از حداقل های محلی استفاده کنید.

4. **Compare Langevin dynamics at different temperatures.**نمونه از پتانسیل دو چاه$U(x) = (x^2 - 1)^2$در دمای پایین نمونه ها در یک چاه جمع می شوند در دمای بالا به هر دو جای پخش می شوند. دمای حساس را پیدا کنید که در آن زنجیره بین چاه ها مخلوط می شود.

5. **Implement the forward diffusion process.**با یک سیگنال یک بعدی (به عنوان مثال، یک موج سینوس) شروع کنید. صدا را به طور تدریجی بیش از 100 مرحله با یک برنامه صدا خطی اضافه کنید. نشان دهید که چگونه سیگنال به صدا خالص کاهش می یابد. سپس یک نشان دهنده ساده را اجرا کنید که روند را معکوس می کند (حتی یک ساده که فقط صدا تخمین زده را حذف می کند).

### اشاره ها

**Exercise 1.**تماس بگیرید`random_walk_1d(10000, seed=i)`برای 1000 دانه و نگه داشتن هر موقعیت نهایی. بررسی کنید که متوسط نزدیک به 0 است، انحراف استاندارد نزدیک به 100 است و حدود 68٪ از موقعیت های نهایی بین -100 و 100 است.

**Exercise 2.**هر جفت کلمه را بشمارید، سپس هر ردیف را با کل آن تقسیم کنید تا هر ردیف به 1 اضافه شود. ماتریس را به `MarkovChain(transition_matrix, state_names)`و تماس بگیرید`simulate`چک کن که هر جفت کلمه ای که در محصولت هست در corpus هم وجود داره

**Exercise 3.**با احتمالي يک نقطه بدتر را قبول کن$\exp(-\Delta / T)$و چند برابرش کنم$T$با فاکتوری مانند 0.99 بعد از هر مرحله، یک تابع را امتحان کنید که حداقل آن را می دانید و پاسخ خود را با یک جستجوی خوب شبکه بررسی کنید.

**Exercise 4.**گرادینت اینه$4x(x^2 - 1)$، پس اينو به`langevin_dynamics(grad_U, x0, dt, temperature, n_steps)`. حاشيه در$x = 0$اون قد 1 داره، پس شمارش کن تا چه موقع علامت$x$برای تغییرات در دمای از 0.05 به 1 و دنبال دمای جایی که شمارش شروع به افزایش سریع می کند.

**Exercise 5.**تماس بگیرید`diffusion_forward(signal, 100)`و محاسبه$\bar{\alpha}$در آخرین مرحله به عنوان محصول از$(1 - \beta_t)$اگه$\bar{\alpha}$فاصله اش از صفر فاصله داره، هنوز صدا خالص نیست، پس بلندش کن`beta_end`اول از اين که صداي واقعي رو امتحان کن$(x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon)/\sqrt{\bar{\alpha}_t}$بايد به طور دقیق سيگنال اصلي رو برگردونه

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Random walk | "Coin-flip movement" | A process where position changes by random increments at each step |
| Markov property | "Memoryless" | The future depends only on the present state, not on the history |
| Transition matrix | "The probability table" | $P[i][j]$ = probability of moving from state $i$ to state $j$ |
| Stationary distribution | "The long-run average" | The distribution $\pi$ where $\pi P = \pi$ -- the chain's equilibrium |
| Brownian motion | "Random jiggling" | The continuous-time limit of a random walk, $B(t) \sim \mathcal{N}(0, t)$ |
| Langevin dynamics | "Gradient descent with noise" | Update rule that combines deterministic gradient and random perturbation |
| MCMC | "Walking toward the target" | Constructing a Markov chain whose stationary distribution is the one you want |
| Metropolis-Hastings | "Propose and accept/reject" | MCMC algorithm that uses acceptance ratios to ensure convergence |
| Temperature | "The randomness knob" | Parameter controlling the tradeoff between exploration and exploitation |
| Diffusion process | "Noise in, noise out" | Forward: gradually add noise. Reverse: gradually remove it. Generates data. |

## خواندن بیشتر

- **Ho, Jain, Abbeel (2020)**-- "مودلهاي احتمالي انتشار را رد ميكنم". مقاله DDPM كه انقلاب مدل انتشار را آغاز کرد. مشتق واضح از زنجیره هاي ماركوف جلو و عقب.
- **Song & Ermon (2019)**-- "نمذره سازی نسل به وسیله تخمین زدنی درجه بندی توزیع داده ها". رویکرد مبتنی بر نمره با استفاده از دینامیک لنجوین برای نمونه گیری.
- **Roberts & Rosenthal (2004)**-- "سلسلات مارکوف و الگوریتم های MCMC" نظریه ی پشت این که MCMC چه زمانی و چرا کار می کند.
- **Norris (1997)**-- "سلسلۀ مارکوف" کتابی استاندارد که شامل تقارب، توزیع ثابت و زمان ضربه می شود.
- **Welling & Teh (2011)**-- "علم بائزی از طریق دینامیک لنجین درجه بندی استوکاستیک".

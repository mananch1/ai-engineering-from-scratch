# الحسابات للتعلم الآلي

> المشتقات تخبرك في أي طريق هو الهبوط هذا كل ما تحتاج الشبكة العصبية إلى تعلمه

**Type:** Learn
**Language:**بايثون
**Prerequisites:** Phase 1, Lessons 01-03
**Time:** ~60 minutes

## أهداف التعلم

- الحساب المشتقات الرقمية والتحليلية للعملات المشتركة للطاقة المعدنية ($x^2$(التي تُستخدم في التشغيل)
- تنفيذ انخفاض التدرج من الصفر لتقليل وظيفة الخسارة في 1D و 2D
- استنباط تراجع النموذج الخطوي وتدريبها عن طريق تحديثات الوزن اليدوية
- شرح المصفوفة الهسسيّة، مقربات سلسلة تايلور، وربطها بأساليب التحسين

## المشكلة

لديك شبكة عصبية مع ملايين الوزن كل وزن هو عقدة عليك أن تجد أي اتجاه لتدور كل عقدة واحدة لتجعل النموذج أقل خطأ قليلاً

بدون الحسابات، تدريب شبكة عصبية يعني تجربة التغييرات العشوائية وتأمل في الأفضل. مع المشتقات، أنت تعرف بالضبط كيف كل وزن يؤثر على الخطأ. أنت تحول كل الزر في الطريق الصحيح، في كل مرة.

## المفهوم

### ما هو المشتق؟

المشتقات تقيس معدل التغيير.$y = f(x)$، المشتق$f'(x)$يخبرك: إذا دفعت$x$بكمية صغيرة، كم هو$y$تغيير؟

من الناحية الجيموترية، المشتق هي ميل الخط المتعلق في نقطة.

**$f(x) = x^2$:**

| $x$ | $f(x)$ | $f'(x)$ (slope) |
|---|------|---------------|
| 0 | 0    | 0 (flat, at the bottom) |
| 1 | 1    | 2 |
| 2 | 4    | 4 (tangent line slope at this point) |
| 3 | 9    | 6 |

في$x=2$، المنحدر هو 4 إذا تحرك$x$قليلاً إلى اليمين$y$يزيد من حوالي أربع مرات هذا المبلغ.$x=0$،المهد هو 0 أنت في أسفل الوعاء.

التعريف الرسمي:

```math
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
```

في الرمز، أنت تخطي الحد و فقط استخدام صغير جدا $h$هذا هو المشتق الرقمي

### المشتقات الجزئية: متغير واحد في كل مرة

المواد الحقيقية لديها العديد من المدخلات. فقدان الشبكة العصبية يعتمد على آلاف الوزن. المشتق جزئي يحتفظ جميع المتغيرات ثابتة باستثناء واحدة، ثم يأخذ المشتق فيما يتعلق بذلك واحد.

```math
\begin{aligned}
f(x, y) &= x^2 + 3xy + y^2 \\[4pt]
\frac{\partial f}{\partial x} &= 2x + 3y \quad \text{(treat } y \text{ as a constant)} \\
\frac{\partial f}{\partial y} &= 3x + 2y \quad \text{(treat } x \text{ as a constant)}
\end{aligned}
```

كل مشتق جزئي يجيب: إذا دفعت فقط هذا الوزن واحد، كيف يتغير الخسارة؟

### المتحدر: متجه لجميع المشتقات الجزئية

الدرجة الجانبية تجمع كل مشتق جزئي في متجه واحد.$f(x, y, z)$، المرجع هو:

```math
\nabla f = \left[ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \right]
```

يُحدّد التسلّل في اتجاه صعود أكثر خطورة، لتحقيق الحدّ الأدنى من وظيفة، اذهب في الاتجاه المعاكس.

**Contour plot of $f(x,y) = x^2 + y^2$:**

تشكل الوظيفة شكل وعاء مع دائرات مركزة كخطوط شكل. الحد الأدنى هو (0, 0).

| Point | $\nabla f$ | $-\nabla f$ (descent direction) |
|-------|--------|----------------------------|
| (1, 1) | [2, 2] (points uphill, away from minimum) | [-2, -2] (points downhill, toward minimum) |
| (0, 0) | [0, 0] (flat, at the minimum) | [0, 0] |

هذا هو انخفاض التراجع في صورة حساب التراجع، سلبه، اتخاذ خطوة.

### الارتباط مع التحسين

تدريب شبكة عصبية هو التحسين. لديك وظيفة الخسارة.$L(w_1, w_2, \ldots, w_n)$هذا يُقيس مدى خطأ النموذج.

```
Gradient descent update rule:

  w_new = w_old - learning_rate * dL/dw

For every weight:
  1. Compute the partial derivative of loss with respect to that weight
  2. Subtract a small multiple of it from the weight
  3. Repeat
```

معدل التعلم يحدد حجم الخطوة، كبير جداً و أنت تتجاوز، صغير جداً و أنت تتجول.

**Loss landscape (1D slice):**

وظيفة الخسارة$L(w)$يشكّل منحنى مع القمم والوعي كوزن $w$يختلف

| Feature | Description |
|---------|-------------|
| Global minimum | The lowest point on the entire curve -- the best solution |
| Local minimum | A valley that is lower than its neighbors but not the lowest overall |
| Slope | Gradient descent follows the slope downhill from any starting point |

يتبع التنحدر التدريجي التسلل الهبوطي. يمكن أن يعلق في الحد الأدنى المحلي، ولكن في المساحات العالية الأبعاد (ملايين الوزن) هذه نادرًا ما تكون مشكلة عملية.

### المشتقات الرقمية مقابل التحليلية

هناك طريقتان لحساب مشتق

التحليلي: تطبيق قواعد الحساب يدويا.$f(x) = x^2$، المشتق هي$f'(x) = 2x$بالضبط، بسرعة

العدد: تقريري باستخدام التعريف. الحساب $f(x+h)$و$f(x-h)$لفترة صغيرة$h$، ثم استخدم الفرق

```
Numerical (central difference):

f'(x) ~= f(x + h) - f(x - h)
          -----------------------
                  2h

h = 0.0001 works well in practice
```

المشتقات الرقمية بطيئة ولكن تعمل لأي وظيفة. المشتقات التحليلية سريعة ولكن تتطلب منك استنباط الصيغة. استخدام إطار الشبكات العصبية نهج ثالث: التمييز الآلي، الذي يحسب المشتقات الدقيقة ميكانيكيا. سترى ذلك في المرحلة 3.

### المشتقات المستخدمة يدوياً للعمل البسيط

هذه هي المشتقات التي ستراها مرارا وتكرارا في ML.

```
Function        Derivative       Used in
--------        ----------       -------
f(x) = x^2     f'(x) = 2x      Loss functions (MSE)
f(x) = wx + b  f'(w) = x        Linear layer (gradient w.r.t. weight)
                f'(b) = 1        Linear layer (gradient w.r.t. bias)
                f'(x) = w        Linear layer (gradient w.r.t. input)
f(x) = e^x     f'(x) = e^x     Softmax, attention
f(x) = ln(x)   f'(x) = 1/x     Cross-entropy loss
f(x) = 1/(1+e^-x)  f'(x) = f(x)(1-f(x))   Sigmoid activation
```

لأجل$f(x) = x^2$:

```
f(x) = x^2    f'(x) = 2x

  x    f(x)   f'(x)   meaning
  -2    4      -4      slope tilts left (decreasing)
  -1    1      -2      slope tilts left (decreasing)
   0    0       0      flat (minimum!)
   1    1       2      slope tilts right (increasing)
   2    4       4      slope tilts right (increasing)
```

لأجل$f(w) = wx + b$مع$x=3$،$b=1$:

```
f(w) = 3w + 1    f'(w) = 3

The derivative with respect to w is just x.
If x is big, a small change in w causes a big change in output.
```

### قاعدة السلسلة

عندما تكون الوظائف مرتبة، قاعدة السلسلة تخبرك كيفية التمييز.

```
If y = f(g(x)), then dy/dx = f'(g(x)) * g'(x)

Example: y = (3x + 1)^2
  outer: f(u) = u^2       f'(u) = 2u
  inner: g(x) = 3x + 1    g'(x) = 3
  dy/dx = 2(3x + 1) * 3 = 6(3x + 1)
```

شبكات العصبية هي سلسلة من الوظائف: المدخل -> خطي -> تفعيل -> خطي -> تفعيل -> خسارة. التنشر الخلفي هو قاعدة سلسلة تطبق مرارا من الخروج إلى المدخل. وهذا هو الخوارزمية بأكملها.

### المصفوفة الهيسي

المرتفعات تخبرك بالمهد، والحسية تخبرك بالمنعطف

المثبتات الجزئية من النظام الثاني هي المصفوفة.$f(x_1, x_2, \ldots, x_n)$, دخول$(i, j)$من الهيسي هو:

```math
H_{ij} = \frac{\partial^2 f}{\partial x_i \, \partial x_j}
```

لـ وظيفة متغيرة$f(x, y)$:

```math
H = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x^2} & \dfrac{\partial^2 f}{\partial x \, \partial y} \\[8pt] \dfrac{\partial^2 f}{\partial y \, \partial x} & \dfrac{\partial^2 f}{\partial y^2} \end{bmatrix}
```

**What the Hessian tells you at a critical point (where gradient = 0):**

| Hessian property | Meaning | Example surface |
|-----------------|---------|-----------------|
| Positive definite (all eigenvalues > 0) | Local minimum | Bowl pointing up |
| Negative definite (all eigenvalues < 0) | Local maximum | Bowl pointing down |
| Indefinite (mixed eigenvalues) | Saddle point | Horse saddle shape |

**Example:** $f(x, y) = x^2 - y^2$(عمل في السرير)

```
df/dx = 2x       df/dy = -2y
d^2f/dx^2 = 2    d^2f/dy^2 = -2    d^2f/dxdy = 0

H = | 2   0 |
    | 0  -2 |

Eigenvalues: 2 and -2 (one positive, one negative)
--> Saddle point at (0, 0)
```

مقارنة مع$f(x, y) = x^2 + y^2$(قوطة):

```
H = | 2  0 |
    | 0  2 |

Eigenvalues: 2 and 2 (both positive)
--> Local minimum at (0, 0)
```

**Why the Hessian matters in ML:**

تستخدم طريقة نيوتن خطوات التحسين الأفضل من تراجع التراجع. بدلاً من اتباع التنحى فحسب، فإنها تأخذ بعين الاعتبار منحنى:

```math
\begin{aligned}
\text{Newton's update:} \quad & w_{\text{new}} = w_{\text{old}} - H^{-1} \cdot \text{gradient} \\
\text{Gradient descent:} \quad & w_{\text{new}} = w_{\text{old}} - \text{lr} \cdot \text{gradient}
\end{aligned}
```

طريقة نيوتن تتقارب أسرع لأن "إعادة تقييم" هيسيان التدفق -- الاتجاهات الرافية تحصل على خطوات أصغر، الاتجاهات المسطحة تحصل على خطوات أكبر.

الصيد: لشبكة عصبية مع$N$المعلمات، هي Hessian هو$N \times N$نموذج مع مليون ملامح يحتاج إلى 1 تريليون مدخل المصفوفة. لهذا السبب نستخدم التقارير.

| Method | What it uses | Cost | Convergence |
|--------|-------------|------|-------------|
| Gradient descent | First derivatives only | $O(N)$ per step | Slow (linear) |
| Newton's method | Full Hessian | $O(N^3)$ per step | Fast (quadratic) |
| L-BFGS | Approximate Hessian from gradient history | $O(N)$ per step | Medium (superlinear) |
| Adam | Per-parameter adaptive rates (diagonal Hessian approx) | $O(N)$ per step | Medium |
| Natural gradient | Fisher information matrix (statistical Hessian) | $O(N^2)$ per step | Fast |

في الممارسة العملية، آدم هو المحسن الافتراضي للتعلم العميق. إنه يقترب من المعلومات من الدرجة الثانية بأسعار رخيصة عن طريق تتبع المتوسط الجاري وتباين التدرج لكل پیرامتر.

### تقارب سلسلة تايلور

يمكن تقريب أي وظيفة سلاسة محليا عن طريق الكتلة:

```math
f(x + h) = f(x) + f'(x)\,h + \frac{1}{2} f''(x)\,h^2 + \frac{1}{6} f'''(x)\,h^3 + \cdots
```

كلما تضيف المزيد من المصطلحات، كلما كانت التقارير أفضل -- ولكن فقط بالقرب من النقطة$x$. . .

**Why Taylor series matter for ML:**

- **First-order Taylor = gradient descent.**عندما تستخدم$f(x + h) \approx f(x) + f'(x) h$إنّك تقوم بتقريب خطيّ، إنّ التنحى التدريجيّ يقلل من هذا النموذج الخطيّ للاختيار$h = -\text{lr} \cdot f'(x)$. . .

- **Second-order Taylor = Newton's method.**استخدام$f(x + h) \approx f(x) + f'(x) h + \frac{1}{2} f''(x) h^2$، تحصل على نموذج مربع. تقليل ذلك يعطي$h = -f'(x)/f''(x)$خطوة نيوتن

- **Loss function design.**إن MSE و entropy متسطّة، مما يعني أنّ توسعاتها Taylor تتصرف بشكل جيد. هذه ليست حادثة. الخسائر السلسة تجعل التكيفّيّة قابلة للتنبؤ.

```
Approximation order    What it captures    Optimization method
-------------------    -----------------   -------------------
0th order (constant)   Just the value      Random search
1st order (linear)     Slope               Gradient descent
2nd order (quadratic)  Curvature           Newton's method
Higher orders          Finer structure     Rarely used in ML
```

المفهوم الرئيسي: كل التحسين القائم على التراجع هو حقا حول تقريب وظيفة الخسارة محليا وتحقيق الحد الأدنى من هذا التقريب.

### الإجماليات في ML

المشتقات تخبرك بمعدلات التغيير. الكاملات تحسب التراكم -- المساحة تحت منحنى.

في ML، نادراً ما تقوم بحساب التكاملات يدوياً، لكن المفهوم موجود في كل مكان:

**Probability.**بالنسبة لمتغير عشوائي مستمر مع كثافة $p(x)$:

```math
P(a < X < b) = \int_a^b p(x) \, dx
```

المساحة تحت منحنى كثافة الاحتمال بين $a$و$b$هو احتمال الهبوط في تلك النطاق.

**Expected value.**النتيجة المتوسطة الموزعة بالاحتمال:

```math
\mathbb{E}[f(X)] = \int f(x) \, p(x) \, dx
```

الخسارة المتوقعة على توزيع البيانات هي جزء لا يتجزأ. التدريب يقلل من التقريب التجريبي لهذا.

**KL divergence.**تقيس مدى اختلاف توزيعين:

```math
\mathrm{KL}(p \,\|\, q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx
```

يستخدم في VAEs، نزيف المعرفة، والإستنتاج البايسي.

**Normalization constants.**في استنتاج بايزي:

```math
p(w \mid \text{data}) = \frac{p(\text{data} \mid w) \, p(w)}{\int p(\text{data} \mid w) \, p(w) \, dw}
```

المعاد هو جزء متكامل على جميع قيم المعلمات المحتملة. غالبا ما يكون غير قابلة للتحدي، وهذا هو السبب في أننا نستخدم التقريبات مثل MCMC والإستنتاج المتغير.

| Integral concept | Where it appears in ML |
|-----------------|----------------------|
| Area under curve | Probability from density functions |
| Expected value | Loss functions, risk minimization |
| KL divergence | VAEs, policy optimization, distillation |
| Normalization | Bayesian posteriors, softmax denominator |
| Marginal likelihood | Model comparison, evidence lower bound (ELBO) |

### قاعدة سلسلة متعددة المتغيرات في الرسم البياني الحسابي

لا تنطبق قاعدة السلسلة على الوظائف المتعددة في خط. في شبكة عصبية، تتراوح المتغيرات وتدمج. إليك كيف تتدفق المشتقات من خلال مرور بسيط للأمام:

```mermaid
graph LR
    x["x (input)"] -->|"*w"| z1["z1 = w*x"]
    z1 -->|"+b"| z2["z2 = w*x + b"]
    z2 -->|"sigmoid"| a["a = sigmoid(z2)"]
    a -->|"loss fn"| L["L = -(y*log(a) + (1-y)*log(1-a))"]
```

الممر الخلفي يحسب التراجع من يمين إلى يسار:

```mermaid
graph RL
    dL["dL/dL = 1"] -->|"dL/da"| da["dL/da = -y/a + (1-y)/(1-a)"]
    da -->|"da/dz2 = a(1-a)"| dz2["dL/dz2 = dL/da * a(1-a)"]
    dz2 -->|"dz2/dw = x"| dw["dL/dw = dL/dz2 * x"]
    dz2 -->|"dz2/db = 1"| db["dL/db = dL/dz2 * 1"]
```

كل سهم يضاعف بمشتقات محلية. تراجع لأي معادلة هو ناتج من جميع المشتقات المحلية على طول المسار من الخسارة إلى تلك المعادلة. عندما تتفرق المسارات وتدمج، يمكنك جمع المساهمات (قاعدة سلسلة متعددة المتغيرات).

هذا كل التنشر الخلفي هو: قاعدة سلسلة تطبق بشكل منهجي من خلال الرسم البياني الحسابي، من الخروج إلى المدخلات.

### المصفوفة جاكوبية

عندما تقوم وظيفة بتخريط متجه إلى متجه (مثل طبقة شبكة عصبية) ، فإن مشتقها هو المصفوفة. يحتوي جيكوبيان على كل مشتق جزئي لكل خروج فيما يتعلق بكل مدخل.

لأجل$f: \mathbb{R}^n \to \mathbb{R}^m$، الـ (جاكوبيان)$J$هو$m \times n$المصفوفة:

| | $x_1$ | $x_2$ | ... | $x_n$ |
|---|---|---|---|---|
| $f_1$ | $\partial f_1/\partial x_1$ | $\partial f_1/\partial x_2$ | ... | $\partial f_1/\partial x_n$ |
| $f_2$ | $\partial f_2/\partial x_1$ | $\partial f_2/\partial x_2$ | ... | $\partial f_2/\partial x_n$ |
| ... | ... | ... | ... | ... |
| $f_m$ | $\partial f_m/\partial x_1$ | $\partial f_m/\partial x_2$ | ... | $\partial f_m/\partial x_n$ |

لن تقوم بحساب الـ "جاكوبيانز" يدوياً لشبكات عصبية. بيتورش يتعامل معها. لكن معرفة وجودها تساعدك على فهم الأشكال في الانتشار الخلفي: إذا كانت الطبقة تخطيط$\mathbb{R}^n$إلى$\mathbb{R}^m$، هو جاكوبيان هو$m \times n$. التراجع يتدفق إلى الوراء من خلال نقل هذه المصفوفة

### لماذا هذا مهم للشبكات العصبية

كل وزن في شبكة عصبية يحصل على تراجع. تراجع يخبرك كيفية ضبط هذا الوزن لتقليل الخسارة.

```mermaid
graph LR
    subgraph Forward["Forward Pass"]
        I["input"] --> W1["W1"] --> R["relu"] --> W2["W2"] --> S["softmax"] --> L["loss"]
    end
```

```mermaid
graph RL
    subgraph Backward["Backward Pass"]
        dL["dL/dloss"] --> dW2["dL/dW2"] --> d2["..."] --> dW1["dL/dW1"]
    end
```

كل تحديث للوزن:
- `W1 = W1 - lr * dL/dW1`
- `W2 = W2 - lr * dL/dW2`

المخطط الأمامي يحسب التنبؤ والخسارة، المخطط الخلفي يحسب تراجع الخسارة فيما يتعلق بكل وزن، ثم كل وزن يأخذ خطوة صغيرة أسفل التلال. كرر لملايين الخطوات. وهذا هو التعلم العميق.

```figure
derivative-tangent
```

## بناءها

### الخطوة الأولى: المشتق العددي من الصفر

```python
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f'(x) numerical={numerical:.6f}  analytical={analytical:.1f}")
```

المشتق العددي يطابق المشتق التحليلي مع العديد من الأماكن العشرية.

### الخطوة الثانية: مشتقات جزئية ومحافظات

```python
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"Numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Analytical gradient at (1,2): [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")
```

### الخطوة الثالثة: التراجع التدريجي للعثور على الحد الأدنى من f ((x) = x^2

```python
x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

بدءاً من$x=5$كل خطوة تقترب من$x=0$(الأقل)

### الخطوة الرابعة: انخفاض درجي على وظيفة 2D

```python
def f_2d(point):
    x, y = point
    return x**2 + y**2

point = [4.0, 3.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
```

### الخطوة 5: مقارنة المشتقات العددية والتحليلية

```python
import math

test_functions = [
    ("x^2",      lambda x: x**2,          lambda x: 2*x),
    ("x^3",      lambda x: x**3,          lambda x: 3*x**2),
    ("sin(x)",   lambda x: math.sin(x),   lambda x: math.cos(x)),
    ("e^x",      lambda x: math.exp(x),   lambda x: math.exp(x)),
    ("1/x",      lambda x: 1/x,           lambda x: -1/x**2),
]

x = 2.0
print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
print("-" * 50)
for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")
```

### الخطوة 6: حساب الهسسي عددا

```python
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x ** 2 - y ** 2

def bowl(x, y):
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)
print(f"Saddle Hessian: {H_saddle}")  # [[2, 0], [0, -2]] -- mixed signs
print(f"Bowl Hessian:   {H_bowl}")    # [[2, 0], [0, 2]]  -- both positive
```

يحتوي الحسسي لـ 2 و -2 (أشارة مختلطة تؤكد نقطة السرير). والقوطة لديها قيم خاصة 2 و 2 (كلتا إيجابية، تؤكد الحد الأدنى).

### الخطوة 7: تقريب تايلور في العمل

```python
import math

def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
    print(f"h={h:.1f}  sin(h)={true_val:.4f}  order1={t1:.4f}  order2={t2:.4f}")
```

قريبة$x_0=0$،$\sin(x) \approx x$(تايلور من الدرجة الأولى) التقريب ممتاز للصغار$h$لكن يتحطم للكبير$h$هذا هو السبب في أن تراجع التراجع يعمل بشكل أفضل مع معدلات التعلم الصغيرة - كل خطوة يفترض التقريب الخطي هو دقيق.

### الخطوة الثامنة: لماذا هذا مهم لشبكة عصبية

```python
import random

random.seed(42)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")
```

كل حلقة تدريبية مبنية على التراجع تتبع هذا النمط: التنبؤ، فقدان الحساب، تراجع الحساب، وزن التحديث.

## استخدمها

مع NumPy، نفس العمليات أسرع وأكثر وضوحا:

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b
    error = pred - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w -= lr * dw
    b -= lr * db

print(f"Learned: y = {w:.2f}x + {b:.2f}")
```

لقد بنيت للتو انخفاض التراجعية من الصفر PyTorch تلقائي الحسابات التراجعية، ولكن حلقة التحديث هو نفسه.

## التمارين

1. تنفيذ`numerical_second_derivative(f, x)`استخدام `numerical_derivative`اتصل مرتين. تحقق من أن المشتق الثاني من $x^3$في$x=2$هو 12
2. استخدم النزول المتحرك للعثور على الحد الأدنى من$f(x, y) = (x - 3)^2 + (y + 1)^2$ابدأ من$(0, 0)$يجب أن تتحرك الإجابة إلى$(3, -1)$. . .
3. إضافة الزخم إلى حلقة هبوط التدرج: الحفاظ على متجه السرعة الذي يتراكم في التدرج السابق. مقارنة سرعة التقارب مع ودون الزخم على $f(x) = x^4 - 3x^2$. . .

### إشارات

**Exercise 1.**تعريف المساعد الذي يعود`numerical_derivative(f, x)`ومرّ هذا المساعد إلى`numerical_derivative`مرة أخرى. إذا كان النتيجة ضجة، استخدم خطوة أكبر مثل `h=1e-4`لكلتا الاتصالات .$x=2$يجب أن يكون قريب من 12

**Exercise 2.**اكتب التراجع$(2(x - 3), 2(y + 1))$أو استخدام`numerical_gradient`، و اتصل`gradient_descent_nd`من$(0, 0)$. طبع النقطة كل 10 خطوات ، يجب أن تتحرك نحو$(3, -1)$والخسارة يجب أن تنخفض إلى ما يقرب من 0.

**Exercise 3.**حافظ على السرعة`v`, تحديثها على`v = momentum * v - lr * grad`، ثم أضف`v`إلى`x`. عد الخطوات التي يحتاجها كل طريقة للوصول إلى تراجع قريب من 0 ؛ الحد الأدنى هو $x = \pm\sqrt{1.5} \approx \pm 1.225$. . .

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Derivative | "The slope" | The rate of change of a function at a point. Tells you how much the output changes per unit change in input. |
| Partial derivative | "Derivative of one variable" | The derivative with respect to one variable while all others are held constant. |
| Gradient | "Direction of steepest ascent" | A vector of all partial derivatives. Points in the direction that increases the function fastest. |
| Gradient descent | "Go downhill" | Subtract the gradient (times a learning rate) from the parameters to reduce the loss. The core of neural network training. |
| Learning rate | "Step size" | A scalar that controls how big each gradient descent step is. Too large: diverge. Too small: converge slowly. |
| Chain rule | "Multiply the derivatives" | The rule for differentiating composed functions: $\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$. The mathematical basis of backpropagation. |
| Jacobian | "Matrix of derivatives" | When a function maps vectors to vectors, the Jacobian is the matrix of all partial derivatives of outputs with respect to inputs. |
| Numerical derivative | "Finite differences" | Approximating a derivative by evaluating the function at two nearby points and computing the slope between them. |
| Backpropagation | "Reverse-mode autodiff" | Computing gradients layer by layer from output to input using the chain rule. How neural networks learn. |
| Hessian | "Matrix of second derivatives" | The matrix of all second-order partial derivatives. Describes the curvature of a function. Positive definite Hessian at a critical point means local minimum. |
| Taylor series | "Polynomial approximation" | Approximating a function near a point using its derivatives: $f(x+h) \approx f(x) + f'(x)h + \frac{1}{2}f''(x)h^2 + \cdots$ The basis for understanding why gradient descent and Newton's method work. |
| Integral | "Area under the curve" | The accumulation of a quantity over a range. In ML, integrals define probabilities, expected values, and KL divergence. |

## المزيد من القراءة

- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus)- البصرية للشتات والتكاملات وقاعدة السلسلة
- [Stanford CS231n: Backpropagation](https://cs231n.github.io/optimization-2/)- كيف تتدفق التدرج عبر طبقات الشبكة العصبية

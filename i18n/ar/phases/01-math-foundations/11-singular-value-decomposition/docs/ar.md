# التفكك في القيمة الفريدة

> (إس.دي.إس) هو سكين الجبر السويسري لكل ماتريك واحدة كل عالم بيانات يحتاج إلى واحدة

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors & Matrices Operations), 03 (Matrix Transformations)
**Time:** ~120 minutes

## أهداف التعلم

- تنفيذ SVD عبر تكرار الطاقة وتفسير المعنى الهندسي من $U$،$\Sigma$و$V^T$
- تطبيق SVD المقطوعة لضغط الصورة وقياس نسبة الضغط مقابل خطأ إعادة الإعمار
- احسب مستوى "مور-بنروز" من خلال "SVD" لحل أنظمة أقل مربعات
- ربط SVD إلى PCA، وأنظمة التوصيات (عوامل متخفية) ، والتحليل النيمي المتخفية في NLP

## المشكلة

لديك ماتريكس 1000x2000. ربما تكون تصنيفات فيلم المستخدم. ربما تكون جدول ترددات في المستند. ربما تكون قيم البيكسل لصور. ربما تحتاج إلى ضغطها، وتخفيفها، وإيجاد هيكل مخفي فيها، أو حل نظام أقل مربعات معها. التكوين الخاص يعمل فقط على المصفوفات المربعة. حتى إذا كان ذلك، فإنه يتطلب من المصفوف أن يكون لديه مجموعة كاملة من المتجهات الخاصة المستقلة خطيا.

يعمل SVD على أي مصفوفة. أي شكل. أي رتبة. لا توجد شروط. إنه يفكك المصفوفة إلى ثلاثة عوامل تكشف عن هندسة ما تفعله المصفوفة إلى الفضاء. إنه الأكثر عامة وأكثر فائدة في جميع الجبر الخطى.

## المفهوم

### ما يفعله SVD بشكل هندسي

كل ماتريكس، بغض النظر عن الشكل، تقوم بعمليات ثلاث في التسلسل: تدور، وتحجم، وتدور.

```math
\underset{\substack{m \times n \\ \text{(any)}}}{A} = \underset{\substack{m \times m \\ \text{(rotate)}}}{U} \cdot \underset{\substack{m \times n \\ \text{(scale)}}}{\Sigma} \cdot \underset{\substack{n \times n \\ \text{(rotate)}}}{V^T}
```

مع أي ماتريكس $A$، و سد يعتبر ذلك في:
- $V^T$يدور المتجهات في مساحة المدخل ($n$-بعد)
- $\Sigma$المقاييس على طول كل محور (ممتد أو مضغوطات)
- $U$يدور النتيجة في مساحة الخروج ($m$-بعد)

```mermaid
graph LR
    A["Input space (n-dim)\nData cloud\n(arbitrary orientation)"] -->|"V^T\n(rotate)"| B["Scaled space\nAligned with axes\nthen scaled by Sigma"]
    B -->|"U\n(rotate)"| C["Output space (m-dim)\nRotated to output\norientation"]
```

فكري بالأمر بهذه الطريقة. تقوم بتقديم المصفوفة إلى SVD. وتقول لك: "تأخذ هذه المصفوفة كرة من المدخلات، وتدورها أولاً$V^T$ثم تمدّها إلى أليبسوسيد$\Sigma$، ثم يدور الاليبسوسيد$U$." القيم الفردية هي أطوال محور البناية.

### التفكك الكامل

لـ ماتريكس$A$مع شكل$m \times n$:

```math
\begin{aligned}
& A = U \cdot \Sigma \cdot V^T \\[6pt]
& \text{where:} \\
& \quad U \text{ is } m \times m, \text{ orthogonal } (U^T U = I) \\
& \quad \Sigma \text{ is } m \times n, \text{ diagonal (singular values on the diagonal)} \\
& \quad V \text{ is } n \times n, \text{ orthogonal } (V^T V = I) \\[6pt]
& \text{The singular values } \sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0 \\
& \text{where } r = \mathop{\mathrm{rank}}(A)
\end{aligned}
```

عمودات $U$يُسمى المتجهات المتفردة اليسرى.$V$يُدعى المتجهات المتوحدة اليمنى.$\Sigma$تسمى القيم الفردية. وهي دائما غير سلبية وتقوم بتنظيمها بشكل تقليدي في ترتيب متناقص.

### المتجهات الفردية اليسرى، القيم الفردية، المتجهات الفردية اليمنى

كل مكون من SVD له معنى هندسي متميز.

**Right singular vectors (columns of $V$):**هذه تشكل أساسًا طبيعيًا للأماكن المدخولية ($\mathbb{R}^n$(). هي الاتجاهات في مساحة المدخل التي تقوم المصفوفة بتخطيطها إلى الاتجاهات المباشرة في مساحة الخروج. فكر في هذه الاتجاهات كنظام التنسيق الطبيعي للمجال.

**Singular values (diagonal of $\Sigma$):**هذه هي عوامل التوسع. القيمة الفردية الثانية تخبرك كم تمتد المصفوفة المتجهات على طول المتجهة الفردية الثانية اليمنى. القيمة الفردية من الصفر تعني أن المصفوفة تحطم هذا الاتجاه بالكامل.

**Left singular vectors (columns of $U$):**هذه تشكل أساسًا طبيعيًا للأماكن الخارجة ($\mathbb{R}^m$). المتجه الفردي الأيسر الثاني هو الاتجاه في مساحة الخروج حيث يتراجع المتجه الفردي الأيمن الثاني (بعد التوسع).

العلاقة بينهما:

```
A * v_i = sigma_i * u_i

The matrix A takes the i-th right singular vector v_i,
scales it by sigma_i, and maps it to the i-th left singular vector u_i.
```

هذا يعطيك صورة من التنسيقات إلى التنسيقات لما تفعله أي ماتريكس

### شكل المنتج الخارجي

يمكن كتابة SVD كجمع من المصفوفات الدرجة-1:

```
A = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T + ... + sigma_r * u_r * v_r^T

Each term sigma_i * u_i * v_i^T is a rank-1 matrix (an outer product).
The full matrix is the sum of r such matrices, where r is the rank.
```

هذا النموذج هو أساس التقريب منخفض الرتب. كل عبارة يضيف طبقة واحدة من الهيكل. العبارة الأولى تسجل نمط واحد أهم. الثانية تسجل الأهم التالي. وهكذا. تقسيم هذا المبلغ يمنحك أفضل تقريب ممكن في أي صف معين.

```math
\begin{aligned}
\text{Rank-1 approx:} \quad & A_1 = \sigma_1 u_1 v_1^T \\
& \text{(captures the dominant pattern)} \\[6pt]
\text{Rank-2 approx:} \quad & A_2 = \sigma_1 u_1 v_1^T + \sigma_2 u_2 v_2^T \\
& \text{(captures the two most important patterns)} \\[6pt]
\text{Rank-}k\text{ approx:} \quad & A_k = \text{sum of top } k \text{ terms} \\
& \text{(optimal by the Eckart-Young theorem)}
\end{aligned}
```

### العلاقة مع التكوين الخاص

إن SVD و Eigendecomposition مرتبطة بعمق.$A$تأتي مباشرة من القيم الخاصة والمتجهات الخاصة$A^T A$و$A A^T$. . .

```
A^T A = V * Sigma^T * U^T * U * Sigma * V^T
      = V * Sigma^T * Sigma * V^T
      = V * D * V^T

where D = Sigma^T * Sigma is a diagonal matrix with sigma_i^2 on the diagonal.

So:
- The right singular vectors (V) are eigenvectors of A^T A
- The singular values squared (sigma_i^2) are eigenvalues of A^T A

Similarly:
A A^T = U * Sigma * V^T * V * Sigma^T * U^T
      = U * Sigma * Sigma^T * U^T

So:
- The left singular vectors (U) are eigenvectors of A A^T
- The eigenvalues of A A^T are also sigma_i^2
```

هذا الاتصال يخبرك بثلاث أشياء:
1. القيم الفردية هي دائما حقيقية وغير سلبية (إنها جذور مربعة من القيم الخاصة للمصفوفة شبه محددة إيجابية).
2. يمكنك حساب SVD عن طريق التكوين الخاص من$A^T A$لكن هذا يربع رقم الحالة و يفقد دقة الرقمية. خوارزميات SVD المخصصة تتجنب هذا.
3. متى$A$هو مربع ومتوازي الإيجابي شبه محدد، SVD و Eigenende Komposition هي نفس الشيء.

### التقريبات المتقصرة: تقريب منخفض

نظرية إيكارت يونغ ميرسكي تقول أن أفضل$\text{rank-}k$التقريب إلى $A$(في كل من فروبينيوس والمعايير الطيفية) يتم الحصول عليها عن طريق الحفاظ على الجزء العلوي فقط $k$القيم الفردية والمتجهات المقابلة لها:

```math
\begin{aligned}
& A_k = U_k \cdot \Sigma_k \cdot V_k^T \\[6pt]
& \text{where:} \\
& \quad U_k \text{ is } m \times k \quad \text{(first } k \text{ columns of } U\text{)} \\
& \quad \Sigma_k \text{ is } k \times k \quad \text{(top-left } k \times k \text{ block of } \Sigma\text{)} \\
& \quad V_k \text{ is } n \times k \quad \text{(first } k \text{ columns of } V\text{)} \\[6pt]
& \text{Approximation error} = \sigma_{k+1} \quad \text{(in spectral norm)} \\
& \text{Approximation error} = \sqrt{\sigma_{k+1}^2 + \cdots + \sigma_r^2} \quad \text{(in Frobenius norm)}
\end{aligned}
```

هذه ليست مجرد تقارب "جيد" ، بل هي على الأرجح أفضل تقارب ممكن للرتب$k$لا يوجد آخر$\text{rank-}k$المصفوفة أقرب إلى$A$. . .

| Component | Relative magnitude | Kept in rank-3 approx? |
|-----------|-------------------|------------------------|
| $\sigma_1$ | Largest | Yes |
| $\sigma_2$ | Large | Yes |
| $\sigma_3$ | Medium-large | Yes |
| $\sigma_4$ | Medium | No (error) |
| $\sigma_5$ | Medium-small | No (error) |
| $\sigma_6$ | Small | No (error) |
| $\sigma_7$ | Very small | No (error) |
| $\sigma_8$ | Tiny | No (error) |

أبقوا أعلى ثلاثة$A_3$يحتوي على أكبر ثلاثة قيم فردية. الخطأ = القيم المتبقية ($\sigma_4$من خلال$\sigma_8$)

إذا كانت القيم الفردية تتحلل بسرعة،$k$إذا تدهور ببطء، فإن المصفوفة لا تملك بنية منخفضة.

### ضغط الصورة مع SVD

صورة على نطاق الرمادي هي ماتريكية من كثافة البيكسل. صورة 800 × 600 لديها 480,000 قيم. SVD يسمح لك تقريرها مع أقل بكثير.

```
Original image: 800 x 600 = 480,000 values

SVD with rank k:
  U_k:      800 x k values
  Sigma_k:  k values
  V_k:      600 x k values
  Total:    k * (800 + 600 + 1) = k * 1401 values

  k=10:   14,010 values   (2.9% of original)
  k=50:   70,050 values  (14.6% of original)
  k=100: 140,100 values  (29.2% of original)

  The compression ratio improves as k gets smaller,
  but visual quality degrades.
```

المعلومات الرئيسية: الصور الطبيعية لها قيم فردية تتدهور بسرعة. الأقوال الفردية الأولى تتقاط الهيكل الواسع (الشكالات والتحركات). والآخرين يلتقطون التفاصيل الدقيقة والضوضاء. تقصير في المرتبة 50 غالبا ما ينتج صورة تبدو متطابقة تقريبًا مع الأصلية مع استخدام 85٪ أقل من التخزين.

### الجهاز التنفيذي للأنظمة التوصية

جائزة نتفليكس جعلت هذا مشهوداً. لديك ماتريكس تصنيف المستخدمين للأفلام حيث تفتقر معظم الإدخالات.

```
             Movie1  Movie2  Movie3  Movie4  Movie5
  User1      [  5      ?       3       ?       1  ]
  User2      [  ?      4       ?       2       ?  ]
  User3      [  3      ?       5       ?       ?  ]
  User4      [  ?      ?       ?       4       3  ]

  ? = unknown rating
```

الفكرة: هذه المصفوفة المصفوفة لديها رتبة منخفضة. لا يمتلك المستخدمون ذوق مستقل تماما. هناك عدد قليل من العوامل الخفية (العمل مقابل الدراما، القديم مقابل الجديد، الدماغ مقابل الدموية) التي تفسر معظم الاختيارات.

يزرق SVD على ماتريك التصنيفات (المملئة) إلى:
- $U$: ملفات المستخدم في الفضاء المتخفي
- $\Sigma$: أهمية كل عامل متخفي
- $V^T$: ملفات الفيلم في الفضاء المتخفي

تصنيف المستخدم المتوقع لفيلم هو نسبة نقطة من ملف الشخصية المستخدم مع ملف الشخصية الفيلم (موزنًا بقيم فردية). يملأ التقريب المنخفض المرتبة الإدخالات المفقودة.

في الممارسة العملية، تستخدم المتغيرات مثل SVD أو ALS المتزايد من سيمون فانك (التبديل على الأقل مربعات) التي تتعامل مع البيانات المفقودة مباشرة. ولكن الفكرة الأساسية هي نفسها: تدهور العوامل الخفية عن طريق SVD.

### التفاصيل في النمط النووي: التحليل النيزكي المتخفي

تحليل اللاتنت المفصل (LSA) ، والذي يسمى أيضاً مؤشر اللاتنت المفصل (LSI) ، يطبق SVD على ماتريكس الوثيقة المحددة.

```
             Doc1   Doc2   Doc3   Doc4
  "cat"      [  3      0      1      0  ]
  "dog"      [  2      0      0      1  ]
  "fish"     [  0      4      1      0  ]
  "pet"      [  1      1      1      1  ]
  "ocean"    [  0      3      0      0  ]

After SVD with rank k=2:

  Each document becomes a point in 2D "concept space."
  Each term becomes a point in the same 2D space.
  Documents about similar topics cluster together.
  Terms with similar meanings cluster together.

  "cat" and "dog" end up near each other (land pets).
  "fish" and "ocean" end up near each other (water concepts).
  Doc1 and Doc3 cluster if they share similar topics.
```

كانت LSA واحدة من أوائل الطرق الناجحة لاستيعاب التشابه الدلالي من النص الخام. تعمل لأنه تميل إلى ظهور مصطلحات متجانسة في وثائق مماثلة، لذلك تقوم SVD بتجميعها في نفس الأبعاد الخفية. يمكن اعتبار تضمين الكلمات الحديثة (Word2Vec، GloVe) نذراً لهذه الفكرة.

### الـ SVD للحد من الضوضاء

البيانات الضوضاء لديها إشارة مركزة في أعلى القيم الفردية والضوضاء تنتشر على جميع القيم الفردية.

**Clean signal singular values:**

| Component | Magnitude | Type |
|-----------|-----------|------|
| $\sigma_1$ | Very large | Signal |
| $\sigma_2$ | Large | Signal |
| $\sigma_3$ | Medium | Signal |
| $\sigma_4$ | Near zero | Negligible |
| $\sigma_5$ | Near zero | Negligible |

**Noisy signal singular values (noise adds to all):**

| Component | Magnitude | Type |
|-----------|-----------|------|
| $\sigma_1$ | Very large | Signal |
| $\sigma_2$ | Large | Signal |
| $\sigma_3$ | Medium | Signal |
| $\sigma_4$ | Small | Noise |
| $\sigma_5$ | Small | Noise |
| $\sigma_6$ | Small | Noise |
| $\sigma_7$ | Small | Noise |

```mermaid
graph TD
    A["All singular values"] --> B{"Clear gap?"}
    B -->|"Above gap"| C["Signal: keep these (top k)"]
    B -->|"Below gap"| D["Noise: discard these"]
    C --> E["Reconstruct with A_k to get denoised version"]
```

يستخدم هذا في معالجة الإشارات والقياس العلمي وتنظيف البيانات. في أي وقت يكون لديك ماتريكس فاسدة من ضجيج إضافي، فإن SVD المقطوع هو وسيلة مبدئية لفرق الإشارة عن الضجيج.

### الاختلافات السوداء عبر SVD

"المركز المضاد "مور - بينروز$A^+$يُعمّل عكس المصفوفة إلى المصفوفات غير المربعية والفردية.

```
If A = U * Sigma * V^T, then:

A+ = V * Sigma+ * U^T

where Sigma+ is formed by:
  1. Transpose Sigma (swap rows and columns)
  2. Replace each non-zero diagonal entry sigma_i with 1/sigma_i
  3. Leave zeros as zeros

For A (m x n):      A+ is (n x m)
For Sigma (m x n):  Sigma+ is (n x m)
```

إنّ الكون السودوي يحلّ مشاكل أقلّ مربعات$Ax = b$لا يوجد حل دقيق (نظام محدد جدا) ، ثم $x = A^+ b$هو حل أقل مربعات (يقلل $\lVert Ax - b \rVert$)

```
Overdetermined system (more equations than unknowns):

  [1  1]         [3]
  [2  1] x   =   [5]       No exact solution exists.
  [3  1]         [6]

  x_ls = A+ b = V * Sigma+ * U^T * b

  This gives the x that minimizes the sum of squared residuals.
  Same result as the normal equations (A^T A)^(-1) A^T b,
  but numerically more stable.
```

### مزايا الاستقرار الرقمي

الحسابات الخاصة بتكوين $A^T A$يربع القيم الفردية (قيم المملكة الخاصة)$A^T A$هل هي$\sigma_i^2$هذا يربع رقم الحالة، مما يزيد من الأخطاء الرقمية.

```
Example:
  A has singular values [1000, 1, 0.001]
  Condition number of A: 1000 / 0.001 = 10^6

  A^T A has eigenvalues [10^6, 1, 10^{-6}]
  Condition number of A^T A: 10^6 / 10^{-6} = 10^{12}

  Computing SVD directly: works with condition number 10^6
  Computing via A^T A:     works with condition number 10^{12}
                           (6 extra digits of precision lost)
```

تعمل خوارزميات SVD الحديثة (الوجبات الثنائية من Golub-Kahan) مباشرة على $A$، لا تتشكل أبداً$A^T A$لهذا يجب أن تفضل دائماً`np.linalg.svd(A)`- لقد انتهت`np.linalg.eig(A.T @ A)`. . .

### اتصال مع PCA

(بي سي ايه) هو (سڤيد) على البيانات المركزة، هذه ليست مقارنة، إنها حرفيا نفس الحسابات.

```
Given data matrix X (n_samples x n_features), centered (mean subtracted):

Covariance matrix: C = (1/(n-1)) * X^T X

PCA finds eigenvectors of C. But:

  X = U * Sigma * V^T    (SVD of X)

  X^T X = V * Sigma^2 * V^T

  C = (1/(n-1)) * V * Sigma^2 * V^T

So the principal components are exactly the right singular vectors V.
The explained variance for each component is sigma_i^2 / (n-1).

In sklearn, PCA is implemented using SVD, not eigendecomposition.
It is faster and more numerically stable.
```

هذا يعني أن كل ما تعلمته عن تقليل الأبعاد في الدروس 10 هو SVD تحت الغطاء. PCA هو التطبيق الأكثر شيوعا من SVD في التعلم الآلي.

```figure
svd-rank-reconstruction
```

## بناءها

### الخطوة 1: SVD من الصفر باستخدام التكرار الطاقة

الفكرة: للعثور على أكبر قيمة فردية ومجهراتها، استخدم التكرار القوى على $A^T A$(أو $A A^T$) ثم تخفف المصفوفة وتكرر للقيمة الفردية التالية.

```python
import numpy as np

def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)

    eigenvalue = v @ M @ v
    return eigenvalue, v

def svd_from_scratch(A, k=None):
    m, n = A.shape
    if k is None:
        k = min(m, n)

    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for _ in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=200)

        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V
```

### الخطوة الثانية: اختبار ومقارنة مع NumPy

```python
np.random.seed(42)
A = np.random.randn(5, 4)

U_ours, S_ours, V_ours = svd_from_scratch(A)
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

print("Our singular values:", np.round(S_ours, 4))
print("NumPy singular values:", np.round(S_np, 4))

A_reconstructed = U_ours @ np.diag(S_ours) @ V_ours.T
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.8f}")
```

### الخطوة 3: عرض لضغط الصورة

```python
def compress_image_svd(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return compressed

image = np.random.seed(42)
rows, cols = 200, 300
image = np.random.randn(rows, cols)

for k in [1, 5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)
    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    original_size = rows * cols
    compressed_size = k * (rows + cols + 1)
    ratio = compressed_size / original_size
    print(f"k={k:>3d}  error={error:.4f}  storage={ratio:.1%}")
```

### الخطوة الرابعة: تقليل الضوضاء

```python
np.random.seed(42)
clean = np.outer(np.sin(np.linspace(0, 4*np.pi, 100)),
                 np.cos(np.linspace(0, 2*np.pi, 80)))
noise = 0.3 * np.random.randn(100, 80)
noisy = clean + noise

U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
denoised = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]

print(f"Noisy error:    {np.linalg.norm(noisy - clean):.4f}")
print(f"Denoised error: {np.linalg.norm(denoised - clean):.4f}")
print(f"Improvement:    {(1 - np.linalg.norm(denoised - clean) / np.linalg.norm(noisy - clean)):.1%}")
```

### الخطوة 5: الاختلاف

```python
A = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
b = np.array([3, 5, 6], dtype=float)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1.0 / S)
A_pinv = Vt.T @ S_inv @ U.T

x_svd = A_pinv @ b
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
x_pinv = np.linalg.pinv(A) @ b

print(f"SVD pseudoinverse solution:  {x_svd}")
print(f"np.linalg.lstsq solution:   {x_lstsq}")
print(f"np.linalg.pinv solution:    {x_pinv}")
```

## استخدمها

الظهور الكامل يعمل في `code/svd.py`. إشغيله لترى SVD تطبيق على ضغط الصورة، وأنظمة التوصيات، التحليل الامتناع، وتقليل الضوضاء.

```bash
python svd.py
```

نسخة جوليا في`code/svd.jl`يظهر نفس المفاهيم باستخدام الوطني جوليا `svd()`وظيفة و`LinearAlgebra`الحزمة

```bash
julia svd.jl
```

## أرسله

هذا الدرس ينتج عن:
- `outputs/skill-svd.md`- مهارة لمعرفة متى وكيفية تطبيق SVD في المشاريع الحقيقية

## التمارين

1. تنفيذ SVD كامل من الصفر دون استخدام التكرار القوى. بدلا من ذلك، حساب التكوين الخاص من $A^T A$للحصول على$V$والقيم الفردية، ثم الحساب$U = A V \Sigma^{-1}$. مقارنة الدقة الرقمية مع إصدار التكرار القوى الخاص بك ومع NumPy.

2. تحميل صورة حقيقية على نطاق الرمادي (أو تحويل واحدة إلى نطاق الرمادي). ضغطها على الصفوف 1، 5، 10، 25، 50، 100. لكل صف، احسب نسبة الضغط والخطأ النسبي. العثور على الصف الذي يصبح الصورة مقبولة بصريا.

3. قم ببناء نظام توصيات صغير. قم بإنشاء ماتريكية تصنيفات فيلم مستخدم 10 × 8 مع بعض الإدخالات المعروفة. املأ الإدخالات المفقودة بوسائل الصف. احسب SVD واستعادة تقريب الدرجة 3. استخدم المصفوفة المعدة لإتنبؤ بالتصنيفات المفقودة. تحقق من أن التنبؤات معقولة.

4. قم بإنشاء ماتريكية 100 × 50 من المستندات المختلفة مع 3 مواضيع اصطناعية. لكل موضوع 5 مصطلحات مرتبطة. أضف الضوضاء. تطبيق SVD وتحقق من أن القيم الثلاثة الأولى من الفردية أكبر بكثير من البقية. مشروع المستندات في الفضاء الخاطئ 3D وتحقق من أن المستندات من نفس مجموعة الموضوع معا.

5. إنشاء صفة منخفضة صفوف (صفوف 3، حجم 50x40) وإضافة ضجيج غوسيان في مستويات مختلفة ($\sigma$= 0.1، 0.5، 1.0، 2.0) لكل مستوى ضجيج، العثور على المرتبة المثلى من التقطيع عن طريق مسح $k$من 1 إلى 40 و قياس خطأ إعادة الإعمار ضد المصفوفة النظيفة. رسم كيفية المثالي $k$تغيرات مع مستوى الضوضاء.

### إشارات

**Exercise 1.**الحساب$A^T A$، الحصول على قيمها الخاصة والمتجهات الخاصة مع `np.linalg.eigh`والقيم الفردية هي الجذور المربعة للقيم الخاصة، تحقق من ذلك$U \Sigma V^T$إعادة البناء$A$و مقارنة الخطأ مع`svd_from_scratch`و`np.linalg.svd`. . .

**Exercise 2.**اتصل`truncated_svd`لكل صف، أعيد بناء الصورة`reconstruct`، واستخدامها`compression_ratio(m, n, k)`الخطأ النسبي يجب أن ينخفض مع نمو الصف، والصف الأول حيث لا ترى بلاك واضح هو الجواب.

**Exercise 3.**قم بتسجيل الخلايا المفقودة، وملئها بمعدل الصف، ثم أعيد بناء مقربة من الصف 3 مع `truncated_svd`و`reconstruct`ثم قياس النتيجة: احتساب التنبؤات خارج نطاق التصنيف، وتطبيق أكبر تغيير في خلية معروفة. مقربة 3 رتبة لا تضمن أي من الممتلكات، لذلك قص التنبؤات إلى النطاق إذا كنت بحاجة إلى تصنيفات صالحة.

**Exercise 4.**أعط كل موضوع كتلة من 5 تعبيرات مع أعداد عالية لمستنداته ، ثم أضف ضجيجًا صغيرًا. طبع القيم الفردية: يجب أن تكون 3 الأولى أكبر بكثير من 4 . ثم ألون إحداثيات 3D حسب الموضوع وتحقق من أن كل موضوع يشكل مجموعة واحدة.

**Exercise 5.**قم بتحويل المصفوفة النظيفة إلى نسبة 50x3 و 3x40 المصفوفة عشوائية، ثم أضف الضجيج لكل منها $\sigma$لكل واحد$k$، مقارنة الرتب$k$إعادة بناء مع المصفوفة النظيفة ، توقع أفضل $k$قرب 3 لضوضاء منخفضة وأصغر لضوضاء عالية.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| SVD | "Factor any matrix" | Decompose $A$ into $U \Sigma V^T$ where $U$ and $V$ are orthogonal and $\Sigma$ is diagonal with non-negative entries. Works for any matrix of any shape. |
| Singular value | "How important this component is" | The $i$-th diagonal entry of $\Sigma$. Measures how much the matrix stretches along the i-th principal direction. Always non-negative, sorted in decreasing order. |
| Left singular vector | "Output direction" | A column of $U$. The direction in output space that the $i$-th right singular vector maps to (after scaling by $\sigma_i$). |
| Right singular vector | "Input direction" | A column of $V$. The direction in input space that the matrix maps to the $i$-th left singular vector (after scaling by $\sigma_i$). |
| Truncated SVD | "Low-rank approximation" | Keep only the top $k$ singular values and their vectors. Produces the provably best $\text{rank-}k$ approximation to the original matrix (Eckart-Young theorem). |
| Rank | "True dimensionality" | The number of non-zero singular values. Tells you how many independent directions the matrix actually uses. |
| Pseudoinverse | "Generalized inverse" | $V \Sigma^+ U^T$. Inverts non-zero singular values, leaves zeros as zeros. Solves least-squares problems for non-square or singular matrices. |
| Condition number | "How sensitive to errors" | $\sigma_{\max} / \sigma_{\min}$. A large condition number means small input changes cause large output changes. SVD reveals this directly. |
| Latent factor | "Hidden variable" | A dimension in the low-rank space discovered by SVD. In recommendations, a latent factor might correspond to genre preference. In NLP, it might correspond to a topic. |
| Frobenius norm | "Total matrix size" | Square root of the sum of squared entries. Equals the square root of the sum of squared singular values. Used to measure approximation error. |
| Eckart-Young theorem | "SVD gives the best compression" | For any target rank $k$, the truncated SVD minimizes the approximation error over all possible $\text{rank-}k$ matrices. |
| Power iteration | "Find the biggest eigenvector" | Repeatedly multiply a random vector by the matrix and normalize. Converges to the eigenvector with the largest eigenvalue. The building block of many SVD algorithms. |

## المزيد من القراءة

- [Gilbert Strang: Linear Algebra and Its Applications, Chapter 7](https://math.mit.edu/~gs/linearalgebra/)- علاج الدماغ المريض بالتهابات المزمنة
- [3Blue1Brown: But what is the SVD?](https://www.youtube.com/watch?v=vSczTbgc8Rc)- الحس البدني الهندسي لـ SVD
- [We Recommend a Singular Value Decomposition](https://www.ams.org/publicoutreach/feature-column/fcarc-svd)- نظرة عامة متاحة من الجمعية الأمريكية للرياضيات
- [Netflix Prize and Matrix Factorization](https://sifter.org/~simon/journal/20061211.html)- المقال الأصلي من سايمون فانك على موقع SVD للحصول على توصيات
- [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)- التطبيق الأصلي لـ NLP لـ SVD
- [Numerical Linear Algebra by Trefethen and Bau](https://people.maths.ox.ac.uk/trefethen/text.html)- المعيار الذهبي لفهم خوارزميات SVD وخصائصها الرقمية

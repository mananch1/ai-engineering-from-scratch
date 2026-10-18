# Định hướng siêu tham số

> Các siêu tham số là những nút mà bạn xoay trước khi bắt đầu tập luyện.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 2, Lesson 11 (Ensemble Methods)
**Time:** ~90 minutes

## Mục tiêu học tập

- Thực hiện tìm kiếm lưới, tìm kiếm ngẫu nhiên và tối ưu hóa Bayesian từ đầu và so sánh hiệu quả mẫu của chúng
- Giải thích tại sao tìm kiếm ngẫu nhiên vượt trội hơn tìm kiếm lưới khi hầu hết các siêu tham số có chiều kích hiệu quả thấp
- Xây dựng một vòng tối ưu hóa Bayesian sử dụng mô hình thay thế và hàm thu thập để hướng dẫn tìm kiếm
- Thiết kế một chiến lược điều chỉnh siêu tham số để tránh quá phù hợp với bộ xác thực thông qua xác thực chéo thích hợp

## Vấn đề

Mô hình tăng gradient của bạn có tốc độ học tập, số lượng cây, độ sâu tối đa, mẫu ít nhất mỗi lá, tỷ lệ mẫu phụ, và tỷ lệ mẫu cột.$5^6 = 15{,}625$mỗi tập mất 10 giây, đó là 43 giờ tính toán để thử tất cả.

Tìm kiếm lưới là cách tiếp cận rõ ràng và tồi tệ nhất trên quy mô. Tìm kiếm ngẫu nhiên làm tốt hơn với ít tính toán hơn. Phối ưu hóa Bayesian làm tốt hơn nữa bằng cách học hỏi từ các đánh giá trước đây. Biết chiến lược nào để sử dụng, và các siêu tham số thực sự quan trọng, tiết kiệm nhiều ngày thời gian GPU lãng phí.

## Khái niệm

### Các tham số so với các tham số siêu

Các tham số được học trong quá trình đào tạo (não trọng lượng, thiên vị, ngưỡng chia).

| Hyperparameter | What it controls | Typical range |
|---------------|-----------------|---------------|
| Learning rate | Step size per update | 0.001 to 1.0 |
| Number of trees/epochs | How long to train | 10 to 10,000 |
| Max depth | Model complexity | 1 to 30 |
| Regularization ($\lambda$) | Overfitting prevention | 0.0001 to 100 |
| Batch size | Gradient estimation noise | 16 to 512 |
| Dropout rate | Fraction of neurons dropped | 0.0 to 0.5 |

### Tìm kiếm lưới

Tìm kiếm lưới đánh giá mọi sự kết hợp của các giá trị được chỉ định. Nó là đầy đủ và dễ hiểu, nhưng cân bằng theo số lượng siêu tham số.

```
Grid for 2 hyperparameters:

  learning_rate: [0.01, 0.1, 1.0]
  max_depth:     [3, 5, 7]

  Evaluations: 3 x 3 = 9 combinations

  (0.01, 3)  (0.01, 5)  (0.01, 7)
  (0.1,  3)  (0.1,  5)  (0.1,  7)
  (1.0,  3)  (1.0,  5)  (1.0,  7)
```

Tìm kiếm lưới có một lỗ hổng cơ bản: nếu một siêu tham số quan trọng và một khác không quan trọng, hầu hết các đánh giá đều lãng phí. Bạn chỉ nhận được 3 giá trị độc đáo của tham số quan trọng từ 9 đánh giá.

### Tìm kiếm ngẫu nhiên

Tìm kiếm ngẫu nhiên các mẫu siêu tham số từ phân phối thay vì lưới. Với cùng ngân sách của 9 đánh giá, bạn nhận được 9 giá trị độc đáo của mỗi siêu tham số.

```mermaid
flowchart LR
    subgraph Grid Search
        G1[3 unique learning rates]
        G2[3 unique max depths]
        G3[9 total evaluations]
    end

    subgraph Random Search
        R1[9 unique learning rates]
        R2[9 unique max depths]
        R3[9 total evaluations]
    end
```

Tại sao số lượng ngẫu nhiên đánh lưới (Bergstra & Bengio, 2012):

- Hầu hết các siêu tham số có kích thước hiệu quả thấp. Chỉ có 1-2 trong số 6 siêu tham số thường quan trọng cho một vấn đề nhất định.
- Đánh giá chất thải tìm kiếm lưới trên các kích thước không quan trọng.
- Tìm kiếm ngẫu nhiên bao gồm các chiều kích quan trọng dày đặc hơn cho cùng một ngân sách.
- Trong 60 thử nghiệm ngẫu nhiên, bạn có 95% cơ hội tìm thấy một điểm trong 5% tối ưu (nếu có trong không gian tìm kiếm).

### Bayesian Optimization

Tìm kiếm ngẫu nhiên bỏ qua kết quả. Nó không tìm hiểu rằng tỷ lệ học tập cao gây ra sự khác biệt hoặc rằng độ sâu 3 liên tục vượt qua độ sâu 10.

```mermaid
flowchart TD
    A[Define search space] --> B[Evaluate initial random points]
    B --> C[Fit surrogate model to results]
    C --> D[Use acquisition function to pick next point]
    D --> E[Evaluate the model at that point]
    E --> F{Budget exhausted?}
    F -->|No| C
    F -->|Yes| G[Return best hyperparameters found]
```

Hai thành phần chính:

**Surrogate model:**Một mô hình giá rẻ để đánh giá (thường là một quy trình Gaussian) gần với chức năng khách quan đắt tiền. Nó cung cấp cả một dự đoán và ước tính độ không chắc chắn ở bất kỳ điểm nào trong không gian tìm kiếm.

**Acquisition function:**Quyết định xem xét tiếp theo là ở đâu bằng cách cân bằng giữa khai thác (số tìm kiếm gần những điểm tốt được biết đến) và khám phá (số tìm kiếm nơi có sự không chắc chắn cao).

- **Expected Improvement (EI):**Chúng ta mong đợi sự cải thiện như thế nào so với những gì tốt nhất hiện tại tại tại thời điểm này?
- **Upper Confidence Bound (UCB):**Dự đoán cộng với số không chắc chắn.
- **Probability of Improvement (PI):**Có khả năng điểm này vượt qua điểm hiện tại tốt nhất là gì?

Tối ưu hóa Bayesian thường tìm thấy các tham số siêu tốt hơn so với tìm kiếm ngẫu nhiên với 2-5 lần ít đánh giá hơn. Chi phí chung của việc lắp đặt mô hình thay thế là không đáng kể so với đào tạo mô hình thực tế.

### Giữ sớm

Không phải mỗi cuộc tập luyện đều cần phải kết thúc. Nếu một cấu hình rõ ràng xấu sau 10 thời kỳ, hãy dừng nó và tiếp tục. Đây là dừng sớm trong bối cảnh tìm kiếm siêu tham số.

Chiến lược:
- **Patience-based:**Ngưng nếu mất hiệu quả không cải thiện trong N thời kỳ liên tiếp
- **Median pruning:**Giữ lại nếu kết quả trung bình của thử nghiệm là tồi tệ hơn trung bình của các thử nghiệm hoàn thành ở cùng một bước
- **Hyperband:**Đưa ra ngân sách nhỏ cho nhiều cấu hình, sau đó tăng dần ngân sách cho những thứ tốt nhất

Hyperband đặc biệt hiệu quả. Nó bắt đầu 81 cấu hình với mỗi 1 thời kỳ, giữ phần ba đầu, cho họ 3 thời kỳ, giữ phần ba đầu, v.v. Điều này tìm thấy các cấu hình tốt 10-50 lần nhanh hơn so với đánh giá tất cả các cấu hình cho ngân sách đầy đủ.

### Các lập trình học tập

Tốc độ học tập hầu như luôn luôn là siêu tham số quan trọng nhất.

| Scheduler | Formula | When to use |
|-----------|---------|-------------|
| Step decay | Multiply by 0.1 every N epochs | Classic CNN training |
| Cosine annealing | $\text{lr} \cdot 0.5 \cdot (1 + \cos(\pi t / T))$ | Modern default |
| Warmup + decay | Linear increase then cosine decay | Transformers |
| One-cycle | Increase then decrease over one cycle | Fast convergence |
| Reduce on plateau | Reduce by factor when metric stalls | Safe default |

### Tầm quan trọng của các siêu tham số

Không phải tất cả các siêu tham số đều quan trọng như nhau. Nghiên cứu về rừng ngẫu nhiên (Probst et al., 2019) và tăng độ cho thấy các mô hình nhất quán:

**High importance:**
- Tốc độ học tập (luôn luôn điệu đầu tiên)
- Số lượng ước tính / thời kỳ ( Sử dụng dừng sớm thay vì điều chỉnh)
- Độ mạnh của sự điều chỉnh

**Medium importance:**
- Độ sâu tối đa / số lớp
- Min mẫu mỗi lá / phân hủy trọng lượng
- Tỷ lệ mẫu phụ

**Low importance:**
- Tính năng tối đa (đối với rừng ngẫu nhiên)
- Chọn chức năng kích hoạt cụ thể
- Kích thước lô (trong phạm vi hợp lý)

Đưa ra những thứ quan trọng trước, để lại phần còn lại là mặc định.

### Chiến lược thực tế

```mermaid
flowchart TD
    A[Start with defaults] --> B[Coarse random search: 20-50 trials]
    B --> C[Identify important hyperparameters]
    C --> D[Fine random or Bayesian search: 50-100 trials in narrowed space]
    D --> E[Final model with best hyperparameters]
    E --> F[Retrain on full training data]
```

Phương trình làm việc cụ thể:

1. **Start with library defaults.**Họ được lựa chọn bởi những người có kinh nghiệm và thường là 80% đường đến đó.
2. **Coarse random search.**Phạm vi rộng, thử nghiệm 20-50, sử dụng dừng sớm để tiêu diệt những người xấu chạy nhanh.
3. **Analyze results.**Các siêu tham số nào tương quan với hiệu suất?
4. **Fine search.**Tối ưu hóa Bayesian hoặc tìm kiếm ngẫu nhiên tập trung trong không gian hạn chế. 50-100 thử nghiệm.
5. **Retrain on all training data**với các siêu tham số tốt nhất được tìm thấy.

### Sự tích hợp hợp hợp lệ chéo

Định nghĩa các siêu tham số trên một phân chia xác nhận duy nhất là rủi ro. Các siêu tham số tốt nhất có thể phù hợp với gấp xác nhận cụ thể.

- **Outer loop**(học định): chia dữ liệu thành tàu + giá và thử nghiệm.
- **Inner loop**(tuning): chia train+val thành train và val. Tìm ra các siêu tham số tốt nhất.

```mermaid
flowchart TD
    D[Full Dataset] --> O1[Outer Fold 1: Test]
    D --> O2[Outer Fold 2: Test]
    D --> O3[Outer Fold 3: Test]
    D --> O4[Outer Fold 4: Test]
    D --> O5[Outer Fold 5: Test]

    O1 --> I1[Inner 5-fold CV on remaining data]
    I1 --> T1[Best hyperparams for fold 1]
    T1 --> E1[Evaluate on outer test fold 1]

    O2 --> I2[Inner 5-fold CV on remaining data]
    I2 --> T2[Best hyperparams for fold 2]
    T2 --> E2[Evaluate on outer test fold 2]
```

Mỗi gấp bên ngoài tìm thấy các siêu tham số tốt nhất của riêng mình độc lập.

Với sklearn:

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

inner_cv = GridSearchCV(
    GradientBoostingRegressor(),
    param_grid={
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
        "n_estimators": [50, 100, 200],
    },
    cv=5,
    scoring="neg_mean_squared_error",
)

outer_scores = cross_val_score(
    inner_cv, X, y, cv=5, scoring="neg_mean_squared_error"
)

print(f"Nested CV MSE: {-outer_scores.mean():.4f} +/- {outer_scores.std():.4f}")
```

Điều này đắt tiền (5 gấp bên ngoài x 5 gấp bên trong x 27 điểm lưới = 675 điểm phù hợp với mô hình), nhưng nó cung cấp cho bạn một ước tính hiệu suất đáng tin cậy.

### Những lời khuyên hữu ích

**Start with the learning rate.**Nó luôn là siêu tham số quan trọng nhất cho các phương pháp dựa trên gradient. Tốc độ học tập kém làm cho mọi thứ khác không liên quan. Dũng chỉnh các siêu tham số khác theo mặc định và quét tốc độ học tập trước.

**Use log-uniform distributions for learning rate and regularization.**Sự khác biệt giữa 0.001 và 0.01 quan trọng như sự khác biệt giữa 0.1 và 1.0. Tìm kiếm theo đường thẳng ngân sách lãng phí ở phần lớn.

**Use early stopping instead of tuning n_estimators.**Đối với tăng cường và mạng thần kinh, hãy đặt n_estimators hoặc epochs cao và để dừng sớm quyết định khi dừng lại.

**Budget allocation.**Hãy dành 60% ngân sách điều chỉnh cho 2 siêu tham số quan trọng nhất, dành 40% còn lại cho mọi thứ khác.

**Scale matters.**Không bao giờ tìm kích thước lô trên thang log (16, 32, 64 là tốt). Luôn tìm tốc độ học tập trên thang log. So sánh phân bố tìm kiếm với cách các siêu tham số ảnh hưởng đến mô hình.

| Model Type | Top Hyperparameters | Recommended Search | Budget |
|-----------|--------------------|--------------------|--------|
| Random Forest | n_estimators, max_depth, min_samples_leaf | Random search, 50 trials | Low (fast training) |
| Gradient Boosting | learning_rate, n_estimators, max_depth | Bayesian, 100 trials + early stopping | Medium |
| Neural Network | learning_rate, weight_decay, batch_size | Bayesian or random, 100+ trials | High (slow training) |
| SVM | C, gamma (RBF kernel) | Grid on log scale, 25-50 trials | Low (2 params) |
| Lasso/Ridge | alpha | 1D search on log scale, 20 trials | Very low |
| XGBoost | learning_rate, max_depth, subsample, colsample | Bayesian, 100-200 trials + early stopping | Medium |

**When in doubt:**tìm kiếm ngẫu nhiên với 2 lần số lượng các siêu tham số như thử nghiệm (ví dụ, 6 siêu tham số = 12 thử nghiệm tối thiểu). Bạn sẽ ngạc nhiên khi tìm kiếm ngẫu nhiên với 50 thử nghiệm vượt qua tìm kiếm lưới được thiết kế cẩn thận.

```figure
k-fold-cv
```

## Hãy xây dựng nó

### Bước 1: Tìm kiếm lưới từ đầu

Mã trong `code/tuning.py`thực hiện tìm kiếm lưới, tìm kiếm ngẫu nhiên, và một trình tối ưu hóa Bayesian đơn giản từ đầu.

```python
def grid_search(model_fn, param_grid, X_train, y_train, X_val, y_val):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_score = -float("inf")
    best_params = None
    n_evals = 0

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        model = model_fn(**params)
        model.fit(X_train, y_train)
        score = evaluate(model, X_val, y_val)
        n_evals += 1

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, n_evals
```

### Bước 2: Tìm kiếm ngẫu nhiên từ đầu

```python
def random_search(model_fn, param_distributions, X_train, y_train,
                  X_val, y_val, n_iter=50, seed=42):
    rng = np.random.RandomState(seed)
    best_score = -float("inf")
    best_params = None

    for _ in range(n_iter):
        params = {k: sample(v, rng) for k, v in param_distributions.items()}
        model = model_fn(**params)
        model.fit(X_train, y_train)
        score = evaluate(model, X_val, y_val)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, n_iter
```

### Bước 3: Bayesian Optimization (Thiển giản)

Ý tưởng cốt lõi: phù hợp với một quy trình Gaussian để quan sát (hyperparameter, điểm) cặp, sau đó sử dụng một hàm thu thập để quyết định xem tiếp theo là ở đâu.

```python
class SimpleBayesianOptimizer:
    def __init__(self, search_space, n_initial=5):
        self.search_space = search_space
        self.n_initial = n_initial
        self.X_observed = []
        self.y_observed = []

    def _kernel(self, x1, x2, length_scale=1.0):
        dists = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * dists / length_scale ** 2)

    def _fit_gp(self, X_new):
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        y_mean = y_obs.mean()
        y_centered = y_obs - y_mean

        K = self._kernel(X_obs, X_obs) + 1e-4 * np.eye(len(X_obs))
        K_star = self._kernel(X_new, X_obs)

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
        mu = K_star @ alpha + y_mean

        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-6)

        return mu, var

    def _expected_improvement(self, mu, var, best_y):
        sigma = np.sqrt(var)
        z = (mu - best_y) / (sigma + 1e-10)
        ei = sigma * (z * norm_cdf(z) + norm_pdf(z))
        return ei

    def suggest(self):
        if len(self.X_observed) < self.n_initial:
            return sample_random(self.search_space)

        candidates = [sample_random(self.search_space) for _ in range(500)]
        X_cand = np.array([to_vector(c) for c in candidates])
        mu, var = self._fit_gp(X_cand)
        ei = self._expected_improvement(mu, var, max(self.y_observed))
        return candidates[np.argmax(ei)]

    def observe(self, params, score):
        self.X_observed.append(to_vector(params))
        self.y_observed.append(score)
```

GP thay thế cho hai thứ tại mỗi điểm ứng cử viên: điểm dự đoán (mu) và điểm không chắc chắn (var).

### Bước 4: So sánh tất cả các phương pháp

Thực hiện cả ba phương pháp trên cùng một mục tiêu tổng hợp và so sánh. So sánh này sử dụng một gói đơn giản gọi mỗi tối ưu hóa với một chức năng mục tiêu trực tiếp (không có đào tạo mô hình), do đó API khác với các triển khai dựa trên mô hình ở trên:

```python
def synthetic_objective(params):
    lr = params["learning_rate"]
    depth = params["max_depth"]
    return -(np.log10(lr) + 2) ** 2 - (depth - 4) ** 2 + 10

param_grid = {
    "learning_rate": [0.001, 0.01, 0.1, 1.0],
    "max_depth": [2, 3, 4, 5, 6, 7, 8],
}

grid_best = None
grid_score = -float("inf")
grid_history = []
for combo in itertools.product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), combo))
    score = synthetic_objective(params)
    grid_history.append((params, score))
    if score > grid_score:
        grid_score = score
        grid_best = params

param_dist = {
    "learning_rate": ("log_float", 0.001, 1.0),
    "max_depth": ("int", 2, 8),
}

rand_best = None
rand_score = -float("inf")
rand_history = []
rng = np.random.RandomState(42)
for _ in range(28):
    params = {k: sample(v, rng) for k, v in param_dist.items()}
    score = synthetic_objective(params)
    rand_history.append((params, score))
    if score > rand_score:
        rand_score = score
        rand_best = params

optimizer = SimpleBayesianOptimizer(param_dist, n_initial=5)
bayes_history = []
for _ in range(28):
    params = optimizer.suggest()
    score = synthetic_objective(params)
    optimizer.observe(params, score)
    bayes_history.append((params, score))
bayes_score = max(s for _, s in bayes_history)

print(f"{'Method':<20} {'Best Score':>12} {'Evaluations':>12}")
print("-" * 50)
print(f"{'Grid Search':<20} {grid_score:>12.4f} {len(grid_history):>12}")
print(f"{'Random Search':<20} {rand_score:>12.4f} {len(rand_history):>12}")
print(f"{'Bayesian Opt':<20} {bayes_score:>12.4f} {len(bayes_history):>12}")
```

Với cùng một ngân sách, tối ưu hóa Bayesian thường tìm thấy điểm số tốt nhất nhanh nhất vì nó không lãng phí các đánh giá ở các khu vực rõ ràng xấu. Tìm kiếm ngẫu nhiên bao gồm nhiều địa điểm hơn tìm kiếm lưới. Tìm kiếm lưới chỉ thắng khi bạn có rất ít các siêu tham số và có thể đủ khả năng để đầy đủ.

## Sử dụng nó

### Optuna thực hành

Optuna là thư viện được khuyến cáo cho việc điều chỉnh siêu tham số nghiêm trọng. Nó hỗ trợ cắt, tìm kiếm phân tán và hình ảnh ra khỏi hộp.

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    n_est = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 2, 10)

    model = GradientBoostingRegressor(
        learning_rate=lr,
        n_estimators=n_est,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)
    return mean_squared_error(y_val, model.predict(X_val))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best MSE: {study.best_value:.4f}")
```

Các tính năng chính của Optuna:
- `suggest_float(..., log=True)`cho các tham số được tìm kiếm tốt nhất trên thang log (tốc độ học tập, quy định)
- `suggest_int`cho các tham số nguyên
- `suggest_categorical`cho các lựa chọn riêng biệt
- MedianPruner tích hợp để ngăn chặn sớm các thử nghiệm xấu
- `study.trials_dataframe()`cho phân tích

### Optuna với cắt

Việc cắt cắt ngăn chặn các thử nghiệm không hứa hẹn sớm, tiết kiệm được tính toán lớn.

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("lr", 1e-4, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }

    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3,
                             scoring="neg_mean_squared_error")
    mean_score = -scores.mean()

    trial.report(mean_score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return mean_score

pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=200)
```

- `MedianPruner`dừng một thử nghiệm nếu giá trị trung gian của nó tồi tệ hơn trung bình của tất cả các thử nghiệm hoàn thành ở cùng một bước.`trial.report()`báo cáo các số liệu trung gian và`trial.should_prune()`để kiểm tra xem liệu xét nghiệm có nên dừng lại hay không.`n_startup_trials=10`đảm bảo ít nhất 10 thử nghiệm hoàn thành hoàn toàn trước khi cắt bắt đầu.

### Sklern's Built-in Tuners

Đối với các thí nghiệm nhanh chóng, sklearn cung cấp `GridSearchCV`- `RandomizedSearchCV`, và`HalvingRandomSearchCV`- Có thể là:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    "learning_rate": loguniform(1e-4, 0.5),
    "max_depth": randint(2, 10),
    "n_estimators": randint(50, 500),
}

search = RandomizedSearchCV(
    GradientBoostingRegressor(),
    param_dist,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best CV MSE: {-search.best_score_:.4f}")
```

Sử dụng `loguniform`từ học tập để tăng tốc độ học tập và quy định.`randint`cho các siêu tham số nguyên.`n_jobs=-1`cờ song song trên tất cả các lõi CPU.

### Những sai lầm phổ biến trong việc điều chỉnh siêu tham số

**Data leakage through preprocessing.**Nếu bạn cài đặt một bộ quy mô trên toàn bộ bộ dữ liệu trước khi xác nhận chéo, thông tin từ khoang xác nhận rò rỉ vào đào tạo.`Pipeline`Vì vậy nó chỉ phù hợp với lớp huấn luyện.

**Overfitting to the validation set.**Thực hiện hàng ngàn thử nghiệm hiệu quả tập trung vào bộ xác thực. Sử dụng xác nhận chéo tổ hợp để ước tính hiệu suất cuối cùng, hoặc giữ ra một bộ thử nghiệm riêng biệt mà bạn không bao giờ chạm vào trong thời gian điều chỉnh.

**Searching too narrow a range.**Nếu giá trị tốt nhất của bạn nằm ở giới hạn của không gian tìm kiếm của bạn, bạn đã tìm kiếm không đủ rộng. giá trị tối ưu có thể nằm ngoài phạm vi của bạn. Luôn kiểm tra xem các tham số tốt nhất có ở các cạnh không.

**Ignoring interaction effects.**Tốc độ học tập và số lượng ước tính tương tác mạnh mẽ trong việc tăng cường. Tốc độ học tập thấp cần nhiều ước tính hơn.

**Not using early stopping for iterative models.**Đối với tăng gradient và mạng thần kinh, đặt n_estimators hoặc epochs lên một giá trị cao và sử dụng dừng sớm.

## Các bài tập

1. Thực hiện tìm kiếm lưới và tìm kiếm ngẫu nhiên với cùng một tổng ngân sách (ví dụ, 50 đánh giá). So sánh điểm số tốt nhất được tìm thấy. Thực hiện thí nghiệm 10 lần với hạt giống khác nhau.

2. Thực hiện Hyperband từ đầu. Bắt đầu với 81 cấu hình, mỗi người được đào tạo cho 1 thời đại. Giữ 1/3 trên cùng tại mỗi vòng và gấp ba ngân sách của họ. So sánh tổng tính toán (tổng số tất cả thời đại trên tất cả các cấu hình) với chạy 81 cấu hình cho ngân sách đầy đủ.

3. Thêm một lập trình học tập tốc độ (cousin annealing) vào gradient tăng cường thực hiện từ Bài học 11.

4. Sử dụng Optuna để điều chỉnh một RandomForestClassifier trên một tập dữ liệu thực (ví dụ, tập dữ liệu ung thư vú của sklearn). Sử dụng `optuna.visualization.plot_param_importances(study)`để xem các siêu tham số nào quan trọng nhất. Nó phù hợp với xếp hạng quan trọng từ bài học này?

5. Thực hiện một chức năng thu thập đơn giản (Thiến thiện dự kiến) và chứng minh việc khám phá đối với khai thác.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Hyperparameter | "A setting you choose" | A value set before training that controls the learning process, not learned from data |
| Grid search | "Try every combination" | Exhaustive search over a specified parameter grid. Exponential cost. |
| Random search | "Just sample randomly" | Sample hyperparameters from distributions. Covers important dimensions better than grid search. |
| Bayesian optimization | "Smart search" | Uses a surrogate model of the objective to decide where to evaluate next, balancing exploration and exploitation |
| Surrogate model | "A cheap approximation" | A model (usually Gaussian process) that approximates the expensive objective function from observed evaluations |
| Acquisition function | "Where to look next" | Scores candidate points by balancing expected improvement with uncertainty. EI and UCB are common choices. |
| Early stopping | "Stop wasting time" | Terminate training early when validation performance stops improving |
| Hyperband | "Tournament bracket for configs" | Adaptive resource allocation: start many configs with small budgets, keep the best and increase their budgets |
| Learning rate scheduler | "Change lr during training" | A function that adjusts the learning rate over the course of training for better convergence |

## Đọc thêm

- [Bergstra & Bengio: Random Search for Hyper-Parameter Optimization (2012)](https://jmlr.org/papers/v13/bergstra12a.html)- tờ báo cho thấy lưới đánh ngẫu nhiên
- [Snoek et al., Practical Bayesian Optimization of Machine Learning Algorithms (2012)](https://arxiv.org/abs/1206.2944)-- Optimize Bayesian cho ML
- [Li et al., Hyperband: A Novel Bandit-Based Approach (2018)](https://jmlr.org/papers/v18/16-558.html)- giấy Hyperband
- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902)- tờ Optuna
- [Probst et al., Tunability: Importance of Hyperparameters (2019)](https://jmlr.org/papers/v20/18-444.html)-- những siêu tham số nào quan trọng

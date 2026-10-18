# Việc xử lý dữ liệu không cân bằng

> Khi 99% dữ liệu của bạn là "tự nhiên", độ chính xác là một lời nói dối.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 2, Lessons 01-09 (especially evaluation metrics)
**Time:** ~90 minutes

## Mục tiêu học tập

- Thực hiện SMOTE từ đầu và giải thích cách làm thế nào quá mô hình tổng hợp khác biệt với trùng lặp ngẫu nhiên
- Đánh giá các phân loại không cân bằng bằng bằng cách sử dụng Hàm tương quan F1, AUPRC và Matthews thay vì độ chính xác
- So sánh trọng lượng lớp, điều chỉnh ngưỡng và chiến lược lấy mẫu lại và chọn cách tiếp cận phù hợp cho tỷ lệ mất cân bằng nhất định
- Xây dựng một đường ống dữ liệu không cân bằng hoàn toàn kết hợp SMOTE, trọng lượng lớp và tối ưu hóa ngưỡng

## Vấn đề

Bạn xây dựng một mô hình phát hiện gian lận, nó có độ chính xác 99,9%, bạn ăn mừng, sau đó bạn nhận ra nó dự đoán "không gian lận" cho mỗi giao dịch.

Đây không phải là lỗi. Đó là điều hợp lý để làm khi chỉ có 0,1% giao dịch là gian lận. mô hình học được rằng luôn luôn đoán lớp đa số giảm thiểu lỗi tổng thể. Nó là kỹ thuật chính xác và hoàn toàn vô dụng.

Điều này xảy ra ở mọi nơi quan trọng phân loại thực sự. Chẩn đoán bệnh: tỷ lệ dương tính 1% . Phạm hỏng mạng: 0,01% tấn công. Thiết bị thiếu sót: 0,5% bị lỗi. Trình lọc spam: 20% spam. Dự đoán churn: 5% churners.

Sự chính xác thất bại bởi vì nó đối xử với tất cả các dự đoán chính xác bằng nhau. Định nghĩa đúng một giao dịch hợp pháp và bắt kịp gian lận chính xác đều được coi là một điểm chính xác. Nhưng bắt kịp gian lận là toàn bộ lý do mô hình tồn tại. Chúng ta cần các số liệu, kỹ thuật và chiến lược đào tạo buộc mô hình chú ý đến lớp hiếm nhưng quan trọng.

## Khái niệm

### Tại sao sự chính xác không thành công

Hãy xem xét một tập dữ liệu với 1000 mẫu: 990 âm, 10 dương. Một mô hình luôn dự đoán âm:

|  | Predicted Positive | Predicted Negative |
|--|---|---|
| Actually Positive | 0 (TP) | 10 (FN) |
| Actually Negative | 0 (FP) | 990 (TN) |

```math
\text{Accuracy} = (0 + 990) / 1000 = 99.0\%
```

Mô hình này không bắt được gian lận, không mắc bệnh, không có lỗi, nhưng độ chính xác là 99%.

### Các số liệu tốt hơn

**Precision**= $\text{TP} / (\text{TP} + \text{FP})$Trong số tất cả những gì được đánh dấu là tích cực, có bao nhiêu thực sự?

**Recall**= $\text{TP} / (\text{TP} + \text{FN})$Trong số tất cả những thứ tích cực, chúng ta đã bắt bao nhiêu?

**F1 Score**= $2 \cdot \text{precision} \cdot \text{recall} / (\text{precision} + \text{recall})$- Tỷ lệ trung bình hợp nhất, gây ra sự mất cân bằng giữa độ chính xác và nhớ lại nhiều hơn là trung bình toán học.

**F-beta Score**= $(1 + \beta^2) \cdot \text{precision} \cdot \text{recall} / (\beta^2 \cdot \text{precision} + \text{recall})$Khi nào?$\beta > 1$, nhớ lại quan trọng hơn.$\beta < 1$F2 là phổ biến trong phát hiện gian lận (sự mất gian lận là tồi tệ hơn một báo động sai).

**AUPRC**(Thành vực dưới đường cong nhớ chính xác). Giống như AUC-ROC nhưng thông tin hơn cho dữ liệu không cân bằng. Một phân loại ngẫu nhiên có AUPRC bằng tỷ lệ lớp tích cực (không phải 0,5 như ROC). Điều này làm cho sự cải thiện dễ dàng hơn để thấy.

**Matthews Correlation Coefficient**= $(\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}) / \sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}$. dao động từ -1 đến +1. chỉ đưa ra điểm số cao khi mô hình làm tốt trên cả hai lớp.

Đối với mô hình "luôn dự đoán tiêu cực" trên: $\text{precision} = 0/0$(không xác định, thường được đặt lên 0), $\text{recall} = 0/10 = 0$- $\text{F1} = 0$- $\text{MCC} = 0$Những số liệu này xác định đúng mô hình là vô giá trị.

### Đường ống dữ liệu không cân bằng

```mermaid
flowchart TD
    A[Imbalanced Dataset] --> B{Imbalance Ratio?}
    B -->|Mild: 80/20| C[Class Weights]
    B -->|Moderate: 95/5| D[SMOTE + Threshold Tuning]
    B -->|Severe: 99/1| E[SMOTE + Class Weights + Threshold]
    C --> F[Train Model]
    D --> F
    E --> F
    F --> G[Evaluate with F1 / AUPRC / MCC]
    G --> H{Good Enough?}
    H -->|No| I[Try Different Strategy]
    H -->|Yes| J[Deploy with Monitoring]
    I --> B
```

### SMOTE: Kỹ thuật lấy mẫu quá mức của thiểu số tổng hợp

Việc lấy mẫu quá nhiều ngẫu nhiên làm trùng lặp các mẫu thiểu số hiện có. Điều này hoạt động nhưng có nguy cơ quá phù hợp vì mô hình nhìn thấy các điểm giống nhau nhiều lần.

SMOTE tạo ra các mẫu thiểu số tổng hợp mới có thể tin được nhưng không phải là bản sao.

1. Đối với mỗi mẫu thiểu số $x$, tìm ra nó $k$hàng xóm gần nhất trong số các mẫu thiểu số khác
2. Chọn một hàng xóm ngẫu nhiên
3. Tạo một mẫu mới trên phân đoạn đường giữa $x$Và người hàng xóm đó

Công thức: `new_sample = x + random(0, 1) * (neighbor - x)`

Điều này liên kết giữa các điểm thiểu số thực sự, tạo ra các mẫu trong cùng một khu vực không gian tính năng mà không chỉ sao chép dữ liệu hiện có.

```mermaid
flowchart LR
    subgraph Original["Original Minority Points"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE Generation"]
        direction TB
        S1["Pick x1, neighbor x2"]
        S2["random t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["Augmented Set"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["synthetic (1.2, 2.2)"]
    end
    SMOTE --> Result
```

### Các chiến lược lấy mẫu so sánh

**Random Oversampling**: lấy mẫu thiểu số trùng lặp để phù hợp với số lượng đa số.
- Lợi thế: đơn giản, không mất thông tin
- Khối tác: các bản sao chính xác gây ra quá phù hợp, tăng thời gian đào tạo

**Random Undersampling**: loại bỏ các mẫu đa số để phù hợp với số lượng thiểu số.
- Lợi thế: đào tạo nhanh, đơn giản
- Nhược điểm: loại bỏ dữ liệu đa số có thể hữu ích, sự khác biệt cao hơn

**SMOTE**: tạo mẫu thiểu số tổng hợp thông qua việc phân cực.
- Lợi thế: tạo ra các điểm dữ liệu mới, giảm quá phù hợp so với việc lấy mẫu quá nhiều ngẫu nhiên
- Khối tác: có thể tạo ra các mẫu tiếng ồn gần ranh giới quyết định, không tính đến phân phối lớp đa số

| Strategy | Data Changed | Risk | When to Use |
|----------|-------------|------|-------------|
| Oversample | Minority duplicated | Overfitting | Small datasets, moderate imbalance |
| Undersample | Majority removed | Information loss | Large datasets, want fast training |
| SMOTE | Synthetic minority added | Boundary noise | Moderate imbalance, enough minority samples for k-NN |

### Tín trọng lớp

Thay vì thay đổi dữ liệu, thay đổi cách mô hình xử lý lỗi. Đưa trọng lượng cao hơn cho phân loại sai nhóm thiểu số.

Đối với một vấn đề nhị phân với 950 mẫu âm và 50 mẫu dương:
- trọng lượng cho lớp âm = $n_{\text{samples}} / (2 \cdot n_{\text{negative}}) = 1000 / (2 \cdot 950) = 0.526$
- Tín trọng cho lớp tích cực = $n_{\text{samples}} / (2 \cdot n_{\text{positive}}) = 1000 / (2 \cdot 50) = 10.0$

Nhóm tích cực có trọng lượng 19x. Việc phân loại sai một mẫu tích cực có giá trị tương đương với việc phân loại sai 19 mẫu tiêu cực. Mô hình buộc phải chú ý đến lớp thiểu số.

Trong sự lùi lại hậu cần, điều này thay đổi hàm mất mát:

```math
\text{weighted\_loss} = -\sum \bigl( w_i \cdot [y_i \log(p_i) + (1-y_i) \log(1-p_i)] \bigr)
```

nơi $w_i$tùy thuộc vào lớp mẫu $i$- Tôi không biết.

Các trọng lượng lớp học bằng toán học với việc lấy mẫu quá mức trong kỳ vọng, nhưng không tạo ra các điểm dữ liệu mới. Điều này làm cho chúng nhanh hơn và tránh nguy cơ quá phù hợp của các mẫu trùng lặp.

### Định hướng ngưỡng

Hầu hết các phân loại xuất hiện một xác suất. ngưỡng mặc định là 0,5: nếu $P(\text{positive}) \ge 0.5$Khi các lớp không cân bằng, ngưỡng tối ưu thường thấp hơn nhiều.

Quá trình:
1. Trình mẫu
2. Nhận dự đoán xác suất trên bộ xác thực
3. Các ngưỡng lọc từ 0,0 đến 1,0
4. Xét F1 (hoặc số liệu bạn chọn) tại mỗi ngưỡng
5. Chọn ngưỡng tối đa hóa số lượng của bạn

```mermaid
flowchart LR
    A[Model] --> B[Predict Probabilities]
    B --> C[Sweep Thresholds 0.0 to 1.0]
    C --> D[Compute F1 at Each]
    D --> E[Pick Best Threshold]
    E --> F[Use in Production]
```

Một mô hình có thể phát hành $P(\text{fraud}) = 0.15$với một giao dịch gian lận. ở ngưỡng 0.5, nó được phân loại là không gian lận. ở ngưỡng 0.10, nó được bắt đúng.

### Học hỏi có giá trị

Việc tổng hợp trọng lượng lớp thay vì chi phí đồng nhất, chỉ định chi phí phân loại sai:

| | Predict Positive | Predict Negative |
|--|---|---|
| Actually Positive | 0 (correct) | $C_{\text{FN}} = 100$ |
| Actually Negative | $C_{\text{FP}} = 1$ | 0 (correct) |

Thiếu giao dịch gian lận (FN) tốn kém hơn 100 lần so với báo động sai (FP). mô hình tối ưu hóa cho tổng chi phí, không phải tổng số lỗi.

Đây là cách tiếp cận có nguyên tắc nhất khi bạn có thể ước tính chi phí trong thế giới thực. Một chẩn đoán ung thư bị bỏ qua có chi phí rất khác với một báo động sai dẫn đến một sinh thiết bổ sung. Làm cho chi phí này rõ ràng buộc các sự thỏa hiệp đúng đắn.

### Hình ảnh dòng chảy quyết định

```mermaid
flowchart TD
    A[Start: Imbalanced Dataset] --> B{How imbalanced?}
    B -->|"< 70/30"| C["Mild: try class weights first"]
    B -->|"70/30 to 95/5"| D["Moderate: SMOTE + class weights"]
    B -->|"> 95/5"| E["Severe: combine multiple strategies"]
    C --> F{Enough data?}
    D --> F
    E --> F
    F -->|"< 1000 samples"| G["Oversample or SMOTE, avoid undersampling"]
    F -->|"1000-10000"| H["SMOTE + threshold tuning"]
    F -->|"> 10000"| I["Undersampling OK, or class weights"]
    G --> J[Train + Evaluate with F1/AUPRC]
    H --> J
    I --> J
    J --> K{Recall high enough?}
    K -->|No| L[Lower threshold]
    K -->|Yes| M{Precision acceptable?}
    M -->|No| N[Raise threshold or add features]
    M -->|Yes| O[Ship it]
```

```figure
class-imbalance
```

## Hãy xây dựng nó

### Bước 1: Tạo một bộ dữ liệu không cân bằng

```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```

### Bước 2: SMOTE từ đầu

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```

### Bước 3: Tiêu chuẩn quá và thiếu mẫu ngẫu nhiên

```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```

### Bước 4: Khản hồi hậu cần với trọng lượng lớp

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```

### Bước 5: Định hướng ngưỡng

```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### Bước 6: Các chức năng đánh giá

```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```

### Bước 7: So sánh tất cả các phương pháp tiếp cận

```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Baseline: no treatment
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# Oversampled
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# Class weights
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# Threshold tuning (tune on held-out validation set, not test set)
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```

Các tập tin mã chạy tất cả trong một kịch bản và in kết quả.

## Sử dụng nó

Với việc học scikit và học không cân bằng, những kỹ thuật này là một dòng:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

Các thực hiện từ đầu cho thấy chính xác những gì mỗi kỹ thuật làm. SMOTE chỉ là k-NN phân cực trên lớp thiểu số. lớp trọng lượng nhân sự mất mát. điều chỉnh ngưỡng là một vòng quay trước cắt. Không có phép thuật.

## Chuyển nó

Bài học này mang lại:
- `outputs/skill-imbalanced-data.md`-- một danh sách kiểm tra quyết định để xử lý các vấn đề phân loại không cân bằng

## Các bài tập

1. **Borderline-SMOTE**: sửa đổi thực hiện SMOTE để chỉ tạo mẫu tổng hợp cho các điểm thiểu số gần biên giới quyết định (những điểm k gần nhất của họ bao gồm các mẫu lớp đa số). So sánh kết quả với SMOTE tiêu chuẩn trên một tập dữ liệu nơi các lớp chồng chéo.

2. **Cost matrix optimization**: thực hiện học tập nhạy cảm với chi phí khi các mô hình chi phí là một tham số. Tạo một hàm lấy một mô hình chi phí và trả lại các dự đoán tối ưu để giảm thiểu chi phí dự kiến.

3. **Threshold calibration**: thực hiện quy mô Platt (giải chỉnh một sự lùi hậu cần trên các sản phẩm thô của mô hình để tạo ra xác suất chuẩn bị). So sánh đường cong nhớ chính xác trước và sau khi chuẩn bị.

4. **Ensemble with balanced bagging**: đào tạo nhiều mô hình, mỗi mô hình trên một mẫu bootstrap cân bằng (tất cả thiểu số + phụ tập hợp ngẫu nhiên của đa số).

5. **Imbalance ratio experiment**: lấy một bộ dữ liệu cân bằng và tăng dần tỷ lệ mất cân bằng (50/50, 70/30, 90/10, 95/5, 99/1). Đối với mỗi tỷ lệ, tập với và không có SMOTE.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Class imbalance | "One class has way more samples" | The distribution of classes in the dataset is significantly skewed, causing models to favor the majority class |
| SMOTE | "Synthetic oversampling" | Creates new minority samples by interpolating between existing minority samples and their k-nearest minority neighbors |
| Class weights | "Making errors on rare classes more expensive" | Multiplying the loss function by class-specific weights so the model penalizes minority misclassification more heavily |
| Threshold tuning | "Moving the decision boundary" | Changing the probability cutoff for classification from the default 0.5 to a value that optimizes the desired metric |
| Precision-recall tradeoff | "You cannot have both" | Lowering the threshold catches more positives (higher recall) but also flags more false positives (lower precision), and vice versa |
| AUPRC | "Area under the PR curve" | Summarizes the precision-recall curve into a single number; more informative than AUC-ROC when classes are heavily imbalanced |
| Matthews Correlation Coefficient | "The balanced metric" | A correlation between predicted and actual labels that produces a high score only when the model performs well on both classes |
| Cost-sensitive learning | "Different mistakes cost different amounts" | Incorporating real-world misclassification costs into the training objective so the model optimizes for total cost, not error count |
| Random oversampling | "Duplicate the minority" | Repeating minority class samples to balance class counts; simple but risks overfitting to duplicated points |

## Đọc thêm

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813)-- bài báo SMOTE ban đầu, vẫn là công trình được trích dẫn nhiều nhất về việc học không cân bằng
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907)-- khảo sát toàn diện bao gồm các phương pháp lấy mẫu, chi phí và thuật toán
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)-- Thư viện Python với các biến thể SMOTE, chiến lược lấy mẫu thấp và tích hợp đường ống
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)-- khi nào và tại sao nên thích đường cong PR hơn đường cong ROC cho các vấn đề không cân bằng

# Bayes ngây thơ

> Giả định "tâm nhiên" là sai, và nó vẫn hoạt động. Đó là vẻ đẹp của nó.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## Mục tiêu học tập

- Thực hiện Multinomial Naive Bayes từ đầu với Laplace smoothing cho phân loại văn bản
- Giải thích tại sao giả định độc lập ngây thơ là sai về mặt toán học nhưng tạo ra xếp hạng lớp học chính xác trong thực tế
- So sánh đa số, Bernoulli, và Gaussian Naive Bayes biến thể và chọn đúng cho một loại tính năng nhất định
- Đánh giá Bayes ngây thơ đối với sự lùi lại hậu cần trên dữ liệu hiếm có chiều cao và giải thích sự thỏa hiệp sự thiên vị-các biến tại nơi làm việc

## Vấn đề

Bạn cần phân loại văn bản. Email vào spam hoặc không spam. đánh giá khách hàng vào tích cực hoặc tiêu cực. Vé hỗ trợ vào các loại. Bạn có hàng ngàn tính năng (một từ) và dữ liệu đào tạo hạn chế.

Hầu hết các phân loại viên bị ngạt ở đây. Khản hồi hậu cần đủ mẫu để ước tính hàng ngàn trọng lượng một cách đáng tin cậy. Cây quyết định chia rẽ trên một từ một lần và quá phù hợp. KNN trong 10.000 chiều là vô nghĩa bởi vì mỗi điểm đều cách nhau với mọi điểm khác.

Bayes ngây thơ sẽ xử lý chuyện này. Nó đưa ra một giả định sai về mặt toán học (nếu mỗi tính năng độc lập với mọi tính năng khác do lớp học), và nó vẫn vượt qua các mô hình " thông minh hơn" về phân loại văn bản, đặc biệt là với các tập hợp đào tạo nhỏ. Nó đào tạo trong một lần qua dữ liệu. Nó có thể đạt tới hàng triệu tính năng. Nó tạo ra ước tính xác suất (mặc dù thường được định đo kém do giả định độc lập).

Hiểu tại sao một giả định sai dẫn đến dự đoán tốt dạy bạn một điều cơ bản về học máy: mô hình tốt nhất không phải là đúng nhất, nó là một với sự thỏa hiệp biến thiên tốt nhất cho dữ liệu của bạn.

## Khái niệm

### Lý thuyết Bayes (Quick Review)

Định lý Bayes làm đảo ngược xác suất có điều kiện:

```math
P(\text{class} \mid \text{features}) = \frac{P(\text{features} \mid \text{class}) \cdot P(\text{class})}{P(\text{features})}
```

Chúng tôi muốn`P(class | features)`-- xác suất rằng một tài liệu thuộc về một lớp được đưa ra từ trong đó.
- `P(features | class)`-- khả năng nhìn thấy những từ này trong các tài liệu của lớp này
- `P(class)`-- xác suất trước của lớp (spam phổ biến như thế nào nói chung?)
- `P(features)`-- bằng chứng, giống nhau cho tất cả các lớp, để chúng ta có thể bỏ qua nó khi so sánh

Nhóm có lớp cao nhất`P(class | features)`thắng.

### Sự giả định độc lập ngây thơ

Máy tính `P(features | class)`với một từ vựng 10.000 từ, bạn sẽ cần ước tính phân bố trên$2^{10{,}000}$Không thể.

Hiểu tưởng ngây thơ: mỗi tính năng đều độc lập theo điều kiện khi xem xét lớp học.

```math
P(w_1, w_2, \ldots, w_n \mid \text{class}) = P(w_1 \mid \text{class}) \cdot P(w_2 \mid \text{class}) \cdots P(w_n \mid \text{class})
```

Thay vì một phân phối chung không thể, bạn ước tính$n$Phân bố đơn giản trên tính năng. Mỗi người chỉ cần một con số.

Hiểu định này rõ ràng là sai. Từ "máy" và "làm học" không độc lập trong bất kỳ tài liệu nào. Nhưng trình phân loại không cần ước tính xác suất chính xác. Nó cần xếp hạng chính xác - lớp nào có xác suất cao nhất. Hiểu định độc lập giới thiệu những lỗi hệ thống, nhưng những sai lầm đó ảnh hưởng đến tất cả các lớp tương tự, vì vậy xếp hạng vẫn chính xác.

### Tại sao nó vẫn còn hiệu quả

Ba lý do:

1. **Ranking over calibration.**Việc phân loại chỉ cần lớp xếp hạng hàng đầu để đúng.$P(\text{spam}) = 0.99999$khi xác suất thực là 0,7, phân loại vẫn chọn spam đúng. chúng ta không cần xác suất chính xác. chúng ta cần người chiến thắng chính xác.

2. **High bias, low variance.**Hiểu thuyết độc lập là một giả định mạnh mẽ. Nó hạn chế mô hình rất nhiều, điều này ngăn chặn quá phù hợp. Với dữ liệu đào tạo hạn chế, một mô hình sai một chút nhưng ổn định đánh bại một mô hình lý thuyết đúng nhưng vô cùng không ổn định. Đây là sự đổi giá sự thiên vị trong hành động.

3. **Feature redundancy cancels out.**Các tính năng tương quan cung cấp bằng chứng dư thừa. Các phân loại nhân đôi đếm bằng chứng này, nhưng nó đếm gấp đôi cho lớp đúng cũng như. Nếu "cỗ máy" và "làm học" luôn xuất hiện cùng nhau, cả hai đều cung cấp bằng chứng cho lớp "công nghệ". NB đếm hai lần, nhưng nó đếm hai lần cho lớp đúng.

Lý do thứ tư, thực tế: Bayes ngây thơ là cực kỳ nhanh. Trình luyện là một lần đi qua tần số đếm dữ liệu. Dự đoán là một nhân số tử liệu. Bạn có thể đào tạo trên một triệu tài liệu trong vài giây. Tốc độ này có nghĩa là bạn có thể lặp lại nhanh hơn, thử nhiều bộ tính năng hơn, và chạy nhiều thí nghiệm hơn so với các mô hình chậm hơn.

### Các môn toán từng bước

Hãy bắt đầu bằng một ví dụ cụ thể. giả sử chúng ta có hai lớp: spam và không spam. Từ vựng của chúng ta có ba từ: "bởi không", "tiền", "các cuộc họp".

Dữ liệu đào tạo:
- Email spam đề cập đến "tự do" 80 lần, "tiền" 60 lần, "các cuộc họp" 10 lần (150 từ tổng thể)
- Các email không spam đề cập đến "tự do" 5 lần, "tiền" 10 lần, "các cuộc họp" 100 lần (115 từ tổng thể)
- 40% email là spam, 60% là không spam

Với Laplace làm trơn ($\alpha=1$):

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

Email mới có chứa: "tự do" (2 lần), "tiền" (1 lần), "quá trình" (0 lần).

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

Spam thắng bằng một tỷ lệ lớn. từ "tự do" xuất hiện hai lần là bằng chứng mạnh mẽ cho spam. lưu ý rằng "quá trình" không xuất hiện đóng góp không cho cả hai log tổng ($0 \cdot \log P$(với nhiều chữ số khác nhau, các từ vắng mặt không có hiệu quả.

### Ba biến thể

Bayes ngây thơ có 3 loại.`P(feature | class)`khác nhau.

#### Bayes đa số ngây thơ

Mô hình mỗi tính năng như một con số. Tốt nhất cho dữ liệu văn bản nơi tính năng là tần số từ hoặc giá trị TF-IDF.

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{count of } \text{word}_i \text{ in class} + \alpha}{\text{total words in class} + \alpha \cdot \text{vocab\_size}}
```

- `alpha`là Laplace smoothing (được giải thích dưới đây).

#### Gaussian Naive Bayes

Mô hình mỗi tính năng như một phân phối bình thường.

```math
P(x_i \mid \text{class}) = \frac{1}{\sqrt{2 \pi \cdot \text{var}}} \cdot \exp\left(-\frac{(x_i - \text{mean})^2}{2 \cdot \text{var}}\right)
```

Mỗi lớp có trung bình và sự khác biệt riêng cho mỗi tính năng. Điều này hoạt động tốt khi các tính năng thực sự theo đường cong chuông trong mỗi lớp.

#### Bernoulli ngây thơ Bayes

Mô hình mỗi tính năng như nhị phân (đây hoặc vắng mặt).

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{docs in class containing } \text{word}_i + \alpha}{\text{total docs in class} + 2 \alpha}
```

Không giống như Multinomial, Bernoulli rõ ràng trừng phạt sự vắng mặt của một từ. Nếu "tự do" thường xuất hiện trong thư rác nhưng vắng mặt trong email này, Bernoulli tính rằng đó là bằng chứng chống lại thư rác.

### Khi nào nên sử dụng mỗi biến thể

| Variant | Feature Type | Best For | Example |
|---------|-------------|----------|---------|
| Multinomial | Counts or frequencies | Text classification, bag-of-words | Email spam, topic classification |
| Gaussian | Continuous values | Tabular data with normal-ish features | Iris classification, sensor data |
| Bernoulli | Binary (0/1) | Short text, binary feature vectors | SMS spam, presence/absence features |

### Laplace Smoothing

Điều gì xảy ra khi một từ xuất hiện trong dữ liệu thử nghiệm nhưng không xuất hiện trong dữ liệu đào tạo cho một lớp cụ thể?

Không làm trơn:`P(word | class) = 0/N = 0`Một số 0 nhân qua toàn bộ sản phẩm tạo ra`P(class | features) = 0`Một lời nói không được nhìn thấy sẽ phá hủy toàn bộ dự đoán, bất kể có nhiều bằng chứng nào khác hỗ trợ nó.

Lượt độ Laplace thêm một con số nhỏ `alpha`(thường là 1) cho mỗi tính năng:

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{count}(\text{word}_i, \text{class}) + \alpha}{\text{total\_words\_in\_class} + \alpha \cdot \text{vocab\_size}}
```

Với $\alpha=1$, mỗi từ có ít nhất một xác suất nhỏ. từ "discombobulate" xuất hiện trong một email thử nghiệm không còn giết chết xác suất spam.

Alpha cao hơn có nghĩa là làm trơn hơn (các phân phối thống nhất hơn). alpha thấp hơn có nghĩa là mô hình tin tưởng dữ liệu nhiều hơn. Alpha là một siêu tham số bạn điều chỉnh.

Ảnh hưởng của alpha:

| Alpha | Effect | When to use |
|-------|--------|-------------|
| 0.001 | Almost no smoothing, trust the data | Very large training set, no unseen features expected |
| 0.1 | Light smoothing | Large training set |
| 1.0 | Standard Laplace smoothing | Default starting point |
| 10.0 | Heavy smoothing, flattens distributions | Very small training set, many unseen features expected |

### Lượng tính toán log-space

Bội số hàng trăm xác suất (mỗi số ít hơn 1) gây ra dòng chảy thấp ở điểm nổi.

Giải pháp: làm việc trong không gian log. Thay vì nhân xác suất, hãy thêm các logarithm của chúng:

```math
\log P(\text{class} \mid x_1, x_2, \ldots, x_n) = \log P(\text{class}) + \sum_i \log P(x_i \mid \text{class})
```

Điều này biến dự đoán thành một sản phẩm điểm:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

Đó là lý do tại sao dự đoán của Bayes ngây thơ rất nhanh -- đó là hoạt động tương tự như mô hình tuyến tính một lớp.

### Bayes ngây thơ vs Lịch lý

Cả hai đều là phân loại đường thẳng cho văn bản. Sự khác biệt là trong những gì chúng mô hình.

| Aspect | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| Type | Generative (models $P(X \mid Y){}$) | Discriminative (models $P(Y \mid X){}$) |
| Training | Count frequencies | Optimize loss function |
| Small data | Better (strong prior helps) | Worse (not enough to estimate weights) |
| Large data | Worse (wrong assumption hurts) | Better (flexible boundary) |
| Features | Assumes independence | Handles correlations |
| Speed | Single pass, very fast | Iterative optimization |
| Calibration | Poor probabilities | Better probabilities |

Quy tắc: bắt đầu với Bayes ngây thơ. Nếu bạn có đủ dữ liệu và cao nguyên NB, chuyển sang sự lùi hậu cần.

### Đường ống phân loại

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

Thực tế, chúng ta làm việc trong không gian log để tránh dòng chảy dưới điểm nổi. thay vì nhân nhiều xác suất nhỏ, chúng ta thêm các logarithm của chúng:

```math
\log P(\text{class} \mid \text{features}) = \log P(\text{class}) + \sum_i \log P(\text{feature}_i \mid \text{class})
```

```figure
naive-bayes
```

## Hãy xây dựng nó

Mã trong `code/naive_bayes.py`thực hiện cả MultinomialNB và GaussianNB từ đầu.

### MultinomialNB

Việc thực hiện từ đầu:

1. **fit(X, y)**: Đối với mỗi lớp, đếm tần số của mỗi tính năng. Thêm thanh toán Laplace. Xét xác suất log. Cấp trước lớp (log tần số lớp).

2. **predict_log_proba(X)**: Đối với mỗi mẫu, tính toán $\log P(\text{class})$+ tổng của $\log P(\text{feature}_i \mid \text{class})$cho tất cả các lớp. Đây là một số nhân tử liệu: X @ log_probs.T + log_priors.

3. **predict(X)**: Trở lại lớp có xác suất log cao nhất.

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

Điều quan trọng: sau khi được gắn, dự đoán chỉ là nhân tử liệu cộng với một thiên vị.

### GaussianNB

Đối với các tính năng liên tục, chúng tôi ước tính trung bình và sự biến động cho mỗi lớp cho mỗi tính năng:

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

Dự đoán sử dụng Gaussian PDF cho mỗi tính năng, nhân trên các tính năng (được thêm vào không gian log).

### Demo: Định dạng văn bản

Mã tạo ra dữ liệu túi từ tổng hợp mô phỏng hai lớp (những bài viết công nghệ so với các bài viết thể thao). Mỗi lớp có phân bố tần số từ khác nhau. MultinomialNB phân loại chúng bằng cách sử dụng số lượng từ.

Các dữ liệu tổng hợp hoạt động như sau: chúng tôi tạo ra 200 "lời" (cột tính năng). Từ 0-39 có tần suất cao trong các bài viết công nghệ và thấp trong thể thao. Từ 80-119 có tần suất cao trong thể thao và thấp trong công nghệ. Từ 40-79 là tần suất trung bình trong cả hai. Điều này tạo ra một kịch bản thực tế nơi một số từ là chỉ số hạng mạnh và những từ khác là tiếng ồn.

### Demo: Các tính năng liên tục

Mã tạo ra dữ liệu giống như Iris (3 lớp, 4 tính năng, cụm Gaussian). GaussianNB phân loại bằng cách sử dụng trung bình và biến thể cho mỗi lớp. Mỗi lớp có trung tâm khác nhau (vêctơ trung bình) và sự lan rộng khác nhau (hác biến), bắt chước dữ liệu trong thế giới thực nơi các phép đo khác nhau theo hệ thống giữa các loại.

Bộ luật cũng chứng minh:
- **Smoothing comparison:**Đào tạo MultinomialNB với các giá trị alpha khác nhau để thể hiện tác động của sức mạnh làm trơn trên độ chính xác.
- **Training size experiment:**Làm thế nào độ chính xác NB cải thiện khi dữ liệu đào tạo tăng từ 20 đến 1600 mẫu. NB đạt được độ chính xác tốt ngay cả với rất ít mẫu - đây là lợi thế chính của nó.
- **Confusion matrix:**Độ chính xác, nhớ lại và điểm số F1 cho thấy NB mắc sai lầm.

### Tốc độ dự đoán

Dự đoán Bayes ngây thơ là một số nhân tử liệu.$n$mẫu với $d$đặc điểm và$k$Các lớp:
- MultinomialNB: một matrix nhân $(n \times d) \mathbin{@} (d \times k) = O(n \cdot d \cdot k)$
- GaussianNB: $n \cdot k$Các đánh giá Gaussian PDF, mỗi lần $d$đặc điểm = $O(n \cdot d \cdot k)$

Cả hai đều tuyến tính trong mọi chiều kích. So sánh với KNN (cần tính khoảng cách đến tất cả các điểm đào tạo) hoặc SVM với lõi RBF (cần đánh giá lõi đối với tất cả các vector hỗ trợ). NB nhanh hơn theo thứ tự quy mô tại thời gian dự đoán.

## Sử dụng nó

Với sklearn, cả hai biến thể là một dòng:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

Đối với phân loại văn bản với sklearn:

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

Mã trong `naive_bayes.py`so sánh các triển khai từ đầu với sklearn trên cùng một dữ liệu để xác minh sự chính xác.

### TF-IDF với Naive Bayes

Số từ thô cho mỗi từ trọng lượng bằng nhau cho mỗi sự kiện. Nhưng các từ phổ biến như "the" và "is" xuất hiện thường xuyên trong mỗi lớp - chúng không mang thông tin. TF-IDF (Term Frequency - Inverse Document Frequency) giảm trọng lượng từ phổ biến và tăng trọng lượng từ hiếm, phân biệt đối xử.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

Các giá trị TF-IDF không âm tính, vì vậy chúng hoạt động với MultinomialNB. Sự kết hợp của TF-IDF + MultinomialNB là một trong những đường cơ sở mạnh nhất cho phân loại văn bản. Nó thường đánh bại các mô hình phức tạp hơn trên tập dữ liệu với ít hơn 10.000 mẫu đào tạo.

### BernoulliNB cho văn bản ngắn

Đối với văn bản ngắn (tweet, SMS, tin nhắn trò chuyện), BernoulliNB có thể vượt qua MultinomialNB. Các văn bản ngắn có số lượng từ thấp, vì vậy thông tin tần số mà MultinomialNB dựa vào là tiếng ồn. BernoulliNB chỉ quan tâm đến sự hiện diện hoặc vắng mặt, điều này đáng tin cậy hơn với văn bản ngắn.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

- `binary=True`cờ trong CountVectorizer chuyển đổi tất cả các con số thành 0/1. Không có nó, BernoulliNB vẫn hoạt động nhưng đang nhìn thấy con số mà nó không được thiết kế cho.

### Định so sánh NB Khả năng

NB xác suất là không được chuẩn bị tốt.$P(\text{spam}) = 0.95$Nếu bạn cần ước tính xác suất đáng tin cậy (ví dụ, để đặt ngưỡng hoặc kết hợp với các mô hình khác), sử dụng Skelarn's CalibratedClassifierCV:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

Điều này phù hợp với một sự lùi hậu cần trên đỉnh điểm thô của NB bằng cách sử dụng xác thực chéo.

### Gotcha thông thường

1. **Negative feature values.**MultinomialNB yêu cầu các tính năng không âm. Nếu bạn có các giá trị tiêu cực (như TF-IDF với một số cài đặt hoặc các tính năng tiêu chuẩn hóa), hãy sử dụng GaussianNB thay vào đó, hoặc chuyển các tính năng để tích cực.

2. **Zero variance features.**GaussianNB chia theo sự biến động. Nếu một tính năng có sự biến động không cho một lớp (tất cả các giá trị đều giống nhau), tính toán xác suất bị phá vỡ. Mã thêm một thuật ngữ thanh thản nhỏ (1e-9) vào tất cả các biến động để ngăn chặn điều này.

3. **Class imbalance.**Nếu 99% email không phải là spam, trước đó $P(\text{not-spam}) = 0.99$là mạnh đến nỗi nó làm cho chứng cứ khả năng tràn ngập. Bạn có thể đặt class prior bằng tay hoặc sử dụng class_prior tham số trong sklearn.

4. **Feature scaling.**MultinomialNB không cần quy mô (nó hoạt động trên đếm). GaussianNB cũng không cần quy mô (nó ước tính thống kê tính năng).

## Chuyển nó

Bài học này mang lại:
- `outputs/skill-naive-bayes-chooser.md`-- một kỹ năng quyết định để chọn đúng biến thể NB
- `code/naive_bayes.py`-- MultinomialNB và GaussianNB từ đầu, với so sánh sklearn

### Khi Bayes ngây thơ thất bại

NB thất bại khi giả định độc lập gây ra xếp hạng không chính xác (không chỉ có xác suất không chính xác).

1. **Strong feature interactions.**Nếu lớp phụ thuộc vào sự kết hợp của hai tính năng nhưng không phải là một mình (chương trình giống như XOR), NB sẽ bỏ qua nó hoàn toàn.

2. **Highly correlated features with opposing evidence.**Nếu tính năng A nói "spam" và tính năng B nói "không spam", nhưng A và B tương quan hoàn hảo (bạn luôn đồng ý trong thực tế), NB sẽ thấy bằng chứng mâu thuẫn nơi không có.

3. **Very large training sets.**Với đủ dữ liệu, các mô hình phân biệt đối xử như sự lùi hậu cần học được ranh giới quyết định thực sự và vượt qua NB. Giả định độc lập đã giúp với dữ liệu nhỏ bây giờ giữ lại mô hình.

Trong thực tế, các chế độ thất bại này hiếm khi xảy ra cho phân loại văn bản. Các tính năng văn bản rất nhiều, yếu kém riêng lẻ, và các lỗi của giả định độc lập có xu hướng bị hủy bỏ. Đối với dữ liệu bảng với một số tính năng tương quan mạnh mẽ, hãy xem xét sự lùi lại hậu cần hoặc mô hình dựa trên cây trước.

## Các bài tập

1. **Smoothing experiment.**Đào tạo MultinomialNB trên dữ liệu văn bản với giá trị alpha là 0.01, 0.1, 1.0, 10.0 và 100.0. Độ chính xác của bản vẽ so với alpha.

2. **Feature independence test.**Hãy lấy một tập dữ liệu văn bản thực sự. Chọn hai từ rõ ràng liên quan ("cỗ máy" và "làm học").$P(\text{word}_1 \mid \text{class}) \cdot P(\text{word}_2 \mid \text{class})$và so sánh với $P(\text{word}_1 \text{ AND } \text{word}_2 \mid \text{class})$Thuyết độc lập có sai đến mức nào?

3. **Bernoulli implementation.**Lớn thêm mã bằng lớp BernoulliNB. Chuyển đổi túi từ thành nhị phân (đây/không có) và so sánh độ chính xác so với MultinomialNB trên dữ liệu văn bản.

4. **NB vs Logistic Regression.**Căn luyện cả hai trên dữ liệu văn bản. Bắt đầu với 100 mẫu đào tạo và tăng lên 10.000. Độ chính xác của cốt truyện so với kích thước tập hợp đào tạo cho cả hai.

5. **Spam filter.**Xây dựng một phân loại spam hoàn chỉnh: mã hóa văn bản email nguyên liệu, xây dựng từ vựng, tạo các tính năng túi từ, đào tạo MultinomialNB, đánh giá với độ chính xác và nhớ lại (không chỉ chính xác - tại sao?).

## Các điều khoản chính

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

## Đọc thêm

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html)-- tất cả ba biến thể với chi tiết toán học
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)-- so sánh cổ điển của Multinomial vs Bernoulli cho văn bản
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)-- cải tiến NB cho văn bản
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)-- chứng minh NB hội tụ nhanh hơn LR với ít dữ liệu

# Quá trình Stochastic

> Sự ngẫu nhiên với cấu trúc, toán học đằng sau những bước đi ngẫu nhiên, chuỗi Markov và mô hình phân tán.

**Type:** Learn
**Language:**Python
**Prerequisites:** Phase 1, Lessons 06-07 (probability, Bayes)
**Time:** ~75 minutes

## Mục tiêu học tập

- Chụp đi bộ ngẫu nhiên 1D và 2D và xác minh $\sqrt{n}$quy mô của sự di chuyển
- Xây dựng một mô phỏng chuỗi Markov và tính toán phân bố tĩnh của nó thông qua cấu trúc riêng
- Thực hiện Metropolis-Hastings MCMC và động lực Langevin để lấy mẫu từ các phân phối mục tiêu
- Kết nối quá trình phân tán về phía trước với chuyển động Brownian và giải thích cách quá trình ngược tạo ra dữ liệu

## Vấn đề

Nhiều hệ thống AI liên quan đến sự ngẫu nhiên mà phát triển theo thời gian, không phải sự ngẫu nhiên tĩnh -- sự ngẫu nhiên cấu trúc, theo dõi nơi mỗi bước phụ thuộc vào những gì đã xảy ra trước đó.

Các mô hình ngôn ngữ tạo ra các token một lần. Mỗi token phụ thuộc vào bối cảnh trước đó. mô hình đưa ra phân phối xác suất, lấy mẫu từ nó, và di chuyển về phía trước. Đó là một quá trình stochastic.

Các mô hình phân phối thêm tiếng ồn vào một hình ảnh từng bước cho đến khi nó trở thành tĩnh hoàn toàn. Sau đó chúng đảo ngược quá trình, từ chối từng bước cho đến khi một hình ảnh mới xuất hiện.

Các đại lý học tập tăng cường thực hiện các hành động trong một môi trường. Mỗi hành động dẫn đến một trạng thái mới với một số xác suất. Đại lý theo một chính sách ngẫu nhiên trong một thế giới ngẫu nhiên.

Việc lấy mẫu MCMC -- xương sống của suy luận Bayesian -- xây dựng một chuỗi Markov mà phân bố tĩnh là phía sau bạn muốn lấy mẫu từ.

Tất cả đều dựa trên bốn ý tưởng cơ bản:
1. Đi bộ ngẫu nhiên - quá trình stochastic đơn giản nhất
2. Dòng chuỗi Markov -- sự ngẫu nhiên được cấu trúc với một matrix chuyển tiếp
3. Động lực Langevin - giảm gradient với tiếng ồn
4. Metropolis-Hastings - lấy mẫu từ bất kỳ phân phối nào

## Khái niệm

### Đi bộ ngẫu nhiên

Bắt đầu từ vị trí 0. Ở mỗi bước, ném một đồng xu công bằng. Đầu: di chuyển sang bên phải (+1). đuôi: di chuyển sang trái (-1).

Sau đó$n$bước, vị trí của bạn là tổng của $n$        $\pm 1$giá trị. vị trí dự kiến là 0 (lễ đi bộ không thiên vị).$\sqrt{n}$- Tôi không biết.

Điều này trái với trực giác. Đi bộ là công bằng - không có trôi dạt theo cả hai hướng. Nhưng theo thời gian, nó đi lang thang xa hơn và xa hơn từ nơi nó bắt đầu. Phản độ tiêu chuẩn sau$n$bước là $\sqrt{n}$- Tôi không biết.

```
Step 0:  Position = 0
Step 1:  Position = +1 or -1
Step 2:  Position = +2, 0, or -2
...
Step 100: Expected distance from origin ~ 10 (sqrt(100))
Step 10000: Expected distance from origin ~ 100 (sqrt(10000))
```

**In 2D**, bước đi di chuyển lên, xuống, trái, hoặc phải với xác suất tương tự.$\sqrt{n}$Scaling áp dụng cho khoảng cách từ nguồn.

**Why $\sqrt{n}$?**Mỗi bước là +1 hoặc -1 với xác suất tương đương.$n$bước, vị trí $S_n = X_1 + X_2 + \cdots + X_n$nơi mỗi $X_i$là $\pm 1$. Sự khác biệt của mỗi bước là 1, và các bước là độc lập, vì vậy $\mathop{\mathrm{Var}}(S_n) = n$. Phác lệch chuẩn = $\sqrt{n}$Theo định lý giới hạn trung tâm,$S_n / \sqrt{n}$hội tụ với phân bố bình thường tiêu chuẩn.

Đây là$\sqrt{n}$quy mô hiển thị ở khắp mọi nơi trong ML. SGD noise scale là 1/sqrt(batch_size).$\sqrt{d}$. Quả gốc vuông là chữ ký của các sự cộng ngẫu nhiên độc lập.

**Connection to Brownian motion.**Đi bộ ngẫu nhiên với kích thước bước $1/\sqrt{n}$và $n$bước mỗi đơn vị thời gian.$n$đi đến vô tận, bước đi hội tụ với chuyển động Brownian$B(t)$-- một quá trình liên tục trong đó $B(t)$thường được phân phối với trung bình 0 và biến số $t$- Tôi không biết.

Hành động Brown là nền tảng toán học của sự phân tán. Nó mô hình hóa sự rung chuyển ngẫu nhiên của các hạt trong một chất lỏng, biến động của giá cổ phiếu, và - quan trọng hơn - quá trình tiếng ồn trong các mô hình phân tán.

**Gambler's ruin.**Một người đi bộ ngẫu nhiên bắt đầu từ vị trí $k$, với các rào cản hấp thụ ở 0 và $N$- Có khả năng đạt được gì?$N$Trước 0?$P(\text{reach } N) = k/N$Nó kết nối với lý thuyết của martingales -- một bước đi ngẫu nhiên là martingale (tín giá trị tương lai = giá trị hiện tại).

### Dòng dây chuyền Markov

Một chuỗi Markov là một hệ thống chuyển đổi giữa các trạng thái theo xác suất cố định.

```math
P(X_{t+1} = j \mid X_t = i, X_{t-1} = \ldots) = P(X_{t+1} = j \mid X_t = i)
```

Đây là tính năng Markov. nó có nghĩa là bạn có thể mô tả toàn bộ động lực bằng một số liệu chuyển tiếp$P$- Có thể là:

```math
P[i][j] = \text{probability of going from state } i \text{ to state } j
```

Mỗi hàng của $P$số tiền lên đến 1 (bạn phải đi đâu đó).

**Example -- Weather:**

```
States: Sunny (0), Rainy (1), Cloudy (2)

P = [[0.7, 0.1, 0.2],    (if sunny: 70% sunny, 10% rainy, 20% cloudy)
     [0.3, 0.4, 0.3],    (if rainy: 30% sunny, 40% rainy, 30% cloudy)
     [0.4, 0.2, 0.4]]    (if cloudy: 40% sunny, 20% rainy, 40% cloudy)
```

Bắt đầu từ bất kỳ trạng thái nào. Sau nhiều chuyển đổi, phân phối các trạng thái hội tụ với phân phối tĩnh.$\pi$, nơi $\pi P = \pi$Đây là phương tiện riêng bên trái của$P$với giá trị chính 1.

Đối với chuỗi thời tiết, phân phối tĩnh là [0,55, 0,18, 0,27] -- trong thời gian dài, nó có mặt trời 55% thời gian bất kể trạng thái khởi đầu.

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

**Computing the stationary distribution.**Có hai cách tiếp cận:

1. **Power method**: nhân bất kỳ phân phối ban đầu nào bằng $P$Sau nhiều lần lặp lại, nó hội tụ lại.
2. **Eigenvalue method**: tìm ra phương tiện tự bên trái của $P$với giá trị bản thân 1. Đây là vector bản thân của $P^T$với giá trị chính 1.

Cả hai cách tiếp cận đều đòi hỏi chuỗi đáp ứng các điều kiện hội tụ.

**Convergence conditions.**Một chuỗi Markov hội tụ với một phân phối tĩnh độc đáo nếu:
- **Irreducible**: mọi bang đều có thể tiếp cận từ mọi bang khác
- **Aperiodic**: chuỗi không có chu kỳ với thời gian cố định

Hầu hết các chuỗi mà bạn gặp trong ML đáp ứng cả hai điều kiện.

**Absorbing states.**Một trạng thái hấp thụ nếu một khi bạn vào đó, bạn không bao giờ rời khỏi ($P[i][i] = 1$Trải theo các quy trình mô hình chuỗi Markov với các trạng thái cuối cùng - một trò chơi kết thúc, một khách hàng đang lắc lắc, một chuỗi mã thông báo tấn công vào mã thông báo cuối văn bản.

**Mixing time.**Số bước cho đến khi chuỗi "gần" với phân phối tĩnh?$P$(1 trừ giá trị tự lớn thứ hai) điều khiển thời gian trộn.

### Kết nối với các mô hình ngôn ngữ

Tạo token trong một mô hình ngôn ngữ là một quá trình Markov. Với bối cảnh hiện tại, mô hình sẽ phát ra phân phối trên token tiếp theo. Nhiệt độ kiểm soát độ sắc:

```math
P(\text{token}_i) = \frac{\exp(\text{logit}_i / \text{temperature})}{\sum_j \exp(\text{logit}_j / \text{temperature})}
```

- Nhiệt độ = 1,0: phân phối tiêu chuẩn
- Nhiệt độ < 1,0: sắc hơn (đối đa xác định)
- Nhiệt độ > 1,0: phẳng hơn (thường tình cờ hơn)
- Nhiệt độ $\to 0$: argmax (cười tham)

Top-k lấy mẫu truncates đến $k$token có xác suất cao nhất. top-p (nucleus) lấy mẫu cắt giảm thành tập hợp nhỏ nhất của token có xác suất tích lũy vượt quá $p$Cả hai đều thay đổi xác suất chuyển đổi Markov.

### Động thái Brown

Giới hạn thời gian liên tục của bước ngẫu nhiên.$B(t)$có ba tính chất:
1. $B(0) = 0$
2. $B(t) - B(s)$thường được phân phối với trung bình 0 và biến số $t - s$(để $t > s$(văn)
3. Các sự gia tăng trên các khoảng không chồng chéo là độc lập

Phong trào Brown liên tục nhưng không thể phân biệt được ở đâu cả - nó rung động ở mọi thang.

Trong mô phỏng riêng biệt, bạn ước tính chuyển động của Browni bằng:

```math
B(t + dt) = B(t) + \sqrt{dt} \, z, \quad \text{where } z \sim \mathcal{N}(0, 1)
```

- $\sqrt{dt}$Scaling là điều quan trọng. Nó xuất phát từ định lý giới hạn trung tâm được áp dụng cho các bước ngẫu nhiên.

### Langevin Dynamics

Sự giảm dần tìm thấy tối thiểu của một hàm.$\exp(-U(x)/T)$, nơi $U$là một chức năng năng và$T$là nhiệt độ.

```math
x_{t+1} = x_t - dt \, \nabla U(x_t) + \sqrt{2 T \, dt} \, z_t
```

Hai lực tác động lên hạt:
1. **Gradient force**($-dt \cdot \nabla U$): đẩy về hướng năng lượng thấp (như giảm gradient)
2. **Random force**($\sqrt{2T dt} \cdot z$): đẩy theo hướng ngẫu nhiên (khám phá)

Ở nhiệt độ $T = 0$Khi nhiệt độ cao, nó gần như là một bước đi ngẫu nhiên.

**Connection to diffusion models.**Quá trình tiến bộ của mô hình phân tán là:

```math
x_t = \sqrt{\alpha_t} \, x_{t-1} + \sqrt{1 - \alpha_t} \cdot \text{noise}
```

Đây là chuỗi Markov liên tục trộn dữ liệu với tiếng ồn.$x_T$là tiếng ồn Gaussian thuần túy.

Quá trình ngược -- từ tiếng ồn trở lại dữ liệu -- cũng là chuỗi Markov, nhưng xác suất chuyển đổi của nó được tìm hiểu bởi một mạng lưới thần kinh. mạng học cách dự đoán tiếng ồn được thêm vào mỗi bước, sau đó trừ nó.

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

### MCMC: Markov Chain Monte Carlo

Đôi khi bạn cần lấy mẫu từ một phân phối$p(x)$Bayesian là ví dụ điển hình -- bạn biết xác suất gấp đôi trước, nhưng các định hình bình thường hóa là khó xử lý.

**Metropolis-Hastings**xây dựng một chuỗi Markov mà phân phối tĩnh là$p(x)$- Có thể là:

1. Bắt đầu từ một vị trí nào đó $x$
2. đề xuất vị trí mới $x'$từ một phân phối đề xuất $Q(x' \mid x)$
3. Tỷ lệ chấp nhận tính toán: $a = p(x') Q(x \mid x') / (p(x) Q(x' \mid x))$
4. Hãy chấp nhận$x'$với khả năng$\min(1, a)$Nếu không thì ở lại đây.$x$- Tôi không biết.
5. Lặp lại.

Nếu$Q$là đối xứng (ví dụ: $Q(x' \mid x) = Q(x \mid x') = \mathcal{N}(x, \sigma^2){}$), tỷ lệ này đơn giản hóa thành $a = p(x') / p(x)$Bạn chỉ cần tỷ lệ xác suất -- các biến cố bình thường hóa hủy bỏ.

Dòng dây chuyền được đảm bảo sẽ hội tụ đến $p(x)$Nhưng sự hội tụ có thể chậm nếu đề xuất quá nhỏ (quay ngẫu nhiên) hoặc quá lớn (đánh chối cao).

**Why it works.**Tỷ lệ chấp nhận đảm bảo sự cân bằng chi tiết: khả năng ở $x$và chuyển đến $x'$bằng với xác suất ở $x'$và chuyển đến $x$Sự cân bằng chi tiết cho thấy rằng$p(x)$là phân bố tĩnh của chuỗi.$p(x)$- Tôi không biết.

**Practical considerations:**
- **Burn-in**: loại bỏ thứ nhất $N$chuỗi cần thời gian để đạt được phân phối tĩnh từ điểm khởi đầu của nó.
- **Thinning**: giữ tất cả $k$- mẫu thứ hai để giảm sự tương quan tự động.
- **Multiple chains**: chạy nhiều chuỗi từ các điểm khởi đầu khác nhau. Nếu chúng hội tụ đến cùng một phân bố, bạn có bằng chứng về hội tụ.
- **Acceptance rate**: cho các đề xuất của Gaussian trong $d$Các quy mô của chuỗi, tỷ lệ chấp nhận tối ưu là khoảng 23% (Roberts & Rosenthal, 2001). quá cao có nghĩa là chuỗi hầu như không di chuyển. quá thấp có nghĩa là nó từ chối mọi thứ.

### Quá trình Stochastic trong AI

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

## Hãy xây dựng nó

### Bước 1: Máy mô phỏng đi bộ ngẫu nhiên

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

1D walk lưu trữ tổng cộng. mỗi bước là +1 hoặc -1. sau đó $n$bước, vị trí là tổng.$n$, vì vậy sự lệch tiêu chuẩn tăng lên như $\sqrt{n}$- Tôi không biết.

### Bước 2: Dòng dây chuyền Markov

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

Phân bố tĩnh là vector tự bên trái của $P$với giá trị bản thân 1. Chúng ta tìm ra nó bằng cách tính toán các vector bản thân của $P^T$(trả chuyển biến các vector tự bên trái thành các vector tự bên phải).

### Bước 3: Động lực Langevin

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

Tốc độ đẩy $x$Phong trào của các loại hình này được chứng minh bởi sự phân bố của các mẫu tương ứng với các loại hình khác nhau.$\exp(-U(x)/\text{temperature})$- Tôi không biết.

### Bước 4: Metropolis-Hastings

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

Các thuật toán đề xuất một điểm mới, kiểm tra xem nó có xác suất cao hơn (hoặc chấp nhận với xác suất tương xứng với tỷ lệ), và lặp lại. Tỷ lệ chấp nhận nên khoảng 23-50% cho sự trộn tốt.

## Sử dụng nó

Thực tế, bạn sử dụng thư viện đã được thiết lập cho các thuật toán này. Nhưng hiểu cơ học là quan trọng để gỡ lỗi và điều chỉnh.

```python
import numpy as np

rng = np.random.RandomState(42)
walk = np.cumsum(rng.choice([-1, 1], size=10000))
print(f"Final position: {walk[-1]}")
print(f"Expected distance: {np.sqrt(10000):.1f}")
print(f"Actual distance: {abs(walk[-1])}")
```

### Numpy cho các matrices chuyển tiếp

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

Lặp gấp số phân phối ban đầu bằng $P$Sau đủ lần lặp lại, nó hội tụ với phân bố tĩnh bất kể bạn bắt đầu ở đâu. Đây là phương pháp năng lượng để tìm ra các cánh tay trái chủ đạo.

### Kết nối với các khung thực tế

- **PyTorch diffusion:**- `DDPMScheduler`Trong khuôn mặt ôm `diffusers`thực hiện các chuỗi Markov phía trước và ngược
- **NumPyro / PyMC:**Sử dụng MCMC (NUTS samplingler, cải thiện trên Metropolis-Hastings) cho suy luận Bayesian
- **Gymnasium (RL):**Chức năng bước môi trường xác định quá trình quyết định Markov

### Kiểm tra sự hội tụ chuỗi Markov

```python
import numpy as np

P = np.array([[0.9, 0.1], [0.3, 0.7]])

eigenvalues = np.linalg.eigvals(P)
spectral_gap = 1 - sorted(np.abs(eigenvalues))[-2]
print(f"Eigenvalues: {eigenvalues}")
print(f"Spectral gap: {spectral_gap:.4f}")
print(f"Approximate mixing time: {1/spectral_gap:.1f} steps")
```

Khoảng cách quang phổ cho bạn biết chuỗi quên đi trạng thái ban đầu nhanh như thế nào. Khoảng cách 0,2 nghĩa là khoảng 5 bước để trộn. Khoảng cách 0,01 nghĩa là khoảng 100 bước. Luôn kiểm tra điều này trước khi chạy mô phỏng dài - một chuỗi trộn chậm tính toán chất thải.

## Chuyển nó

Bài học này mang lại:
- `outputs/prompt-stochastic-process-advisor.md`-- một lời nhắc giúp xác định khung quy trình stochastic áp dụng cho một vấn đề nhất định

## Kết nối

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

Các mô hình phân tán xứng đáng được chú ý đặc biệt. DDPM (Ho et al., 2020) xác định chuỗi Markov phía trước:

```math
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I\right)
```

nơi $\beta_t$là một lịch trình tiếng ồn.$T$bước,$x_T$là khoảng $\mathcal{N}(0, I)$Quá trình ngược được định đo bởi một mạng thần kinh dự đoán tiếng ồn:

```math
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I\right)
```

Mỗi bước của thế hệ là một bước trong một chuỗi Markov học.

SGLD (Stochastic Gradient Langevin Dynamics) kết hợp giảm gradient mini-batch với tiếng ồn Langevin. Thay vì tính toán độ tốc đầy đủ, bạn sử dụng ước tính stochastic và thêm âm thanh chuẩn. Khi tốc độ học tập giảm, SGLD chuyển từ tối ưu hóa sang lấy mẫu -- bạn có được mẫu sau Bayesian gần như miễn phí. Đây là một trong những cách đơn giản nhất để có được ước tính không chắc chắn từ một mạng lưới thần kinh.

Ý tưởng quan trọng trong tất cả các mối liên hệ này: các quy trình stochastic không chỉ là các công cụ lý thuyết. Chúng là cơ chế tính toán bên trong các hệ thống AI hiện đại. Khi bạn điều chỉnh nhiệt độ của một LLM, bạn đang điều chỉnh một chuỗi Markov. Khi bạn đào tạo một mô hình phân tán, bạn đang học cách đảo ngược một quá trình giống như chuyển động Brownian. Khi bạn chạy suy luận Bayesian, bạn đang xây dựng một chuỗi hội tụ với phía sau.

## Các bài tập

1. **Simulate 1000 random walks of 10000 steps.**Chụp bản phân bố các vị trí cuối cùng. Kiểm tra nó là khoảng Gaussian với trung bình 0 và lệch chuẩn $\sqrt{10000} = 100$- Tôi không biết.

2. **Build a text generator using a Markov chain.**Trình luyện trên một tập hợp nhỏ: cho mỗi từ, đếm chuyển tiếp đến từ tiếp theo. Xây dựng các trận chuyển tiếp. Tạo ra các câu mới bằng cách lấy mẫu từ chuỗi.

3. **Implement simulated annealing**sử dụng Metropolis-Hastings. Bắt đầu ở nhiệt độ cao (tận dụng hầu hết mọi thứ) và dần dần làm mát (tận dụng chỉ cải tiến). Sử dụng nó để tìm tối thiểu của một chức năng với nhiều tối thiểu địa phương.

4. **Compare Langevin dynamics at different temperatures.**Mô hình từ tiềm năng cống hai lốp$U(x) = (x^2 - 1)^2$Ở nhiệt độ thấp, các mẫu được tập hợp trong một giếng. Ở nhiệt độ cao, chúng được trải rộng trên cả hai. Tìm nhiệt độ quan trọng nơi chuỗi trộn lẫn giữa các giếng.

5. **Implement the forward diffusion process.**Bắt đầu với một tín hiệu 1D (ví dụ, một sóng âm đạo). Thêm tiếng ồn dần dần hơn 100 bước với một lịch trình tiếng ồn tuyến tính. Hãy cho thấy tín hiệu suy giảm xuống là tiếng ồn tinh khiết. Sau đó thực hiện một biểu tượng đơn giản đảo ngược quá trình (ngay cả một sự ngây thơ mà chỉ trừ được tiếng ồn ước tính).

### Những gợi ý

**Exercise 1.**Gọi`random_walk_1d(10000, seed=i)`cho 1000 hạt và giữ từng vị trí cuối cùng. kiểm tra rằng trung bình gần 0, lệch chuẩn gần 100, và khoảng 68% vị trí cuối cùng là giữa -100 và 100.

**Exercise 2.**Đếm từng cặp từ, sau đó chia từng hàng bằng tổng số của nó để mỗi hàng cộng lên đến 1.`MarkovChain(transition_matrix, state_names)`và gọi`simulate`Hãy kiểm tra xem mỗi cặp từ trong đầu ra của bạn cũng xuất hiện trong corpus.

**Exercise 3.**Hãy chấp nhận điểm tồi tệ hơn với khả năng$\exp(-\Delta / T)$và nhân lên$T$bằng một nhân tố như 0,99 sau mỗi bước. kiểm tra trên một hàm mà bạn biết tối thiểu, và kiểm tra câu trả lời của bạn với một tìm kiếm lưới tinh tế.

**Exercise 4.**Tốc độ là $4x(x^2 - 1)$, nên hãy chuyển nó cho `langevin_dynamics(grad_U, x0, dt, temperature, n_steps)`- Chân rào ở$x = 0$có chiều cao 1, vì vậy đếm bao nhiêu lần dấu hiệu của $x$thay đổi nhiệt độ từ 0,05 đến 1 và tìm nhiệt độ nơi số lượng bắt đầu tăng nhanh chóng.

**Exercise 5.**Gọi`diffusion_forward(signal, 100)`và tính toán$\bar{\alpha}$bước cuối cùng như là sản phẩm của $(1 - \beta_t)$Nếu$\bar{\alpha}$là xa 0, tín hiệu vẫn chưa là tiếng ồn tinh khiết, vì vậy hãy nâng `beta_end`Hãy thử thử máy báo trước với tiếng ồn thật: sau đó$(x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon)/\sqrt{\bar{\alpha}_t}$phải trả lại tín hiệu ban đầu chính xác.

## Các điều khoản chính

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

## Đọc thêm

- **Ho, Jain, Abbeel (2020)**- "Thử bác bỏ các mô hình xác suất phân tán". Bài báo DDPM đã khởi động cuộc cách mạng mô hình phân tán.
- **Song & Ermon (2019)**-- "Mô hình hóa tổng thể bằng cách ước tính các gradient của phân phối dữ liệu". Phương pháp dựa trên điểm sử dụng động lực Langevin cho việc lấy mẫu.
- **Roberts & Rosenthal (2004)**-- "Thống kê Markov và thuật toán MCMC". Lý thuyết đằng sau khi và tại sao MCMC hoạt động.
- **Norris (1997)**- "Markov Chains". - Cuốn sách giáo khoa tiêu chuẩn.
- **Welling & Teh (2011)**"Sự học Bayesian thông qua động lực Langevin Gradient Stochastic". Kết hợp SGD với động lực Langevin cho suy luận Bayesian có thể mở rộng.

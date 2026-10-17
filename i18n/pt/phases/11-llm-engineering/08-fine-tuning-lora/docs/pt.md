# Ajuste perfeito com LoRA e QLoRA

> O ajuste fino completo de um modelo 7B requer 56 GB de VRAM. Você não tem isso. Nem a maioria das empresas. LoRA permite ajustar o mesmo modelo em 6 GB treinando menos de 1% dos parâmetros. Isto não é um compromisso - ele corresponde à qualidade de ajuste fino completo na maioria das tarefas. Todo o ecossistema de ajuste fino de código aberto funciona com este truque.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 06 (Instruction Tuning / SFT)
**Time:** ~75 minutes
**Related:**A fase 10 abrange os ciclos SFT/DPO desde o zero. Esta lição conecta-os aos kits de ferramentas PEFT 2026 (PEFT, TRL, Unsloth, Axolotl, LLaMA-Factory).

## Objetivos de aprendizagem

- Implementar o LoRA, injetando matrizes de adaptadores de baixo nível (A e B) nas camadas de atenção de um modelo pré-treinado
- Calcular as economias de parâmetros do LoRA vs. ajustamento completo: rank $r$com$d_{\text{model}}$dimensões de trens $2 \cdot r \cdot d$Parâmetros em vez de $d^2$
- Ajuste perfeitamente um modelo usando QLoRA (4 bits base quantizada + adaptadores LoRA) para caber na memória de GPU do consumidor
- Reunir os pesos do LoRA para o modelo base para implantação e comparar a velocidade de inferência com e sem adaptadores

## O problema

Você tem um modelo base. Llama 3 8B. Você quer que ele responda aos bilhetes de suporte ao cliente na voz da sua empresa. SFT é a resposta. Mas SFT tem um problema de custo.

O Llama 3 8B tem 8 bilhões de parâmetros. Em fp16, cada parâmetro leva 2 bytes. Isso é 16 GB apenas para carregar os pesos. Durante o treinamento, você também precisa de gradientes (16 GB), estados de otimização para Adam (32 GB para impulso + variância) e ativações. Total: aproximadamente 56 GB de VRAM para um único modelo 8B.

Um A100 de 80 GB mal pode caber nisto.$3-4/hour on cloud providers. Training for 3 epochs on 50,000 examples takes 6-10 hours. That's $30-40 por experiência, executar 10 experiências para obter os hiperparametros corretos e gastar 400 dólares antes de implementar qualquer coisa.

Escala isto para Llama 3 70B e os números ficam absurdos. 140 GB para pesos sozinhos.

Há um problema mais profundo também. A ajuste fina completa modifica cada peso do modelo. Se ajustarmos os dados do suporte ao cliente, podemos degradar as capacidades gerais do modelo. É chamado esquecimento catastrófico. O modelo fica melhor em sua tarefa e pior em tudo o resto.

Precisamos de um método que treine menos parâmetros, use menos memória e não destrua o conhecimento existente do modelo.

## O conceito

### LoRA: Adaptação de baixo grau

Edward Hu e colegas da Microsoft publicaram o LoRA em junho de 2021. A visão do artigo: as atualizações de peso durante o ajuste fino têm baixa classificação intrínseca. Você não precisa atualizar todos os 16,7 milhões de parâmetros em uma matriz de peso 4096x4096.

Aqui está a matemática. Uma camada linear padrão calcula:

```math
y = Wx
```

Onde ?$W$é um $d_{\text{out}} \times d_{\text{in}}$Para uma projeção de atenção de 4096x4096, isso é 16.777.216 parâmetros.

LoRA congela .$W$e acrescenta uma decomposição de baixo grau:

```math
y = Wx + BAx
```

Onde ?$B$É o que é$(d_{\text{out}} \times r)$E ...$A$É o que é$(r \times d_{\text{in}})$- A posição .$r$é muito menor que $d$-- normalmente 8, 16 ou 32.

Para o$r=16$em camadas de 4096x4096,
- Parâmetros originais: 4096 x 4096 = 16.777.216
- Parâmetros de LoRA: (4096 x 16) + (16 x 4096) = 65,536 + 65,536 = 131,072
- Reduzir: 131.072 / 16.777.216 = 0,78%

Estás a treinar 0,78% dos parâmetros e a obter 95-100% da qualidade.

```mermaid
graph LR
    X["Input x"] --> W["Frozen W (d x d)"]
    X --> A["A (r x d)"]
    A --> B["B (d x r)"]
    W --> Plus["+ (merge)"]
    B --> Plus
    Plus --> Y["Output y"]

    style W fill:#1a1a2e,stroke:#e94560,color:#fff
    style A fill:#0f3460,stroke:#16213e,color:#fff
    style B fill:#0f3460,stroke:#16213e,color:#fff
```

A é iniciada com um gaussiano aleatório. B é iniciada com zero. Isso significa que a contribuição do LoRA começa em zero - o modelo começa a treinar a partir de seu comportamento original e gradualmente aprende a adaptação.

### O Fator de Escalada: Alfa

A LoRA introduz um fator de escala $\alpha$que controla o impacto da atualização de baixo nível na saída:

```math
y = Wx + \frac{\alpha}{r} \cdot BAx
```

Quando ?$\alpha = r$, a escala é 1x. Quando $\alpha = 2r$O hiperparâmetro controla a taxa de aprendizagem do caminho LoRA independentemente da taxa de aprendizagem base.

Orientações práticas:
- $\alpha = 2 \cdot \text{rank}$É uma convenção comunitária comum (o papel original utilizado $\alpha = \text{rank}$na maioria dos experimentos)
- $\alpha = \text{rank}$dá 1x escalação, conservadora mas estável
- Mais alto .$\alpha$significa atualizações maiores por etapa, que podem acelerar a convergência ou causar instabilidade

### Onde aplicar LoRA

Um transformador tem muitas camadas lineares, não é preciso adicionar LoRA a todas.

| Target Layers | Trainable Params (7B) | Quality |
|--------------|----------------------|---------|
| q_proj only | 4.7M | Good |
| q_proj + v_proj | 9.4M | Better |
| q_proj + k_proj + v_proj + o_proj | 18.9M | Best for attention |
| All linear (attention + MLP) | 37.7M | Marginal gain, 2x params |

O ponto ideal para a maioria das tarefas: q_proj + v_proj. Isso visa a consulta e as projeções de valor em auto-atenção, que controlam o que o modelo atende e que informações extrai. Adicionar camadas de MLP ajuda para tarefas complexas como geração de código, mas duplica a contagem de parâmetros para retornos decrescentes em tarefas mais simples.

### Seleção de classificação

O grau .$r$Controla a expressividade da adaptação:

| Rank | Trainable Params (per layer) | Best For |
|------|---------------------------|----------|
| 4 | 32,768 | Simple classification, sentiment |
| 8 | 65,536 | Single-domain Q&A, summarization |
| 16 | 131,072 | Multi-domain tasks, instruction following |
| 32 | 262,144 | Complex reasoning, code generation |
| 64 | 524,288 | Diminishing returns for most tasks |
| 128 | 1,048,576 | Rarely justified |

Hu et al. mostraram que$r=4$A adaptação para tarefas simples já é muito importante. $r=8$E ...$r=16$As opções mais comuns na prática.$r=64$raramente melhora a qualidade e começa a perder a vantagem de memória do LoRA.

### QLoRA: Quantização de 4 bits + LoRA

Tim Dettmers e colegas da Universidade de Washington publicaram o QLoRA em maio de 2023. A ideia: quantizar o modelo base congelado para precisão de 4 bits, em seguida, anexar adaptadores LoRA em fp16 no topo.

Isso muda a equação de memória drasticamente:

| Method | Weight Memory (7B) | Training Memory (7B) | GPU Required |
|--------|-------------------|---------------------|-------------|
| Full fine-tune (fp16) | 14GB | ~56GB | 1x A100 80GB |
| LoRA (fp16 base) | 14GB | ~18GB | 1x A100 40GB |
| QLoRA (4-bit base) | 3.5GB | ~6GB | 1x RTX 3090 24GB |

A QLoRA faz três contribuições técnicas:

**NF4 (Normal Float 4-bit)**O NF4 coloca seus 16 níveis de quantização nos quantiles de uma distribuição normal padrão. Isto é idealmente óptimo para dados normalmente distribuídos. Perde menos informações do que a quantização uniforme de 4 bits (INT4) ou Float4.

**Double quantization**A quantização de constantes em si toma memória. Cada bloco de 64 pesos precisa de um fp32 fator de escala (4 bytes). Para um modelo 7B, que é um extra 0,4 GB.

**Paged optimizers**Durante o treinamento, os estados do optimizador (momento e variância de Adam) podem exceder a memória da GPU em sequências longas. Os optimizadores de páginas usam a memória unificada da NVIDIA para automaticamente pagar os estados do optimizador para a RAM da CPU quando a memória da GPU estiver esgotada, e pagá-los de volta quando necessário. Isso evita que OOM caia ao custo de alguma capacidade.

### A questão da qualidade

A redução de parâmetros ou a quantização da base prejudicam a qualidade?

| Method | MMLU (5-shot) | MT-Bench | HumanEval |
|--------|--------------|----------|-----------|
| Full fine-tune (Llama 2 7B) | 48.3 | 6.72 | 14.6 |
| LoRA r=16 | 47.9 | 6.68 | 14.0 |
| QLoRA r=16 (NF4) | 47.5 | 6.61 | 13.4 |
| QLoRA r=64 (NF4) | 48.1 | 6.70 | 14.2 |

Loura em $r=16$O nível de ajustamento é inferior a 1% do ajustamento completo na maioria dos valores de referência.$r=16$perdem outra fracção de um por cento.$r=64$O sistema de memória é essencialmente adequado ao ajuste perfeito, enquanto usa 90% menos memória.

### Custos reais

Llama 3 8B de ajuste fino em 50.000 exemplares (3 épocas):

| Method | GPU | Time | Cost |
|--------|-----|------|------|
| Full fine-tune | 2x A100 80GB | 8 hours | ~$32 |
| LoRA r=16 | 1x A100 40GB | 4 hours | ~$8 |
| QLoRA r=16 | 1x RTX 4090 24GB | 6 hours | ~$5 |
| QLoRA r=16 (Unsloth) | 1x RTX 4090 24GB | 2.5 hours | ~$2 |
| QLoRA r=16 | 1x T4 16GB | 12 hours | ~$4 |

QLoRA em uma GPU de consumo único custa menos do que um almoço. É por isso que a comunidade de ajuste fino de peso aberto explodiu em 2023 e por isso que cada quadro de treinamento abaixo enviará QLoRA por padrão em 2026.

### A pilha de PEFT 2026

| Framework | What it is | Pick when |
|-----------|-----------|-----------|
| **Hugging Face PEFT** | The canonical LoRA/QLoRA/DoRA/IA3 library | You want raw control and your training loop is already on `transformers.Trainer` |
| **TRL** | HF's reinforcement-from-feedback trainers (SFT, DPO, GRPO, PPO, ORPO) | You need DPO/GRPO after SFT; built on top of PEFT |
| **Unsloth** | Triton-kernel rewrite of the forward/backward pass | You want 2-5x speedup + half the VRAM with no accuracy loss; Llama/Mistral/Qwen family |
| **Axolotl** | YAML-config wrapper over PEFT + TRL + DeepSpeed + Unsloth | You want reproducible, version-controlled training runs |
| **LLaMA-Factory** | GUI/CLI/API over PEFT + TRL | You want zero-code fine-tuning; 100+ model families supported |
| **torchtune** | Native PyTorch recipes, no `transformers` dep | You want minimal deps and your org already standardizes on PyTorch |

Regra geral: uso de pesquisa ou experimento único → PEFT. Pipeline de produção repetível → Axolotl com núcleos Unsloth habilitados.

### Adaptadores de fusão

Após o treino, você tem duas coisas: o modelo base congelado e um pequeno adaptador LoRA (normalmente 10-100 MB).

1. **Keep them separate**A partir de agora, o modelo base é carregado, o adaptador é carregado no topo.

2. **Merge them permanently**: Computação $W' = W + (\alpha/r) \cdot BA$O modelo combinado é do mesmo tamanho que o original, sem despesas gerais, sem adaptador para gerenciar.

Para executar várias tarefas (adaptor de suporte ao cliente, adaptador de código, adaptador de tradução), mantenha-as separadas.

Técnicas de fusão avançadas para a combinação de vários adaptadores:

- **TIES-Merging**(Yadav et al. 2023): Trimparametros de pequena magnitude, resolve conflitos de sinais, então se funde. Reduz a interferência entre os adaptadores.
- **DARE**(Yu et al. 2023): Desliga aleatoriamente os parâmetros do adaptador antes de fundir e reescala o resto. Surpreendentemente eficaz na combinação de capacidades.
- **Task arithmetic**A adição de um adaptador "código" e um adaptador "matemática" geralmente produz um modelo bom em ambos.

### Quando não se deve ajustar

A ajuste é a terceira opção, não a primeira.

**First: prompt engineering.**Escreva um sistema melhor de instrução, adicione alguns exemplos de tiros, use cadeia de pensamento, isso não custa nada e leva minutos, se o instrução lhe dá 80% do caminho, provavelmente não precisa de ajustar.

**Second: RAG.**Se o modelo precisar saber os seus dados específicos (documentos, base de conhecimentos, catálogo de produtos), a recuperação é mais barata e mais mantida do que a elaboração de pesos.

**Third: fine-tuning.**Use isso quando você precisa do modelo para adotar um estilo, formato ou padrão de raciocínio específico que não pode ser alcançado através de solicitação. Quando você precisa de saída estruturada consistente. Quando você precisa destilar um modelo maior em um menor. Quando a latência importa e você não pode pagar os tokens extras de algumas fotos de solicitação.

```mermaid
graph TD
    Start["Need better model behavior?"] --> PE["Try prompt engineering"]
    PE -->|"Works"| Done["Ship it"]
    PE -->|"Not enough"| RAG["Need external knowledge?"]
    RAG -->|"Yes"| RAGBuild["Build RAG pipeline"]
    RAG -->|"No, need style/format change"| FT["Fine-tune with LoRA/QLoRA"]
    RAGBuild -->|"Works"| Done
    RAGBuild -->|"Also need style change"| FT
    FT --> Done

    style Start fill:#1a1a2e,stroke:#e94560,color:#fff
    style Done fill:#0f3460,stroke:#16213e,color:#fff
```

```figure
lora-params
```

## Construí-lo

Implementamos o LoRA do zero em PyTorch puro, sem bibliotecas, sem magia, construímos a camada LoRA, injetamos-a num modelo, treinamos-a e reintegramos os pesos.

### Passo 1: A camada de LoRA

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling
```

A é iniciada com valores aleatórios escalados. B é iniciada com zero. O produto BA começa em zero, então o modelo começa com seu comportamento original.

### Passo 2: Layer linear embutida em LoRA

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

A camada linear original está congelada. Somente os parâmetros LoRA (A e B) são treináveis.

### Passo 3: Injectar LoRA num modelo

```python
def inject_lora(model, target_modules, rank=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers
```

Primeiro, congelar todos os parâmetros do modelo. Depois, caminhe pela árvore do modelo, encontre camadas lineares que correspondam aos seus nomes de destino e substitua-as por versões embaladas em LoRA. As matrizes LoRA A e B são os únicos parâmetros treinaveis em todo o modelo.

### Passo 4: Conte os parâmetros

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }
```

### Passo 5: Reunir os pesos

```python
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)
```

Depois da fusão, as camadas LoRA desaparecem. O modelo é do mesmo tamanho que o original com a adaptação cozida nos pesos.

### Passo 6: Quantização QLoRA simulada

```python
def quantize_to_nf4(tensor, block_size=64):
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales

def dequantize_from_nf4(quantized, scales, original_shape):
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)
```

Isso simula a quantização de 4 bits, mapeando pesos em 16 níveis discretos dentro de blocos de 64.

### Passo 7: Loop de treinamento

```python
def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses
```

### Passo 8: Demo completa

```python
def demo():
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    params_before = count_parameters(model)

    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    params_after = count_parameters(model)

    losses = train_lora(model, data, epochs=20, lr=1e-3)

    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }
```

A demonstração cria um modelo pequeno, injeta o LoRA em duas camadas, treina-o e reintegra os pesos.

## Usá-lo

Com o ecossistema de Rosto Abraçado, o LoRA num modelo real leva cerca de 20 linhas:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Para QLoRA, adicionar quantização de bits e bytes:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = get_peft_model(model, lora_config)
```

O modelo base vive agora em 4 bits, os adaptadores LoRA treinam em fp16, e tudo cabe em 6 GB.

Para o treino com o " Hugging Face Trainer ":

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./lora-adapter")
```

O adaptador guardado é de 10 a 100 MB. O modelo base permanece intocado. Você pode compartilhar adaptadores no Hugging Face Hub sem redistribuir o modelo completo.

## Envia-o

Esta lição produz:
- `outputs/prompt-lora-advisor.md`-- um prompt que ajuda a decidir a classificação do LoRA, módulos alvos e hiperparâmetros para a sua tarefa específica
- `outputs/skill-fine-tuning-guide.md`- Uma habilidade que ensina os agentes a árvore de decisão para quando e como ajustar

## Exercícios

1. **Rank ablation study.**Execute a demonstração com as linhas 2, 4, 8, 16, 32 e 64. Plot final perda vs. rank. Encontre o ponto de retornos diminuindo onde dobrar a classificação não mais metade da perda. Para uma tarefa de classificação simples em 256-dim características, isso deve ser em torno de r = 8-16.

2. **Target module comparison.**Modifique inject_lora para atingir apenas a camada "0", apenas a camada "2", apenas a camada "4", e todas as três. Treine cada variante por 20 épocas. Compare a velocidade de convergência e a perda final. Isso reflete a decisão real de atingir q_proj vs v_proj vs todas as camadas lineares.

3. **Quantization error analysis.**Tome as matrizes de peso do modelo treinado antes e depois quantize_to_nf4 / dequantize_from_nf4. Calcule o erro médio quadrado, o erro máximo absoluto e a correlação entre pesos originais e reconstruídos. Experimente com valores de block_size de 32, 64, 128 e 256.

4. **Multi-adapter serving.**Treinar dois adaptadores LoRA em subconjuntos diferentes dos dados (mesmo índices versus índices ímpares). Salvar ambos os adaptadores. carregar o modelo base uma vez, em seguida, trocar adaptadores e verificar que cada produz saídas diferentes na mesma entrada. É assim que os sistemas de produção servem vários modelos de sintonia fina de uma base.

5. **Merge vs. unmerged inference.**Compare a saída do modelo LoRA antes e depois de merge_lora_weights nas mesmas 100 entradas. Verifique que as saídas são idênticas (dentro da tolerância de ponto flutuante de 1e-5).

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| LoRA | "Efficient fine-tuning" | Low-Rank Adaptation: freeze base weights, train two small matrices A and B whose product approximates the full weight update |
| QLoRA | "Fine-tune on a laptop" | Quantized LoRA: load the base model in 4-bit NF4, train LoRA adapters in fp16 on top, enabling 7B fine-tuning in 6GB VRAM |
| Rank (r) | "How much the model can learn" | The inner dimension of the A and B matrices; controls expressiveness vs. parameter count |
| Alpha | "LoRA learning rate" | Scaling factor applied to the LoRA output; $\alpha/r$ scales the adaptation's contribution to the final output |
| NF4 | "4-bit quantization" | Normal Float 4: a 4-bit data type with quantization levels at normal distribution quantiles, optimal for neural network weights |
| Adapter | "The small trained part" | The LoRA A and B matrices saved as a separate file (10-100MB), loadable on top of any copy of the base model |
| Target modules | "Which layers to LoRA" | The specific linear layers (q_proj, v_proj, etc.) where LoRA adapters are injected |
| Merging | "Bake it in" | Computing $W + (\alpha/r) \cdot BA$ and replacing the original weight, eliminating the adapter overhead at inference |
| Paged optimizers | "Don't OOM during training" | Offloading optimizer states (Adam momentum, variance) to CPU when GPU memory is exhausted |
| Catastrophic forgetting | "Fine-tuning broke everything else" | When updating all weights causes the model to lose previously learned capabilities |

## Mais leitura

- Hu et al., "LoRA: Adaptação de baixo nível de grandes modelos de linguagem" (2021) -- o artigo original que introduz o método de decomposição de baixo nível, testado no GPT-3 175B com um nível tão baixo quanto 4
- Dettmers et al., "QLoRA: Eficiente Fine-Tuning de Modelos de Língua Quantizada" (2023) -- introduz NF4, dupla quantização e optimizadores de páginas, permitindo 65B fine-tuning em uma única GPU de 48GB
- Documentação da biblioteca PEFT (huggingface.co/docs/peft) - a biblioteca padrão para LoRA, QLoRA e outros métodos eficientes em parâmetros no ecossistema Hugging Face
- Yadav et al., "TIES-Merging: Resolving Interference When Merging Models" (2023) -- técnicas para combinar vários adaptadores LoRA sem degradação de qualidade
- [Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023)](https://arxiv.org/abs/2305.18290)-- Derivação DPO; a fase de ajuste de preferências que vem após a FFT, sem modelo de recompensa necessário.
- [TRL documentation](https://huggingface.co/docs/trl/)-- referência oficial para `SFTTrainer`- Não .`DPOTrainer`- Não .`KTOTrainer`, e a superfície de integração com PEFT/bitsandbytes/Unsloth.
- [Unsloth documentation](https://docs.unsloth.ai/)-- núcleos fundidos que duplicam o rendimento de ajuste fino e reduzem a metade a memória; a camada de desempenho sob o TRL.
- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/)-- Treinador multi-GPU SFT/DPO/QLoRA configurado com YAML; alternativa de configuração como código a scripts manuscritos.

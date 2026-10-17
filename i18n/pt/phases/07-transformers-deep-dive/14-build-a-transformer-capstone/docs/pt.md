# Construir um Transformador a partir do zero  A Capstone

> Trinta lições, um modelo, sem atalhos.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## O problema

Já leu todos os artigos, implementou atenção, divisões de várias cabeças, codificações posicionais, blocos de codificação e decodificação, perdas de BERT e GPT, MoE, cache KV. Agora faça com que trabalhem juntos em uma tarefa real.

A pedra angular: treinar um pequeno transformador de end-to-end apenas para um decodificador em uma tarefa de modelagem de linguagem de nível de personagens. Ele lê Shakespeare. Ele gera novo Shakespeare. É pequeno o suficiente para treinar em um laptop em menos de 10 minutos. É correto o suficiente que trocar um conjunto de dados maior e treinamento mais longo lhe dá um LM real.

Este é o "nanoGPT" do curso. Não é original. O tutorial de 2023 de Karpathy é a implementação de referência que cada aluno escreve pelo menos uma vez. Levamos a forma e reformula-a em torno do que cobrimos.

## O conceito

![Transformer-from-scratch block diagram](../assets/capstone.svg)

A arquitetura, com notas:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### O que enviamos

- `GPTConfig` um local para a configuração de todos os hiperparâmetros.
- `MultiHeadAttention` Causal, batch, com caminho opcional de estilo Flash (PyTorch's `scaled_dot_product_attention`)).
- `SwiGLUFFN` FFN moderno.
- `Block` Atenção pré-norma, embalada residual + FFN.
- `GPT` embutidos, blocos empilhados, cabeçalho LM, gerar().
- Localização com AdamW, cosino LR, corte de gradiente.
- Tokenizer de nível Char sobre textos de Shakespeare.

### O que não enviamos

- RoPE  implementado conceitualmente na lição 04. Aqui usamos embutimentos posicionais aprendidos para simplicidade.
- O cache KV durante a geração  cada passo de geração recalcula a atenção sobre o prefixo completo. Mais lento, mas mais simples. Os exercícios pedem que você adicione um cache KV.
- Atenção Flash  PyTorch 2.0+ auto-dispatches se as entradas coincidem; usamos `F.scaled_dot_product_attention`- Não .
- MoE  FFN único por bloco.

### Metricas de metas

Em um laptop Mac M2, um laptop de 4 camadas, 4 cabeças, d_model=128 GPT treinado para 2.000 passos em `tinyshakespeare.txt`- Não .

- A perda de treino converge de ~ 4,2 (aleatório) para ~ 1,5 em cerca de 6 minutos.
- A produção amostrada parece em forma de Shakespeare: palavras arcaicas, interrupções de linha, nomes próprios como "ROMEO:" surgem.
- A perda de valor (a última 10% do texto não recebida) acompanha de perto a perda de formação; não há sobreajuste neste tamanho/orçamento.

```figure
n5-block-stack
```

## Construí-lo

Esta aula usa PyTorch.`torch`(Construção do CPU está bem).`code/main.py`O roteiro diz:

- Descarga`tinyshakespeare.txt`se faltarem (ou se estiverem a ler uma cópia local).
- Tokenizer de carros de nível byte.
- Trem/val dividido em 90/10.
- Loop de treinamento com bf16 autocast no hardware suportado.
- A amostragem após o treino termina.

### Passo 1: dados

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 caracteres únicos, pequeno vocabulário, cabe a um tamanho de 4 bytes, sem BPE, sem drama de tokenizer.

### Passo 2: modelo

Veja .`code/main.py`O bloco é um livro de texto da lição 05  pré-norma, RMSNorm, SwiGLU, MHA causal.

### Passo 3: ciclo de treinamento

Obter um lote aleatório de janelas de tokens de comprimento 256, para frente, transmissão por entropia, para trás, passo AdamW, registro, repetição.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Passo 4: amostra

Dado um pedido, repetidamente encaminhado, amostra de logits top-p, anexe, e continuar.

### Passo 5: ler a saída

Depois de 2.000 passos:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Não é Shakespeare, mas em forma de Shakespeare, uma vitória clara por 800 mil parâmetros e 6 minutos num laptop.

## Usá-lo

Esta pedra-chave é uma arquitetura de referência.

1. **Swap the tokenizer.**Utilize BPE (por exemplo `tiktoken.get_encoding("cl100k_base")`O tamanho da vocab aumentou de 65 para 50 mil.
2. **Train on a bigger corpus.**Utilização`OpenWebText`ou `fineweb-edu`10B tokens em um único A100 leva cerca de 24 horas para um GPT de 125M-param.
3. **Add RoPE + KV cache + Flash Attention.**Os exercícios abaixo vão guiá-lo através de cada um.

Isto acaba como um GPT de 125M que gera inglês fluente. Não é um modelo de fronteira. Mas o mesmo caminho de código  apenas maior  é o que Karpathy, EleutherAI e o Instituto Allen usam para treinar pontos de controle de pesquisa em 2026.

## Envia-o

Veja .`outputs/skill-transformer-review.md`A habilidade revisa uma implementação transformadora a partir do zero para verificar a correcção em todas as 13 lições anteriores.

## Exercícios

1. **Easy.**Corra .`code/main.py`Verifique se a perda de validação da fase final do modelo treinado é inferior a 2.0. Mudança `max_steps`De 2.000 a 5.000 , a perda de val continua a melhorar?
2. **Medium.**Substitua as inserções posicionais aprendidas com RoPE. Aplique a rotação para Q e K dentro `MultiHeadAttention`A perda de val é pelo menos tão baixa.
3. **Medium.**Implementar um cache KV no loop de amostragem. Gerar 500 tokens com e sem cache. O relógio de parede deve melhorar em 520x em um laptop.
4. **Hard.**Adicione uma segunda cabeça ao modelo que prevê o próximo token mais um (MTP  Multi-Token Prediction de DeepSeek-V3). Treinar em conjunto.
5. **Hard.**Substitua o único FFN por bloco por um MoE de 4 especialistas. Roteador + roteamento top-2. Veja como a perda de val muda em parâmetros ativos correspondentes.

## Termos-chave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy's tutorial repo" | Minimal decoder-only transformer training code, ~300 LOC; the canonical reference. |
| tinyshakespeare | "The standard toy corpus" | ~1.1 MB of text; every character-LM tutorial since 2015 uses it. |
| Tied embeddings | "Share input/output matrix" | LM head weight = transpose of token embedding matrix; saves parameters, improves quality. |
| bf16 autocast | "Training precision trick" | Run forward/back in bf16, keep optimizer state in fp32; standard since 2021. |
| Gradient clipping | "Stops spikes" | Cap global grad norm at 1.0; prevents training blowups. |
| Cosine LR schedule | "The 2020+ default" | LR ramps up linearly (warmup) then decays cosine-shaped to 10% of peak. |
| MFU | "Model FLOP Utilization" | Achieved FLOPs / theoretical peak; 40% dense, 30% MoE is strong in 2026. |
| Val loss | "Held-out loss" | Cross-entropy on data the model never saw; overfit detector. |

## Mais leitura

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) a aplicação clássica com anotações.

# Patches de codificação visual

> Um modelo de visão que lê pixels precisa de um tokenizer para pixels. Embed patch é esse tokenizer. Corte a imagem em uma grade de quadrados, aplanie cada quadrado, projetá-lo através de uma camada linear, e adicione um sinal de posição 2D para que o transformador saiba onde cada quadrado estava na imagem original.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 lessons 30-37 (Track B foundations)
**Time:** ~90 minutes

## Objetivos de aprendizagem

- Tokenize uma imagem em uma sequência de comprimento fixo de embutidos de parches.
- Implementar um `Conv2d`- Baseada em projeção de parche que corresponde à matemática de desdobrar-depois-linear.
- Construir uma posição sinusoidal 2D determinista incorporando assim ordem simbólica codifica posição espacial.
- Verificar o número de parches, a forma de inserção e `Conv2d`- Desdobrar a equivalência numa fixação sintética.

## O problema

Um transformador come uma sequência de vetores. Uma imagem é uma grade de três canais. Ler cada pixel como um token explode o comprimento da sequência: uma imagem RGB de 224x224 é de 150.528 tokens, que um transformador de 12 camadas não pode permitir a atenção. A leitura da imagem como um vetor plano gigante descarta localidade, da qual a camada de atenção não pode recuperar. O trabalho do encoder front end é comprimir a grade de pixels em algumas centenas de tokens que cada um resumem uma região quadrada.

A inserção de parche resolve isso com uma projeção linear. Uma imagem de 224x224 cortada em parches 16x16 produz uma grade de 14x14 de 196 parches.`(3, 16, 16) = 768`Os valores dos pixels em um vetor, então uma camada linear mapeia-o para a dimensão oculta do modelo.`hidden`(comumente 768) mais um token CLS. Essa é uma sequência que o resto da rede pode mastigar.

## O conceito

```mermaid
flowchart LR
  Image[224x224x3 image] --> Cut[cut into 16x16 patches]
  Cut --> Grid[14x14 grid of patches]
  Grid --> Flatten[flatten each patch]
  Flatten --> Proj[linear projection]
  Proj --> Tokens[196 tokens of dim hidden]
  Tokens --> Pos[add 2D sinusoidal position]
  Pos --> Out[final token sequence]
```

### Por que correções, não pixels

A atenção é quadrática no comprimento da sequência.`196 * 196 = 38,416`pontuações de atenção por cabeça por camada; uma sequência de 150.528 tokens custa `150,528 * 150,528 = 22.6 billion`Os patches compram uma redução de 590.000x no cálculo da atenção, e uma única região 16x16 carrega sinal suficiente para tarefas de visão de alto nível. O custo é a perda de detalhes espaciais finos dentro de um patch, razão pela qual as pilhas multimodal a jusante geralmente executam um segundo ramo de alta resolução quando a localização fina importa.

### Por que uma projeção linear é suficiente

Cada parche é tratado como um vetor independente. A projeção aprende uma base: detectores de borda, filtros de cor, texturas simples.`768 * 768 = 589,824`Os sistemas de codificação de peso aberto são mais rápidos e os sistemas de codificação de peso aberto são mais rápidos.

### O `Conv2d`truque

A.`Conv2d(in_channels=3, out_channels=hidden, kernel_size=patch_size, stride=patch_size)`sem enchimento dá o mesmo resultado numérico que o de desdobrar-e-desdobrar-linear, porque cada posição de saída produz os pixels do parche contra um filtro. A convolução é a projeção do parche, e a maioria das bases de código de produção enviam-no para esse caminho porque é mais rápido na GPU e usa uma remodelação menos.

### Embedings de posição

Os tokens não têm ordem fora da projeção. a incorporação sinusoidal 2D dá a cada token um sinal fixo que codifica o seu`(row, col)`A metade da dimensão de incorporação codifica a posição da linha com sin/cos em múltiplas frequências; a outra metade codifica a posição da coluna. A codificação é determinista para que você possa trocar resoluções sem reestruturação, e interpola limpa para redes que o modelo nunca viu no tempo de treinamento.

| Component | Shape | Parameters |
|-----------|-------|------------|
| Patch projection (`Conv2d`) | `(hidden, 3, patch, patch)` | `3 * P * P * hidden + hidden` |
| Position embedding (fixed) | `(num_patches, hidden)` | 0 (computed, not learned) |
| CLS token (learned) | `(1, hidden)` | `hidden` |

Para ViT-Base/16 em resolução 224: 590.592 parâmetros na projeção, 768 no token CLS, e zero para a posição sinusoidal.

### Equivalência como controlo de sanidade

O passo de parche tem duas ortografias: a `Conv2d`A projeção de um código de código é uma projeção de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código

```figure
ch-patch-tokenizer
```

## Construí-lo

`code/main.py`Implementos:

- `PatchEmbed`, um `nn.Module`embalagem`Conv2d`para projecção de parches.
- `sinusoidal_2d(grid_h, grid_w, dim)`, uma função sem estado que constrói a tabela de posições 2D.
- `VisionFrontEnd`, que compõe a inserção de parche, a prependio CLS e a adição de posição em uma passagem para a frente.
- A.`synthesize_image(seed)`auxiliar que constrói uma fixação determinista de 224x224x3 a partir de `numpy.random`- Não .
- Uma demonstração que expande uma imagem de um dispositivo através da frente e imprime a forma de saída, a norma do token CLS e uma linha da posição de inserção.

- É o que é ?

```bash
python3 code/main.py
```

Resultado: a fixação 224x224 é tokenizada para uma sequência de forma `(1, 197, 768)`O primeiro token é o CLS; os seguintes 196 são patch tokens.

## Usá-lo

O mesmo patch front end aparece em todos os modelos modernos de linguagem de visão: CLIP ViT-L/14, SigLIP, DINOv2, a família Qwen-VL e a pilha InternVL todos começam a partir de um `Conv2d`A diferença entre as famílias vive para baixo (CLS vs no-CLS pooling, tokens de registro, tamanhos variados de patch 14 vs 16, resolução dinâmica através de posições interpoladas).

## Teste

`code/test_main.py`Cobertura:

- Parâmetros de contagem de parâmetros`(image_size / patch_size) ** 2`
- correspondências de forma de saída `(batch, num_patches + 1, hidden)`
- O `Conv2d`projeção é igual a manual desdobrar-e-linear em um pequeno dispositivo
- Tabela de posição sinusoidal é determinista em todas as chamadas
- Transmissões de tokens CLS através de lotes de dim sem vazamento

- E depois ?

```bash
python3 -m unittest code/test_main.py
```

## Exercícios

1. Substitua a posição sinusoidal por um aprendiz`nn.Parameter`As posições aprendidas ganham em resolução fixa; as sinusoides ganham quando mudamos de resolução após o treino.

2. Troca o `Conv2d`para um explícito `nn.Unfold`- E mais .`nn.Linear`E afirmar que as saídas correspondem à tolerância de flutuação.

3. Adicionar suporte para tamanhos de parches não quadrados (por exemplo, 32x16 para entradas de grande dimensão) e verificar se a tabela de posições lida com redes não quadradas.

4. Profile o passo do parche em tamanhos de lote 1, 8, 64. A projeção do parche raramente é o gargalo de engarrafamento; as camadas de atenção para baixo dominam.

5. Treinar a frente como um extractor de características congelado em um conjunto de dados de forma sintética de 4 classes (círculos, quadrados, triângulos, estrelas).

## Termos-chave

| Term | What it means |
|------|---------------|
| Patch | A square sub-region of the image, typically 14x14 or 16x16 |
| Patch embedding | Linear projection of one flattened patch to the hidden dim |
| Sequence length | Number of tokens after patch tokenization, usually plus CLS |
| Sinusoidal position | Fixed sin/cos signal that encodes 2D grid coordinates |
| CLS token | Learned vector prepended to the sequence as the pooling head |

## Mais leitura

- Uma imagem vale 16x16 palavras (ViT, 2021) para o enquadramento original com inserção de parche.
- Atenção é tudo que você precisa (2017) para a fórmula de posição sinusoidal adaptada aqui para 2D.
- Papel DINOv2 para tokens de registro, uma extensão que pode ser adicionada como exercício 6.

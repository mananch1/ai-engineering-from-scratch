# Modelagem Autoregressiva Visual (VAR): Previsão em Nova Escala

> Os modelos de difusão amostram iterativamente no tempo (denotando passos). As amostras VAR iterativamente na escala  prevê um token 1x1, em seguida, 2x2, em seguida, 4x4, até a resolução final, cada escala condicionando sobre a anterior. O artigo de 2024 mostrou que o VAR combina com as leis de escalagem de estilo GPT para geração de imagem e supera o DiT no mesmo orçamento de computação. Esta lição constrói o mecanismo central.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## O problema

A geração autoregressiva dominou a modelagem de linguagem porque escala de forma previsível: mais computação, mais parâmetros, menor perplexidade, melhores saídas. A geração de imagens teve duas tentativas principais de AR antes de 2024: PixelRNN/PixelCNN (pixel-by-pixel) e DALL-E 1 / Parti / MuseGAN (token-by-token em códigos VQ-VAE).

Ambos sofreram um problema de ordem de geração. Pixels e tokens são dispostos em uma grade 2D, mas o modelo AR tem que visitá-los em uma ordem raster 1D. Um pixel de canto inicial não tem idéia do que a imagem acaba se tornando.

O VAR corrige o problema da ordem de geração alterando o que está sendo gerado. Em vez de prever tokens de imagem um por um no espaço, o VAR prevê uma imagem inteira com resoluções crescentes. Etapa 1: prevê um token 1x1 (a imagem geral "resumo"). Etapa 2: prevê uma grade de tokens 2x2 (funções mais grosseiras). Etapa 3: prevê uma grade de 4x4. Etapa K: prevê a grade final (H/8) x ((W/8)).

Cada escala atende a todas as escalas anteriores (causalmente em "ordem de escala") e paralelo dentro de sua própria escala.

## O conceito

### O Tokenizer VQ-VAE em Multicálculo

A VAR precisa de um**multi-scale discrete tokenizer**Para uma imagem .$x$, produz uma sequência de redes de tokens de resolução progressivamente maior:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

Cada um .$z_k$Utiliza o mesmo livro de códigos (dimensional típico 4096-16384). A tokenização em cada escala não é independente  é treinada de modo que a soma dos resíduos em cada escala reconstrui $f$- Não .

```math
f \approx \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_1), \text{target\_size}) + \cdots + \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_K), \text{target\_size})
```

Isto é um**residual VQ**Variante.$k$Captura que balança$1 \ldots k-1$O decodificador leva a soma de todas as incrustações da escala e produz a imagem.

O tokenizer VQ em várias escalas é treinado uma vez (como VQGAN) e depois congelado.

### Próxima previsão

O modelo gerativo é um transformador que vê tokens de todas as escalas anteriores e prevê os tokens na próxima escala.

Estrutura de sequência de entrada:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

As inserções de posição codificam tanto o índice de escala quanto a posição espacial dentro da escala.$k$, posição $(i, j)$Pode assistir a todos os tokens em escalas $1 \ldots k$e para tokens em escala $k$A VAR utiliza atenção posicional fixa sem causalidade intra-escala  todas as posições dentro de uma escala são previstas em paralelo.

Perda de formação: em cada escala $k$, prever os tokens .$z_k$A perda de entropia cruzada nos códigos VQ discretos. A mesma estrutura que a GPT exceto a "sequência" é agora estruturada em escala.

### Geração

Na inferência:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

Para o$K = 10$As imagens são de 256x256 e são de aproximadamente 10 passes em relação a 28-50 da DiT.

### Por que a próxima escala vence a próxima

Três vitórias estruturais:
1. **Coarse-to-fine aligns with natural image statistics.**A percepção visual humana e os conjuntos de dados de imagem apresentam regularidades dependentes da escala: a estrutura de baixa frequência é estável e previsível; os detalhes de alta frequência são condicionados ao conteúdo de baixa frequência.
2. **Parallel generation within scale.**Ao contrário do token AR do estilo GPT, o VAR produz todos os tokens em uma escala em um passo.
3. **No generation order bias.**Tokens na escala k vêem toda a escala k-1; não há "esquerdade" ou "acima" que força os tokens iniciais a se comprometer antes que o contexto tardío esteja disponível.

### Lei de Escalada

Tian et al. demonstrou que o VAR segue uma curva de escalagem de poder para a FID na ImageNet  assim como o GPT faz para a perplexidade. O duplicado de parâmetros ou de cálculo reduz de forma confiável o erro à metade. Este foi o primeiro modelo gerador de imagens a exibir este tipo de comportamento de escala tão limpo quanto os modelos de linguagem. O resultado é que as previsões em escala VAR se tornam previsíveis a partir de computação, não suposições empíricas por arquitetura.

### Relação com a difusão

A VAR e a difusão compartilham a mesma história de compressão de dados: ambos dividem o problema de geração em uma sequência de subproblemas mais fáceis.

- Difusão: adicionar gradualmente ruído, aprender a desviar um passo.
- VAR: adicionar gradualmente resolução, aprender a prever a próxima escala.

Eles são eixos diferentes através do problema. Ambos produzem distribuições condicionais tratáveis. Empiricamente, VAR é mais rápido na inferência (menos passes, todos paralelos dentro de uma escala) e corresponde ou supera o DiT na ImageNet condicional de classe.

```figure
gx-var-next-scale
```

## Construí-lo

- Não .`code/main.py`Você vai:
1. Construir um pequeno .**multi-scale VQ tokenizer**em dados sintéticos de "imagem" (2 anéis gaussianos em dimensão).
2. Treinar um**VAR-style transformer**para a próxima escala-previsão dos tokens.
3. Amostra por chamada ao transformador 4 vezes (4 escalas) e decodificação.
4. Verificar que a formação em ordem de escala faz a geração paralela dentro de uma escala.

É uma implementação de brinquedo. O ponto é ver a máscara de atenção estruturada em escala e a geração paralela dentro da escala realmente a funcionar.

## Envia-o

Esta lição produz`outputs/skill-var-tokenizer-designer.md` habilidade para a concepção de um tokenizer de várias escalas: número de escalas, proporções de escala, tamanho do livro de códigos, partilha residual, arquitetura de decodificadores.

## Exercícios

1. **Scale count ablation.**Treinar VAR com 4, 6, 8, 10 escalas. Medir a qualidade da reconstrução vs número de passes autoregressivos. Mais escalas = restantes mais finos = melhor qualidade mas mais passes.

2. **Codebook size.**Trem tokenizers com tamanhos de código de 512, 4096, 16384.

3. **Parallel-within-scale check.**Para um VAR treinado, medir o padrão de atenção explicitamente. Dentro da escala k, o modelo atende a posições em escala transversal, mas não intra-escala? Verifique a implementação da máscara.

4. **VAR vs DiT scaling.**Para a mesma tarefa condicional de classe ImageNet, treine VAR e DiT em orçamentos de parâmetros iguais (por exemplo, 33M, 130M, 458M). Plot FID vs computação. VAR deve avançar em frente de DiT em cada tamanho  reproduzir o resultado do papel em pequena escala.

5. **Text conditioning.**Extender VAR para tomar uma incorporação de texto (CLIP agrupado) como uma entrada de condicionamento extra através da adaLN. Esta é a receita HART.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | Image generation by next-scale prediction over a pyramid of VQ token grids |
| Next-scale prediction | "Predict coarser, then finer" | The model predicts tokens at increasing resolution scales, conditioning on all previous scales |
| Multi-scale VQ tokenizer | "Residual VQ" | VQ-VAE that produces K token grids of increasing resolution, with decoder summing all scales |
| Scale k | "Pyramid level k" | One of K resolution levels, from 1x1 at k=1 up to (H/p)x(W/p) at k=K |
| Parallel-within-scale | "One forward per scale" | All tokens at scale k are predicted in one transformer pass, not autoregressively |
| Causal-across-scales | "Scale-ordered attention" | Token at scale $k$ can attend to all of scales $1 \ldots k$ but not scales $k+1 \ldots K$ |
| Residual VQ | "Additive tokenization" | Each scale's tokens encode the residual left by lower scales; decoder sums all scale embeddings |
| VAR scaling law | "Image GPT scaling" | FID follows a predictable power law in compute, like language models' perplexity |
| HART | "Hybrid VAR + text" | Text-conditional VAR variant combining MaskGIT-style iterative decoding with VAR's scale structure |
| Scale position embedding | "(scale, row, col) triple" | Positional encoding carries both the scale index and spatial coordinates within the scale |

## Mais leitura

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) o documento VAR, referência canónica
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) DiT, linha de base de comparação de difusão
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) VQGAN, a família de tokenizadores VAR extensão de tokenizadores em várias escala
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) VQ-VAE, a base da tokenização de imagens discretas
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) VAR condicional de texto

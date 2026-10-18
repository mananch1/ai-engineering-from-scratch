# Análise paralela e de bolhas de oleodutos

> O paralelismo tensor divide a matriz multiplicar-se em filas. O paralelismo de pipeline divide o modelo em filas, um estágio por fila. Os microbates fluem através do pipeline. O tempo vazio no início e no fim é a bolha; minimizando-a é toda a nave.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 Track C lessons 42-49
**Time:** ~90 min

## Objetivos de aprendizagem

- Dividir um modelo sequencial em N fases e simular um pipeline para frente em N filas.
- A programação M micro-baches através do gasoduto usando o programação GPipe (completo apenas para a frente, em seguida para trás) e calcular a fracção de bolhas.
- Compare bolha com o cronograma 1F1B entrelaçado usado em Megatron-LM e PipeDream.
- A atribuição de estágios defenda: o cálculo igual por estágio importa mais do que o número igual de parâmetros por estágio.

## O problema

Um modelo de parâmetro 70B em fp16 precisa de 140 GB de parâmetros sozinho. Nenhuma GPU de consumo o mantém. ZeRO-3 divide os parâmetros em filas, mas ainda precisa de cada fila para reunir a camada completa para cada passo a frente, pagando $\log(N)$O modelo é dividido em N fases e colocado em cada categoria. A frente da camada 1 termina na categoria 0 e entrega o tensor de ativação para a categoria 1; a categoria 1 corre na categoria 2 e as mãos para a categoria 2; e assim por diante.

A bolha é o tempo de inatividade no início do oleoduto (a espera do primeiro microbatch para chegar à última fase) e no final (a espera do último microbatch para drenar de volta). Com microbates M e estágios N, a fração de bolhas por estágio é (N-1)/(M+N-1). Em M=8, N=4 é 27%. Em M=64, N=4 é 4,5%. A bolha encolhe quando temos muitos microbatches por passo, o que significa pequenos tamanhos de batches por microbatch, que é a restrição que impulsiona o projeto de microbatches.

## O conceito

```mermaid
flowchart LR
  R0[rank 0: stage 0 / layer 0] --> R1[rank 1: stage 1 / layer 1]
  R1 --> R2[rank 2: stage 2 / layer 2]
  R2 --> R3[rank 3: stage 3 / loss]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### Programação do GPipe

Preencha o tubo para a frente com todos os microbates M antes de iniciar qualquer retroceder; em seguida, deslize para trás em sentido inverso. As ativas de cada microbatch devem ser mantidas até ao seu retorno, para que a memória cresça linearmente com M. Para frente, são necessários ciclos M+N-1, para trás, outros ciclos M+N-1. O trabalho útil por fase é de 2M ciclos; por fase bolha é de 2 ((N-1) ciclos. A fração de bolha é (N-1) / ((M+N-1) quando cada avanço e atraso demoram uma unidade de tempo. Escolher M muito maior que N esconde a bolha.

### 1F1B programação

Interleave: assim que o microbatch avançar até o último estágio, comece para trás e deixe fluir de volta. O horário alternará uma para frente e outra para trás por estágio. A bolha ainda é N-1, mas a memória de ativação é limitada pela profundidade do tubo, não pela contagem de microbatches. As linhas de produção utilizam 1F1B (Megatron, PipeDream). A lição implementa o GPipe primeiro porque é mais simples, e 1F1B como um exercício.

### Por que a computação igual por estágio importa

Se o estágio 0 demora 50 ms e o estágio 1 demora 100 ms, cada ciclo é bloqueado no estágio 1. Os outros estágios estão inactivos 50 ms por ciclo esperando o estágio 1 para ser liberado.

### Microbatch versus batch

Um pipeline executar M microbatches de tamanho B cada. O tamanho de lote efetivo é M*B. O gradiente no final de um passo de pipeline é o gradiente nos exemplos combinados M*B. A fração de bolha depende de M; o optimizador vê M*B. A sintonização M significa a troca de bolha (menor com M alto) contra a memória por microbatch (memória de ativação mais alta com M alto para GPipe).

```figure
cd-pipeline-bubble
```

## Construí-lo

`code/main.py`Implementos:

- `PipelineStage`Uma pequena .`nn.Module`que contenha os parâmetros de um estágio e expõe `forward(activation)`- Não .
- `Pipeline(stages, num_microbatches)`O programa GPipe é organizado em fases simuladas, utilizando um relógio de parede simulado por etapa.
- `bubble_fraction(num_stages, num_microbatches)`: de forma fechada (N-1) / ((M+N-1).
- Uma demonstração em 4 etapas que imprime a traça por microbate e a fração de bolha medida.

- É o que é ?

```bash
python3 code/main.py
```

Resultado: um gráfico de Gantt etapa por microbatech e a porcentagem de bolhas contra a previsão de forma fechada.

## Padrões de produção em silêncio

Três padrões endurecem o oleoduto paralelo o suficiente para o transporte.

**Activation checkpointing pairs with pipeline.**Com M microbatches em voo no GPipe, a memória de ativação é M vezes um microbatch.

**Stage balance is measured, not assumed.**As equipes de produção executam um perfil que mede a computação real por camada (FLOPs e relógio de parede) no hardware alvo, e depois partição por essa medição.`--num-layers-per-stage`A bandeira aceita uma lista que permita contagens desiguais de camadas quando os estágios têm custos por camada diferentes.

**Send-recv schedule must avoid deadlock.**Um pipeline que tem cada estágio enviado antes de receber impasses no fio. A solução padrão é intercalar: estágios de ranking par enviam primeiro, em seguida, recv, estágios de ranking ímpar recv primeiro, em seguida, enviam. Os horários de lições são classificados explicitamente para que o padrão seja visível.

## Usá-lo

Padrões de produção:

- **Megatron-LM.**A referência para o paralelismo de pipeline em escala. utiliza 1F1B e suporta tensor + pipeline + dados paralelos combinados.
- **DeepSpeed Pipeline.**Integrado com o ZeRO; o oleoduto ZeRO-1 + é uma combinação comum para os maiores modelos abertos.
- **PyTorch Pipe.**O envelope de oleoduto nativo PyTorch, construído em`torch.distributed.pipeline.sync.Pipe`- Não .

## Envia-o

A lição 80 armazena os fragmentos de parâmetros por etapa no checkpoint fragmentado. A lição 81 compõe o pipeline DDP + ZeRO + na demonstração de ponta a ponta (em espírito; a demonstração mantém a pipeline simulada para tempo de execução).

## Exercícios

1. Implementar 1F1B e verificar que a fração da bolha coincide com o GPipe, mas a memória de ativação é limitada.
2. Profila o tempo real por fase num modelo mais profundo e reequilibra os estágios por meio do relógio de parede medido.
3. Adicionar a acumulação de gradientes em microbatches de pipeline e verificar que o gradiente é igual ao gradiente do equivalente lote completo para a frente.
4. Combine o pipeline com o checkpoint de ativação e mede a queda de memória versus o custo de cálculo.
5. Combine o pipeline com o DDP (cada nível de pipeline é replicado em um grupo paralelo de dados) e racionalize através do cronograma 2D.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Pipeline | "Model parallel along depth" | One stage per rank, activations flow stage to stage |
| Bubble | "Pipeline idle time" | (N-1) steps at start + end where some stages have no work |
| Microbatch | "Slice of the batch" | One forward/backward unit; bubble shrinks as M grows |
| GPipe | "Fill then drain" | All M forwards before any backward; high activation memory |
| 1F1B | "Interleaved schedule" | One forward one backward per stage; bounded activation memory |

## Mais leitura

- [Huang et al, GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Narayanan et al, PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)
- [Megatron-LM pipeline parallel docs](https://github.com/NVIDIA/Megatron-LM)
- Fase 19 Lição 76 - as primitivas de envio/recovery que o cronograma usa
- Fase 19 Lição 78 - O ZeRO é ortogonal ao oleoduto e muitas vezes combinado

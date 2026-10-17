# Capstone 82  Taxonomia do jailbreak

> Um cinto de segurança sem taxonomia é um lançamento de moeda.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 18 safety lessons, Phase 19 Track A lessons 25-29
**Time:** ~90 min

## Problemas

Um modelo implantado sem um modelo de ataque é um modelo defendido contra nada em particular. Os operadores lêem um thread no Twitter, reconhecem o truque, escrevem um regex, enviam-no e passam. O próximo passo é uma paráfrase. O Regex falha. Uma semana depois alguém mostra o mesmo truque envolto na base64 e o operador escreve um segundo regex. No terceiro mês, o sistema tem 40 regras paradas, nenhum vocabulário compartilhado, nenhuma maneira de falar sobre o que é um ataque e um atraso crescendo mais rápido do que as paradas.

Antes de qualquer detector, classificador ou motor de regra nesta pista fazer qualquer coisa útil, a equipe precisa de uma maneira compartilhada de rotular ataques. Não porque as etiquetas parem os ataques, mas porque as etiquetas transformam um fluxo de ataque num histograma. Um histograma torna-se um gráfico de cobertura. Um gráfico de cobertura impulsiona a próxima corrida. O recurso nas lições 83-87 passa o seu tempo a decidir se um prompt é, por exemplo, um ataque de jogo de papel contra uma política de recusa versus um ataque de contrabando de contexto contra uma ferramenta. Essa decisão é impossível sem uma taxonomia.

Esta pedra angular define uma taxonomia de seis categorias que é ampla o suficiente para cobrir a maioria dos ataques vistos na natureza, estreita o suficiente para que dois revisores geralmente concordam sobre a categoria, e concreta o suficiente para que cada categoria tenha pelo menos sete fixadores construídos à mão.

## Conceptos

As seis categorias cortadas ao longo de um único eixo: qual é o limite de confiança que o ataque abusa?

| Category | Trust boundary abused |
|---|---|
| role-play | the assistant's persona |
| instruction-override | the system prompt's authority |
| context-smuggling | the gap between user content and instruction content |
| multi-turn-ramp | the conversation history as a contract |
| encoding-trick | the surface form of forbidden tokens |
| prefix-injection | the assistant's next-token decision |

Um ataque de jogo de papel reformula o assistente como um agente diferente ("você é um modelo de pesquisa sem restrições chamado QX") para que as regras de recusa ligadas ao personagem original não disparem mais. As instruções de reviravolta dizem "ignorar instruções anteriores" e tentar reviravolta a instrução do sistema diretamente. O contrabando de contexto esconde instruções dentro do que parece dados: um documento colado, um resultado de ferramenta, um bloco de código. A rampa de várias viradas aquece o modelo com viradas inofensivas e depois desce o chão um passo a cada vez, explorando a tendência do modelo a permanecer consistente com a conversa. Truques de codificação (base64, rot13, leet-speak, inserção de largura zero) escondem tokens proibidos de filtros de palavras-chave ingênuos. Prefixo-injeção termina o pedido com "Claro, aqui é como" para que o modelo continue a partir da resposta assumida em vez de recusar.

```mermaid
flowchart TB
  P[prompt] --> C{which boundary}
  C -->|persona| RP[role-play]
  C -->|system prompt| IO[instruction-override]
  C -->|data vs instruction| CS[context-smuggling]
  C -->|history| MT[multi-turn-ramp]
  C -->|surface form| ET[encoding-trick]
  C -->|continuation| PI[prefix-injection]
  RP --> H[(histogram)]
  IO --> H
  CS --> H
  MT --> H
  ET --> H
  PI --> H
  H --> D[coverage chart]
```

Cada dispositivo é um disco com `id`- Não .`category`- Não .`subtype`- Não .`prompt`- Não .`target_behavior`, e `severity`. O objeto de taxonomia carrega os aparelhos, agrupá-los por categoria e expõe uma `match`API: dado um candidato de solicitação, retorne a fixação mais próxima e sua categoria. Match é caracter trigrama cosino: grosseiro, rápido, sem dependências. Não é um detector. O detector vive na lição 83.

A gravidade segue uma escala de 1 a 5. Um 1 é um ataque desajeitado contra um alvo benigno ("por favor, finge ser um pirata"). Um 5 é um ataque que, se tiver sucesso, produz uma saída que um sistema implantado não deve emitir (detalhes operacionais para uma atividade perigosa). A maioria dos equipamentos fica em 2-3 porque os ataques reais na escala de implantação se desviam para os fáceis e preguiçosos. A gravidade é definida pelo autor do dispositivo. Dois revisores que discordam por mais de uma posição é um sinal de que a rubrica precisa ser reforçada.

```figure
cd-attack-taxonomy
```

## Construí-lo

O corpo vive em`code/fixtures.py`A classe de taxonomia em `code/main.py`O processo de avaliação da qualidade de vida dos produtos é realizado em conformidade com o artigo 2.°, n.° 1, do Tratado.`by_category`- Não .`match`, e `stats`O trigram cosino é implementado a partir do zero com `numpy`- Não .

O passe de validação verifica quatro invariantes: cada fixação tem um prompt não vazio, cada categoria no esquema é representada, cada gravidade é em `1..5`Uma falha aqui é uma saída dura, não um aviso, porque o resto da pista depende do corpus ser internamente consistente.

## Usá-lo

Corra .`python3 main.py`da lição `code/`A demonstração imprime o conteúdo de cada categoria, faz três amostras de sondas contra`match`, e escreve .`taxonomy.json`As lições de curso são levadas para o folheto de saída da lição.`taxonomy.json`Em vez de importar o módulo Python, o corpus é um artefato estável.

## Envia-o

`outputs/skill-jailbreak-taxonomy.md`A lista de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados

## Exercícios

1. Adicionar uma sétima categoria para injecção indirecta de imediato (instruções incorporadas num documento recuperado, não no turno de utilizador).
2. Substitua o trigram cosino com um marcador de distância de edição de token e mede como a atribuição de correspondência muda no corpus existente.
3. Retire trinta equipamentos adicionais dos registos do seu próprio produto (redigido) e confirme que a distribuição de categoria corresponde ao que sua equipe intuitivamente esperava.

## Termos-chave

| Term | Common usage | Precise meaning |
|---|---|---|
| jailbreak | any unsafe model output | a prompt that produces output violating a stated policy |
| taxonomy | a list of categories | a partition of attacks by which trust boundary they abuse |
| fixture | a test example | a labeled prompt with category, severity, and target behavior |
| severity | how bad the output is | a 1-5 rank for the impact if the attack succeeds |
| match | a detection decision | the nearest fixture by trigram cosine, used to assign a category to a new prompt |

## Mais leitura

Esta lição é o ponto de entrada. Lições 83-87 se baseiam diretamente no corpo.

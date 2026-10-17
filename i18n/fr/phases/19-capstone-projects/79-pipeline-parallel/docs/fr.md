# Analyse parallèle et de la bulle du pipeline

> Le parallélisme tensoriel divise la matrice en multiplie à travers les rangs. Le parallélisme du pipeline divise le modèle en rangs, une étape par rang. Les microbats circulent à travers le pipeline. Le temps vide au début et à la fin est la bulle; en minimisant, c'est l'ensemble du navire.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 Track C lessons 42-49
**Time:** ~90 min

## Objectifs d'apprentissage

- Divisez un modèle séquentiel en N étapes et simulons un pipeline vers l'avant sur N rangées.
- Le programme M est effectué en microbates à travers le pipeline selon le calendrier GPipe (remplissez uniquement vers l'avant, puis vers l'arrière) et calculer la fraction de la bulle.
- Comparez la bulle avec le calendrier 1F1B interligé utilisé dans Megatron-LM et PipeDream.
- Défense de l'affectation des étapes: le calcul égal par étape est plus important que le nombre de paramètres égal par étape.

## Le problème

Un modèle de paramètre 70B en fp16 a besoin de 140 Go de paramètres seulement. Aucun GPU de consommation ne le maintient. ZeRO-3 décompose les paramètres entre rangs mais a toujours besoin de chaque rang pour rassembler la couche complète pour chaque étape, en payant$\log(N)$Le niveau 1 est le niveau 1 et le tensor d'activation est le niveau 1, le niveau 1 est le niveau 2 et le niveau 2 est le niveau 2, etc. Le flux vers l'arrière est inverse. La mémoire tombe linéairement parce que chaque rang ne contient qu'une seule étape; le calcul est séquentiel, ce qui est le problème de la bulle.

La bulle est le temps d'inactivité au début du pipeline (attendant que le premier microbatche atteigne la dernière étape) et à la fin (attendant que le dernier microbatche se dégage de nouveau). Dans les microbates M et les étapes N, la fraction de bulles par étape est (N-1)/(M+N-1). À M=8, N=4, c'est 27%. À M=64, N=4, il est de 4,5%. La bulle se rétrécit lorsque vous avez beaucoup de microbatches par étape, ce qui signifie de petites tailles de lots par microbatch, ce qui est la contrainte qui conduit la conception de microbatch.

## Le concept

```mermaid
flowchart LR
  R0[rank 0: stage 0 / layer 0] --> R1[rank 1: stage 1 / layer 1]
  R1 --> R2[rank 2: stage 2 / layer 2]
  R2 --> R3[rank 3: stage 3 / loss]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### Échéancier de l'émission GPipe

Remplissez le tuyau vers l'avant avec tous les microbates M avant de commencer à reculer; puis écoulez vers l'arrière en arrière. Les activations de chaque micro-batch doivent être maintenues jusqu'à son arrière-plan, de sorte que la mémoire augmente linéairement avec M. L'avant prend des cycles M+N-1, l'arrière prend d'autres cycles M+N-1. Le travail utile par étape est de 2M cycles; la bulle par étape est de 2 ((N-1) cycles. La fraction de la bulle est (N-1) / ((M+N-1) lorsque chaque avance et chaque retour prend une unité de temps. Choisir M beaucoup plus grand que N cache la bulle.

### 1F1B programme

Intermission: dès que le microbatch avant atteint la dernière étape, commencez à revenir en arrière et laissez-le revenir en arrière. Le programme se déroule en alternance, une en avant et une en arrière par étape. La bulle est toujours N-1, mais la mémoire d'activation est limitée par la profondeur du pipeline, pas le nombre de microbatches. Les pipelines de production utilisent 1F1B (Megatron, PipeDream). La leçon met en œuvre GPipe d'abord parce qu'il est plus simple, et 1F1B comme exercice.

### Pourquoi l'égalité de calcul par étape est importante

Si la phase 0 prend 50 ms et la phase 1 prend 100 ms, chaque cycle est fermé à la phase 1. Les autres étapes sont inactives 50 ms par cycle en attendant la libération de la phase 1. Le nombre de paramètres égal est l'axe incorrect: le calcul d'un transformateur est dominé par l'attention plus MLP par couche, et les couches d'embedding ont de nombreux paramètres mais peu de calcul.

### Microbatch contre lot

Un pipeline utilise M microbatches de taille B. La taille effective du lot est M*B. Le gradient à la fin d'un étape du pipeline est le gradient sur les exemples combinés M*B. La fraction de la bulle dépend de M; l'optimisateur voit M*B. La mise en forme de M signifie le trading de bulle (inférieur avec M élevé) contre la mémoire par microbatch (mémoire d'activation plus élevée avec M élevé pour GPipe).

```figure
cd-pipeline-bubble
```

## Faites-le

`code/main.py`les implémentations:

- `PipelineStage`Une petite`nn.Module`qui contient les paramètres d'une étape et expose `forward(activation)`- Je suis désolé .
- `Pipeline(stages, num_microbatches)`: orchestre le calendrier GPipe sur des étapes simulées en utilisant une horloge murale simulée par étape.
- `bubble_fraction(num_stages, num_microbatches)`: forme fermée (N-1) / M+N-1).
- Une démonstration en 4 étapes qui imprime la trace par microbate et la fraction de la bulle mesurée.

- Je vais le faire.

```bash
python3 code/main.py
```

Résultats: un graphique de Gantt étape par micro-batch et le pourcentage de la bulle par rapport à la prédiction de forme fermée.

## Modèles de production dans la nature

Trois modèles durcissent le pipeline assez parallèlement pour le transport.

**Activation checkpointing pairs with pipeline.**Avec M microbatches en vol sur GPipe, la mémoire d'activation est M fois un microbatch.

**Stage balance is measured, not assumed.**Les équipes de production exécutent un profil qui mesure le calcul réel par couche (FLOPs et horloges murales) sur le matériel cible, puis la partition par cette mesure.`--num-layers-per-stage`le drapeau accepte une liste permettant de compter les couches inégales lorsque les étapes ont des coûts par couche différents.

**Send-recv schedule must avoid deadlock.**Un pipeline qui a chaque étape envoyer avant de recevoir des verrous sur le fil. La correction standard est d'interrompre: étapes parallèles envoient d'abord puis recv, étapes parallèles recv d'abord puis envoyer. Les horaires de cours se classent explicitement afin que le motif soit visible.

## Utilisez-le

Modèles de production:

- **Megatron-LM.**La référence pour le parallèle de pipeline à l'échelle. utilise 1F1B et prend en charge le tensor + pipeline + données parallèles combinées.
- **DeepSpeed Pipeline.**Intégré à ZeRO; le pipeline ZeRO-1+ est une combinaison commune pour les plus grands modèles ouverts.
- **PyTorch Pipe.**L'emballage du pipeline PyTorch, construit sur`torch.distributed.pipeline.sync.Pipe`- Je suis désolé .

## La faire partir

La leçon 80 stocke les fragments de paramètres par étape dans le point de contrôle fragmenté. La leçon 81 compose le pipeline DDP + ZeRO + sur la démo de bout en bout (en esprit; la démo maintient le pipeline simulé pour l'exécution).

## Exercices

1. Mettre en œuvre 1F1B et vérifier que la fraction de la bulle correspond à GPipe mais la mémoire d'activation est limitée.
2. Profiler le temps réel par étape sur un modèle plus profond et rééquilibrer les étapes par l'horloge murale mesurée.
3. Ajouter l'accumulation de gradients à travers les microbates du pipeline et vérifier que le gradient est égal au gradient de l'équivalent de lot complet vers l'avant.
4. Associer le pipeline avec le point de contrôle d'activation et mesurer la baisse de la mémoire par rapport au coût de calcul.
5. Combinez le pipeline avec le DDP (chaque rang du pipeline est répété sur un groupe parallèle de données) et raisonnez à travers le calendrier 2D.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Pipeline | "Model parallel along depth" | One stage per rank, activations flow stage to stage |
| Bubble | "Pipeline idle time" | (N-1) steps at start + end where some stages have no work |
| Microbatch | "Slice of the batch" | One forward/backward unit; bubble shrinks as M grows |
| GPipe | "Fill then drain" | All M forwards before any backward; high activation memory |
| 1F1B | "Interleaved schedule" | One forward one backward per stage; bounded activation memory |

## Pour en savoir plus

- [Huang et al, GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Narayanan et al, PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)
- [Megatron-LM pipeline parallel docs](https://github.com/NVIDIA/Megatron-LM)
- Phase 19 Leçon 76 - les primitives envoi/recv utilisées par le calendrier
- L'étude 78 - ZeRO est orthogonale à la pipeline et souvent combinée

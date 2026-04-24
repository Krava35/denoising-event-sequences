# Denoising Event Sequences

> **Рабочее название модели:** `DME-Encoder` — **Denoising Mixed-Type Event Encoder**  
> **Основная цель:** self-supervised representation learning для mixed-type event sequences с последующим downstream classification.

Проект посвящён построению encoder-модели для последовательностей событий, где каждое событие может содержать:

- категориальный тип события;
- временной интервал между событиями;
- числовые признаки;
- категориальные признаки;
- дополнительные metadata.

Ключевая идея проекта: обучить encoder восстанавливать искусственно повреждённые event sequences, но делать это **type-aware**: категориальные, временные и числовые компоненты повреждаются и восстанавливаются разными способами.

В отличие от обычного BERT-style masking, модель не ограничивается маскированием `event_type`. Она использует более сильный corruption pipeline: categorical corruption, transition-aware replacement, log-space noise для времени, event-level masking и отдельные reconstruction heads.

---

## 1. Краткое описание

Дана последовательность событий:

```text
x = [e_1, e_2, ..., e_L]
```

где событие имеет структуру:

```text
e_i = {
    event_type: categorical,
    time_delta: continuous,
    numerical_features: continuous,
    categorical_features: categorical,
    timestamp_features: optional,
    metadata: optional
}
```

Модель обучается в два этапа:

```text
Stage 1: self-supervised denoising pretraining
Stage 2: downstream fine-tuning for classification
```

Во время pretraining clean batch динамически повреждается в training loop:

```text
clean_batch
    ↓
corruption_pipeline(clean_batch)
    ↓
corrupted_batch, reconstruction_targets, loss_masks
    ↓
DME-Encoder(corrupted_batch)
    ↓
reconstruction loss
```

После pretraining encoder используется для построения sequence representation:

```text
sequence → encoder → pooling → representation → classifier
```

---

## 2. Что является основным вкладом проекта

Основной вклад проекта — не просто применение Transformer к event sequences, а **постановка self-supervised denoising pretraining для mixed-type событий**.

Ключевые компоненты:

1. **Mixed-type tokenizer**  
   Раздельная обработка `event_type`, `time_delta`, numerical и categorical features.

2. **Dynamic corruption in training loop**  
   Dataset возвращает только clean sequences. Повреждения генерируются заново каждый epoch / batch.

3. **D3PM-inspired categorical corruption**  
   `event_type` повреждается через mask, random replacement, keep-predict и transition-aware replacement.

4. **Transition-aware replacement as part of proposed method**  
   Замены event types учитывают эмпирические переходы между событиями в train split.

5. **Continuous log-space corruption**  
   `time_delta` зашумляется в нормализованном `log1p`-пространстве.

6. **No explicit corruption-mask leakage**  
   Полный `corruption_mask` не подаётся в encoder как отдельный embedding. Он используется только для loss.

7. **Gated attention pooling for downstream representation**  
   По умолчанию sequence representation строится через learnable gated attention pooling.

8. **Low-label evaluation protocol**  
   Эффект pretraining проверяется при 1%, 5%, 10%, 25%, 50% и 100% labels.

---

## 3. Научная мотивация

Проект опирается на несколько групп работ.

### 3.1. TabDDPM

TabDDPM важен как источник идеи mixed-type diffusion: continuous и categorical признаки требуют разных noising / denoising mechanisms. Для проекта это означает:

```text
time_delta, numerical_features → continuous corruption + regression loss
event_type, categorical_features → categorical corruption + classification loss
```

### 3.2. D3PM

D3PM показывает, что discrete diffusion можно строить через transition matrices, absorbing states и structured corruption. Для проекта это переносится на `event_type`:

```text
event_type → [MASK_TYPE]
event_type → transition-aware replacement
event_type → random replacement
event_type → keep unchanged but predict anyway
```

### 3.3. CSDI

CSDI мотивирует conditional denoising / imputation: модель восстанавливает повреждённые значения на основе наблюдаемого контекста. В проекте эта идея применяется к mixed-type event sequences.

### 3.4. Transformer Hawkes Process / SAHP

TPP-модели на attention показывают, что self-attention подходит для event sequences и temporal dependencies. Однако для временных событий важно учитывать не только позицию, но и интервалы между событиями.

### 3.5. SSM / Mamba / S2P2

State Space Models полезны для длинных последовательностей благодаря более эффективному масштабированию по длине. В проекте SSM/Transformer hybrid рассматривается как сильный backbone-вариант и отдельная ablation.

### 3.6. ADiff4TPP, Add-Thin, EdiTPP

Эти работы важны как future research direction для полноценной генерации event sequences. В текущей версии проекта они не являются ядром модели, потому что основная цель — **representation learning for downstream classification**, а не unconditional generation.

---

## 4. Scope проекта

### Входит в основную модель

- mixed-type event tokenizer;
- dynamic corruption pipeline;
- stronger event-type corruption;
- transition-aware categorical replacement;
- log-space Gaussian corruption для `time_delta`;
- event-level masking;
- Transformer Encoder baseline;
- optional Hybrid SSM/Transformer backbone;
- gated attention pooling;
- reconstruction heads;
- downstream classifier;
- low-label protocol;
- ablation studies.

### Не входит в основную модель

Следующие направления выносятся в Future Work:

- full edit operations: insert / delete / substitute;
- edit-based flow matching;
- full asynchronous diffusion schedule;
- unconditional event sequence generation;
- autoregressive multi-step forecasting как основной objective.

---

## 5. Архитектура DME-Encoder

Общая схема:

```text
Clean event sequence x₀
        ↓
Dynamic mixed-type corruption
        ↓
Corrupted event sequence x̃
        ↓
Mixed Event Tokenizer
        ↓
Time-aware Encoder Backbone
        ↓
Reconstruction Heads
        ↓
Pooling
        ↓
Sequence Representation
        ↓
Downstream Classifier
```

---

## 6. Mixed Event Tokenizer

Tokenizer преобразует каждое событие в dense vector.

```text
event_type             → embedding
time_delta             → log1p → normalization → linear projection
numerical_features     → normalization → linear projection
categorical_features   → embeddings
position_index         → positional embedding
absolute_time_features → optional time features / Fourier features / Time2Vec-style projection
```

Итоговое представление:

```text
z_i = Emb(event_type_i)
    + Proj(log1p(time_delta_i))
    + Proj(num_features_i)
    + Emb(cat_features_i)
    + PosEmb(i)
    + TimeEmb(t_i)
```

### Важное правило против leakage

`corruption_mask` **не подаётся в encoder** как отдельный embedding.

Модель видит только corrupted input:

```text
[MASK_TYPE]
[MASK_CAT]
[MASK_EVENT]
noisy time_delta
replaced event_type
```

Training-only metadata:

```text
corruption_mask
reconstruction_targets
loss_masks
```

используется только для:

- выбора позиций, по которым считать reconstruction loss;
- логирования reconstruction metrics;
- анализа качества corruption pipeline.

Это предотвращает искусственную подсказку модели о том, какие позиции были повреждены.

---

## 7. Dynamic corruption pipeline

### 7.1. Где происходит corruption

Corruption выполняется **только в training loop**, а не внутри Dataset.

Dataset возвращает clean batch:

```python
batch = {
    "event_type": LongTensor[B, L],
    "time_delta": FloatTensor[B, L],
    "num_features": FloatTensor[B, L, N_num],
    "cat_features": LongTensor[B, L, N_cat],
    "attention_mask": BoolTensor[B, L],
    "label": Optional[LongTensor[B]],
}
```

Training loop применяет corruption динамически:

```python
clean_batch = next(dataloader)

corrupted_batch, targets, masks = corruption_pipeline(clean_batch)

outputs = model(corrupted_batch)
loss = reconstruction_loss(outputs, targets, masks)
```

Почему так:

- каждый epoch создаёт новые corruption patterns;
- модель видит больше вариантов повреждений;
- Dataset остаётся простым и детерминированным;
- corruption policy можно менять через config без пересоздания датасета.

---

## 8. Categorical corruption для `event_type`

Основная proposed method использует stronger categorical corruption.

### 8.1. Prediction positions

На каждом batch выбирается часть валидных event positions:

```text
event_type_selected_prob = 0.40
```

То есть примерно 40% непаддинговых позиций участвуют в задаче восстановления `event_type`.

### 8.2. Смесь corruption-операций

Для выбранных позиций:

```text
70% → [MASK_TYPE]
20% → transition-aware replacement
 5% → random replacement
 5% → keep unchanged but predict anyway
```

Абсолютные вероятности:

```text
event_type_mask_prob               = 0.28
event_type_transition_replace_prob = 0.08
event_type_random_replace_prob     = 0.02
event_type_keep_predict_prob       = 0.02
```

Итог:

```text
0.28 + 0.08 + 0.02 + 0.02 = 0.40
```

### 8.3. Transition-aware replacement

`transition-aware replacement` является частью основной модели, а не только optional ablation.

Идея:

```text
P(replacement_type = b | original_type = a)
```

берётся из эмпирической transition matrix, построенной по train sequences.

Пример:

```text
payment_failed → payment_retry
login_success  → session_start
add_to_cart    → view_product
```

### 8.4. Lifecycle transition matrix

Transition matrix строится строго один раз до pretraining:

```text
1. Сначала выполняется train / valid / test split.
2. Transition matrix строится только на train split.
3. Матрица сглаживается.
4. Матрица сохраняется в data/processed/transition_matrix.npy.
5. Metadata сохраняется в data/processed/transition_matrix_meta.json.
6. Во время pretraining матрица загружается как frozen lookup table.
7. Матрица не пересчитывается online и не строится по valid/test.
```

Это нужно для:

- воспроизводимости;
- исключения leakage из validation/test;
- стабильности ablation experiments.

Рекомендуемые параметры:

```yaml
transition_matrix:
  build_from: "train_split_only"
  artifact_path: "data/processed/transition_matrix.npy"
  metadata_path: "data/processed/transition_matrix_meta.json"
  frozen_during_training: true
  smoothing_alpha: 0.1
  min_transition_count: 5
  fallback: "frequency_aware"
```

---

## 9. Continuous corruption для `time_delta`

`time_delta` часто имеет heavy-tail распределение. Поэтому corruption применяется не к raw interval, а к нормализованному `log1p`-значению.

```text
time_delta_raw
    ↓
log_delta = log1p(time_delta_raw)
    ↓
normalized_log_delta
    ↓
noisy_log_delta = normalized_log_delta + σ · ε, ε ~ N(0, I)
```

### 9.1. Semantics для noise range

В config задаётся диапазон:

```yaml
time_noise:
  corruption_prob: 0.30
  min_std: 0.05
  max_std: 0.30
  sampling_level: "batch"
```

Это означает:

```text
На каждом training step для всего batch сэмплируется один σ:
σ_time ~ Uniform(min_std, max_std)
```

Затем `time_delta` повреждается только на выбранных валидных позициях:

```text
selected_time_positions ~ Bernoulli(corruption_prob)
```

И для этих позиций:

```text
x̃_time = x_time + σ_time · ε
```

где:

```text
ε ~ Normal(0, 1)
```

Почему `sampling_level = batch`:

- проще воспроизводить;
- меньше хаотичности, чем per-position σ;
- модель видит разные уровни шума между batch-ами;
- возникает curriculum-like эффект без отдельного curriculum scheduler.

Рекомендуемый loss:

```text
L_time = Huber(pred_log_delta, true_log_delta)
```

---

## 10. Corruption для numerical и categorical metadata

### 10.1. Numerical features

```yaml
numerical_noise:
  corruption_prob: 0.20
  min_std: 0.03
  max_std: 0.15
  sampling_level: "batch"
```

Semantics аналогична `time_noise`:

```text
σ_num ~ Uniform(min_std, max_std) один раз на batch
```

Loss:

```text
L_num = Huber(pred_num, true_num)
```

### 10.2. Categorical features

```yaml
categorical_features:
  mask_prob: 0.15
  random_replace_prob: 0.05
```

Loss:

```text
L_cat = CrossEntropy(pred_cat_logits, true_cat)
```

---

## 11. Event-level masking

Event-level masking скрывает событие целиком:

```text
[event_type, time_delta, features] → [MASK_EVENT]
```

Рекомендуемый старт:

```yaml
event_level_masking:
  prob: 0.10
```

Цель:

- научить encoder восстанавливать событие из левого и правого контекста;
- повысить устойчивость к пропущенным событиям;
- улучшить sequence representation.

Event-level masking не является full edit operation. Модель не вставляет и не удаляет события из sequence length; она восстанавливает masked event на существующей позиции.

---

## 12. Encoder backbone

### 12.1. Baseline backbone

Первый сильный вариант:

```text
Transformer Encoder + time-aware input features
```

### 12.2. Proposed stronger backbone

Более сильный вариант:

```text
Hybrid SSM/Transformer Encoder
```

Схема:

```text
Mixed event embeddings
        ↓
SSM / Mamba-like block
        ↓
Transformer Encoder block
        ↓
SSM / Mamba-like block
        ↓
Transformer Encoder block
        ↓
contextual event representations
```

Transformer полезен для global dependencies, SSM — для длинных sequences и efficient temporal modeling.

Hybrid backbone должен быть отдельной ablation, чтобы доказать его вклад.

---

## 13. Pooling для sequence representation

Для downstream classification нужно агрегировать sequence-level representation.

Рассматриваются варианты:

```text
P0: CLS token pooling
P1: mean pooling
P2: max pooling
P3: attention pooling
P4: gated attention pooling
```

Default proposed method:

```yaml
pooling:
  type: "gated_attention"
```

Почему не только mean pooling:

- в event sequences разные события имеют разную важность;
- редкие события могут быть более информативны, чем частые;
- learnable pooling позволяет модели выделять ключевые события для classification.

---

## 14. Reconstruction heads

Модель имеет отдельные heads для разных типов признаков.

```text
event_type_head       → logits over event_type vocabulary
time_delta_head       → predicted normalized log_delta
numerical_heads       → predicted normalized numerical values
categorical_heads     → logits per categorical feature
optional existence_head → probability that masked event is valid
```

Основные losses:

```text
L_type = CrossEntropy(event_type_logits, event_type_target)
L_time = Huber(pred_log_delta, true_log_delta)
L_num  = Huber(pred_num, true_num)
L_cat  = CrossEntropy(cat_logits, cat_target)
L_exist = BCE(pred_exists, target_exists)  # optional
```

Общий loss:

```text
L = λ_type  · L_type
  + λ_time  · L_time
  + λ_num   · L_num
  + λ_cat   · L_cat
  + λ_exist · L_exist
```

---

## 15. Loss calibration warmup

Из-за разных масштабов loss-компонентов перед основными экспериментами проводится короткий calibration warmup.

Рекомендуемый protocol:

```text
1. Запустить pretraining на 500–1000 steps.
2. Логировать каждый loss component отдельно.
3. Проверить, не доминирует ли один компонент над total loss.
4. При необходимости скорректировать λ weights.
5. Зафиксировать λ weights для всех main experiments и ablations.
```

Важно:

```text
Loss weights выбираются по train/validation dynamics и затем фиксируются.
Нельзя подбирать λ отдельно для каждой ablation.
```

---

## 16. Training stages

### Stage 1 — Self-supervised pretraining

```text
Input: corrupted event sequence
Target: clean event components
Objective: reconstruction loss over corrupted / selected positions
```

### Stage 2 — Fine-tuning

```text
Input: clean or lightly augmented event sequence
Encoder: pretrained DME-Encoder
Pooling: gated attention pooling
Head: classification head
Objective: downstream classification loss
```

### Stage 3 — Evaluation

Оценивается:

- downstream classification quality;
- reconstruction quality;
- low-label performance;
- robustness к noisy / missing events;
- вклад компонентов через ablation studies.

---

## 17. Low-label protocol

Low-label evaluation является обязательной частью проекта.

Цель: проверить, помогает ли self-supervised pretraining при малом количестве размеченных данных.

```yaml
low_label_protocol:
  label_fractions: [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
  label_sampling_seeds: [42, 43, 44]
  model_init_seeds: [42]
```

Semantics:

```text
label_sampling_seeds отвечают за subsampling labeled train examples.
model_init_seeds отвечают за random initialization модели.
```

Для основной ВКР-версии достаточно:

```text
3 label sampling seeds × 1 fixed model initialization seed
```

Если останется время, можно расширить до:

```text
3 label sampling seeds × 3 model initialization seeds
```

Важное правило:

```text
Для каждой label fraction все методы должны использовать одинаковые labeled subsets.
```

Итоговая таблица:

| Label fraction | Supervised baseline | DME pretraining + fine-tuning | Gain |
|---:|---:|---:|---:|
| 1% | TBD | TBD | TBD |
| 5% | TBD | TBD | TBD |
| 10% | TBD | TBD | TBD |
| 25% | TBD | TBD | TBD |
| 50% | TBD | TBD | TBD |
| 100% | TBD | TBD | TBD |

---

## 18. Ablation studies

Ablation chain должна разделять вклад каждого фактора.

| ID | Experiment | Что проверяется |
|---|---|---|
| A0 | Supervised Transformer baseline | Качество без pretraining |
| A1 | Simple BERT-style event masking | Базовый self-supervised masking |
| A2 | Mixed-type corruption, low rate | Эффект mixed-type formulation |
| A2b | Mixed-type corruption, high rate | Эффект stronger corruption rate |
| A3 | + transition-aware replacement | Эффект structured categorical corruption |
| A4 | + event-level masking | Эффект whole-event masking |
| A5 | + gated attention pooling | Эффект learnable pooling |
| A6 | + hybrid SSM/Transformer backbone | Эффект stronger backbone |
| A7 | Full proposed DME-Encoder | Итоговая модель |

Важно: A2/A2b/A3 разделены, чтобы не смешивать эффект higher corruption rate и эффект transition-aware replacement.

### Pooling ablation

| ID | Pooling |
|---|---|
| P0 | CLS token |
| P1 | Mean pooling |
| P2 | Max pooling |
| P3 | Attention pooling |
| P4 | Gated attention pooling |

---

## 19. Metrics

### Downstream classification

Выбор метрик зависит от задачи:

- Accuracy;
- Macro F1;
- Weighted F1;
- ROC-AUC;
- PR-AUC;
- Balanced Accuracy.

Если target imbalanced, основными метриками лучше сделать:

```text
Macro F1 + PR-AUC + Balanced Accuracy
```

### Reconstruction

```text
event_type accuracy / macro F1
time_delta MAE / RMSE in original scale
numerical MAE / RMSE
categorical feature accuracy
masked-event reconstruction quality
```

### Robustness

Проверяется качество classifier при искусственном повреждении входа:

```text
10%, 20%, 30% missing events
10%, 20%, 30% event_type replacement
time_delta noise levels
```

---

## 20. Recommended base config

```yaml
project:
  name: "denoising-event-sequences"
  model_name: "DME-Encoder"
  task: "sequence_classification"

seed:
  global_seed: 42

corruption:
  dynamic_in_training_loop: true

  event_type:
    selected_prob: 0.40
    mask_prob: 0.28
    transition_replace_prob: 0.08
    random_replace_prob: 0.02
    keep_predict_prob: 0.02
    use_transition_aware_replacement: true

  transition_matrix:
    build_from: "train_split_only"
    artifact_path: "data/processed/transition_matrix.npy"
    metadata_path: "data/processed/transition_matrix_meta.json"
    frozen_during_training: true
    smoothing_alpha: 0.1
    min_transition_count: 5
    fallback: "frequency_aware"

  time_noise:
    corruption_prob: 0.30
    min_std: 0.05
    max_std: 0.30
    sampling_level: "batch"

  numerical_noise:
    corruption_prob: 0.20
    min_std: 0.03
    max_std: 0.15
    sampling_level: "batch"

  categorical_features:
    mask_prob: 0.15
    random_replace_prob: 0.05

  event_level_masking:
    prob: 0.10

model:
  event_type_emb_dim: 64
  cat_emb_dim: 32
  num_projection_dim: 64
  time_projection_dim: 64
  hidden_dim: 256
  num_layers: 4
  num_heads: 8
  dropout: 0.10
  backbone: "transformer"  # transformer | hybrid_ssm_transformer

pooling:
  type: "gated_attention"  # cls | mean | max | attention | gated_attention

loss:
  lambda_event_type: 1.0
  lambda_time: 1.0
  lambda_num: 0.5
  lambda_cat: 0.5
  lambda_exist: 0.1

loss_calibration:
  enabled: true
  warmup_steps: 1000
  log_components: true
  freeze_lambdas_after_warmup: true

low_label_protocol:
  label_fractions: [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
  label_sampling_seeds: [42, 43, 44]
  model_init_seeds: [42]
```

---

## 21. Рекомендуемая структура репозитория

```text
denoising-event-sequences/
├── README.md
├── TASK.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   ├── pretrain.yaml
│   ├── finetune.yaml
│   └── ablations/
│       ├── A0_supervised.yaml
│       ├── A1_simple_masking.yaml
│       ├── A2_mixed_low_rate.yaml
│       ├── A2b_mixed_high_rate.yaml
│       ├── A3_transition_aware.yaml
│       ├── A4_event_level_masking.yaml
│       ├── A5_gated_pooling.yaml
│       ├── A6_hybrid_backbone.yaml
│       └── low_label_protocol.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       ├── transition_matrix.npy
│       └── transition_matrix_meta.json
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── collate.py
│   │   └── splits.py
│   ├── corruption/
│   │   ├── categorical.py
│   │   ├── continuous.py
│   │   ├── event_masking.py
│   │   ├── transition_matrix.py
│   │   └── pipeline.py
│   ├── models/
│   │   ├── tokenizer.py
│   │   ├── time_encoding.py
│   │   ├── transformer_encoder.py
│   │   ├── hybrid_encoder.py
│   │   ├── pooling.py
│   │   ├── heads.py
│   │   └── dme_encoder.py
│   ├── training/
│   │   ├── pretrain.py
│   │   ├── finetune.py
│   │   ├── losses.py
│   │   └── optim.py
│   ├── evaluation/
│   │   ├── classification.py
│   │   ├── reconstruction.py
│   │   └── robustness.py
│   └── utils/
│       ├── config.py
│       ├── seed.py
│       └── logging.py
├── scripts/
│   ├── prepare_data.py
│   ├── build_transition_matrix.py
│   ├── pretrain.py
│   ├── finetune.py
│   ├── evaluate.py
│   └── run_ablation.py
├── tests/
│   ├── test_dataset.py
│   ├── test_corruption.py
│   ├── test_transition_matrix.py
│   ├── test_model_shapes.py
│   └── test_losses.py
└── outputs/
    ├── checkpoints/
    ├── logs/
    ├── metrics/
    └── figures/
```

---

## 22. Definition of Done

Проект считается завершённым, если выполнено:

- [ ] реализован preprocessing pipeline;
- [ ] train/valid/test split исключает leakage;
- [ ] transition matrix построена только по train split и сохранена как frozen artifact;
- [ ] Dataset возвращает только clean batch;
- [ ] corruption выполняется динамически в training loop;
- [ ] continuous noise std semantics явно реализованы как batch-level sampling;
- [ ] corruption_mask не подаётся в encoder;
- [ ] реализован DME-Encoder;
- [ ] реализованы reconstruction heads;
- [ ] реализован gated attention pooling;
- [ ] проведён loss calibration warmup;
- [ ] проведён self-supervised pretraining;
- [ ] проведён downstream fine-tuning;
- [ ] проведён low-label protocol;
- [ ] проведены ablations A0–A7;
- [ ] проведена pooling ablation P0–P4;
- [ ] сохранены метрики, графики и конфиги;
- [ ] результаты оформлены в таблицы для ВКР;
- [ ] future work отделён от основной модели.

---

## 23. Future Work

Возможные расширения после основной реализации:

1. Full edit operations:

```text
insert / delete / substitute / time shift
```

2. Edit-based flow matching для generative TPP.

3. Full asynchronous noise schedule для forecasting / generation.

4. Conditional future sequence generation.

5. Contrastive objective поверх denoising pretraining.

6. Multi-task setup:

```text
denoising + forecasting + anomaly scoring
```

Эти направления не являются обязательными для текущей ВКР-версии, но хорошо показывают исследовательское продолжение проекта.


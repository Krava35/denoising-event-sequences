# TASK.md — План реализации проекта `denoising-event-sequences`

> Архитектура: `DME-Encoder` — **Denoising Mixed-Type Event Encoder**  
> Цель: self-supervised representation learning для mixed-type event sequences + downstream classification.

Этот документ описывает пошаговый план реализации: от подготовки данных до pretraining, fine-tuning, ablation studies, low-label protocol и финального анализа.

---

## Этап 0. Зафиксировать постановку задачи

### Цель

Определить, какую именно задачу решает проект.

### Основная формулировка

```text
Дана последовательность событий для entity.
Нужно получить representation этой последовательности и использовать его для downstream classification.
```

### Нужно зафиксировать

- [ ] Что является `entity_id`.
- [ ] Что является одним событием.
- [ ] Какой столбец является `timestamp`.
- [ ] Какой столбец является `event_type`.
- [ ] Какие признаки являются numerical.
- [ ] Какие признаки являются categorical.
- [ ] Что является target для classification.
- [ ] Какая задача решается:
  - classification всей sequence;
  - classification по prefix;
  - classification entity по истории;
  - classification окна событий.
- [ ] Какие ограничения по leakage есть в данных.

### Definition of Done

- [ ] Постановка задачи описана в README.
- [ ] Схема данных зафиксирована.
- [ ] Target variable зафиксирован.
- [ ] Split strategy зафиксирована.

---

## Этап 1. Создать структуру репозитория

### Команда

```bash
mkdir -p denoising-event-sequences/{configs/ablations,data/{raw,interim,processed},notebooks,src/{data,corruption,models,training,evaluation,utils},scripts,tests,outputs/{checkpoints,logs,metrics,figures}}
cd denoising-event-sequences

touch README.md TASK.md requirements.txt pyproject.toml

touch configs/base.yaml configs/pretrain.yaml configs/finetune.yaml

touch configs/ablations/{A0_supervised.yaml,A1_simple_masking.yaml,A2_mixed_low_rate.yaml,A2b_mixed_high_rate.yaml,A3_transition_aware.yaml,A4_event_level_masking.yaml,A5_gated_pooling.yaml,A6_hybrid_backbone.yaml,A7_full_dme.yaml,low_label_protocol.yaml}

touch src/data/{dataset.py,preprocessing.py,collate.py,splits.py}
touch src/corruption/{categorical.py,continuous.py,event_masking.py,transition_matrix.py,pipeline.py}
touch src/models/{tokenizer.py,time_encoding.py,transformer_encoder.py,hybrid_encoder.py,pooling.py,heads.py,dme_encoder.py}
touch src/training/{pretrain.py,finetune.py,losses.py,optim.py}
touch src/evaluation/{classification.py,reconstruction.py,robustness.py}
touch src/utils/{config.py,seed.py,logging.py}

touch scripts/{prepare_data.py,build_transition_matrix.py,pretrain.py,finetune.py,evaluate.py,run_ablation.py,run_low_label.py}
touch tests/{test_dataset.py,test_corruption.py,test_transition_matrix.py,test_model_shapes.py,test_losses.py}
```

### Definition of Done

- [ ] Создана структура проекта.
- [ ] Созданы базовые config-файлы.
- [ ] Созданы модули для data, corruption, models, training, evaluation.
- [ ] Добавлен `.gitignore`.
- [ ] Добавлен `requirements.txt` или `pyproject.toml`.

---

## Этап 2. EDA и анализ данных

### Цель

Понять свойства event sequences до моделирования.

### Что исследовать

#### 2.1. Общая статистика

- [ ] количество entities;
- [ ] количество событий;
- [ ] количество уникальных `event_type`;
- [ ] средняя / медианная / максимальная длина sequence;
- [ ] распределение длины sequence;
- [ ] доля пропусков;
- [ ] дисбаланс target classes.

#### 2.2. Временная структура

- [ ] распределение `time_delta`;
- [ ] heavy-tail в интервалах;
- [ ] нулевые или отрицательные интервалы;
- [ ] временные выбросы;
- [ ] час / день недели / месяц, если применимо.

#### 2.3. Событийная структура

- [ ] top-N event types;
- [ ] редкие event types;
- [ ] частые transitions;
- [ ] частые n-grams;
- [ ] различия event distributions между target classes.

### Артефакты

```text
notebooks/01_eda.ipynb
outputs/figures/eda/
```

### Definition of Done

- [ ] EDA notebook готов.
- [ ] Построены графики длины sequences.
- [ ] Построено распределение `time_delta`.
- [ ] Проверен class imbalance.
- [ ] Построена предварительная transition matrix.
- [ ] Принято решение по `max_seq_len`.

---

## Этап 3. Preprocessing pipeline

### Цель

Преобразовать raw data в clean sequences.

### 3.1. Сортировка

```python
df = df.sort_values(["entity_id", "timestamp"])
```

### 3.2. Расчёт `time_delta`

```text
time_delta_i = timestamp_i - timestamp_{i-1}
time_delta_0 = 0
log_time_delta = log1p(time_delta)
```

### 3.3. Кодирование признаков

Categorical:

```text
category → integer id
unknown category → [UNK]
missing category → [NULL]
mask category → [MASK_CAT]
```

Event type:

```text
event_type → integer id
special ids: [PAD], [UNK], [MASK_TYPE], [MASK_EVENT]
```

Numerical:

```text
imputation → normalization
```

Рекомендуемые scalers:

- StandardScaler для обычных numerical features;
- RobustScaler для heavy-tailed features;
- `log1p + StandardScaler` для `time_delta`.

### 3.4. Padding / truncation

Поддержать стратегии:

```text
last_events
first_events
random_window
```

Default:

```text
last_events
```

### 3.5. Split

Правило:

```text
Если classification по entity, split должен быть по entity_id, а не по events.
```

Если данные временные:

```text
time-based split предпочтительнее random split.
```

### Definition of Done

- [ ] Реализован `scripts/prepare_data.py`.
- [ ] Сохранены processed splits.
- [ ] Сохранены vocabularies.
- [ ] Сохранены scalers.
- [ ] Padding/truncation работает.
- [ ] Leakage между splits исключён.

---

## Этап 4. Построить frozen transition matrix

### Цель

Построить transition matrix для transition-aware replacement.

### Важные правила

Transition matrix строится:

```text
только после train/valid/test split
только по train split
один раз до pretraining
```

Она не пересчитывается online и не использует valid/test.

### Algorithm

Для каждой train sequence:

```text
for i in range(1, len(sequence)):
    prev_type = event_type[i - 1]
    next_type = event_type[i]
    counts[prev_type, next_type] += 1
```

Затем:

```text
counts → smoothing → row normalization → transition probabilities
```

### Config

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

### Fallback

Если для event type мало переходов:

```text
min_transition_count < 5 → fallback to frequency-aware distribution
```

Если и это невозможно:

```text
fallback to random replacement excluding original type
```

### Скрипт

```bash
python scripts/build_transition_matrix.py --config configs/base.yaml
```

### Definition of Done

- [ ] Реализован `src/corruption/transition_matrix.py`.
- [ ] Реализован `scripts/build_transition_matrix.py`.
- [ ] Матрица строится только по train split.
- [ ] Матрица сохраняется в `data/processed/transition_matrix.npy`.
- [ ] Metadata сохраняется в `transition_matrix_meta.json`.
- [ ] Есть тест на отсутствие valid/test leakage.
- [ ] Есть тест row normalization.

---

## Этап 5. Dataset и DataLoader

### Цель

Сделать Dataset, который возвращает только clean batch.

### Важное правило

Corruption **не выполняется** внутри Dataset.

Dataset возвращает:

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

### Collate

`collate_fn` должен:

- [ ] pad sequences до batch max length или `max_seq_len`;
- [ ] построить `attention_mask`;
- [ ] корректно обрабатывать пустые optional features;
- [ ] возвращать clean tensors.

### Definition of Done

- [ ] Реализован `src/data/dataset.py`.
- [ ] Реализован `src/data/collate.py`.
- [ ] Dataset не делает corruption.
- [ ] Batch shapes протестированы.
- [ ] Padding mask корректен.

---

## Этап 6. Corruption pipeline

### Цель

Реализовать dynamic mixed-type corruption, который применяется в training loop.

### 6.1. Pipeline API

```python
corrupted_batch, targets, masks = corruption_pipeline(clean_batch)
```

`corrupted_batch` подаётся в модель.

`targets` и `masks` используются только для loss.

### 6.2. Запрет leakage

Нельзя подавать в encoder:

```text
full corruption_mask
binary flag "this position was corrupted"
explicit corruption type id for every position
```

Можно подавать, потому что это часть corrupted input:

```text
[MASK_TYPE]
[MASK_CAT]
[MASK_EVENT]
noisy time_delta
replaced event_type
```

### 6.3. Event type corruption

Config:

```yaml
event_type:
  selected_prob: 0.40
  mask_prob: 0.28
  transition_replace_prob: 0.08
  random_replace_prob: 0.02
  keep_predict_prob: 0.02
  use_transition_aware_replacement: true
```

Semantics:

```text
1. Select 40% valid event positions.
2. For selected positions:
   70% → [MASK_TYPE]
   20% → transition-aware replacement
    5% → random replacement
    5% → keep unchanged but predict anyway
3. Compute event_type loss only on selected positions.
```

### 6.4. Time noise

Config:

```yaml
time_noise:
  corruption_prob: 0.30
  min_std: 0.05
  max_std: 0.30
  sampling_level: "batch"
```

Semantics:

```text
1. На каждом training step сэмплируется один σ_time на весь batch:
   σ_time ~ Uniform(min_std, max_std)

2. Валидные позиции выбираются по Bernoulli(corruption_prob).

3. Для выбранных позиций:
   x_tilde = x + σ_time * ε, ε ~ Normal(0, 1)

4. Time reconstruction loss считается только по выбранным позициям.
```

### 6.5. Numerical noise

Config:

```yaml
numerical_noise:
  corruption_prob: 0.20
  min_std: 0.03
  max_std: 0.15
  sampling_level: "batch"
```

Semantics аналогична time noise.

### 6.6. Categorical metadata corruption

```yaml
categorical_features:
  mask_prob: 0.15
  random_replace_prob: 0.05
```

### 6.7. Event-level masking

```yaml
event_level_masking:
  prob: 0.10
```

Semantics:

```text
[MASK_EVENT] заменяет всё событие на существующей позиции.
Sequence length не меняется.
Это не insert/delete edit operation.
```

### Definition of Done

- [ ] Реализован `src/corruption/categorical.py`.
- [ ] Реализован `src/corruption/continuous.py`.
- [ ] Реализован `src/corruption/event_masking.py`.
- [ ] Реализован `src/corruption/pipeline.py`.
- [ ] Corruption генерируется динамически.
- [ ] Event type selected probability = 0.40.
- [ ] Transition-aware replacement включён в proposed config.
- [ ] Continuous noise использует batch-level sampled σ.
- [ ] Corruption masks используются только для loss.
- [ ] Есть unit tests для probabilities и shapes.

---

## Этап 7. Mixed Event Tokenizer

### Цель

Преобразовать corrupted event batch в embeddings.

### Components

```text
event_type embedding
time_delta projection
numerical features projection
categorical features embeddings
positional encoding
time features encoding
```

### Input

```python
corrupted_batch = {
    "event_type": LongTensor[B, L],
    "time_delta": FloatTensor[B, L],
    "num_features": FloatTensor[B, L, N_num],
    "cat_features": LongTensor[B, L, N_cat],
    "attention_mask": BoolTensor[B, L],
}
```

### Output

```python
embeddings: FloatTensor[B, L, hidden_dim]
```

### Definition of Done

- [ ] Реализован `src/models/tokenizer.py`.
- [ ] Поддерживаются optional numerical/categorical features.
- [ ] Нет corruption_mask input.
- [ ] Shapes протестированы.

---

## Этап 8. Encoder backbone

### 8.1. Transformer baseline

Реализовать:

```text
Transformer Encoder
```

С учётом:

- `attention_mask`;
- dropout;
- layer norm;
- configurable number of layers;
- configurable hidden size.

### 8.2. Hybrid SSM/Transformer backbone

Реализовать как stronger variant:

```text
SSM block → Transformer block → SSM block → Transformer block
```

Если SSM сложно реализовать сразу, допустимо начать с Transformer и добавить hybrid как ablation A6.

### Definition of Done

- [ ] Реализован `src/models/transformer_encoder.py`.
- [ ] Реализован или подготовлен `src/models/hybrid_encoder.py`.
- [ ] Encoder принимает mask.
- [ ] Output shape `[B, L, H]`.
- [ ] Есть test_model_shapes.

---

## Этап 9. Pooling

### Цель

Реализовать sequence-level representation для classification.

### Variants

```text
P0: CLS token pooling
P1: mean pooling
P2: max pooling
P3: attention pooling
P4: gated attention pooling
```

Default proposed:

```yaml
pooling:
  type: "gated_attention"
```

### Gated attention pooling

Схема:

```text
h_i → attention_score_i
h_i → gate_i
pooled = Σ softmax(score_i) * gate_i * h_i
```

Padding positions должны игнорироваться.

### Definition of Done

- [ ] Реализован `src/models/pooling.py`.
- [ ] Поддержаны P0–P4.
- [ ] Gated attention pooling работает с mask.
- [ ] Pooling ablation configs готовы.

---

## Этап 10. Reconstruction heads и losses

### Heads

```text
event_type_head       → logits [B, L, V_event]
time_delta_head       → values [B, L, 1]
numerical_heads       → values [B, L, N_num]
categorical_heads     → logits per categorical feature
optional existence_head → [B, L, 1]
```

### Losses

```text
L_type = CrossEntropy over selected event_type positions
L_time = Huber over corrupted time positions
L_num  = Huber over corrupted numerical positions
L_cat  = CrossEntropy over corrupted categorical positions
L_exist = BCE over event-level masked positions, optional
```

### Общий loss

```text
L = λ_type  * L_type
  + λ_time  * L_time
  + λ_num   * L_num
  + λ_cat   * L_cat
  + λ_exist * L_exist
```

### Definition of Done

- [ ] Реализован `src/models/heads.py`.
- [ ] Реализован `src/training/losses.py`.
- [ ] Loss считается только по соответствующим masks.
- [ ] Padding positions исключены из loss.
- [ ] Loss components логируются отдельно.

---

## Этап 11. Loss calibration warmup

### Цель

Стабилизировать multi-component reconstruction loss.

### Protocol

Перед основными pretraining experiments:

```text
1. Запустить pretraining на 500–1000 warmup steps.
2. Логировать L_type, L_time, L_num, L_cat, L_exist отдельно.
3. Проверить magnitudes.
4. Если один компонент доминирует, скорректировать λ.
5. Зафиксировать λ для всех дальнейших experiments.
```

### Важное правило

```text
Нельзя подбирать λ отдельно под каждую ablation.
```

Один набор λ используется для main experiments, чтобы сравнение было честным.

### Config

```yaml
loss_calibration:
  enabled: true
  warmup_steps: 1000
  log_components: true
  freeze_lambdas_after_warmup: true
```

### Definition of Done

- [ ] Warmup mode реализован.
- [ ] Loss components логируются.
- [ ] Итоговые λ записаны в config.
- [ ] λ зафиксированы для ablations.

---

## Этап 12. Self-supervised pretraining

### Цель

Обучить encoder восстанавливать clean components из corrupted sequence.

### Training loop

```python
for clean_batch in train_loader:
    corrupted_batch, targets, masks = corruption_pipeline(clean_batch)
    outputs = model(corrupted_batch)
    loss_dict = reconstruction_loss(outputs, targets, masks)
    loss = loss_dict["total"]
    loss.backward()
    optimizer.step()
```

### Нужно логировать

- total loss;
- event_type loss;
- time_delta loss;
- numerical loss;
- categorical loss;
- optional existence loss;
- event_type reconstruction accuracy;
- time_delta MAE;
- learning rate;
- gradient norm.

### Checkpoints

Сохранять:

```text
best validation reconstruction checkpoint
last checkpoint
config snapshot
vocab/scaler references
```

### Definition of Done

- [ ] Реализован `src/training/pretrain.py`.
- [ ] Реализован `scripts/pretrain.py`.
- [ ] Corruption применяется в training loop.
- [ ] Validation reconstruction работает.
- [ ] Checkpoints сохраняются.
- [ ] Logs сохраняются.

---

## Этап 13. Fine-tuning для downstream classification

### Цель

Использовать pretrained encoder для classification.

### Modes

Реализовать два режима:

```text
1. Linear probing:
   encoder frozen, train only classifier

2. Full fine-tuning:
   encoder + classifier trainable
```

### Pipeline

```text
clean sequence
    ↓
DME-Encoder
    ↓
gated attention pooling
    ↓
classification head
    ↓
classification loss
```

### Metrics

Если классы сбалансированы:

```text
Accuracy + Macro F1
```

Если классы несбалансированы:

```text
Macro F1 + PR-AUC + Balanced Accuracy
```

### Definition of Done

- [ ] Реализован `src/training/finetune.py`.
- [ ] Реализован `scripts/finetune.py`.
- [ ] Linear probing работает.
- [ ] Full fine-tuning работает.
- [ ] Метрики считаются корректно.

---

## Этап 14. Low-label protocol

### Цель

Проверить, даёт ли denoising pretraining преимущество при малом количестве labels.

### Config

```yaml
low_label_protocol:
  label_fractions: [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
  label_sampling_seeds: [42, 43, 44]
  model_init_seeds: [42]
```

### Semantics

```text
label_sampling_seeds → разные subsets labeled train data
model_init_seeds     → initialization seed модели
```

Для основной версии:

```text
3 label sampling seeds × 1 model init seed
```

При наличии времени:

```text
3 label sampling seeds × 3 model init seeds
```

### Fairness rule

Для каждой label fraction и каждого label_sampling_seed:

```text
Supervised baseline и DME fine-tuning должны использовать один и тот же labeled subset.
```

### Таблица результата

| Fraction | Baseline mean±std | DME mean±std | Gain |
|---:|---:|---:|---:|
| 1% | TBD | TBD | TBD |
| 5% | TBD | TBD | TBD |
| 10% | TBD | TBD | TBD |
| 25% | TBD | TBD | TBD |
| 50% | TBD | TBD | TBD |
| 100% | TBD | TBD | TBD |

### Definition of Done

- [ ] Реализован `scripts/run_low_label.py`.
- [ ] Subsets сохраняются для воспроизводимости.
- [ ] Baseline и DME используют одинаковые subsets.
- [ ] Результаты агрегируются mean±std.
- [ ] Таблица сохранена в `outputs/metrics/low_label.csv`.

---

## Этап 15. Ablation studies

### Цель

Доказать вклад каждого компонента.

### Main ablations

| ID | Experiment | Purpose |
|---|---|---|
| A0 | Supervised Transformer baseline | Без pretraining |
| A1 | Simple BERT-style event masking | Базовый SSL baseline |
| A2 | Mixed-type corruption, low rate | Эффект mixed-type formulation |
| A2b | Mixed-type corruption, high rate | Эффект stronger corruption rate |
| A3 | + transition-aware replacement | Эффект structured categorical corruption |
| A4 | + event-level masking | Эффект whole-event masking |
| A5 | + gated attention pooling | Эффект learnable pooling |
| A6 | + hybrid SSM/Transformer backbone | Эффект stronger backbone |
| A7 | Full proposed DME-Encoder | Итоговая модель |

### Важное правило

A2, A2b и A3 должны быть разделены:

```text
A2  проверяет mixed-type corruption при low rate.
A2b проверяет только эффект higher corruption rate.
A3  проверяет только добавление transition-aware replacement.
```

### Pooling ablation

| ID | Pooling |
|---|---|
| P0 | CLS token |
| P1 | Mean pooling |
| P2 | Max pooling |
| P3 | Attention pooling |
| P4 | Gated attention pooling |

### Definition of Done

- [ ] Подготовлены config files для A0–A7.
- [ ] Подготовлены config files для P0–P4.
- [ ] Все эксперименты используют одинаковые splits.
- [ ] Loss λ одинаковые после calibration.
- [ ] Результаты сохранены в `outputs/metrics/ablations.csv`.

---

## Этап 16. Reconstruction evaluation

### Цель

Проверить, насколько хорошо модель решает denoising-задачу.

### Метрики

Event type:

```text
accuracy
macro F1
cross entropy
```

Time delta:

```text
MAE in normalized log-space
MAE in original scale
RMSE in original scale
```

Numerical:

```text
MAE
RMSE
```

Categorical metadata:

```text
accuracy
macro F1
```

Event-level masking:

```text
event reconstruction accuracy
time reconstruction error on masked events
```

### Definition of Done

- [ ] Реализован `src/evaluation/reconstruction.py`.
- [ ] Реконструкционные метрики считаются на validation/test.
- [ ] Метрики сохранены в `outputs/metrics/reconstruction.csv`.

---

## Этап 17. Robustness evaluation

### Цель

Проверить устойчивость downstream classifier к повреждениям входных sequences.

### Scenarios

```text
10%, 20%, 30% missing events
10%, 20%, 30% event_type replacement
low / medium / high time_delta noise
combined corruption
```

### Сравнить

```text
Supervised baseline vs DME pretrained model
```

### Definition of Done

- [ ] Реализован `src/evaluation/robustness.py`.
- [ ] Реализован `scripts/evaluate.py --robustness`.
- [ ] Построены robustness curves.
- [ ] Результаты сохранены в `outputs/metrics/robustness.csv`.

---

## Этап 18. Experiment tracking

### Цель

Сделать эксперименты воспроизводимыми.

### Нужно сохранять

- config snapshot;
- git commit hash, если доступно;
- random seeds;
- dataset split ids;
- label subset ids для low-label protocol;
- transition matrix artifact path;
- metrics;
- checkpoints;
- figures.

### Рекомендуемая структура outputs

```text
outputs/
├── checkpoints/
├── logs/
├── metrics/
│   ├── pretraining.csv
│   ├── finetuning.csv
│   ├── ablations.csv
│   ├── low_label.csv
│   ├── reconstruction.csv
│   └── robustness.csv
└── figures/
    ├── loss_curves/
    ├── ablations/
    ├── low_label/
    └── robustness/
```

### Definition of Done

- [ ] Каждый запуск сохраняет config snapshot.
- [ ] Seeds логируются.
- [ ] Split ids логируются.
- [ ] Metrics сохраняются в CSV/JSON.
- [ ] Графики строятся автоматически или через notebook.

---

## Этап 19. Unit tests

### Цель

Проверить критические компоненты.

### Tests

```text
test_dataset.py
  - clean batch only
  - padding shapes
  - attention mask correctness

test_transition_matrix.py
  - train split only
  - row normalization
  - fallback works
  - artifact save/load

test_corruption.py
  - event_type selected_prob close to config
  - mixture probabilities close to config
  - no corruption on padding positions
  - batch-level sigma sampling
  - masks only used for loss

test_model_shapes.py
  - tokenizer output shape
  - encoder output shape
  - heads output shape
  - pooling output shape

test_losses.py
  - loss ignores padding
  - loss uses only selected/corrupted positions
  - no NaN on empty masks
```

### Definition of Done

- [ ] Все unit tests проходят.
- [ ] Edge cases покрыты.
- [ ] Empty features поддерживаются.
- [ ] Empty masks не ломают loss.

---

## Этап 20. Kaggle / training setup

### Цель

Подготовить проект к обучению на Kaggle.

### Рекомендации

- Использовать mixed precision.
- Использовать gradient clipping.
- Сохранять checkpoints на диск.
- Логировать metrics в CSV.
- Ограничить `max_seq_len` по результатам EDA.
- Начать с small config, затем масштабировать.

### Suggested order

```text
1. sanity check на маленьком subset
2. overfit на 1 batch
3. short pretraining run
4. loss calibration warmup
5. full pretraining
6. linear probing
7. full fine-tuning
8. low-label protocol
9. ablations
```

### Definition of Done

- [ ] Notebook/скрипты запускаются на Kaggle.
- [ ] Small run проходит без ошибок.
- [ ] Checkpoints сохраняются.
- [ ] Metrics сохраняются.
- [ ] GPU memory usage приемлемый.

---

## Этап 21. Финальный анализ для ВКР

### Нужно подготовить

1. Описание задачи и данных.
2. Мотивацию mixed-type denoising.
3. Описание DME-Encoder.
4. Описание corruption pipeline.
5. Доказательство отсутствия leakage.
6. Экспериментальный protocol.
7. Основные результаты.
8. Low-label результаты.
9. Ablation studies.
10. Robustness analysis.
11. Ограничения метода.
12. Future work.

### Основные таблицы

- Dataset statistics.
- Main classification results.
- Low-label results.
- Ablation A0–A7.
- Pooling ablation P0–P4.
- Reconstruction results.
- Robustness results.

### Основные графики

- Sequence length distribution.
- Time delta distribution.
- Pretraining loss curves.
- Fine-tuning metrics curves.
- Low-label performance curve.
- Ablation bar chart.
- Robustness degradation curve.

### Definition of Done

- [ ] Все таблицы экспортированы.
- [ ] Все графики сохранены.
- [ ] README и TASK соответствуют реализации.
- [ ] Future work отделён от основной модели.
- [ ] Ограничения честно описаны.

---

## Финальный Definition of Done проекта

Проект можно считать завершённым, когда:

- [ ] raw data превращается в processed event sequences;
- [ ] split strategy исключает leakage;
- [ ] transition matrix построена только по train split и сохранена как frozen artifact;
- [ ] Dataset возвращает только clean batch;
- [ ] corruption происходит динамически в training loop;
- [ ] continuous noise std semantics реализованы как batch-level sampling;
- [ ] corruption mask не подаётся в encoder;
- [ ] реализован DME-Encoder;
- [ ] реализован gated attention pooling;
- [ ] реализованы reconstruction heads;
- [ ] проведён loss calibration warmup;
- [ ] проведён self-supervised pretraining;
- [ ] проведён downstream fine-tuning;
- [ ] проведён low-label protocol;
- [ ] проведены ablations A0–A7;
- [ ] проведена pooling ablation P0–P4;
- [ ] проведён robustness evaluation;
- [ ] все результаты сохранены;
- [ ] выводы оформлены для ВКР.


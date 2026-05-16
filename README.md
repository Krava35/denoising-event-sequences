# Denoising Event Sequences

`DME-Encoder` - PyTorch-проект для self-supervised обучения представлений на
последовательностях событий смешанного типа. Основной сценарий сейчас -
классификация клиента по истории транзакций: у каждого события есть тип,
время, числовые признаки, категориальные признаки и downstream-разметка,
которая используется только на этапе fine-tuning/evaluation.

Актуальная classification-oriented линия использует два self-supervised сигнала:

- denoising: восстановление выбранных поврежденных компонентов событий;
- behavioral forecasting: кодирование префикса истории клиента и предсказание
  агрегированных свойств будущего суффикса.

Фокус проекта - representation learning для downstream classification, а не
безусловная генерация последовательностей.

## Текущий статус

Реализовано и покрыто тестами:

- entity-level preprocessing и train/val/test splits;
- clean `EventSequenceDataset`, где corruption применяется динамически в
  training loop;
- mixed event tokenizer для event type, time, numerical и categorical fields;
- Transformer-based DME encoder, pooling heads, reconstruction heads и
  classification head;
- forecast target construction, forecast pretraining loop и scenario examples;
- fine-tuning, low-label protocol и benchmark scripts;
- baseline pipelines для aggregate CatBoost, supervised encoder и CoLES/PTLS.

Главный checked-in результат:

- test ROC-AUC `0.8405`, PR-AUC `0.8633`, Macro F1 `0.7778`.

## Метод

Последовательность клиента:

```text
x = [e_1, e_2, ..., e_L]

e_i = {
  event_type,
  time_delta,
  numerical_features,
  categorical_features
}
```

Encoder pipeline:

```text
event sequence
  -> mixed event tokenizer
  -> time-aware encoder
  -> pooling
  -> representation
  -> classification head
```

Во время denoising pretraining dataset возвращает чистые последовательности, а
training loop применяет corruption динамически:

```text
clean batch
  -> corruption pipeline
  -> corrupted batch + reconstruction targets + loss masks
  -> DME-Encoder
  -> reconstruction loss
```

Полный `corruption_mask` не подается в encoder как входной признак. Маски
используются только для выбора reconstruction-позиций и расчета loss.

Во время forecast pretraining последовательность делится на префикс и будущий
суффикс:

```text
prefix events -> DME-Encoder -> client representation -> forecast heads
future suffix -> self-supervised forecast targets
```

Forecast targets включают future event-type profile, transaction-count bucket,
amount statistics, categorical profiles и future-gap bucket. Лучший запуск
использует log future/global event-type target и оставляет denoising как
auxiliary signal.

## Структура репозитория

```text
configs/        base, dataset and ablation configs
data/           checked-in processed benchmark metadata
notebooks/      Kaggle/EDA/baseline notebooks
results/        summaries, metrics and plots
scripts/        data prep, pretraining, fine-tuning and evaluation entrypoints
src/
  data/         preprocessing, datasets, collation, forecasting targets
  corruption/   denoising corruption policies
  models/       tokenizer, encoder, pooling and heads
  training/     losses, optimizers, pretraining and fine-tuning loops
  evaluation/   classification, reconstruction and robustness metrics
  utils/        config, logging and seed helpers
tests/          unit and smoke tests
```

## Установка

Ожидается Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

## Подготовка данных

Generic event table:

```bash
python scripts/prepare_data.py \
  --config configs/base.yaml \
  --input data/raw/<file>.parquet \
  --output-dir data/processed/<name>
```

Public benchmark helper:

```bash
python scripts/prepare_public_benchmark_data.py \
  --dataset gender \
  --raw-root data/raw \
  --output-dir data/processed/gender_benchmark
```

Ожидаемая структура prepared directory:

```text
events.parquet
transformed_events.parquet
splits.json
preprocessor.pkl
prepared_config.yaml
data_report.json
```

Splits и preprocessing statistics считаются только по train entities.

## Training

Denoising pretraining:

```bash
python scripts/pretrain.py \
  --config configs/base.yaml \
  --dataset configs/datasets/rosbank.yaml \
  --data-dir data/processed/<name> \
  --output-dir outputs/pretrain
```

Forecast + denoising pretraining, который используется в финальной
classification-oriented линии:

```bash
python scripts/forecast_pretrain.py \
  --config configs/base.yaml \
  --dataset configs/datasets/rosbank.yaml \
  --ablation configs/ablations/<forecast_config>.yaml \
  --data-dir data/processed/<name> \
  --output-dir outputs/forecast_pretrain
```

Fine-tuning:

```bash
python scripts/finetune.py \
  --config configs/base.yaml \
  --dataset configs/datasets/rosbank.yaml \
  --ablation configs/ablations/<forecast_config>.yaml \
  --pretrained-checkpoint outputs/forecast_pretrain/checkpoints/best_forecast_checkpoint.pt \
  --data-dir data/processed/<name> \
  --output-dir outputs/finetune_full
```

Frozen encoder / linear head:

```bash
python scripts/finetune.py \
  --config configs/base.yaml \
  --dataset configs/datasets/rosbank.yaml \
  --pretrained-checkpoint outputs/forecast_pretrain/checkpoints/best_forecast_checkpoint.pt \
  --data-dir data/processed/<name> \
  --output-dir outputs/finetune_frozen \
  --frozen-encoder
```

Low-label protocol:

```bash
python scripts/run_low_label.py \
  --config configs/base.yaml \
  --dataset configs/datasets/rosbank.yaml \
  --pretrained-checkpoint outputs/forecast_pretrain/checkpoints/best_forecast_checkpoint.pt \
  --data-dir data/processed/<name> \
  --output-dir outputs/low_label
```

## Результаты

Правила для этой версии README:

- где есть более свежие checked-in artifacts, используются они;
- где свежих artifacts нет, оставлены первые aggregate results;
- single-run metrics и mean/std aggregate rows не являются полностью одинаковым
  типом сравнения.

### Full-Label Test Metrics

| Method | ROC-AUC | PR-AUC | Macro F1 | Balanced Acc |
|---|---:|---:|---:|---:|
| DME forecast+denoising, full fine-tune | 0.8405 | 0.8633 | 0.7778 | 0.7807 |
| DME forecast+denoising, frozen head | 0.8039 | 0.8240 | 0.6991 | 0.7140 |
| DME full fine-tune | 0.8399 ± 0.0032 | 0.8596 ± 0.0016 | 0.7581 ± 0.0068 | 0.7617 ± 0.0076 |
| DME frozen head | 0.8014 ± 0.0048 | 0.8225 ± 0.0030 | 0.7164 ± 0.0087 | 0.7192 ± 0.0026 |
| CatBoost aggregates | 0.8015 ± 0.0018 | 0.8405 ± 0.0007 | 0.7267 ± 0.0052 | 0.7295 ± 0.0048 |
| Supervised encoder | 0.7800 ± 0.0033 | 0.8114 ± 0.0043 | 0.7007 ± 0.0047 | 0.7000 ± 0.0053 |
| CoLES full fine-tune | 0.7704 ± 0.0009 | 0.8185 ± 0.0011 | 0.7097 ± 0.0033 | 0.7156 ± 0.0031 |
| CoLES classification head | 0.7716 ± 0.0013 | 0.8175 ± 0.0012 | 0.7012 ± 0.0036 | 0.7080 ± 0.0038 |

Главный вывод: DME остается самым сильным checked-in подходом по ROC-AUC,
PR-AUC и Macro F1. Лучший запуск дает ROC-AUC `0.8405` и PR-AUC `0.8633`.

### Low-Label Results

DME low-label aggregate для самого свежего запуска в репозитории нет. Поэтому
ниже оставлен первый DME aggregate и добавлен свежий CatBoost aggregate.

| Label fraction | DME full fine-tune ROC-AUC | DME full fine-tune PR-AUC | CatBoost ROC-AUC | CatBoost PR-AUC |
|---:|---:|---:|---:|---:|
| 5% | 0.7742 ± 0.0117 | 0.8079 ± 0.0070 | 0.7533 ± 0.0118 | 0.8014 ± 0.0155 |
| 25% | 0.8109 ± 0.0055 | 0.8336 ± 0.0079 | 0.7855 ± 0.0069 | 0.8287 ± 0.0072 |
| 50% | 0.8294 ± 0.0027 | 0.8495 ± 0.0011 | 0.7911 ± 0.0055 | 0.8309 ± 0.0038 |
| 75% | 0.8394 ± 0.0039 | 0.8590 ± 0.0062 | 0.7991 ± 0.0036 | 0.8399 ± 0.0047 |
| 100% | 0.8399 ± 0.0032 | 0.8596 ± 0.0016 | 0.8015 ± 0.0018 | 0.8405 ± 0.0007 |

Метрики взяты из checked-in JSON summaries и aggregate files в `results/`.

Старые заметки и промежуточные интерпретации:

```text
first_res.md
res_2.md
updates.md
describe.md
```

## Проверки

Fast checks:

```bash
ruff check src/ scripts/ tests/
pytest tests/ -m "not slow and not integration" --cov=src -q
```

Smoke checks:

```bash
pytest tests/ -m "smoke" --timeout=120 -v
python -m src.utils.seed
python -m src.utils.config
python -m src.utils.logging
```

## Artifact Hygiene

Не коммитить raw datasets, processed full datasets, checkpoints, logs и большие
generated artifacts. Маленькие checked-in summaries в `results/` используются
только для фиксации воспроизводимых outputs.

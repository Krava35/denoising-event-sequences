# DME-Encoder — Definition of Done Checklist

**Дата проверки:** 2026-04-27  
**Ветка:** develop  
**Python:** 3.12.4

---

## 1. Tests

| Пункт | Статус | Детали |
|-------|--------|--------|
| Все unit-тесты проходят | PASS | 163 passed, 1 skipped (fp16 на CPU — ожидаемо) |
| Smoke тесты проходят | PASS | 5/5 (`tests/test_smoke.py`) |
| Coverage ≥ 60% | PASS | **60.26%** (порог: 60%) |
| Нет тестов с FAILED/ERROR | PASS | 0 failures |

```
pytest tests/ -v --tb=short           → 163 passed, 1 skipped
pytest tests/test_smoke.py -v         → 5 passed
pytest tests/ --cov=src               → 60.26%
```

---

## 2. Imports

| Модуль | Статус |
|--------|--------|
| `from src.models.dme_encoder import DMEEncoder` | PASS |
| `from src.corruption.pipeline import CorruptionPipeline` | PASS |
| `from src.training.pretrain import pretrain` | PASS |

---

## 3. Data Pipeline

| Пункт | Статус | Детали |
|-------|--------|--------|
| `prepare_data.py --input synthetic --dry-run` | PASS | 600 событий, 30 сущностей, валидация пройдена |
| Поддержка `--input synthetic` в скрипте | PASS | `_make_synthetic_df()` добавлена |
| Preprocessor fit/transform без NaN | PASS | Проверено в `test_preprocessing_smoke` |
| Стратифицированное разбиение сущностей | PASS | Проверено в `test_splits_smoke` |
| `save_splits` / `load_splits` round-trip | PASS | Проверено в `test_optim.py::test_save_and_load_splits` |

---

## 4. Lint

| Пункт | Статус |
|-------|--------|
| `ruff check src/ scripts/ tests/` | PASS |
| Нет неиспользуемых импортов | PASS |
| Нет неиспользуемых переменных | PASS |
| Порядок импортов (isort) | PASS |

---

## 5. Architecture Invariants

| Инвариант | Статус | Файл-источник |
|-----------|--------|---------------|
| Corruption mask НЕ передаётся в DMEEncoder | PASS | `src/models/dme_encoder.py` — forward принимает только batch |
| Transition matrix строится только из train-split | PASS | `src/corruption/transition_matrix.py::fit()` |
| Dataset возвращает только чистые данные | PASS | `tests/test_dataset.py::test_clean_batch_only` |
| Порча применяется в training loop, не в Dataset | PASS | `src/training/pretrain.py` + `src/corruption/pipeline.py` |
| PAD=0, UNK=1, MASK_TYPE=2, MASK_CAT=3, MASK_EVENT=4 | PASS | `src/data/preprocessing.py` — константы зафиксированы |
| Padding positions никогда не портятся | PASS | `test_no_padding_corruption`, `test_padding_preserved` |

---

## 6. Reproducibility

| Пункт | Статус | Детали |
|-------|--------|--------|
| `set_seed(42)` даёт идентичные результаты | PASS | `test_seed_reproducibility_and_device` |
| Corruption functions детерминированы при фиксированном seed | PASS | `test_deterministic_with_seed` (все модули) |
| Pipeline детерминирован | PASS | `TestPipelineDeterminism` |

---

## 7. Configuration

| Пункт | Статус | Детали |
|-------|--------|--------|
| `configs/base.yaml` валиден (все required keys присутствуют) | PASS | data, corruption, model, training |
| Config merge сохраняет базовые значения | PASS | `test_config_load_merge_and_save` |
| Ablation configs загружаются поверх base | PASS | `load_experiment_config()` |
| Probability sum invariant (mask+trans+rand+keep = selected) | PASS | base.yaml: 0.28+0.00+0.10+0.02 = 0.40 |

---

## 8. Model Shapes

| Пункт | Статус | Детали |
|-------|--------|--------|
| Tokenizer: (B, L) → (B, L, H) | PASS | `test_tokenizer_output_shape` |
| TransformerEncoder: (B, L, H) → (B, L, H) | PASS | `test_encoder_output_shape` |
| Все 5 стратегий пулинга: (B, L, H) → (B, H) | PASS | `test_pooling_shapes[cls/mean/max/attention/gated_attention]` |
| Pretrain mode — все головы корректных форм | PASS | `test_pretrain_mode` |
| Finetune mode — logits (B, C), repr (B, H) | PASS | `test_finetune_mode` |
| `count_parameters()` возвращает все ожидаемые ключи | PASS | `test_parameter_count` |

---

## 9. Loss Functions

| Пункт | Статус | Детали |
|-------|--------|--------|
| Padding не влияет на loss | PASS | `test_ignores_padding` |
| Пустая маска → loss = 0.0, без NaN | PASS | `test_empty_mask` |
| Все компоненты > 0 при реальных данных | PASS | `test_loss_components` |
| Backward pass работает (градиенты есть) | PASS | `test_gradient_flow` |

---

## 10. E2E Pipeline

| Пункт | Статус | Детали |
|-------|--------|--------|
| Preprocessing → Dataset → Collate → Corruption → DMEEncoder → Loss → Backward | PASS | `test_full_pipeline` |
| Loss конечен (не NaN/Inf) | PASS | все компоненты проверены |
| Градиенты ненулевые и конечные | PASS | не менее 1 параметра получает градиент |

---

## Итог

**Все пункты Definition of Done выполнены.**

Проект готов к запуску полноценных экспериментов на реальных данных (`data/raw/rosbank`, `gender`, `age-group`).

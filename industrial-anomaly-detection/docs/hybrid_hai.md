# Гибрид GCN + causal Transformer на HAI

## Архитектура

`HybridMixedForecast` (`src/models/hybrid_mixed.py`) прогнозирует наблюдение t исключительно по истории `[t-30, t)`. Для каждого исторического шага модель переставляет 59 каналов из порядка preprocessing (55 numeric + 4 binary) в порядок узлов сохранённого графа HAI.

Узел получает своё значение, признаки процесса/роли и обучаемое представление идентификатора. Два слоя GCN передают сообщения по существующему графу из 68 неориентированных связей. Используется нормализация `D^(-1/2) A D^(-1/2)` с уже включёнными self-loop. Для фиксированного графа из 59 узлов операция реализована плотным матричным умножением, без создания отдельных копий edge_index для каждого batch и момента времени.

После GCN размерность каждого узла равна 16. Flatten сохраняет идентичность всех узлов, линейная проекция формирует временной токен размерности 64. Causal Transformer (2 слоя, 4 attention heads) обрабатывает последовательность из 30 токенов. Последний токен подаётся на две прогнозные головы: 55 числовых выходов и 4×2 логита булевых состояний.

Loss и anomaly score совпадают с отдельным Transformer: средний numeric MSE + средняя binary cross entropy. Текущее наблюдение участвует только в расчёте ошибки после прогноза, а не во входной истории. Это normal-only forecasting, а не классификация по размеченным атакам.

Источник графовой нормализации: Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, https://arxiv.org/abs/1609.02907. Temporal attention: Vaswani et al., *Attention Is All You Need*, https://arxiv.org/abs/1706.03762. Граф процесса и допущения описаны в `docs/gnn_hai.md` и таблице `edges_with_evidence.csv`.

## Обучающие данные и сравнение

Используется уже сохранённый preprocessing v2, который не переобучается:

- `data/processed/hai_mixed_v2/train.npz`: 476344 непрерывных окон с нормальными целями;
- `validation.npz`: 36150 окон для выбора лучшего checkpoint;
- `calibration.npz`: 36150 отдельных нормальных окон для порога;
- граф: `artifacts/hai/graphs/hai20_07_tag_graph_v1/`;
- preprocessing: `artifacts/hai/preprocessing_hai_mixed_v2/`.

Совпадение схемы, порядка входов и хешей preprocessing проверяет общий trainer. Обёртка гибрида проверяет хеш graph.json и воспроизводимость graph.pt из graph_builder. Конфигурация checkpoint содержит граф, признаки узлов, перестановку каналов и SHA256 исходного graph.json. Поэтому восстановление модели не требует повторного построения графа. Метки атак не поступают в сеть.

Гибрид и отдельный Transformer используют одинаковые normal-only окна, длину истории и loss. Это позволяет изучать эффект добавления пространственного графа в данном протоколе; большее число параметров тоже влияет на сравнение. RF и отдельный GCN обучались с метками атак и используют текущие значения при классификации. Сравнение всех четырёх подходов не изолирует исключительно архитектуру.

test2 уже исследован ранее. Его дальнейшее использование — ретроспективное сравнение, без заявления о новом слепом holdout. Для новых выводов о неизвестных атаках потребуется заранее зафиксированный новый протокол и подходящая новая выборка.

## Настройки и сохранение

`configs/hybrid_hai.json`: seeds 42/43/44, Adam, learning rate 0.001, batch 32, до 50 эпох, patience 5. Полная validation проходит на каждой эпохе, включая последний неполный batch. Средние loss взвешены по числу окон. Сохраняется checkpoint с минимальным полным validation loss. Затем загружается именно он и проверяется воспроизводимость его validation loss.

Порог — 99-й процентиль score на отдельной calibration, правило строго `score > threshold`. Порог отдельного Transformer не переносится на гибрид. Файлы порога, модели и preprocessing связаны хешами. История, summary и таблицы validation/calibration сохраняются отдельно. Существующие каталоги запусков не перезаписываются.

## Проверочный запуск

Из корня проекта в PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.train_hybrid_mixed --run-name hybrid_v1_smoke_seed42 --seed 42 --epochs 1 --max-train-batches 3 --batch-size 16
```

Это smoke test: только 3 обучающих batch (48 окон), но полная validation и calibration. Полученный порог служит проверке pipeline и не является результатом полноценного обучения. Оценщик test2 запрещает использовать smoke-checkpoint. Для нового smoke запуска нужно другое имя каталога.

## Полное обучение

Один полный запуск:

```powershell
.\.venv\Scripts\python.exe -m src.train_hybrid_mixed --run-name hybrid_v1_full_seed42 --seed 42
```

Все три seed последовательно:

```powershell
foreach ($experimentSeed in 42, 43, 44) {
    .\.venv\Scripts\python.exe -m src.train_hybrid_mixed --run-name "hybrid_v1_full_seed$experimentSeed" --seed $experimentSeed
    if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $experimentSeed" }
}
```

Не добавлять `--max-train-batches` к полному обучению. `--epochs`, `--batch-size`, `--threads` и `--device cpu|cuda` доступны явно; фактические настройки записываются в run_config.json. Для сопоставимых экспериментов сохранять фиксированные настройки. Параметры не выбирать по test2.

## Оценка после полного обучения

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_hybrid_mixed --run-name hybrid_v1_full_seed42
```

Оценщик сначала проверяет полноценный checkpoint, отдельную calibration и сохранённый порог. Затем проверяет хеш test2 по прежнему holdout audit и точное совпадение времён/меток с результатами отдельного Transformer. Он сохраняет `test_predictions.csv`, `metrics.json/csv`, четыре диагностических графика и сравнение с предыдущими моделями.

Выходные каталоги:

```text
artifacts/hai/hybrid_mixed/<run-name>/
results/hai/hybrid_mixed/<run-name>/
results/hai/hybrid_mixed/<run-name>/test_evaluation/
```

`comparison.csv` содержит этот seed гибрида, один прежний Transformer и по три seed RF/GCN. `comparison_means.csv` и `comparison.png` показывают средние по доступным строкам, а не результат трёх seed гибрида. Для общего вывода после трёх запусков нужно объединить все три результата гибрида и явно отразить разброс. В текущем подготовительном этапе реальные test2-метрики гибрида ещё не получены.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Новые проверки: causal mask, отсутствие обмена между разными примерами batch, градиенты через GCN и attention, влияние графа на пространственные сообщения, отклонение неверной перестановки каналов, восстановление checkpoint и полный синтетический цикл train → validation → calibration → test без изменения порога. Синтетические метрики проверяют код и не характеризуют качество на HAI.

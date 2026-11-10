"""Summarize completed frozen GCN runs without changing models or thresholds."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from src.config import PROJECT_ROOT as ROOT


def main():
    output = ROOT/'results/hai/gnn/hai_gcn_classifier_v1'
    artifact = ROOT/'artifacts/hai/gnn/hai_gcn_classifier_v1'
    # Correct generic helper descriptions in the first executed run; numeric thresholds stay unchanged.
    for seed in [42, 43, 44]:
        path = artifact/f'seed{seed}/threshold.json'
        threshold = json.loads(path.read_text())
        threshold.pop('validation_windows', None)
        threshold.update(method='normal_calibration_quantile', score='GCN softmax class Attack; not probability-calibrated',
            note='Separate normal calibration split; checkpoint selected on validation, no test tuning')
        path.write_text(json.dumps(threshold, indent=2), encoding='utf-8')
    rows = pd.read_csv(output/'comparison.csv')
    metrics = ['precision', 'recall', 'f1', 'fpr', 'average_precision']
    summary = rows.groupby('model')[metrics].agg(['mean', lambda x: x.std(ddof=0)])
    summary.columns = [f'{a}_{"mean" if b == "mean" else "std"}' for a,b in summary.columns]
    summary.to_csv(output/'comparison_summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, ['f1', 'recall', 'fpr']):
        ax.bar(summary.index, summary[f'{metric}_mean'], yerr=summary[f'{metric}_std'], capsize=4)
        ax.set_title(metric); ax.set_ylim(0, 1); ax.tick_params(axis='x', labelrotation=15)
    fig.suptitle('Same test2 targets, frozen calibration thresholds; mean / population SD')
    fig.tight_layout(); fig.savefig(output/'comparison.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, seed in zip(axes, [42, 43, 44]):
        history = pd.read_csv(artifact/f'seed{seed}/history.csv')
        threshold = json.loads((artifact/f'seed{seed}/threshold.json').read_text())
        ax.plot(history.epoch, history.train_loss, label='weighted train CE')
        ax.plot(history.epoch, history.validation_loss, label='normal validation CE')
        ax.axvline(threshold['best_epoch'], color='gray', linestyle='--', label='best checkpoint')
        ax.set_title(f'seed={seed}'); ax.set_xlabel('epoch'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output/'learning_curves.png', dpi=150); plt.close(fig)
    lines = ['# GCN / Transformer / Random Forest: HAI test2', '',
        'Все модели оценены на одинаковых 152970 временных точках. Порог каждой модели сохранён до оценки test2.', '',
        '| Model | Seed | Precision | Recall | F1 | FPR | AP |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for _, row in rows.iterrows():
        lines.append(f'| {row.model} | {int(row.seed)} | '+ ' | '.join(f'{row[m]:.4f}' for m in metrics) + ' |')
    lines += ['', 'GNN и RF обучены на одинаковых размеченных целях; Transformer — только на нормальных данных. '
        'Это сравнение подходов с разным доступом к меткам, а не чистая проверка архитектуры. '
        'Для GNN/RF три seed; для Transformer один ранее восстановленный checkpoint. '
        'test2 уже был исследован: результаты ретроспективные. Нет настройки по test.', '',
        'Граф: 59 тегов, 68 неориентированных функциональных связей; документированные связи и инженерные '
        'предположения различаются в таблице evidence. Это не схема физической проводки.', '',
        'В train только два интервала атак. Нормальная validation позволяет выбрать стабильную модель на нормальном '
        'режиме, но не измеряет способность распознавать новые атаки. Низкий train loss сам по себе не доказывает обобщение.', '',
        'Признаки GNN и RF включают текущее наблюдение и предыдущие 30 секунд; Transformer прогнозирует текущее '
        'наблюдение по истории. Метрики рассчитаны по точкам без point adjustment.']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(rows[['model','seed']+metrics].to_string(index=False))


if __name__ == '__main__':
    main()

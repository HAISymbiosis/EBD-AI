"""Bias in the data and in fuzzy-rule inference.

These helpers compose with fitted ``ex_fuzzy`` classifiers. They do not
re-export Ex-Fuzzy names. Metrics follow the usual group-fairness
definitions (selection rate, TPR/FPR, demographic parity, equalized odds)
without a Fairlearn dependency.

The workshop notebooks in ``Demos/`` (09–11) show the same workflow that
`WorkshopIgualdad2025 <https://github.com/rferper/WorkshopIgualdad2025>`_
used with Ex-Fuzzy 2.1.3, updated for the current ``ex_fuzzy`` API.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

ArrayLike = Union[np.ndarray, pd.Series, pd.Index, Sequence]


def _as_1d(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.ravel()
    if array.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional, got shape {array.shape}')
    return array


def _binary_positive(y: np.ndarray, positive_label: Any) -> np.ndarray:
    return np.asarray(y) == positive_label


def outcome_rates_by_group(
    y: ArrayLike,
    sensitive: ArrayLike,
    positive_label: Any = 1,
) -> pd.DataFrame:
    """Share of the positive outcome in each sensitive group (data bias).

    Args:
        y: Outcome labels.
        sensitive: Group labels aligned with ``y``.
        positive_label: Value of ``y`` treated as the favourable outcome.

    Returns:
        One row per group with counts and the positive rate.
    """
    y = _as_1d(y, 'y')
    sensitive = _as_1d(sensitive, 'sensitive')
    if y.shape[0] != sensitive.shape[0]:
        raise ValueError('y and sensitive must have the same length')
    frame = pd.DataFrame({'y': y, 'group': sensitive})
    positive = _binary_positive(frame['y'].to_numpy(), positive_label)
    frame['positive'] = positive
    grouped = frame.groupby('group', dropna=False)
    out = grouped.agg(n=('y', 'size'), n_positive=('positive', 'sum'))
    out['positive_rate'] = out['n_positive'] / out['n']
    return out.reset_index()


def group_performance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive: ArrayLike,
    positive_label: Any = 1,
) -> pd.DataFrame:
    """Selection rate and confusion rates of a predictor in each group.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        sensitive: Group labels aligned with the predictions.
        positive_label: Favourable class.

    Returns:
        One row per group with selection rate, TPR, FPR, FNR and TNR.
    """
    y_true = _as_1d(y_true, 'y_true')
    y_pred = _as_1d(y_pred, 'y_pred')
    sensitive = _as_1d(sensitive, 'sensitive')
    if not (y_true.shape[0] == y_pred.shape[0] == sensitive.shape[0]):
        raise ValueError('y_true, y_pred and sensitive must have the same length')

    truth = _binary_positive(y_true, positive_label)
    pred = _binary_positive(y_pred, positive_label)
    rows = []
    for group in pd.unique(sensitive):
        mask = sensitive == group
        t = truth[mask]
        p = pred[mask]
        tp = np.count_nonzero(t & p)
        tn = np.count_nonzero(~t & ~p)
        fp = np.count_nonzero(~t & p)
        fn = np.count_nonzero(t & ~p)
        n_pos = tp + fn
        n_neg = tn + fp
        rows.append({
            'group': group,
            'n': int(mask.sum()),
            'selection_rate': float(p.mean()) if mask.any() else np.nan,
            'tpr': tp / n_pos if n_pos else np.nan,
            'fpr': fp / n_neg if n_neg else np.nan,
            'fnr': fn / n_pos if n_pos else np.nan,
            'tnr': tn / n_neg if n_neg else np.nan,
        })
    return pd.DataFrame(rows)


def _gap(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return float('nan')
    return float(values.max() - values.min())


def _ratio(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty or values.max() == 0:
        return float('nan')
    return float(values.min() / values.max())


def disparity_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive: ArrayLike,
    positive_label: Any = 1,
) -> dict[str, float]:
    """Demographic parity and equalized-odds gaps across groups.

    Differences are ``max - min`` over groups (0 is equal). Ratios are
    ``min / max`` (1 is equal).
    """
    table = group_performance(y_true, y_pred, sensitive, positive_label=positive_label)
    tpr_gap = _gap(table['tpr'])
    fpr_gap = _gap(table['fpr'])
    return {
        'demographic_parity_difference': _gap(table['selection_rate']),
        'demographic_parity_ratio': _ratio(table['selection_rate']),
        'equalized_odds_difference': float(np.nanmax([tpr_gap, fpr_gap])),
        'equalized_odds_ratio': min(_ratio(table['tpr']), _ratio(table['fpr'])),
        'tpr_difference': tpr_gap,
        'fpr_difference': fpr_gap,
    }


def fairness_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive: ArrayLike,
    positive_label: Any = 1,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Group performance table and disparity metrics for a predictor.

    Returns:
        ``(group_performance, disparity_metrics)``.
    """
    table = group_performance(y_true, y_pred, sensitive, positive_label=positive_label)
    gaps = disparity_metrics(y_true, y_pred, sensitive, positive_label=positive_label)
    return table, gaps


def reweigh_weights(
    y: ArrayLike,
    sensitive: ArrayLike,
) -> np.ndarray:
    """Kamiran and Calders (2012) sample weights that unbias P(Y, A).

    ``w(a, y) = P(Y=y) P(A=a) / P(A=a, Y=y)``.
    """
    y = _as_1d(y, 'y')
    sensitive = _as_1d(sensitive, 'sensitive')
    if y.shape[0] != sensitive.shape[0]:
        raise ValueError('y and sensitive must have the same length')
    n = y.shape[0]
    weights = np.ones(n, dtype=float)
    for group in pd.unique(sensitive):
        in_group = sensitive == group
        for label in pd.unique(y):
            in_cell = in_group & (y == label)
            n_cell = int(in_cell.sum())
            if n_cell == 0:
                continue
            p_y = float((y == label).mean())
            p_a = float(in_group.mean())
            weights[in_cell] = (p_y * p_a) / (n_cell / n)
    return weights


def parse_printed_rules(rule_text: str) -> list[str]:
    """Turn ``eval_fuzzy_model`` printed rules into one string per rule.

    Args:
        rule_text: Text returned by ``eval_fuzzy_model(..., return_rules=True)``.

    Returns:
        Rules as ``<antecedent> THEN <consequent index>``, in print order.
    """
    if not rule_text:
        return []
    collection: list[str] = []
    consequent = -1
    for raw in rule_text.splitlines():
        line = raw.strip()
        if not line or line.startswith('------'):
            continue
        if line.startswith('Rules for consequent'):
            consequent += 1
            continue
        antecedent = line.split('WITH')[0].strip()
        if antecedent:
            collection.append(f'{antecedent} THEN {consequent}')
    return collection


def winning_rules_by_group(
    classifier: Any,
    X: Any,
    sensitive: ArrayLike,
    rule_texts: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """How often each winning rule fires in each sensitive group.

    Uses ``classifier.explainable_predict``, which current ``ex_fuzzy``
    returns as an ``ExplainedPrediction`` named tuple (indexable like the
    2.1.3 tuple the workshop notebooks used).

    Args:
        classifier: Fitted ``BaseFuzzyRulesClassifier``.
        X: Feature table aligned with ``sensitive``.
        sensitive: Group labels.
        rule_texts: Optional labels from :func:`parse_printed_rules`.

    Returns:
        Counts and within-group rates per ``(group, rule)``.
    """
    explained = classifier.explainable_predict(X)
    winners = np.asarray(explained[1]).ravel()
    sensitive = _as_1d(sensitive, 'sensitive')
    if winners.shape[0] != sensitive.shape[0]:
        raise ValueError('X and sensitive must have the same number of rows')
    frame = pd.DataFrame({'group': sensitive, 'rule': winners})
    counts = frame.groupby(['group', 'rule']).size().rename('count').reset_index()
    totals = frame['group'].value_counts()
    counts['rate'] = [row.count / totals[row.group] for row in counts.itertuples()]
    if rule_texts is not None:
        labels = list(rule_texts)
        counts['rule_text'] = [
            labels[int(rule)] if 0 <= int(rule) < len(labels) else ''
            for rule in counts['rule']
        ]
    return counts


def plot_outcome_rates(
    rates: pd.DataFrame,
    title: str = 'Positive outcome rate by group',
    ax: Any = None,
):
    """Bar chart of :func:`outcome_rates_by_group`."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    groups = rates['group'].astype(str)
    ax.bar(groups, rates['positive_rate'], color='#5fba7d')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Positive rate')
    ax.set_xlabel('Group')
    ax.set_title(title)
    for i, row in rates.iterrows():
        ax.text(i, row['positive_rate'] + 0.02, f"{row['positive_rate']:.2f}", ha='center')
    return ax


def plot_winning_rules_by_group(
    counts: pd.DataFrame,
    title: str = 'Winning-rule rate by group',
    ax: Any = None,
):
    """Grouped bars of :func:`winning_rules_by_group` rates."""
    import matplotlib.pyplot as plt

    pivot = counts.pivot(index='rule', columns='group', values='rate').fillna(0.0)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(pivot.index))
    groups = list(pivot.columns)
    width = 0.8 / max(len(groups), 1)
    for i, group in enumerate(groups):
        ax.bar(x + i * width, pivot[group].to_numpy(), width=width, label=str(group))
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels([str(v) for v in pivot.index])
    ax.set_xlabel('Winning rule')
    ax.set_ylabel('Share of group')
    ax.set_title(title)
    ax.legend(title='Group')
    return ax


def weighted_mcc_loss(sample_weight: ArrayLike) -> Callable:
    """``customized_loss`` that maximises weighted Matthews correlation.

    ``sample_weight`` must follow the training rows passed to ``fit``.
    """
    weights = _as_1d(sample_weight, 'sample_weight').astype(float)

    def loss(rule_base, X, y, tolerance, alpha: float = 0.0, beta: float = 0.0,
             precomputed_truth=None) -> float:
        from ex_fuzzy import eval_rules as evr
        evaluator = evr.evalRuleBase(rule_base, X, y, precomputed_truth=precomputed_truth)
        evaluator.add_rule_weights()
        preds = evaluator.mrule_base.winning_rule_predict(
            evaluator.X, precomputed_truth=evaluator.precomputed_truth)
        n = len(np.asarray(y))
        return float(matthews_corrcoef(y, preds, sample_weight=weights[:n]))

    return loss


def fairness_regularized_loss(
    sensitive: ArrayLike,
    lam: float = 0.2,
    positive_label: Any = 1,
) -> Callable:
    """``customized_loss``: MCC minus ``lam`` times demographic-parity difference.

    ``sensitive`` must follow the training rows passed to ``fit``.
    """
    groups = _as_1d(sensitive, 'sensitive')

    def loss(rule_base, X, y, tolerance, alpha: float = 0.0, beta: float = 0.0,
             precomputed_truth=None) -> float:
        from ex_fuzzy import eval_rules as evr
        evaluator = evr.evalRuleBase(rule_base, X, y, precomputed_truth=precomputed_truth)
        evaluator.add_rule_weights()
        preds = evaluator.mrule_base.winning_rule_predict(
            evaluator.X, precomputed_truth=evaluator.precomputed_truth)
        n = len(np.asarray(y))
        dpd = disparity_metrics(y, preds, groups[:n], positive_label=positive_label)[
            'demographic_parity_difference'
        ]
        if np.isnan(dpd):
            dpd = 0.0
        return float(matthews_corrcoef(y, preds) - lam * dpd)

    return loss


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    positive_label: Any = 1,
) -> dict[str, Any]:
    """Accuracy, F1 and recall for a binary predictor."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

    y_true = _as_1d(y_true, 'y_true')
    y_pred = _as_1d(y_pred, 'y_pred')
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }

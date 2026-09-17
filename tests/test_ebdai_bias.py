"""Bias helpers and workshop datasets."""
import numpy as np
import pandas as pd
import pytest

from ebdai import (
    disparity_metrics,
    fairness_regularized_loss,
    fairness_report,
    features_and_target,
    load_heart_failure,
    load_loan_approval,
    load_titanic,
    outcome_rates_by_group,
    parse_printed_rules,
    reweigh_weights,
    weighted_mcc_loss,
    winning_rules_by_group,
)
from ex_fuzzy import BaseFuzzyRulesClassifier, eval_tools


def test_outcome_rates_detect_data_bias():
    y = np.array([1, 1, 1, 0, 0, 0])
    a = np.array(['F', 'F', 'F', 'M', 'M', 'M'])
    rates = outcome_rates_by_group(y, a, positive_label=1)
    by_group = rates.set_index('group')['positive_rate']
    assert by_group['F'] == pytest.approx(1.0)
    assert by_group['M'] == pytest.approx(0.0)


def test_disparity_metrics_on_biased_predictor():
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    a = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    gaps = disparity_metrics(y, pred, a)
    assert gaps['demographic_parity_difference'] == pytest.approx(1.0)
    table, same = fairness_report(y, pred, a)
    assert set(table['group']) == {'A', 'B'}
    assert same['demographic_parity_difference'] == pytest.approx(1.0)


def test_reweigh_weights_flatten_label_by_group():
    y = np.array([1, 1, 1, 0, 0, 1, 0, 0])
    a = np.array(['F', 'F', 'F', 'F', 'M', 'M', 'M', 'M'])
    w = reweigh_weights(y, a)
    assert w.shape == y.shape
    assert np.all(w > 0)
    # Weighted P(Y=1 | A) becomes equal across groups.
    frame = pd.DataFrame({'y': y, 'a': a, 'w': w})
    rates = [
        np.average(part.y, weights=part.w)
        for _, part in frame.groupby('a')
    ]
    assert max(rates) - min(rates) == pytest.approx(0.0, abs=1e-10)


def test_parse_printed_rules():
    text = (
        'Rules for consequent: 0\n'
        '------\n'
        'IF x IS Low WITH DS 0.5\n'
        'Rules for consequent: 1\n'
        'IF y IS High WITH DS 0.2\n'
    )
    rules = parse_printed_rules(text)
    assert rules == ['IF x IS Low THEN 0', 'IF y IS High THEN 1']


def test_workshop_datasets_load():
    titanic, sex = load_titanic()
    assert sex == 'Sex'
    assert 'Survived' in titanic.columns
    assert titanic['Age'].isna().sum() == 0
    X, y = features_and_target(titanic, 'Survived')
    assert len(X) == len(y)
    assert 'Survived' not in X.columns

    heart, col = load_heart_failure()
    assert col == 'sex'
    assert heart['sex'].dtype == object

    loan, col = load_loan_approval()
    assert col == 'Gender'
    assert set(loan['Loan_Status'].unique()) <= {0, 1}
    assert loan.isna().sum().sum() == 0


def test_custom_losses_match_fitness_signature():
    y = np.array([1, 0, 1, 0])
    a = np.array(['F', 'F', 'M', 'M'])
    w = reweigh_weights(y, a)
    loss_w = weighted_mcc_loss(w)
    loss_f = fairness_regularized_loss(a, lam=0.2)
    assert callable(loss_w) and callable(loss_f)


def test_winning_rules_and_tiny_fit():
    rng = np.random.default_rng(0)
    n = 80
    sensitive = np.array(['F'] * (n // 2) + ['M'] * (n - n // 2))
    x0 = rng.normal(size=n)
    x1 = (sensitive == 'F').astype(float)
    y = ((x0 + 0.8 * x1) > 0).astype(int)
    X = pd.DataFrame({'x0': x0, 'group': sensitive})
    clf = BaseFuzzyRulesClassifier(
        nRules=4,
        nAnts=2,
        n_linguistic_variables=3,
        verbose=False,
        ds_mode=1,
        n_gen=4,
        pop_size=8,
        patience=2,
        random_state=0,
    )
    clf.fit(X, y)
    report = eval_tools.eval_fuzzy_model(
        clf, X, y, X, y,
        plot_rules=False, print_rules=False, plot_partitions=False,
        return_rules=True, bootstrap_results_print=False,
    )
    texts = parse_printed_rules(report or '')
    counts = winning_rules_by_group(clf, X, X['group'], rule_texts=texts)
    assert not counts.empty
    assert set(counts['group']).issubset({'F', 'M'})
    preds = clf.predict(X)
    table, gaps = fairness_report(y, preds, X['group'])
    assert 'demographic_parity_difference' in gaps
    assert len(table) >= 1

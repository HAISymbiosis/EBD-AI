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


@pytest.mark.parametrize('missing', [np.nan, None, pd.NA])
def test_missing_sensitive_group_is_counted_everywhere(missing):
    from types import SimpleNamespace

    groups = pd.Series(['A', 'A', missing, missing], dtype=object)
    y = np.array([1, 0, 1, 0])
    predictions = np.array([1, 1, 0, 0])
    rates = outcome_rates_by_group(predictions, groups)
    missing_rate = rates.loc[rates.group.isna()].iloc[0]
    assert missing_rate['n'] == 2
    assert missing_rate['positive_rate'] == 0
    table, gaps = fairness_report(y, predictions, groups)
    missing_row = table.loc[table.group.isna()].iloc[0]
    assert missing_row['n'] == 2
    assert missing_row['selection_rate'] == 0
    assert missing_row['tpr'] == 0
    assert missing_row['fpr'] == 0
    assert gaps['demographic_parity_difference'] == 1
    assert gaps['equalized_odds_difference'] == 1

    classifier = SimpleNamespace(explainable_predict=lambda X: (predictions, [0, 1, 1, 1]))
    counts = winning_rules_by_group(classifier, np.zeros((4, 1)), groups)
    assert counts['count'].sum() == 4
    missing_counts = counts.loc[counts.group.isna()].iloc[0]
    assert missing_counts['count'] == 2
    assert missing_counts['rate'] == 1


@pytest.mark.parametrize('missing', [np.nan, None, pd.NA])
def test_reweigh_missing_sensitive_group(missing):
    groups = pd.Series(['A'] * 4 + [missing] * 4, dtype=object)
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    weights = reweigh_weights(y, groups)
    np.testing.assert_allclose(weights, [2/3, 2/3, 2/3, 2, 2, 2/3, 2/3, 2/3])
    assert np.average(y[:4], weights=weights[:4]) == pytest.approx(.5)
    assert np.average(y[4:], weights=weights[4:]) == pytest.approx(.5)


def test_mixed_missing_markers_form_one_group():
    groups = pd.Series(['A', None, pd.NA, np.nan], dtype=object)
    table, _ = fairness_report([1, 0, 1, 0], [1, 0, 1, 0], groups)
    assert len(table) == 2
    assert table.loc[table.group.isna(), 'n'].item() == 3


@pytest.mark.parametrize('labels,positive', [(['denied', 'approved'], 'approved'),
                                          (['approved', 'denied'], 'denied'),
                                          ([10, 20], 20)])
def test_fairness_loss_maps_original_labels(monkeypatch, labels, positive):
    from types import SimpleNamespace
    from ex_fuzzy import eval_rules

    predictions = np.array([1, 1, 0, 0])
    X = np.zeros((4, 1))
    evaluator = SimpleNamespace(
        X=X, precomputed_truth=None, add_rule_weights=lambda: None,
        mrule_base=SimpleNamespace(winning_rule_predict=lambda *args, **kwargs: predictions),
    )
    monkeypatch.setattr(eval_rules, 'evalRuleBase', lambda *args, **kwargs: evaluator)
    groups = pd.Series(['A', 'A', pd.NA, pd.NA], dtype=object)
    loss = fairness_regularized_loss(groups, lam=.2, positive_label=positive, classes=labels)
    assert loss(None, X, predictions, 0) == pytest.approx(.8)
    default_loss = fairness_regularized_loss(groups, lam=.2)
    assert default_loss(None, X, predictions, 0) == pytest.approx(.8)
    with pytest.raises(ValueError, match='training class'):
        fairness_regularized_loss(groups, positive_label=2)(None, X, predictions, 0)
    with pytest.raises(ValueError, match='same length'):
        fairness_regularized_loss(['A'] * 5)(None, X, predictions, 0)


@pytest.mark.parametrize('kwargs', [
    {'positive_label': 'approved'},
    {'positive_label': -1},
    {'positive_label': 'unknown', 'classes': ['approved', 'denied']},
    {'classes': [0, 1, 1]},
    {'classes': [0, np.nan]},
])
def test_fairness_loss_rejects_invalid_label_mapping(kwargs):
    with pytest.raises(ValueError):
        fairness_regularized_loss(['A', 'B'], **kwargs)


@pytest.mark.parametrize('class_order', [None, ['denied', 'approved']])
def test_string_label_fairness_fit_matches_encoded_fit(class_order):
    X = pd.DataFrame({'x': np.linspace(-2, 2, 40)})
    y = np.where(X.x > 0, 'approved', 'denied')
    labels = np.unique(y) if class_order is None else np.array(class_order)
    encoded = np.array([list(labels).index(label) for label in y])
    groups = pd.Series(['A'] * 20 + [pd.NA] * 20, dtype=object)
    kwargs = dict(nRules=4, nAnts=1, n_gen=3, pop_size=8, random_state=42,
                  verbose=False, ds_mode=1)
    named = BaseFuzzyRulesClassifier(class_names=class_order, **kwargs)
    named.customized_loss(fairness_regularized_loss(
        groups, positive_label='approved', classes=labels))
    indexed = BaseFuzzyRulesClassifier(**kwargs)
    indexed.customized_loss(fairness_regularized_loss(
        groups, positive_label=list(labels).index('approved')))
    named.fit(X, y)
    indexed.fit(X, encoded)
    np.testing.assert_array_equal(named.predict(X), labels[indexed.predict(X).astype(int)])

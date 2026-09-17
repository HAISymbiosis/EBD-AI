=====
ebdai
=====

``import ebdai`` is the BDI expansion package (pip version 1.0.1). It does
not replace ``import ex_fuzzy``. The fuzzy rule learner stays in ``ex_fuzzy``
(currently 3.2.0). New explainable-by-design tools, starting with **bias in
the data and in inference**, live under ``ebdai``.

.. code-block:: python

    import ebdai
    from ex_fuzzy import BaseFuzzyRulesClassifier

    print(ebdai.__version__)   # 1.0.1
    import ex_fuzzy
    print(ex_fuzzy.__version__)  # 3.2.0

Two imports
===========

+----------------------+-----------------------------------------------+
| Import               | Role                                          |
+======================+===============================================+
| ``ex_fuzzy``         | Rebased Ex-Fuzzy API: rules, classifiers,     |
|                      | FERL, conformal prediction                    |
+----------------------+-----------------------------------------------+
| ``ebdai``            | BDI expansions that accept ``ex_fuzzy``       |
|                      | objects (bias metrics, workshop datasets)     |
+----------------------+-----------------------------------------------+

Do not import classifiers from ``ebdai``. Each class and enum must exist
once per process.

Bias in the data
================

``outcome_rates_by_group`` reports the positive-label rate in each
sensitive group **before** any model is trained.

.. code-block:: python

    from ebdai import load_titanic, features_and_target, outcome_rates_by_group

    frame, sensitive = load_titanic()
    X, y = features_and_target(frame, 'Survived')
    print(outcome_rates_by_group(y, X[sensitive]))

Workshop tables ship in the package:

- :func:`ebdai.datasets.load_titanic` — survival, sensitive column ``Sex``
- :func:`ebdai.datasets.load_heart_failure` — death, sensitive column ``sex``
- :func:`ebdai.datasets.load_loan_approval` — approval, sensitive column ``Gender``

Bias in inference
=================

After ``BaseFuzzyRulesClassifier.fit``, compare group performance of the
predictions and which rules win for whom.

.. code-block:: python

    from sklearn.model_selection import train_test_split
    from ex_fuzzy import BaseFuzzyRulesClassifier, eval_tools
    from ebdai import fairness_report, parse_printed_rules, winning_rules_by_group

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    clf = BaseFuzzyRulesClassifier(nRules=8, nAnts=3, n_gen=6, pop_size=12,
                                   random_state=42, verbose=False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    table, gaps = fairness_report(y_test, y_pred, X_test[sensitive])
    print(table)
    print(gaps['demographic_parity_difference'])

    report = eval_tools.eval_fuzzy_model(
        clf, X_train, y_train, X_test, y_test,
        plot_rules=False, print_rules=False, plot_partitions=False,
        return_rules=True, bootstrap_results_print=False,
    )
    counts = winning_rules_by_group(
        clf, X_test, X_test[sensitive],
        rule_texts=parse_printed_rules(report or ''),
    )

``fairness_report`` returns per-group selection rate, TPR and FPR, plus
demographic-parity and equalized-odds gaps. It does not depend on Fairlearn.

Mitigation
==========

Kamiran and Calders (2012) sample weights, and a genetic loss that
penalises demographic parity, plug into
``BaseFuzzyRulesClassifier.customized_loss``:

.. code-block:: python

    from ebdai import reweigh_weights, weighted_mcc_loss, fairness_regularized_loss

    weights = reweigh_weights(y_train, X_train[sensitive])
    clf.customized_loss(weighted_mcc_loss(weights))
    clf.fit(X_train, y_train)

    clf.customized_loss(fairness_regularized_loss(X_train[sensitive], lam=0.2))
    clf.fit(X_train, y_train)

Demos
=====

Executed notebooks (they render on GitHub):

- `09 Bias in the data (Titanic) <https://github.com/HAISymbiosis/EBD-AI/blob/main/Demos/09_bias_titanic.ipynb>`_
- `10 Bias in heart-failure labels <https://github.com/HAISymbiosis/EBD-AI/blob/main/Demos/10_bias_heart_failure.ipynb>`_
- `11 Bias in inference and mitigation (loans) <https://github.com/HAISymbiosis/EBD-AI/blob/main/Demos/11_bias_loan_fairness.ipynb>`_

They come from
`WorkshopIgualdad2025 <https://github.com/rferper/WorkshopIgualdad2025>`_
(Raquel Fernandez Peralta and Javier Fumanal Idocin), updated from Ex-Fuzzy
2.1.3 to the current ``ex_fuzzy`` API.

API reference
=============

See :doc:`../api/ebdai` for every public function in :mod:`ebdai.bias` and
:mod:`ebdai.datasets`.

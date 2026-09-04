# third party
import numpy as np
import pandas as pd
import pytest
from fhelpers import generate_fixtures, get_airfoil_dataset
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.eval import PerformanceEvaluatorXGB
from synthcity.plugins import Plugin
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.privacy.plugin_pategan import PATEGAN, Teachers, plugin

plugin_name = "pategan"
plugin_args = {
    "generator_n_layers_hidden": 1,
    "generator_n_units_hidden": 10,
    "lamda": 0.1,
}


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_sanity(test_plugin: Plugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_name(test_plugin: Plugin) -> None:
    assert test_plugin.name() == plugin_name


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_type(test_plugin: Plugin) -> None:
    assert test_plugin.type() == "privacy"


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_hyperparams(test_plugin: Plugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 20


def test_privacy_parameters_are_forwarded_and_recorded() -> None:
    model = PATEGAN(delta=1e-6, lamda=0.25, alpha=17, epsilon=2.0)
    assert model.delta == pytest.approx(1e-6)
    assert model.lamda == pytest.approx(0.25)
    assert model.alpha == 17
    metadata = model.get_accounting_metadata()
    assert metadata["schema_version"] == "pate-accounting-v1"
    assert metadata["privacy_claim_type"] == "formal_dp"
    assert metadata["requested_delta"] == pytest.approx(1e-6)
    assert metadata["requested_alpha"] == 17
    assert metadata["requested_lamda"] == pytest.approx(0.25)
    assert metadata["requested_epsilon"] == pytest.approx(2.0)

    test_plugin = plugin(delta=1e-6, lamda=0.25, alpha=17, epsilon=2.0)
    assert test_plugin.model.alpha == 17
    assert test_plugin.model.delta == pytest.approx(1e-6)


def test_plugin_exposes_accounting_metadata() -> None:
    test_plugin = plugin(delta=1e-6, lamda=0.25, alpha=17, epsilon=2.0)

    metadata = test_plugin.get_accounting_metadata()

    assert metadata["schema_version"] == "pate-accounting-v1"
    assert metadata["privacy_claim_type"] == "formal_dp"
    assert metadata["accountant"] == "pate_moments_v1"
    assert metadata["requested_epsilon"] == pytest.approx(2.0)
    assert metadata["requested_delta"] == pytest.approx(1e-6)
    assert metadata["requested_alpha"] == 17
    assert metadata["requested_lamda"] == pytest.approx(0.25)


def test_accounting_accessor_rejects_incomplete_metadata() -> None:
    model = PATEGAN()
    model.accounting_metadata.pop("effective_epsilon")

    with pytest.raises(RuntimeError, match="accounting metadata is incomplete"):
        model.get_accounting_metadata()


@pytest.mark.parametrize(
    "kwargs",
    [{"lamda": 0}, {"lamda": -1}, {"delta": 0}, {"delta": 1}, {"alpha": 0}],
)
def test_invalid_accounting_parameters_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PATEGAN(**kwargs)


def test_hyperparameter_space_excludes_zero_lamda() -> None:
    lamda = next(
        parameter
        for parameter in plugin.hyperparameter_space()
        if parameter.name == "lamda"
    )
    assert lamda.low > 0


def test_pate_teachers_keep_groups_together() -> None:
    class RecordingTeacher:
        real_groups = []

        def __init__(self, **kwargs):
            del kwargs

        def fit(self, X, y):
            del y
            sample_count = len(X) // 2
            type(self).real_groups.append(set(X[:sample_count, 0]))

    teacher_data = np.repeat(np.arange(4), 2).reshape(-1, 1).astype(float)
    groups = np.repeat([f"p{index}" for index in range(4)], 2)
    teachers = Teachers(n_teachers=2, samples_per_teacher=4)
    teachers.model_template = RecordingTeacher

    teachers.fit(
        teacher_data,
        lambda count: np.zeros((count, teacher_data.shape[1])),
        groups=groups,
    )

    assert len(RecordingTeacher.real_groups) == 2
    assert RecordingTeacher.real_groups[0].isdisjoint(RecordingTeacher.real_groups[1])
    assert set.union(*RecordingTeacher.real_groups) == set(range(4))


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_fit(test_plugin: Plugin) -> None:
    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))


def test_plugin_generate_pategan() -> None:
    test_plugin = plugin(**plugin_args)
    X = get_airfoil_dataset()

    test_plugin.fit(GenericDataLoader(X))

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)
    assert X_gen.shape[1] == X.shape[1]

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)


def test_plugin_generate_constraints() -> None:
    test_plugin = plugin(**plugin_args)
    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))

    constraints = Constraints(
        rules=[
            ("0", "le", 6),
            ("0", "ge", 4.3),
            ("1", "le", 4.4),
            ("1", "ge", 3),
            ("2", "le", 5.5),
            ("2", "ge", 1.0),
            ("3", "le", 2),
            ("3", "ge", 0.1),
        ]
    )

    X_gen = test_plugin.generate(constraints=constraints).dataframe()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)
    assert constraints.filter(X_gen).sum() == len(X_gen)

    X_gen = test_plugin.generate(count=50, constraints=constraints).dataframe()
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert constraints.filter(X_gen).sum() == len(X_gen)
    assert list(X_gen.columns) == list(X.columns)


def test_sample_hyperparams() -> None:
    for i in range(100):
        args = plugin.sample_hyperparameters()

        assert plugin(**args) is not None


@pytest.mark.slow_2
@pytest.mark.slow
def test_eval_performance() -> None:
    results = []

    Xraw, y = load_iris(return_X_y=True, as_frame=True)
    Xraw["target"] = y
    X = GenericDataLoader(Xraw)

    for retry in range(2):
        test_plugin = plugin(
            n_iter=200, generator_n_layers_hidden=1, n_teachers=2, lamda=2e-4
        )
        evaluator = PerformanceEvaluatorXGB(task_type="classification")

        test_plugin.fit(X)
        X_syn = test_plugin.generate()

        results.append(evaluator.evaluate(X, X_syn)["syn_id"])

    print(plugin.name(), np.mean(results))
    assert np.mean(results) > 0.7

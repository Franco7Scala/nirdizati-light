from enum import Enum
from typing import Optional, List, Any, Mapping, Sequence

import numpy as np
from hyperopt import Trials, fmin, tpe, hp
from hyperopt.pyll.base import Apply

from nirdizati_light.predictive_model.predictive_model import PredictiveModel
from nirdizati_light.predictive_model.pm_common import ClassificationMethods


class HyperoptTarget(str, Enum):
    AUC = "auc"
    F1 = "f1_score"
    MAE = "mae"
    ACCURACY = "accuracy"


def _is_hp_node(x: Any) -> bool:
    """Ritorna True se x (ricorsivamente) contiene almeno un nodo hp.* (Apply)."""
    if isinstance(x, Apply):
        return True
    if isinstance(x, Mapping):
        return any(_is_hp_node(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(_is_hp_node(v) for v in x)
    return False


def _wrap_fixed(name: str, value):
    """Se value NON è un nodo hp.*, wrappalo in scelta fissa per evitare spazi degeneri."""
    return value if _is_hp_node(value) else hp.choice(f"fixed_{name}", [value])


def _get_space(model_type) -> dict:
    key = model_type.value if hasattr(model_type, "value") else str(model_type)

    if key == ClassificationMethods.TORCH_MOE.value:
        return {
            "max_num_epochs": hp.quniform("moe_max_num_epochs", 20, 80, 1),
            "lr": hp.loguniform("moe_lr", np.log(1e-4), np.log(5e-2)),
            "dropout": hp.uniform("moe_dropout", 0.0, 0.5),
            "num_experts": hp.quniform("moe_num_experts", 2, 8, 1),
            "hidden": hp.quniform("moe_hidden", 64, 512, 1),
            "weight_decay": hp.loguniform("moe_weight_decay", np.log(1e-6), np.log(1e-2)),
            "patience": hp.quniform("moe_patience", 3, 10, 1),
            "min_delta": hp.uniform("moe_min_delta", 0.0, 0.01),
        }

    return {}


def retrieve_best_model(
    predictive_models: List[PredictiveModel],
    max_evaluations: int,
    target: HyperoptTarget,
    seed: Optional[int] = None,
):
    if not isinstance(predictive_models, list):
        predictive_models = [predictive_models]

    best_candidates = []
    best_target_per_model = []

    rstate = np.random.RandomState(seed) if seed is not None else np.random.default_rng()

    for predictive_model in predictive_models:
        print(f"Running hyperparameter optimization on model {predictive_model.model_type}...")

        space = _get_space(predictive_model.model_type)

        if predictive_model.hyperopt_space is not None:
            for k, v in predictive_model.hyperopt_space.items():
                space[k] = _wrap_fixed(k, v)

        if not space or not _is_hp_node(space):
            cfg = {}
            result = predictive_model.train_and_evaluate_configuration(config=cfg, target=target)
            best_candidates.append(result)
            best_target_per_model.append(result["result"][target.value])
            continue

        trials = Trials()
        fmin(
            fn=lambda cfg: predictive_model.train_and_evaluate_configuration(
                config=cfg,
                target=target,
            ),
            space=space,
            algo=tpe.suggest,
            max_evals=max_evaluations,
            trials=trials,
            rstate=rstate,
        )

        best_trial = trials.best_trial
        cand = best_trial["result"]  # {'result': {...}, 'model': obj, 'config': cfg}
        best_candidates.append(cand)
        best_target_per_model.append(cand["result"][target.value])

    best_model_idx = int(np.argmax(best_target_per_model))
    return (
        best_candidates,
        best_model_idx,
        best_candidates[best_model_idx]["model"],
        best_candidates[best_model_idx]["config"],
    )

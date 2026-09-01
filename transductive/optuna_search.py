import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna


METRIC_RE = re.compile(
    r"\[VALID\]\s+MRR:(?P<v_mrr>[0-9.]+)\s+H@1:(?P<v_h1>[0-9.]+)\s+H@10:(?P<v_h10>[0-9.]+)\s+"
    r"\[TEST\]\s+MRR:(?P<t_mrr>-?[0-9.]+)\s+H@1:(?P<t_h1>-?[0-9.]+)\s+H@10:(?P<t_h10>-?[0-9.]+)"
)


DATASET_DEFAULTS = {
    "family": {
        "topk": 100,
        "layers": 8,
        "fact_ratio": 0.90,
        "lr": 0.0036,
        "decay_rate": 0.999,
        "lamb": 0.000017,
        "hidden_dim": 48,
        "attn_dim": 5,
        "dropout": 0.29,
        "act": "relu",
        "n_batch": 20,
    },
    "umls": {
        "topk": 100,
        "layers": 5,
        "fact_ratio": 0.90,
        "lr": 0.0012,
        "decay_rate": 0.998,
        "lamb": 0.00014,
        "hidden_dim": 64,
        "attn_dim": 5,
        "dropout": 0.01,
        "act": "tanh",
        "n_batch": 10,
    },
    "WN18RR": {
        "topk": 1000,
        "layers": 8,
        "fact_ratio": 0.96,
        "lr": 0.0030,
        "decay_rate": 0.994,
        "lamb": 0.00014,
        "hidden_dim": 64,
        "attn_dim": 5,
        "dropout": 0.02,
        "act": "idd",
        "n_batch": 50,
    },
    "fb15k-237": {
        "topk": 2000,
        "layers": 7,
        "fact_ratio": 0.99,
        "lr": 0.0009,
        "decay_rate": 0.9938,
        "lamb": 0.000080,
        "hidden_dim": 48,
        "attn_dim": 5,
        "dropout": 0.0391,
        "act": "idd",
        "n_batch": 6,
    },
    "nell": {
        "topk": 2000,
        "layers": 6,
        "fact_ratio": 0.95,
        "lr": 0.0011,
        "decay_rate": 0.9938,
        "lamb": 0.000089,
        "hidden_dim": 128,
        "attn_dim": 64,
        "dropout": 0.2593,
        "act": "idd",
        "n_batch": 10,
    },
    "YAGO": {
        "topk": 1000,
        "layers": 8,
        "fact_ratio": 0.995,
        "lr": 0.001,
        "decay_rate": 0.9429713470775948,
        "lamb": 0.000946516892415447,
        "hidden_dim": 64,
        "attn_dim": 2,
        "dropout": 0.19456805575101324,
        "act": "relu",
        "n_batch": 5,
    },
}


DATASET_SEARCH_SPACE = {
    "family": {
        "topk": [50, 100, 200],
        "layers": [6, 7, 8],
        "fact_ratio": (0.85, 0.95),
        "hidden_dim": [32, 48, 64],
        "attn_dim": [3, 5, 8],
        "n_batch": [10, 20, 50],
        "act": ["relu", "tanh", "idd"],
        "lr": (1e-3, 6e-3),
        "decay_rate": (0.990, 0.9999),
        "lamb": (1e-6, 1e-4),
        "dropout": (0.0, 0.4),
    },
    "umls": {
        "topk": [50, 80, 100],
        "layers": [4, 5, 6, 7, 8],
        "fact_ratio": (0.85, 0.95),
        "hidden_dim": [32, 48, 64, 96, 128],
        "attn_dim": [3, 5, 8, 16],
        "n_batch": [5, 10, 20],
        "act": ["relu", "tanh", "idd"],
        "lr": (5e-4, 3e-3),
        "decay_rate": (0.990, 0.9999),
        "lamb": (1e-5, 5e-4),
        "dropout": (0.0, 0.2),
    },
    "WN18RR": {
        "topk": [800, 1000, 1200],
        "layers": [4, 5, 6, 7, 8],
        "fact_ratio": (0.93, 0.98),
        "hidden_dim": [48, 64, 96],
        "attn_dim": [3, 5, 8],
        "n_batch": [20, 50],
        "act": ["relu", "tanh", "idd"],
        "lr": (1e-3, 6e-3),
        "decay_rate": (0.985, 0.999),
        "lamb": (1e-5, 5e-4),
        "dropout": (0.0, 0.2),
    },
    "fb15k-237": {
        "topk": [1000, 1500, 2000],
        "layers": [5, 6, 7, 8],
        "fact_ratio": (0.97, 0.995),
        "hidden_dim": [32, 48, 64],
        "attn_dim": [3, 5, 8],
        "n_batch": [4, 6, 8, 10],
        "act": ["relu", "tanh", "idd"],
        "lr": (3e-4, 3e-3),
        "decay_rate": (0.985, 0.999),
        "lamb": (1e-5, 5e-4),
        "dropout": (0.0, 0.2),
    },
    "nell": {
        "topk": [1000, 1500, 2000],
        "layers": [5, 6, 7, 8],
        "fact_ratio": (0.90, 0.98),
        "hidden_dim": [64, 96, 128],
        "attn_dim": [16, 32, 64],
        "n_batch": [5, 10, 20],
        "act": ["relu", "tanh", "idd"],
        "lr": (5e-4, 3e-3),
        "decay_rate": (0.985, 0.999),
        "lamb": (1e-5, 5e-4),
        "dropout": (0.05, 0.4),
    },
    "YAGO": {
        "topk": [500, 1000, 1500, 2000],
        "layers": [6, 7, 8],
        "fact_ratio": (0.98, 0.998),
        "hidden_dim": [48, 64, 96],
        "attn_dim": [2, 3, 5, 8],
        "n_batch": [3, 5, 8],
        "act": ["relu", "tanh", "idd"],
        "lr": (3e-4, 3e-3),
        "decay_rate": (0.90, 0.99),
        "lamb": (1e-5, 2e-3),
        "dropout": (0.05, 0.4),
    },
}


def suggest_original_params(trial, dataset):
    """Original DiffusionE search space supported by the current train.py."""
    if dataset not in DATASET_SEARCH_SPACE:
        raise ValueError(f"No search space configured for dataset: {dataset}")
    defaults = DATASET_DEFAULTS.get(dataset, {})
    space = DATASET_SEARCH_SPACE[dataset]
    fact_ratio_low, fact_ratio_high = space["fact_ratio"]
    lr_low, lr_high = space["lr"]
    decay_low, decay_high = space["decay_rate"]
    lamb_low, lamb_high = space["lamb"]
    dropout_low, dropout_high = space["dropout"]
    params = {
        "topk": trial.suggest_categorical("topk", unique_sorted(space["topk"])),
        "layers": trial.suggest_categorical("layers", unique_sorted(space["layers"])),
        "fact_ratio": trial.suggest_float("fact_ratio", fact_ratio_low, fact_ratio_high),
        "tau": trial.suggest_categorical("tau", [0.5, 1.0, 2.0]),
        "remove_1hop_edges": False,
        "lr": trial.suggest_float("lr", lr_low, lr_high, log=True),
        "decay_rate": trial.suggest_float("decay_rate", decay_low, decay_high),
        "lamb": trial.suggest_float("lamb", lamb_low, lamb_high, log=True),
        "hidden_dim": trial.suggest_categorical(
            "hidden_dim",
            unique_sorted(space["hidden_dim"])
        ),
        "attn_dim": trial.suggest_categorical(
            "attn_dim",
            unique_sorted(space["attn_dim"])
        ),
        "dropout": trial.suggest_float("dropout", dropout_low, dropout_high),
        "act": trial.suggest_categorical("act", space["act"]),
        "n_batch": trial.suggest_categorical(
            "n_batch",
            unique_sorted(space["n_batch"])
        ),
    }

    if "layers" in defaults:
        trial.set_user_attr("official_layers", defaults["layers"])

    return params


def suggest_relation_codiffusion_params(trial):
    """New relation co-diffusion search space.

    Enable with --enable_rel_search after train.py supports these arguments.
    """
    return {
        "rel_line_topk": trial.suggest_categorical("rel_line_topk", [5, 10, 20, -1]),
        "rel_edge_threshold": trial.suggest_categorical(
            "rel_edge_threshold", [0.0, 0.001, 0.005, 0.01]
        ),
        "rel_tau": trial.suggest_categorical("rel_tau", [0.5, 1.0, 2.0]),
        "rel_residual_alpha": trial.suggest_categorical(
            "rel_residual_alpha", [0.3, 0.4, 0.5, 0.6, 0.7]
        ),
        "rel_diff_weight": trial.suggest_categorical(
            "rel_diff_weight", [0.1, 0.3, 0.5, 0.7, 1.0]
        ),
        "rel_dropout": trial.suggest_float("rel_dropout", 0.0, 0.3),
        "rel_layers_per_gnn": trial.suggest_categorical("rel_layers_per_gnn", [1, 2]),
        "rel_include_inverse": trial.suggest_categorical(
            "rel_include_inverse", [True, False]
        ),
    }


def suggest_phase_interference_params(trial):
    """Phase-interference search space."""
    return {
        "phase_tau": trial.suggest_categorical("phase_tau", [0.5, 1.0, 2.0]),
        "phase_weight": trial.suggest_categorical("phase_weight", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    }


def dataset_name(data_path):
    return Path(data_path.rstrip("/\\")).name


def suggest_params(trial, dataset, enable_rel_search, enable_phase_search):
    params = suggest_original_params(trial, dataset)

    if enable_rel_search:
        params["use_rel_codiffusion"] = True
        params.update(suggest_relation_codiffusion_params(trial))
    if enable_phase_search:
        params["use_phase_interference"] = True
        params.update(suggest_phase_interference_params(trial))

    return params


def unique_sorted(values):
    values = [v for v in values if v is not None]
    return sorted(set(values))


def build_command(args, params):
    cmd = [
        args.python,
        args.train_script,
        "--data_path",
        args.data_path,
        "--train",
        "--gpu",
        str(args.gpu),
        "--seed",
        str(args.seed),
        "--epoch",
        str(args.max_epoch),
        "--eval_interval",
        str(args.eval_interval),
        "--trial_id",
        str(args.current_trial_id),
    ]

    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    for item in args.extra_args:
        cmd.append(item)

    return cmd


def command_to_string(cmd):
    if os.name == "nt":
        return subprocess.list2cmdline([str(item) for item in cmd])
    return shlex.join(str(item) for item in cmd)


def stop_process(proc):
    if proc.poll() is not None:
        return

    if os.name == "nt":
        proc.terminate()
    else:
        os.killpg(proc.pid, signal.SIGTERM)

    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)


def run_trial_command(cmd, cwd, patience, trial):
    best_val = None
    best_test = None
    best_line = None
    eval_count = 0
    stale_count = 0
    output_tail = []
    start_time = time.time()

    popen_kwargs = {
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": None,
        "text": True,
        "bufsize": 1,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name != "nt":
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            output_tail.append(line.rstrip())
            if len(output_tail) > 80:
                output_tail.pop(0)

            match = METRIC_RE.search(line)
            if not match:
                continue

            eval_count += 1
            val_mrr = float(match.group("v_mrr"))
            test_mrr = float(match.group("t_mrr"))

            improved = best_val is None or val_mrr > best_val
            if improved:
                best_val = val_mrr
                best_test = test_mrr
                best_line = line.strip()
                stale_count = 0
            else:
                stale_count += 1

            trial.report(val_mrr, step=eval_count)
            trial.set_user_attr("last_val_mrr", val_mrr)
            trial.set_user_attr("last_test_mrr", test_mrr)
            trial.set_user_attr("best_val_mrr", best_val)
            trial.set_user_attr("best_test_mrr", best_test)
            trial.set_user_attr("eval_count", eval_count)

            if stale_count >= patience:
                trial.set_user_attr("stopped_by_patience", True)
                stop_process(proc)
                break

        return_code = proc.wait()
    finally:
        stop_process(proc)

    trial.set_user_attr("duration_sec", round(time.time() - start_time, 2))
    trial.set_user_attr("return_code", return_code)
    trial.set_user_attr("best_metric_line", best_line)
    trial.set_user_attr("output_tail", "\n".join(output_tail))

    if best_val is None:
        raise RuntimeError("No validation metric was parsed from train.py output.")
    if return_code not in (0, -signal.SIGTERM if hasattr(signal, "SIGTERM") else 0):
        # A patience stop may return non-zero on some systems, but no-metric failures are
        # already handled above. Preserve the best observed value when metrics exist.
        trial.set_user_attr("nonzero_return_with_metric", True)

    return best_val


def load_initial_params(path, inline_values):
    params = []

    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            params.append(loaded)
        elif isinstance(loaded, list):
            params.extend(loaded)
        else:
            raise ValueError("Initial params JSON must be an object or a list of objects.")

    for value in inline_values:
        params.append(parse_param_object(value))

    for item in params:
        if not isinstance(item, dict):
            raise ValueError("Each initial parameter group must be a JSON object.")

    return params


def parse_param_object(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Windows cmd.exe strips unescaped double quotes in inline JSON. Accept a
        # flat JSON-like object such as {topk:100,act:tanh,flag:true}.
        return parse_cmd_flat_object(value)


def parse_cmd_flat_object(value):
    text = value.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise
    body = text[1:-1].strip()
    if not body:
        return {}

    result = {}
    for item in body.split(","):
        if ":" not in item:
            raise ValueError(f"Invalid initial parameter item: {item}")
        key, raw_value = item.split(":", 1)
        key = key.strip().strip("'\"")
        result[key] = parse_cmd_value(raw_value.strip())
    return result


def parse_cmd_value(value):
    value = value.strip().strip("'\"")
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def make_objective(args, study):
    dataset = dataset_name(args.data_path)
    cwd = str(Path(args.workdir).resolve())

    def objective(trial):
        args.current_trial_id = trial.number
        params = suggest_params(trial, dataset, args.enable_rel_search, args.enable_phase_search)
        trial.set_user_attr("dataset", dataset)
        trial.set_user_attr("params", json.dumps(params, sort_keys=True))

        duplicate_trial = next(
            (
                old_trial
                for old_trial in study.get_trials(deepcopy=False)
                if old_trial.number != trial.number
                and old_trial.state
                in (
                    optuna.trial.TrialState.FAIL,
                    optuna.trial.TrialState.PRUNED,
                )
                and old_trial.params == trial.params
            ),
            None,
        )
        if duplicate_trial is not None:
            trial.set_user_attr("duplicate_of", duplicate_trial.number)
            print(
                f"==> Trial {trial.number} duplicates failed/pruned "
                f"Trial {duplicate_trial.number}; skip it.",
                flush=True,
            )
            raise optuna.TrialPruned(
                f"Duplicate of failed/pruned Trial {duplicate_trial.number}."
            )

        cmd = build_command(args, params)
        command_str = command_to_string(cmd)
        trial.set_user_attr("command", command_str)
        print(f"\n==> Trial {trial.number} command:")
        print(command_str)
        print("", flush=True)
        try:
            return run_trial_command(cmd, cwd, args.patience, trial)
        except KeyboardInterrupt as exc:
            trial.set_user_attr("stopped_by_user", True)
            print(
                f"\n==> Trial {trial.number} interrupted; continue to next trial.",
                flush=True,
            )
            raise optuna.TrialPruned("Interrupted by user.") from exc

    return objective


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna database-backed search runner for DiffusionE transductive training."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--max_epoch", "--trial_max_epoch", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_diffusione.db")
    parser.add_argument("--sampler_seed", type=int, default=2026)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--train_script", type=str, default="train.py")
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--enable_rel_search", action="store_true")
    parser.add_argument("--enable_phase_search", action="store_true")
    parser.add_argument("--initial_params", type=str, default=None)
    parser.add_argument(
        "--enqueue_json",
        action="append",
        default=[],
        help='Initial parameter group as JSON, e.g. \'{"topk":100,"layers":8,"fact_ratio":0.9}\'.',
    )
    parser.add_argument(
        "--extra_args",
        nargs="*",
        default=[],
        help="Extra raw arguments appended to train.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.current_trial_id = -1
    dataset = dataset_name(args.data_path)
    study_name = args.study_name or f"diffusione_{dataset}"

    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        n_startup_trials=3,
        multivariate=True,
        group=True,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    for params in load_initial_params(args.initial_params, args.enqueue_json):
        study.enqueue_trial(params)

    study.optimize(
        make_objective(args, study),
        n_trials=args.n_trials,
        gc_after_trial=True,
        catch=(RuntimeError,),
    )

    print("==> Best value:", study.best_value)
    print("==> Best params:")
    print(json.dumps(study.best_params, indent=2, sort_keys=True))
    print("==> Best user attrs:")
    print(json.dumps(study.best_trial.user_attrs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

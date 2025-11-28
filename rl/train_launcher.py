import argparse
import datetime
import itertools
import json
import subprocess


# -------------------------------------------------------------------------
# CURRICULUM DEFINITION
# -------------------------------------------------------------------------
CURRICULUM = [
    # (env_kwargs, timesteps, name, hyperparams)
    ({"num_drones": 5, "gui": False}, 1_500_000, "stage1_easy", None),
    ({"num_drones": 5, "gui": False}, 2_500_000, "stage2_full", None),
    (
        {"num_drones": 5, "gui": False},
        4_500_000,
        "stage3_obstacles",
        {"learning_rate": 1e-4, "target_kl": 0.04},
    ),
]


# -------------------------------------------------------------------------
# HYPERPARAMETER SWEEP DEFINITION
# -------------------------------------------------------------------------
HYPER_SWEEP = {
    "learning_rate": [3e-4, 1e-4],
    "gamma": [0.99, 0.995],
    "n_steps": [1024, 2048],
}


# -------------------------------------------------------------------------
def run_command(cmd):
    print(f"\n===============================")
    print(f"RUNNING:\n{cmd}")
    print(f"===============================\n")
    subprocess.run(cmd, shell=True)
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# CURRICULUM TRAINING
# -------------------------------------------------------------------------
def run_curriculum(run_name):
    print("\n=== STARTING CURRICULUM TRAINING ===\n")

    prev_model = None

    for stage in CURRICULUM:
        env_kwargs, timesteps, tag, hyper = stage

        stage_name = f"{run_name}_{tag}"

        cmd = f"python rl/train_multi.py " \
              f"--run-name {stage_name} " \
              f"--timesteps {timesteps} " \
              f"--num-drones {env_kwargs['num_drones']} "

        if hyper:
            hp_json = json.dumps(hyper)
            cmd += f"--hyper '{hp_json}' "

        if prev_model:
            cmd += f"--load-model {prev_model} "

        run_command(cmd)

        # After training stage: save for next stage
        prev_model = f"models/{stage_name}/ppo_final_model"

    print("\n=== CURRICULUM FINISHED ===\n")
    return prev_model  # final model path


# -------------------------------------------------------------------------
# SWEEP STARTING FROM A MODEL
# -------------------------------------------------------------------------
def run_hyper_sweep(run_name, base_model):
    print("\n=== STARTING HYPERPARAMETER SWEEP ===\n")

    keys = list(HYPER_SWEEP.keys())
    values = list(HYPER_SWEEP.values())

    for combo in itertools.product(*values):
        hp = dict(zip(keys, combo))

        tag = "_".join([f"{k}{v}" for k, v in hp.items()])
        exp_name = f"{run_name}_sweep_{tag}"
        hp_json = json.dumps(hp)

        cmd = (
            f"python rl/train_multi.py "
            f"--run-name {exp_name} "
            f"--timesteps 150000 "
            f"--num-drones 5 "
            f"--hyper '{hp_json}' "
            f"--load-model {base_model}"
        )

        run_command(cmd)

    print("\n=== SWEEP FINISHED ===\n")


# -------------------------------------------------------------------------
# ALL (curriculum → sweep)
# -------------------------------------------------------------------------
def run_all(run_name):
    print("\n=== BEGIN FULL PIPELINE (CURRICULUM → SWEEP) ===\n")

    final_model = run_curriculum(run_name)

    print(f"\n>>> Curriculum final model: {final_model}\n")

    run_hyper_sweep(run_name, final_model)

    print("\n=== FULL PIPELINE COMPLETED ===\n")


# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default=None,
                        help="Base name for run folders.")

    parser.add_argument("--curriculum", action="store_true",
                        help="Run curriculum training.")

    parser.add_argument("--sweep", action="store_true",
                        help="Run hyperparameter sweep only.")

    parser.add_argument("--all", action="store_true",
                        help="Run curriculum then hyperparameter sweep.")

    return parser.parse_args()


# -------------------------------------------------------------------------
def main():
    args = parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{timestamp}"

    if args.all:
        run_all(run_name)
    elif args.curriculum:
        run_curriculum(run_name)
    elif args.sweep:
        print("ERROR: --sweep requires a base model. Use --all or curriculum first.")
    else:
        print("You must choose one: --curriculum | --sweep | --all")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()

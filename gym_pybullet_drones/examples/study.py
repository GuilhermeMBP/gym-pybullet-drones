import optuna

def simple_objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    try:
        study = optuna.create_study(storage='sqlite:///simple_study.db', study_name="simple_study", direction='minimize', load_if_exists=True)
        study.optimize(simple_objective, n_trials=10)

        print("Best hyperparameters: ", study.best_params)
        print("Best value: ", study.best_value)
    except Exception as e:
        print(f"Failed to fetch study (reason={e})")

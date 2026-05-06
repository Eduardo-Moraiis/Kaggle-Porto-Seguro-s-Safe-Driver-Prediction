import pandas as pd
import numpy as np
import json
import optuna
import lightgbm as lgb


from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score
from pathlib import Path


def gini_normalized(auc):
    return 2 * auc.mean() - 1

def objective(trial, X, y):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,

        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),

        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),

        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),

        "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),

        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),

        "max_bin": trial.suggest_int("max_bin", 128, 512),

        "n_jobs": -1
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = LGBMClassifier(
            **params,
            n_estimators=3000
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(100, verbose=False)
            ]
        )

        preds = model.predict_proba(X_valid)[:, 1]
        aucs.append(roc_auc_score(y_valid, preds))

    return 2 * np.mean(aucs) - 1

def run_optuna(X, y):
    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=30,
        show_progress_bar=True
    )

    print("\nMelhores parâmetros:")
    print(study.best_params)
    print(f"\nMelhor Gini: {study.best_value:.6f}")

    return study.best_params, study.best_value

def main():
    print("\nEscolha uma opção:")
    print("1 - Rodar Optuna + salvar config")
    print("2 - Gerar submissão Kaggle a partir do JSON")

    opcao = input("Opção: ")

    if opcao == "1":

        df_train = pd.read_csv('data_sets/train.csv')

        X = df_train.drop(columns=["id", "target"])
        y = df_train["target"]

        print("\nRodando Optuna...")
        best_params, best_gini = run_optuna(X, y)

        # garantir parâmetros base
        best_params.update({
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "n_jobs": -1
        })

        print("\nValidando modelo com cross_validate...")
        model = LGBMClassifier(**best_params, n_estimators=1500)

        scores = cross_validate(
            model,
            X,
            y,
            cv=10,
            scoring=['roc_auc'],
            n_jobs=-1
        )

        gini_score = gini_normalized(scores['test_roc_auc'])

        print(f"\nGini CV: {gini_score:.6f}")

        print("\nSalvando configuração...")
        path = Path("utils")
        path.mkdir(parents=True, exist_ok=True)
        with open("utils/best_params_lgb.json", "w") as f:
            json.dump({
                "best_params": best_params,
                "gini": gini_score
            }, f, indent=4)

    elif opcao == "2":

        df_train = pd.read_csv('data_sets/train.csv')
        df_test = pd.read_csv('data_sets/test.csv')

        X = df_train.drop(columns=["id", "target"])
        y = df_train["target"]
        X_kaggle = df_test.drop(columns=["id"])

        print("\nCarregando parâmetros...")
        with open("utils/best_params_lgb.json", "r") as f:
            json_config = json.load(f)

        params = json_config["best_params"]

        model = LGBMClassifier(**params, n_estimators=1500)

        print("\nTreinando modelo final...")
        model.fit(X, y)

        print("\nGerando submissão...")
        y_pred = model.predict_proba(X_kaggle)[:, 1]

        df_test["target"] = y_pred
        path = Path("submissions")
        path.mkdir(parents=True, exist_ok=True)
        df_test[["id", "target"]].to_csv(
            "submissions/solucao_tp1_lgb_completa_2026.csv",
            index=False
        )

        print("\nArquivo gerado com sucesso")

    else:
        print("Opção inválida")


if __name__ == "__main__":
    main()
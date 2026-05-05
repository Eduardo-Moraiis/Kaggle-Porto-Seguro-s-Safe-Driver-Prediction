#best_params = {'n_estimators': 704, 'max_depth': 5, 'learning_rate': 0.0719502857394281, 'subsample': 0.8704612117424998, 'colsample_bytree': 0.6022693114843498, 'gamma': 1.7921296413939896, 'min_child_weight': 8, 'reg_alpha': 2.252119399276311, 'reg_lambda': 2.562495817814109}
#segundo params = {'n_estimators': 1011, 'learning_rate': 0.01807889810534756, 'max_depth': 7, 'min_child_weight': 3, 'subsample': 0.6981980135543202, 'colsample_bytree': 0.8387016028430684, 'gamma': 3.3319828741163384, 'reg_alpha': 2.504566011478833, 'reg_lambda': 2.985217740698493, 'max_bin': 388}

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from xgboost import XGBClassifier
from boruta import BorutaPy
import shap
import optuna_xg
import optuna
import json

def objective(trial, X, y):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",

        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),

        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),

        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        "gamma": trial.suggest_float("gamma", 0, 5),

        "reg_alpha": trial.suggest_float("reg_alpha", 0, 3),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 3),

        "max_bin": trial.suggest_int("max_bin", 128, 512),

        "random_state": 42,
    }

    model = XGBClassifier(**params)

    scores = cross_validate(model, X, y, scoring=['roc_auc'], cv=10, n_jobs=-1)

    gini_mean = 2 * scores['test_roc_auc'].mean() - 1

    return gini_mean


def run_optuna(X, y):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=30, show_progress_bar=True)

    print("\nMelhores parâmetros:")
    print(study.best_params)
    print(f"Melhor Gini: {study.best_value:.4f}")

    return study.best_params


def boruta_selection(model, X, y):
    print("--------Rodando Boruta--------")

    selector = BorutaPy(
        estimator=model,
        n_estimators='auto',
        verbose=2,
        random_state=42
    )

    selector.fit(X.values, y.values)

    selected_features = X.columns[selector.support_]

    print("\nFeatures selecionadas (Boruta):")
    print(selected_features)

    return selected_features

def evaluate_model(model, X, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    scores = cross_validate(
        model,
        X,
        y,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    gini_mean = 2 * scores["test_score"].mean() - 1
    gini_std = 2 * scores["test_score"].std()

    return gini_mean, gini_std


def main():
    print("\nEscolha uma opção:")
    print("1 - Rodar Boruta + Optuna + salvar config")
    print("2 - Gerar submissão Kaggle a partir do JSON")

    opcao = input("Opção: ")

    if opcao == "1":
        df = pd.read_csv("data_sets/train.csv")
        X = df.drop(columns=["id", "target"])
        y = df["target"]

        base_params = {
            'n_estimators': 276,
            'max_depth': 4,
            'learning_rate': 0.0636,
            'subsample': 0.75,
            'colsample_bytree': 0.66,
            "random_state": 42,
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cuda",
        }

        base_model = XGBClassifier(**base_params)

        print("\nRodando seleção de features (Boruta)...")
        selected_features = boruta_selection(base_model, X, y)

        print("\nShape após seleção:", X[selected_features].shape)

        X_opt = X[selected_features].astype(np.float32)

        print("\nRodando Optuna...")
        best_params = run_optuna(X_opt, y)

        best_params.update({
            "random_state": 42,
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cuda",
        })

        print("\nSalvando configuração...")
        with open("utils/xg_config.json", "w") as f:
            json.dump({
                "best_params": best_params,
                "selected_features": list(selected_features)
            }, f, indent=4)

        print("\nTreinando modelo final...")
        model = XGBClassifier(**best_params)
        gini, gini_std = evaluate_model(model, X_opt, y)

        print(f"\nGini: {gini:.4f}")
        print(f"Gini std: {gini_std:.4f}")

    elif opcao == "2":
        print("\nCarregando configuração salva...")

        with open("utils/xg_config.json", "r") as f:
            config = json.load(f)

        best_params = config["best_params"]
        selected_features = config["selected_features"]

        df = pd.read_csv("data_sets/train.csv")
        df_test = pd.read_csv("data_sets/test.csv")

        X = df.drop(columns=["id", "target"])
        y = df["target"]

        X_train = X[selected_features]
        X_kaggle = df_test[selected_features]

        print("\nTreinando modelo final com todos os dados...")
        model = XGBClassifier(**best_params)
        model.fit(X_train, y)

        print("\nGerando submissão...")
        y_pred = model.predict_proba(X_kaggle)[:, 1]

        df_test["target"] = y_pred
        df_test[["id", "target"]].to_csv("submissions/solucao_tp1_xg_completa_2026.csv", index=False)

        print("\nArquivo solucao_tp1_2026.csv gerado com sucesso")

    else:
        print("Opção inválida!")




if __name__ == "__main__":
    main()
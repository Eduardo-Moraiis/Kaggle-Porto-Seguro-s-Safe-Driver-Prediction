import pandas as pd
import numpy as np
import json
import optuna

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score
from boruta import BorutaPy

def gini_normalized(auc):
    return 2 * auc.mean() - 1

def boruta_selection(X, y):
    print("\nRodando Boruta...")

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        n_jobs=-1,
        random_state=42
    )

    selector = BorutaPy(
        estimator=model,
        n_estimators='auto',
        verbose=2,
        random_state=42
    )

    selector.fit(X.values, y.values)

    selected_features = X.columns[selector.support_]

    print("\nFeatures selecionadas:")
    print(list(selected_features))

    return selected_features

def objective(trial, X, y):
    var_smoothing = trial.suggest_float(
        "var_smoothing",
        1e-12,
        1e-6,
        log=True
    )

    model = GaussianNB(var_smoothing=var_smoothing)

    scores = cross_validate(
        model,
        X,
        y,
        cv=5,
        scoring=["roc_auc"],
        n_jobs=-1
    )

    return gini_normalized(scores["test_roc_auc"])


def run_optuna(X, y):
    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=30
    )

    print("\nMelhor parâmetro:")
    print(study.best_params)
    print(f"Melhor Gini: {study.best_value:.6f}")

    return study.best_params, study.best_value

def main():
    print("\nEscolha uma opção:")
    print("1 - GaussianNB SEM seleção (com Optuna)")
    print("2 - GaussianNB COM seleção (Boruta + Optuna)")
    print("3 - Gerar submissão Kaggle")

    opcao = input("Opção: ")

    if opcao == "1":
        df = pd.read_csv("data_sets/train.csv")

        X = df.drop(columns=["id", "target"])
        y = df["target"]

        print("\nRodando Optuna...")
        best_params, best_gini = run_optuna(X, y)

        model = GaussianNB(**best_params)

        scores = cross_validate(
            model,
            X,
            y,
            cv=5,
            scoring=["roc_auc"],
            n_jobs=-1
        )

        gini = gini_normalized(scores["test_roc_auc"])

        print(f"\nGini final: {gini:.6f}")

        print("\nSalvando config...")
        with open("utils/gnb_config.json", "w") as f:
            json.dump({
                "best_params": best_params,
                "selected_features": None,
                "gini": gini
            }, f, indent=4)

    elif opcao == "2":
        df = pd.read_csv("data_sets/train.csv")

        X = df.drop(columns=["id", "target"])
        y = df["target"]

        selected_features = boruta_selection(X, y)

        X_sel = X[selected_features]

        print("\nRodando Optuna...")
        best_params, best_gini = run_optuna(X_sel, y)

        model = GaussianNB(**best_params)

        scores = cross_validate(
            model,
            X_sel,
            y,
            cv=5,
            scoring=["roc_auc"],
            n_jobs=-1
        )

        gini = gini_normalized(scores["test_roc_auc"])

        print(f"\nGini final: {gini:.6f}")

        print("\nSalvando config...")
        with open("utils/gnb_config.json", "w") as f:
            json.dump({
                "best_params": best_params,
                "selected_features": list(selected_features),
                "gini": gini
            }, f, indent=4)

    elif opcao == "3":
        df_train = pd.read_csv("data_sets/train.csv")
        df_test = pd.read_csv("data_sets/test.csv")

        X = df_train.drop(columns=["id", "target"])
        y = df_train["target"]

        with open("utils/gnb_config.json", "r") as f:
            config = json.load(f)

        params = config["best_params"]
        selected_features = config["selected_features"]

        if selected_features is not None:
            X = X[selected_features]
            X_kaggle = df_test[selected_features]
        else:
            X_kaggle = df_test.drop(columns=["id"])

        model = GaussianNB(**params)

        print("\nTreinando modelo final...")
        model.fit(X, y)

        print("\nGerando submissão...")
        y_pred = model.predict_proba(X_kaggle)[:, 1]

        df_test["target"] = y_pred
        df_test[["id", "target"]].to_csv(
            "submissions/solucao_tp1_gnb_completa_2026.csv",
            index=False
        )

        print("\nArquivo gerado com sucesso")

    else:
        print("Opção inválida!")


if __name__ == "__main__":
    main()
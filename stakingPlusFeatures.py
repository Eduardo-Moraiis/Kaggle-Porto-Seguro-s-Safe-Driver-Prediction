import pandas as pd
import numpy as np
import json

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

import optuna

N_SPLITS = 5
RANDOM_STATE = 42

def gini(y_true, y_pred):
    return 2 * roc_auc_score(y_true, y_pred) - 1


def optimize_logistic(X_stack, y):

    def objective(trial):

        params = {
            "C": trial.suggest_float("C", 0.01, 3.0, log=True),
            "solver": "lbfgs",
            "max_iter": 1000
        }

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        oof_pred = np.zeros(X_stack.shape[0])

        for train_idx, valid_idx in skf.split(X_stack, y):

            X_tr, X_val = X_stack[train_idx], X_stack[valid_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

            model = LogisticRegression(**params)
            model.fit(X_tr, y_tr)

            oof_pred[valid_idx] = model.predict_proba(X_val)[:, 1]

        return gini(y, oof_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("\nMelhor params Logistic:")
    print(study.best_params)

    return study.best_params


def load_configs():
    # parâmetros previamente obtidos via otimização individual de cada modelo
    with open("utils/xg_config.json") as f:
        xgb_config = json.load(f)

    with open("utils/best_params_lgb.json") as f:
        lgb_config = json.load(f)

    with open("utils/gnb_config.json") as f:
        gnb_config = json.load(f)

    return xgb_config, lgb_config, gnb_config


def generate_oof(X, y, X_test,
                 xgb_config, lgb_config, gnb_config):

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros((X.shape[0], 3))
    test = np.zeros((X_test.shape[0], 3))

    xgb_feats = xgb_config["selected_features"]
    gnb_feats = gnb_config.get("selected_features")

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}")

        X_train_full, X_valid_full = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        X_train_xgb = X_train_full[xgb_feats]
        X_valid_xgb = X_valid_full[xgb_feats]
        X_test_xgb = X_test[xgb_feats]

        if gnb_feats is not None:
            X_train_gnb = X_train_full[gnb_feats]
            X_valid_gnb = X_valid_full[gnb_feats]
            X_test_gnb = X_test[gnb_feats]
        else:
            X_train_gnb = X_train_full
            X_valid_gnb = X_valid_full
            X_test_gnb = X_test

        X_train_lgb = X_train_full
        X_valid_lgb = X_valid_full
        X_test_lgb = X_test

        xgb = XGBClassifier(**xgb_config["best_params"])
        lgb = LGBMClassifier(
            **lgb_config["best_params"],
            objective="binary",
            metric="auc",
            verbosity=-1,
            n_estimators=1500
        )
        nb = GaussianNB(**gnb_config["best_params"])

        xgb.fit(X_train_xgb, y_train)
        lgb.fit(X_train_lgb, y_train)
        nb.fit(X_train_gnb, y_train)

        oof[valid_idx, 0] = xgb.predict_proba(X_valid_xgb)[:, 1]
        oof[valid_idx, 1] = lgb.predict_proba(X_valid_lgb)[:, 1]
        oof[valid_idx, 2] = nb.predict_proba(X_valid_gnb)[:, 1]

        test[:, 0] += xgb.predict_proba(X_test_xgb)[:, 1] / N_SPLITS
        test[:, 1] += lgb.predict_proba(X_test_lgb)[:, 1] / N_SPLITS
        test[:, 2] += nb.predict_proba(X_test_gnb)[:, 1] / N_SPLITS

    print("\nGini base models:")
    for i, name in enumerate(["XGB", "LGB", "NB"]):
        print(f"{name}: {gini(y, oof[:, i]):.5f}")

    return oof, test


def stacking(X, y, X_test, xgb_config, lgb_config, gnb_config):

    oof_base, test_base = generate_oof(
        X, y, X_test,
        xgb_config, lgb_config, gnb_config
    )

    print("\nConcatenando features originais + predições (stacking features)...")

    # concatena dataset original + predições dos modelos
    X_stack = np.hstack([X.values, oof_base])
    X_test_stack = np.hstack([X_test.values, test_base])

    print("Shape após concatenação:", X_stack.shape)

    print("\nAplicando normalização...")
    scaler = StandardScaler()
    X_stack = scaler.fit_transform(X_stack)
    X_test_stack = scaler.transform(X_test_stack)

    print("Normalização concluída")

    print("\nRodando Optuna para Logistic...")
    best_params = optimize_logistic(X_stack, y)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    meta_oof = np.zeros(X_stack.shape[0])
    meta_test = np.zeros(X_test_stack.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_stack, y)):
        print(f"\nMeta Fold {fold+1}")

        X_tr, X_val = X_stack[train_idx], X_stack[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        meta_model = LogisticRegression(
            **best_params,
            solver="lbfgs",
            max_iter=1000
        )

        meta_model.fit(X_tr, y_tr)

        meta_oof[valid_idx] = meta_model.predict_proba(X_val)[:, 1]
        meta_test += meta_model.predict_proba(X_test_stack)[:, 1] / N_SPLITS

    print(f"\nGini STACKING (OOF): {gini(y, meta_oof):.5f}")

    return meta_test


def main():
    print("\nCarregando dados...")
    df_train = pd.read_csv("data_sets/train.csv")
    df_test = pd.read_csv("data_sets/test.csv")

    X = df_train.drop(columns=["id", "target"])
    y = df_train["target"]
    X_test = df_test.drop(columns=["id"])

    print("Dados carregados")

    print("\nCarregando configs...")
    xgb_config, lgb_config, gnb_config = load_configs()

    print("\nRodando stacking...")
    y_pred = stacking(X, y, X_test, xgb_config, lgb_config, gnb_config)

    print("\nGerando submissão...")
    df_test["target"] = y_pred
    df_test[["id", "target"]].to_csv("submissions/submission_stackingPlusFeatures_final.csv", index=False)

    print("\nSubmissão gerada com sucesso")


if __name__ == "__main__":
    main()
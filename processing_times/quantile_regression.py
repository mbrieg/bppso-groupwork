import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss


def fit_and_eval_quantiles(data, target, cat_cols, num_cols, seed=42, qs=(0.10, 0.50, 0.90)):
    """
    Fits GradientBoosting quantile regressors for given quantiles qs=(low,mid,high)
    and evaluates coverage/width/pinball on the same quantiles.
    """
    qL, qM, qH = qs

    data = data.dropna(subset=[target] + num_cols).copy()

    X = data[cat_cols + num_cols]
    y = data[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    pre = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),("num", "passthrough", num_cols),])

    def fit_q(alpha):
        return Pipeline(steps=[("pre", pre),
            ("q", GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=seed)),
        ]).fit(X_train, y_train)

    mL = fit_q(qL)
    mM = fit_q(qM)
    mH = fit_q(qH)

    pL = mL.predict(X_test)
    pM = mM.predict(X_test)
    pH = mH.predict(X_test)

    pL = np.minimum(pL, pM)
    pH = np.maximum(pH, pM)

    # values
    coverage = np.mean((y_test >= pL) & (y_test <= pH))
    avg_width = np.mean(pH - pL)

    pinL = mean_pinball_loss(y_test, pL, alpha=qL)
    pinM = mean_pinball_loss(y_test, pM, alpha=qM)
    pinH = mean_pinball_loss(y_test, pH, alpha=qH)

    print(f"Coverage [q{int(qL*100)},q{int(qH*100)}] (target ~{qH-qL:.2f}):", coverage)
    print("Avg interval width (sec):", avg_width)
    print(f"Pinball loss q{int(qL*100)}/q{int(qM*100)}/q{int(qH*100)}:", pinL, pinM, pinH)

    return {
        "models": {qL: mL, qM: mM, qH: mH},
        "qs": qs,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "pred": {qL: pL, qM: pM, qH: pH},
        "metrics": {
            "coverage": float(coverage),
            "avg_width": float(avg_width),
            f"pin{int(qL*100)}": float(pinL),
            f"pin{int(qM*100)}": float(pinM),
            f"pin{int(qH*100)}": float(pinH),
        },
    }


def baseline_empirical_by_activity(X_train, y_train, X_test, y_test,
                                   act_col="concept:name", qs=(0.10, 0.50, 0.90)):
    """
    lookup-table, per-activity quantile baseline
    """
    qL, qM, qH = qs

    train_tmp = X_train.copy()
    train_tmp["y"] = y_train.values

    q_train = train_tmp.groupby(act_col)["y"].quantile([qL, qM, qH]).unstack()
    global_q = np.quantile(y_train, [qL, qM, qH])

    acts = X_test[act_col].values
    bL = np.array([q_train.loc[a, qL] if a in q_train.index else global_q[0] for a in acts], dtype=float)
    bM = np.array([q_train.loc[a, qM] if a in q_train.index else global_q[1] for a in acts], dtype=float)
    bH = np.array([q_train.loc[a, qH] if a in q_train.index else global_q[2] for a in acts], dtype=float)


    # values
    cov_b = np.mean((y_test >= bL) & (y_test <= bH))
    wid_b = np.mean(bH - bL)

    pin_bL = mean_pinball_loss(y_test, bL, alpha=qL)
    pin_bM = mean_pinball_loss(y_test, bM, alpha=qM)
    pin_bH = mean_pinball_loss(y_test, bH, alpha=qH)

    print(f"BASELINE Coverage [q{int(qL*100)},q{int(qH*100)}]:", cov_b)
    print("BASELINE Avg width:", wid_b)
    print(f"BASELINE Pinball q{int(qL*100)}/q{int(qM*100)}/q{int(qH*100)}:", pin_bL, pin_bM, pin_bH)

    return {
        "qs": qs,
        "coverage": float(cov_b),
        "avg_width": float(wid_b),
        f"pin{int(qL*100)}": float(pin_bL),
        f"pin{int(qM*100)}": float(pin_bM),
        f"pin{int(qH*100)}": float(pin_bH),
    }

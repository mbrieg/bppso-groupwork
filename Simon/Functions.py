import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd


def density_plot_from_counts(
        durations_full, act,
        bins=80, zoom_q=1.0, log_x=False, drop_zeros=False, plot_data=True,
        show_model=False, dist="gamma", max_samples=200000, min_samples=200
):
    """
    Density plot (histogram with density=True) for one activity.
    Uses the complete processing time list in format: durations_full[act] = [(count, value), ...]
    drop_zeros=True removes duration==0(for zero-inflated).
    show_model=True overlays the model on the plot
    dist: "gamma" or "lognorm"
    """
    lst = durations_full.get(act, [])
    if not lst:
        print("No data for", act)
        return

    values = np.array([v for cnt, v in lst], dtype=float)
    weights = np.array([cnt for cnt, v in lst], dtype=float)

    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[m], weights[m]

    # drop zeros
    if drop_zeros:
        m0 = values > 0
        values, weights = values[m0], weights[m0]
        if len(values) == 0:
            print("Only zeros -> nothing to plot for", act)
            return

    idx = np.argsort(values)
    values, weights = values[idx], weights[idx]

    # zoom threshold on quantiles
    if zoom_q < 1.0:
        cum = np.cumsum(weights)
        thr = values[np.searchsorted(cum, zoom_q * cum[-1], side="left")]
    else:
        thr = values.max()

    # filter to range
    mask = values <= thr
    v = values[mask]
    w = weights[mask]

    if log_x:
        mask_pos = v > 0
        v, w = v[mask_pos], w[mask_pos]
        if len(v) == 0:
            print("No positive values in range -> log_x not possible for", act)
            return

    # plot empirical density
    if plot_data:
        plt.figure(figsize=(9, 5))
        plt.hist(v, bins=bins, weights=w, density=True, alpha=0.7, label="Empirical")

    # fit + overlay the model
    model = None
    if show_model:
        # Fit always on positive values
        mfit = v > 0
        vfit = v[mfit]
        wfit = w[mfit]

        # capped expand using weights
        pos = []
        for val, cnt in zip(vfit, wfit.astype(int)):
            if len(pos) >= max_samples:
                break
            take = min(int(cnt), max_samples - len(pos))
            if take > 0:
                pos.extend([float(val)] * take)

        pos = np.asarray(pos, dtype=float)

        # Fitting the model to the data
        if len(pos) >= min_samples:
            x = np.linspace(max(v.min(), 1e-9), v.max(), 1000)

            if dist == "gamma":
                a, loc, scale = stats.gamma.fit(pos, floc=0)
                y = stats.gamma.pdf(x, a, loc=loc, scale=scale)
                params = (float(a), float(loc), float(scale))
                model = {"type": "parametric", "dist": "gamma",
                         "params": {"a": float(a), "scale": float(scale)}}
                plt.plot(x, y, label=f"Gamma fit")
            elif dist == "lognorm":
                s, loc, scale = stats.lognorm.fit(pos, floc=0)
                y = stats.lognorm.pdf(x, s, loc=loc, scale=scale)
                params = (float(s), float(loc), float(scale))
                model = {"type": "parametric", "dist": "lognorm",
                         "params": {"s": float(s), "scale": float(scale)}}
                plt.plot(x, y, label=f"Lognorm fit")
            else:
                raise ValueError("dist must be 'gamma' or 'lognorm'")
        else:
            print(f"Too few samples to fit for {act}: {len(pos)} (min={min_samples})")

    if plot_data or show_model:
        plt.title(f"{act} – density (zoom_q={zoom_q}, drop_zeros={drop_zeros}, show_model={show_model})")
        plt.xlabel("seconds")
        plt.ylabel("density")
        if log_x:
            plt.xscale("log")
            plt.xlim(max(v.min(), 1e-6), v.max())
        else:
            plt.xlim(0 if not drop_zeros else v.min(), thr)

        if show_model:
            plt.legend()

        plt.show()

    # retrun the model
    return model


"""
All lifecycle transitions that start, pause and terminate an activity
"""
BEGIN = {"start", "resume"}
END = {"suspend", "complete", "ate_abort", "withdraw"}
TERMINAL = {"complete", "ate_abort", "withdraw"}


def build_processing_times(df, workflow_only=True):
    """
    builds the processing times for activities that have at least 2 lifecycel status
    :param df: dataframe of the event log
    :param workflow_only: W_ Activities only
    :return: seg, proc_inst: seg contains all intermediate processing times per intermediate activity that can be paused
                proc_inst: complete processing time for one activity
    """
    x = df.dropna(subset=["time:timestamp"]).copy()
    x["time:timestamp"] = pd.to_datetime(x["time:timestamp"], utc=True)

    if workflow_only:
        x = x[x["concept:name"].astype(str).str.startswith("W_")]

    x = x.sort_values(["case:concept:name", "concept:name", "time:timestamp"])

    seg_rows = []
    inst_rows = []

    for (case, act), g in x.groupby(["case:concept:name", "concept:name"], sort=False):
        inst = 0
        open_ts = None
        open_res = None
        proc_sum = 0.0
        worked = False  # act was paused?

        for _, r in g.iterrows():
            tr = r["lifecycle:transition"]
            ts = r["time:timestamp"]
            res = r.get("org:resource", np.nan)

            if tr in BEGIN:
                open_ts = ts
                open_res = res
                worked = True

            elif tr in END:
                if open_ts is not None:
                    dur = (ts - open_ts).total_seconds()
                    if dur >= 0:
                        seg_rows.append({
                            "case:concept:name": case,
                            "concept:name": act,
                            "instance": inst,
                            "org:resource": open_res,
                            "seg_start": open_ts,
                            "seg_end": ts,
                            "end_transition": tr,
                            "proc_seconds": dur
                        })
                        proc_sum += dur
                    open_ts = None
                    open_res = None

                # for terminal lifecycle
                if tr in TERMINAL:
                    if worked:
                        inst_rows.append({
                            "case:concept:name": case,
                            "concept:name": act,
                            "instance": inst,
                            "proc_seconds": proc_sum
                        })
                    inst += 1
                    open_ts = None
                    open_res = None
                    proc_sum = 0.0
                    worked = False

    seg = pd.DataFrame(seg_rows)
    proc_inst = pd.DataFrame(inst_rows)
    return seg, proc_inst


def select_model_gamma_vs_lognorm(durations_full, act, max_samples=200000, min_samples=200):
    """
    chooses between gamma and lognorm model to decide which one is better based on AIC value
    :param durations_full: list with all the processing times / list with all waiting times (maybe not implemented)
    :param act: The activity to select the model for
    :param max_samples: max boundary
    :param min_samples: min boundary
    :return: preference for model and AIC values for both models
    """
    lst = durations_full.get(act, [])
    pos = []
    for cnt, val in lst:
        if val > 0:
            take = min(int(cnt), max_samples - len(pos))
            if take <= 0:
                break
            pos.extend([float(val)] * take)

    x = np.asarray(pos, dtype=float)
    if len(x) < min_samples:
        return None

    # fit both
    s, loc_ln, scale_ln = stats.lognorm.fit(x, floc=0)
    a, loc_g, scale_g = stats.gamma.fit(x, floc=0)

    # log-likelihoods
    ll_ln = np.sum(stats.lognorm.logpdf(x, s, loc=loc_ln, scale=scale_ln))
    ll_g = np.sum(stats.gamma.logpdf(x, a, loc=loc_g, scale=scale_g))

    # AIC
    aic_ln = 2*2 - 2*ll_ln
    aic_g  = 2*2 - 2*ll_g

    if aic_ln < aic_g:
        return {"best": "lognorm", "aic_lognorm": float(aic_ln), "aic_gamma": float(aic_g),
                "lognorm": (float(s), float(loc_ln), float(scale_ln)),
                "gamma": (float(a), float(loc_g), float(scale_g)),
                "n_pos_used": int(len(x))}
    else:
        return {"best": "gamma", "aic_lognorm": float(aic_ln), "aic_gamma": float(aic_g),
                "lognorm": (float(s), float(loc_ln), float(scale_ln)),
                "gamma": (float(a), float(loc_g), float(scale_g)),
                "n_pos_used": int(len(x))}

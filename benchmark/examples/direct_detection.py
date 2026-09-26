"""Evaluate pooled direct diagnostic scores against signed PMP errors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, SymmetricalLogLocator
from sklearn.metrics import roc_auc_score


DIAGNOSTICS = ("l2", "linf", "density", "mmd")


def evaluate_direct_detection(
    diagnostics: pd.DataFrame,
    comparison: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    keys: tuple[str, str],
    loss: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label a dataset high error when any signed PMP component leaves its interval."""
    if comparison.index.names != list(keys) or not comparison.index.is_unique:
        raise ValueError("Comparison needs unique dataset keys")
    if diagnostics["loss_function"].ne(loss).any():
        raise ValueError("Diagnostic loss does not match comparison")
    required = [*keys, "diagnostic", "rho", "rho_low"]
    if diagnostics[required].isna().any().any() or diagnostics.duplicated([*keys, "diagnostic"]).any():
        raise ValueError("Diagnostic rows need one finite score per dataset and method")
    expected = {
        (diagnostic, *dataset_key)
        for diagnostic in DIAGNOSTICS
        for dataset_key in comparison.index
    }
    observed = {
        (row.diagnostic, *(getattr(row, key) for key in keys))
        for row in diagnostics.itertuples()
    }
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise ValueError("Expected exactly one score per dataset and diagnostic")
    if not np.isfinite(diagnostics[["rho", "rho_low"]].to_numpy(float)).all():
        raise ValueError("Diagnostic scores must be finite")
    models = tuple(column.removeprefix("p_direct_") for column in comparison if column.startswith("p_direct_"))
    if len(models) != 4 or any(f"p_gold_{model}" not in comparison for model in models):
        raise ValueError("Comparison needs four direct and gold PMP components")
    if (
        len(calibration) != len(models)
        or set(calibration["candidate_model"]) != set(models)
        or calibration["candidate_model"].duplicated().any()
        or calibration["loss_function"].ne(loss).any()
        or calibration["metric"].ne("signed_pmp_error").any()
        or calibration["n_values"].le(0).any()
        or not np.allclose(calibration["lower_quantile"], 0.05)
        or not np.allclose(calibration["upper_quantile"], 0.95)
    ):
        raise ValueError("Expected four candidate-specific signed PMP 90% intervals")
    bounds = calibration.set_index("candidate_model").loc[list(models)]
    lower = bounds["lower_threshold"].to_numpy(dtype=float)
    upper = bounds["threshold"].to_numpy(dtype=float)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper).any():
        raise ValueError("Invalid signed PMP error interval")
    signed = pd.DataFrame({
        f"signed_pmp_error_{model}": comparison[f"p_direct_{model}"] - comparison[f"p_gold_{model}"]
        for model in models
    }, index=comparison.index)
    signed_values = signed.to_numpy(dtype=float)
    if not np.isfinite(signed_values).all():
        raise ValueError("Signed PMP errors must be finite")
    high_error = ((signed_values < lower) | (signed_values > upper)).any(axis=1)
    labels = pd.Series(high_error, index=comparison.index, name="high_pmp_error")
    data = diagnostics.merge(
        signed.join(labels).reset_index(),
        on=list(keys), validate="many_to_one",
    )
    data["diagnostic_positive"] = data["rho"] > 1.0
    for model, low, high in zip(models, lower, upper):
        data[f"pmp_error_lower_threshold_{model}"] = low
        data[f"pmp_error_upper_threshold_{model}"] = high
    threshold_columns = {
        f"pmp_error_{bound}_threshold_{model}": value
        for model, low, high in zip(models, lower, upper)
        for bound, value in (("lower", low), ("upper", high))
    }
    rows = []
    for diagnostic, group in data.groupby("diagnostic", sort=False):
        actual = group["high_pmp_error"].to_numpy(bool)
        predicted = group["diagnostic_positive"].to_numpy(bool)
        tp, fp = int((actual & predicted).sum()), int((~actual & predicted).sum())
        fn, tn = int((actual & ~predicted).sum()), int((~actual & ~predicted).sum())
        rows.append({
            "loss_function": loss,
            "diagnostic": diagnostic,
            "n_datasets": len(group),
            "n_high_error": int(actual.sum()),
            "pmp_error_coverage": 0.90,
            **threshold_columns,
            "diagnostic_rho_threshold": 1.0,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "FNR": fn / (fn + tp) if fn + tp else np.nan,
            "FPR": fp / (fp + tn) if fp + tn else np.nan,
            "ROC_AUC": roc_auc_score(actual, group["rho"]) if actual.any() and (~actual).any() else np.nan,
        })
    return data, pd.DataFrame(rows)


def plot_direct_scores(
    data_by_loss: dict[str, pd.DataFrame],
    *,
    loss_labels: dict[str, str],
    loss_colors: dict[str, str],
    output_stem,
    empirical_dataset: str | None = None,
    facet_style: bool = False,
    reference_style: bool = False,
    point_size: float = 13,
    point_alpha: float = 0.35,
    source_colors: dict[str, str] | None = None,
    source_column: str = "source_model",
    source_labels: dict[str, str] | None = None,
    loss_markers: dict[str, str] | None = None,
    y_scale: str = "linear",
    y_symlog_linthresh: float = 0.01,
    y_limits: tuple[float, float] | None = None,
    pmp_threshold_band: bool = False,
) -> None:
    """Plot candidate PMP components by row and diagnostics by column."""
    if not data_by_loss:
        raise ValueError("At least one trained network is required")
    if y_scale not in {"linear", "symlog"}:
        raise ValueError("y_scale must be 'linear' or 'symlog'")
    if y_scale == "symlog" and (
        not np.isfinite(y_symlog_linthresh) or y_symlog_linthresh <= 0
    ):
        raise ValueError("y_symlog_linthresh must be positive and finite")
    if y_limits is not None and (
        len(y_limits) != 2 or not np.isfinite(y_limits).all()
        or y_limits[0] >= y_limits[1]
    ):
        raise ValueError("y_limits must contain two increasing finite values")
    if source_colors is not None:
        if not reference_style:
            raise ValueError("Source-colored plot needs reference_style")
        if any(source_column not in data for data in data_by_loss.values()):
            raise ValueError(f"Source-colored plot needs a {source_column} column")
        plotted_sources = set().union(
            *(set(data[source_column].unique()) for data in data_by_loss.values())
        )
        if not plotted_sources.issubset(source_colors):
            raise ValueError(f"Missing colors for sources: {plotted_sources - source_colors.keys()}")
        loss_markers = loss_markers or {loss: marker for loss, marker in
                                        zip(data_by_loss, ("o", "s", "^"), strict=False)}
        if set(data_by_loss) != set(loss_markers):
            raise ValueError("Provide exactly one marker for every loss")
        threshold_styles = {loss: style for loss, style in
                            zip(data_by_loss, (":", "--", "-."), strict=False)}
        if len(threshold_styles) != len(data_by_loss):
            raise ValueError("Source-colored plot supports at most three losses")
    facet_style = facet_style or reference_style
    titles = {"l2": r"$L_2$", "linf": r"$L_\infty$", "density": "Density", "mmd": "Kernel MMD"}
    if reference_style:
        titles["mmd"] = "Kernel"
    markers = ("o", "s", "^", "D")
    signed_columns = [
        column for column in next(iter(data_by_loss.values()))
        if column.startswith("signed_pmp_error_")
    ]
    if len(signed_columns) != 4 or any(
        set(signed_columns) != {
            column for column in data if column.startswith("signed_pmp_error_")
        }
        for data in data_by_loss.values()
    ):
        raise ValueError("Expected four signed PMP error components for plotting")
    fig, axes = plt.subplots(
        4, 4,
        figsize=(23.5, 22.0) if reference_style else (17.6, 14.4) if facet_style else (18, 13),
        sharex="col", sharey=True,
        constrained_layout=not facet_style,
    )
    if reference_style:
        fig.subplots_adjust(left=0.105, right=0.925,
                            bottom=(4.0 if source_colors is not None else 2.75) / 22.0,
                            top=1.0 - 1.375 / 22.0, wspace=0.12, hspace=0.14)
    elif facet_style:
        fig.subplots_adjust(left=0.10, right=0.91, bottom=0.10, top=0.91,
                            wspace=0.12, hspace=0.14)
    for row, (marker, column) in enumerate(zip(markers, signed_columns)):
        model = column.removeprefix("signed_pmp_error_")
        point_marker = "o" if facet_style else marker
        if not facet_style:
            axes[row, 0].set_ylabel(model.upper())
        for col, metric in enumerate(DIAGNOSTICS):
            ax = axes[row, col]
            groups = {
                loss: data.loc[data["diagnostic"].eq(metric)]
                for loss, data in data_by_loss.items()
            }
            # The green range is typical for every plotted network.
            low = max(float(group["rho_low"].iloc[0]) for group in groups.values())
            ax.axvspan(low, 1, color="#DCEEDC", alpha=0.7 if reference_style else 0.8, zorder=0)
            ax.axvline(1, color="0.35", linewidth=0.9, linestyle="--")
            for loss, group in groups.items():
                lower = float(group[f"pmp_error_lower_threshold_{model}"].iloc[0])
                upper = float(group[f"pmp_error_upper_threshold_{model}"].iloc[0])
                if pmp_threshold_band:
                    ax.axhspan(lower, upper, color="#9FD8A9", alpha=0.35, zorder=0)
                for bound in (lower, upper):
                    ax.axhline(bound,
                               color="0.4" if source_colors is not None else loss_colors[loss],
                               linewidth=1.1,
                               linestyle=threshold_styles[loss] if source_colors is not None else ":",
                               alpha=0.85)
                if source_colors is not None:
                    for source, source_group in group.groupby(source_column, sort=False):
                        empirical = source == empirical_dataset
                        ax.scatter(source_group["rho"], source_group[column],
                                   marker=loss_markers[loss], s=point_size,
                                   alpha=point_alpha,
                                   facecolors="none" if empirical else source_colors[source],
                                   edgecolors=source_colors[source] if empirical else "0.25",
                                   linewidths=1.2 if empirical else 0.45)
                elif empirical_dataset is None:
                    ax.scatter(group["rho"], group[column], marker=point_marker,
                               s=point_size, alpha=point_alpha,
                               c=loss_colors[loss],
                               edgecolors="0.25" if reference_style else "none",
                               linewidths=0.45 if reference_style else 0)
                else:
                    empirical = group["dataset"].eq(empirical_dataset)
                    if facet_style:
                        ax.scatter(group.loc[~empirical, "rho"], group.loc[~empirical, column],
                                   marker=point_marker, s=point_size, alpha=point_alpha,
                                   facecolors=loss_colors[loss], edgecolors="none", linewidths=0)
                        ax.scatter(group.loc[empirical, "rho"], group.loc[empirical, column],
                                   marker=point_marker, s=point_size, alpha=point_alpha,
                                   facecolors="none", edgecolors=loss_colors[loss],
                                   linewidths=1.2, zorder=4)
                    else:
                        ax.scatter(group.loc[~empirical, "rho"], group.loc[~empirical, column],
                                   marker=point_marker, s=18, alpha=0.5, facecolors=loss_colors[loss],
                                   edgecolors=loss_colors[loss], linewidths=0.4)
                        ax.scatter(group.loc[empirical, "rho"], group.loc[empirical, column],
                                   marker=point_marker, s=28, facecolors="none",
                                   edgecolors=loss_colors[loss], linewidths=1.0)
            if row == 0 and not facet_style:
                ax.set_title(titles[metric])
            ax.set_xscale("symlog", linthresh=1)
            if y_scale == "symlog":
                ax.set_yscale("symlog", linthresh=y_symlog_linthresh)
            if facet_style:
                ax.set_ylim(*(y_limits if y_limits is not None else (-1.05, 1.05)))
                ax.axhline(0, color="0.5", linewidth=0.75, zorder=1)
                ax.tick_params(labelbottom=row == 3, labelleft=col == 0,
                               labelsize=17 if reference_style else 9)
            else:
                ax.set_ylim(*(y_limits if y_limits is not None else (-1, 1)))
            ax.grid(alpha=0.2 if reference_style else 0.15)
    if facet_style:
        for ax in axes.ravel():
            if y_scale == "symlog":
                ax.yaxis.set_major_locator(
                    SymmetricalLogLocator(base=10, linthresh=y_symlog_linthresh)
                )
            else:
                ax.yaxis.set_major_locator(FixedLocator(np.linspace(-1, 1, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
        x_labels = {
            "l2": r"Diagnostic: $L_2$-based",
            "linf": r"Diagnostic: $L_\infty$-based",
            "density": "Diagnostic: density-based",
            "mmd": "Diagnostic: Kernel-based",
        }
        for ax, metric in zip(axes[-1], DIAGNOSTICS):
            ax.set_xlabel(x_labels[metric], fontsize=19 if reference_style else 10)
        fig.supylabel(
            r"Signed PMP error $p_{\mathrm{direct},j}-p_{\mathrm{gold},j}$",
            x=0.022 if reference_style else 0.025,
            fontsize=22 if reference_style else 13,
        )
        for ax, metric in zip(axes[0], DIAGNOSTICS):
            pos = ax.get_position()
            strip_height = 0.55 / fig.get_figheight() if reference_style else 0.026
            strip = fig.add_axes([pos.x0, pos.y1 + 0.004, pos.width, strip_height])
            strip.set_facecolor("0.85" if reference_style else "0.88")
            strip.text(0.5, 0.5, titles[metric], ha="center", va="center",
                       fontsize=21 if reference_style else 11)
            strip.set_xticks([])
            strip.set_yticks([])
            for spine in strip.spines.values():
                spine.set_color("0.35" if reference_style else "0.65")
                spine.set_linewidth(1.0 if reference_style else 0.6)
        for ax, column in zip(axes[:, -1], signed_columns):
            model = column.removeprefix("signed_pmp_error_")
            pos = ax.get_position()
            strip = fig.add_axes([pos.x1 + 0.004, pos.y0,
                                  0.058 if reference_style else 0.035, pos.height])
            strip.set_facecolor("0.85" if reference_style else "0.88")
            strip.text(0.5, 0.5, rf"Assumed $M_{{{model[1:]}}}$",
                       rotation=-90, ha="center", va="center",
                       fontsize=19 if reference_style else 10)
            strip.set_xticks([])
            strip.set_yticks([])
            for spine in strip.spines.values():
                spine.set_color("0.35" if reference_style else "0.65")
                spine.set_linewidth(1.0 if reference_style else 0.6)
    else:
        fig.supxlabel("Diagnostic score")
        fig.supylabel(r"Signed PMP error $p_{\mathrm{direct},j}-p_{\mathrm{gold},j}$")
    if source_colors is not None:
        handles = [
            Line2D([], [], color="0.4", marker=loss_markers[loss],
                   linestyle=threshold_styles[loss], markersize=9,
                   markerfacecolor="0.6", markeredgecolor="0.25",
                   markeredgewidth=0.8, label=loss_labels[loss])
            for loss in data_by_loss
        ]
    else:
        handles = [
            Line2D([], [], color=loss_colors[loss], marker="o", linestyle=":",
                   markeredgecolor="0.25" if reference_style else loss_colors[loss],
                   markeredgewidth=0.8 if reference_style else 1.0,
                   label=loss_labels[loss])
            for loss in data_by_loss
        ]
    if empirical_dataset is not None and source_colors is None:
        if facet_style:
            handles.extend([
                Line2D([], [], color="0.4", marker="o", markeredgecolor="none",
                       linestyle="none", label="Simulated"),
                Line2D([], [], color="0.4", marker="o", markerfacecolor="none",
                       linestyle="none", label="Empirical (hollow)"),
            ])
        else:
            handles.extend([
                Line2D([], [], color="0.4", marker="o", linestyle="none", label="Simulated (filled)"),
                Line2D([], [], color="0.4", marker="o", markerfacecolor="none", linestyle="none", label="Empirical (hollow)"),
            ])
    typical_label = "Typical set (all losses)" if len(data_by_loss) > 1 else "Typical set"
    handles.append(Line2D([], [], color="#DCEEDC", linewidth=8, label=typical_label))
    if pmp_threshold_band:
        handles.append(Patch(facecolor="#9FD8A9", alpha=0.35,
                             label="PMP threshold band (per loss)"))
    if source_colors is not None:
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, 1.8 / fig.get_figheight()),
                   ncol=len(handles), frameon=False, fontsize=18,
                   handlelength=2.2, columnspacing=1.5)
        source_handles = [
            Line2D([], [], linestyle="none", marker="o", markersize=10,
                   markerfacecolor="none" if source == empirical_dataset else color,
                   markeredgecolor=color if source == empirical_dataset else "0.25",
                   markeredgewidth=1.2 if source == empirical_dataset else 0.8,
                   label=source_labels.get(source, source.upper()) if source_labels else source.upper())
            for source, color in source_colors.items() if source in plotted_sources
        ]
        fig.legend(handles=source_handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.15 / fig.get_figheight()),
                   ncol=min(6, len(source_handles)), frameon=False, fontsize=16,
                   columnspacing=1.8, handletextpad=0.5, labelspacing=0.8)
    elif reference_style:
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.3125 / fig.get_figheight()),
                   ncol=len(handles), frameon=False, fontsize=18,
                   handlelength=2.2, columnspacing=1.35, handletextpad=0.6)
    elif facet_style:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
                   ncol=min(len(handles), 5), frameon=False)
    else:
        fig.legend(handles=handles, loc="outside upper center",
                   ncol=min(len(handles), 5), frameon=False)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"),
                    dpi=200 if reference_style else 220,
                    bbox_inches="tight" if reference_style else None)
    plt.close(fig)

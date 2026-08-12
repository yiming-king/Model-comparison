"""Generate the PPT-ready assumed-M4 L2 and Linf 2x3 figures."""

import ppt_summary_m2_posterior_logml_2x2 as plotter


if __name__ == "__main__":
    plotter.ASSUMED_MODEL = "m4"
    for diagnostic_metric in ("l2", "linf"):
        plotter.make_figure(diagnostic_metric)

import bayesflow as bf


def SummaryNetwork(
    summary_dim: int = 30,
    embed_dim: int = 64,
    base_distribution: str | None = "normal",
):
    kwargs = {"summary_dim": summary_dim, "embed_dim": embed_dim}
    if base_distribution is not None:
        kwargs["base_distribution"] = base_distribution
    return bf.networks.DeepSet(**kwargs)


def PosteriorNetwork():
    return bf.networks.CouplingFlow()

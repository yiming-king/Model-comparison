import bayesflow as bf

def SummaryNetwork(summary_dim: int = 30):
    return bf.networks.DeepSet(summary_dim=summary_dim, base_distribution="normal")

def PosteriorNetwork():
    return bf.networks.CouplingFlow()

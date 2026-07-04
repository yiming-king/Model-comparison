import bayesflow as bf
import numpy as np
from bayesflow.types import Shape
from bayesflow.utils.decorators import allow_batch_size
from scipy.special import expit

try:
    from dataset import wagenmakers
except ImportError:
    from ..dataset import wagenmakers

class RDM(bf.simulators.Simulator):
    '''
    The Racing diffusion model (RDM) simulator. Optionally can simulate from a model where the decision boundary
    depends on condition.
    '''
    def __init__(self, alternative, keep_params = True, **kwargs):
        super().__init__(**kwargs)
        self.alternative = alternative
        self.keep_params = keep_params

    @allow_batch_size
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        parameters = self.prior(batch_shape)
        observables = self.likelihood(batch_shape, **self._constrain_parameters(**parameters))

        if self.keep_params:
            output = parameters | observables
        else:
            output = observables

        # output = {key: np.expand_dims(value, axis=-1) if np.ndim(value) == 1 else value for key, value in output.items()}
        return output

    @allow_batch_size
    def prior(self, batch_shape: Shape) -> dict[str, np.ndarray]:
        return dict(
            # 2 boundaries for 2 conditions in the alternative model
            alpha=np.random.normal(loc=0.0, scale=0.5, size=batch_shape + (2,) if self.alternative else batch_shape + (1,)),
            nu=np.random.normal(loc=0.0, scale=0.5, size=batch_shape + (2,)),
            tau=np.random.normal(loc=0.0, scale=1, size=batch_shape + (1,)),
        )

    @staticmethod
    def _constrain_parameters(alpha: np.ndarray, nu: np.ndarray, tau: np.ndarray) -> dict[str, np.ndarray]:
        """
        Convert parameters of the racing diffusion model from an unconstrained space into the constrained space.
        """

        return dict(
            alpha = np.exp(alpha),
            nu    = np.exp(nu),
            tau   = expit(tau)
        )

    @allow_batch_size
    def likelihood(self, batch_shape: Shape, alpha: np.ndarray, nu: np.ndarray, tau: np.ndarray) -> dict[str, np.ndarray]:
        conditions = wagenmakers.conditions.reshape((1,) * len(batch_shape) + (-1,))
        conditions = np.broadcast_to(conditions, batch_shape + (conditions.shape[-1],))  # (batch_shape, num_trials)

        if self.alternative:
            alpha = np.expand_dims(alpha, axis=1) # (batch_shape, 1, 2)
            alpha = np.repeat(alpha, repeats=conditions.shape[-1], axis=1) # (batch_shape, num_trials, 2)
            alpha = np.where(conditions > 0.5, alpha[...,0], alpha[...,1]) #  (batch_shape, num_trials)
            alpha = np.expand_dims(alpha, axis=-1) # (batch_shape, num_trials, 1)
        else:
            alpha = np.expand_dims(alpha, axis=-1) # (batch_shape, 1, 1)
            alpha = np.repeat(alpha, repeats=conditions.shape[-1], axis=1) # (batch_shape, num_trials, 1)

        nu = np.expand_dims(nu, axis=1)

        rt = self.rdm_rng(alpha=alpha, nu=nu, tau=tau)

        return dict(rt=rt, conditions=conditions)

    def rdm_rng(self, alpha: np.ndarray, nu: np.ndarray, tau: np.ndarray) -> float:
        """
        Simulate a RT and accuracy from a Wiener diffusion decision model.
        Note that RT = t + t0 is the sum of the decision and non-decision times
        :param alpha: The decision boundary
        :param nu: The drift rates for the correct and incorrect responses, respectively
        :param tau: Non-decision time <lower=0, upper=1> proportion where tau = t0 / min(rt) -> t0 = tau * min(t) / (1-tau)
        :return: The reaction time. Negative values mean RTs for the opposite 'error' boundary.
        """

        # alpha.shape = (batch_shape, num_trials, 1)
        # nu.shape = (batch_shape, 1, 2)
        # tau.shape = (batch_shape, 1)

        decision_times = self.wald_rng(alpha=alpha, nu=nu) # (batch_shape, num_trials, 2)

        t = np.min(decision_times, axis=-1, keepdims=False)# (batch_shape, num_trials) min rt value
        choice = np.argmin(decision_times, axis=-1, keepdims=False)# (batch_shape, num_trials) index

        # calculate non-decision times and reaction times
        min_t = np.min(t, axis=-1, keepdims=True) #(batch_shape, 1)
        t0 = tau * min_t / (1 - tau)
        rt = t + t0

        rt[np.where(choice == 1)] = - rt[np.where(choice == 1)]

        return rt


    @staticmethod
    def wald_rng(alpha: np.ndarray, nu: np.ndarray) -> np.ndarray:
        mu = alpha / nu
        mu_sq = np.square(mu)
        lam = np.square(alpha)
        zeta = np.random.standard_normal(size=mu.shape)
        zeta_sq = np.square(zeta)

        x = mu + (mu_sq * zeta_sq)/(2 * lam) - mu/(2*lam) * np.sqrt(4*mu*lam*zeta_sq + mu_sq*np.square(zeta_sq))
        z = np.random.uniform(size=mu.shape)

        t = np.zeros_like(mu)
        condition = z <= mu / (mu + x)

        t[np.where(condition)] = x[np.where(condition)]
        t[np.where(np.logical_not(condition))] = (mu_sq / x)[np.where(np.logical_not(condition))]

        return t


rdm = RDM(alternative=False)
rdm_alternative = RDM(alternative=True)
rdm_model_comparison = bf.simulators.ModelComparisonSimulator(
    [RDM(alternative=False, keep_params=False), RDM(alternative=True, keep_params=False)],
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("\nnull\n")
    data = rdm.sample((64,))
    for k, v in data.items():
        print(k, v.shape)

    plt.hist(data["rt"].flatten(), bins=100)# histogram of RTs
    plt.title("RTs (null)")
    plt.show()

    print("\nalt\n")
    data = rdm_alternative.sample((64,))
    for k, v in data.items():
        print(k, v.shape)

    plt.hist(data["rt"].flatten(), bins=100)
    plt.title("RTs (alt)")
    plt.show()

    print("\nmodel comparison\n")
    data = rdm_model_comparison.sample((10,))
    for k, v in data.items():
        print(k, v.shape)

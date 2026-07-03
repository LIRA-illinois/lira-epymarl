import numpy as np
from scipy.stats import sem, t as students_t


def get_mean_confidence_interval(val_array: np.ndarray, alpha: float = 0.05):
    estimated_mean = np.mean(val_array)
    conf_int_vals = students_t.interval(
        1 - alpha, len(val_array) - 1, scale=sem(val_array)
    )

    conf_int = estimated_mean * np.ones(2) + conf_int_vals

    return estimated_mean, conf_int


############
# part 1: estimating the mean model return by sampling evaluation episodes
############
class VarianceEstimatorWelford:
    """class Welford
    This library is python(numpy) implementation of Welford's algorithm,
    which is online and parallel algorithm for calculating variances.

    Welfords method is more numerically stable than the standard method as
    described in the following blog,
        * Accurately computing running variance: www.johndcook.com/blog/standard_deviation

    This library is inspired by the jvf's implementation, which is implemented
    without using numpy library.
        * implementaion done by jvf: github.com/jvf/welford

            taken from https://github.com/a-mitani/welford/blob/main/welford/welford.py

     Accumulator object for Welfords online / parallel variance algorithm.

    Attributes:
        count (int): The number of accumulated samples.
        mean (array(D,)): Mean of the accumulated samples.
        var_s (array(D,)): Sample variance of the accumulated samples.
        var_p (array(D,)): Population variance of the accumulated samples.
    """

    def __init__(self, elements=None) -> None:
        """__init__

        Initialize with an optional data.
        For the calculation efficiency, Welford's method is not used on the initialization process.

        Args:
            elements (array(S, D)): data samples.

        """

        # Initialize instance attributes
        if elements is None:
            self.__shape = None
            # current attribute values
            self.__count = 0
            self.__m = None
            self.__s = None
            # previous attribute values for rollbacking
            self.__count_old = None
            self.__m_old = None
            self.__s_old = None

        else:
            self.__shape = elements[0].shape
            # current attribute values
            self.__count = elements.shape[0]
            self.__m = np.mean(elements, axis=0)
            self.__s = np.var(elements, axis=0, ddof=0) * elements.shape[0]
            # previous attribute values for rollbacking
            self.__count_old = None
            self.__init_old_with_nan()

    @property
    def count(self) -> int:
        return self.__count

    @property
    def mean(self):
        return self.__m

    @property
    def var_s(self):
        return self.__getvars(ddof=1)

    @property
    def var_p(self):
        return self.__getvars(ddof=0)

    def add(self, element, backup_flg=True) -> None:
        """add

        add one data sample.

        Args:
            element (array(D, )): data sample.
            backup_flg (boolean): if True, backup previous state for rollbacking.

        """
        # Initialize if not yet.
        if self.__shape is None:
            self.__shape = element.shape
            self.__m = np.zeros(element.shape)
            self.__s = np.zeros(element.shape)
            self.__init_old_with_nan()
        # argument check if already initialized
        else:
            assert element.shape == self.__shape

        # backup for rollbacking
        if backup_flg:
            self.__backup_attrs()

        # Welford's algorithm
        self.__count += 1
        delta = element - self.__m
        self.__m += delta / self.__count
        self.__s += delta * (element - self.__m)

    def add_all(self, elements, backup_flg: bool=True) -> None:
        """add_all

        add multiple data samples.

        Args:
            elements (array(S, D)): data samples.
            backup_flg (boolean): if True, backup previous state for rollbacking.

        """
        # backup for rollbacking
        if backup_flg:
            self.__backup_attrs()

        for elem in elements:
            self.add(elem, backup_flg=False)

    def rollback(self) -> None:
        self.__count = self.__count_old
        self.__m[...] = self.__m_old[...]
        self.__s[...] = self.__s_old[...]

    def __getvars(self, ddof):
        if self.__count <= 0:
            return None
        min_count = ddof
        if self.__count <= min_count:
            return np.full(self.__shape, np.nan)
        else:
            return self.__s / (self.__count - ddof)

    def __backup_attrs(self) -> None:
        if self.__shape is None:
            pass
        else:
            self.__count_old = self.__count
            self.__m_old[...] = self.__m[...]
            self.__s_old[...] = self.__s[...]

    def __init_old_with_nan(self) -> None:
        self.__m_old = np.empty(self.__shape)
        self.__m_old[...] = np.nan
        self.__s_old = np.empty(self.__shape)
        self.__s_old[...] = np.nan


def sample_ep_return() -> float:
    mean = 0.5
    var = 0.25
    std_dev = np.sqrt(var)
    return np.random.normal(mean, std_dev)


def get_sample_size_from_variance(
    variance: float,
    alpha: float = 0.05,
    d: float = 0.05,
) -> int:
    """We want to be able to choose a number of samples such that we can say P(|\hat{\theta} - \theta| >= d) < alpha.
    Based on equation 4.4 from "Sampling" 2012 by Thompson, this gives the number of required samples to achieve a
    given level of statistical signficiance alpha, given specified distance d and sample variance.

    In our case, we are estimating the

    Args:
        variance (float): variance of a given sample
        d (float, optional): max allowable difference between the true and estimated value. Defaults to 0.01.
        alpha (float, optional): _description_. Defaults to 0.05.

    Returns:
        int: number of samples required
    """

    # z is the upper alpha / 2 point of the standard normal distribution
    ## for alpha = 0.05 and large samples, this is approximately 1.96
    # TODO in the actual code, just pre-compute this number, don't re-compute it each time
    n_samples_z = 100000
    z = students_t.ppf(1 - alpha / 2, n_samples_z)

    # the true number of samples will be lower than n_samples_z, but that is a fine approximation to make
    # since it will stil be a large number of samples
    n_samples = (z**2 * variance) / d**2
    return int(np.ceil(n_samples))


def get_estimated_mean_return():
    """
    estimates the mean return that can be achieved by a given RL policy
    """

    alpha = 0.05
    d = 0.05
    curr_sample_var_estimate = 0.4
    n_eval_eps = get_sample_size_from_variance(
        curr_sample_var_estimate, alpha=alpha, d=d
    )
    print("N samples to estimate mean return:", n_eval_eps)
    # variance_estimator = VarianceEstimatorWelford()

    ep_returns = []

    for ep in range(n_eval_eps):
        sampled_return = sample_ep_return()
        ep_returns.append(sampled_return)

        # variance_estimator.add(np.array(sampled_return))

        # curr_sample_var_estimate = variance_estimator.var_s

        # n_eval_eps = get_n_eval_eps_from_variance(curr_sample_var_estimate)

        # # "warm start" to prevent the evaluation from terminating too quickly
        # if eval_ep_counter < 50:
        #     pass
        # else:
        #     n_eval_eps = get_n_eval_eps_from_variance(curr_sample_var_estimate)
        #     # print(n_eval_eps)
        # eval_ep_counter += 1

    # print("N episodes evaluated ", eval_ep_counter)
    # print("Variance estimate from Welford", variance_estimator.var_s)
    # print("Variance estimate from np.var", np.var(ep_returns))
    # return variance_estimator.var_s

    estimated_mean_return, conf_int = get_mean_confidence_interval(ep_returns, alpha)
    return estimated_mean_return, conf_int


def run_return_variance_experiment() -> None:
    n_eval = 100
    estimated_vars = np.zeros(n_eval)
    var = 0.25
    actual_vars = var * np.ones(n_eval)

    for i in range(n_eval):
        print(i)
        estimated_vars[i] = get_estimated_mean_return()

    abs_diff = np.abs(actual_vars - estimated_vars)
    within_range = np.where(abs_diff < 0.02)[0]
    percent_within = len(within_range) / n_eval
    print(percent_within)


############
# part 2: estimating the current-best model's success probability and final state distribution by sampling evaluation episodes
############
def sample_final_state_dist(n_samples: int):
    elements = [0, 1, 2, 3, 4]
    probs = [0.4, 0.2, 0.2, 0.1, 0.1]
    return np.random.choice(elements, n_samples, p=probs)


def sample_success_dist(n_samples: int):
    elements = [0, 1]
    probs = [0.9, 0.1]
    return np.random.choice(elements, n_samples, p=probs)


def get_estimated_final_joint_state_distribution():
    n_incoming_edges = 5

    success_prob_alpha = 0.05
    success_prob_d = 0.05

    joint_state_prob_alpha = 0.025
    joint_state_prob_d = 0.02

    # my approach: choose the total number of samples per edge based on whichever is greater
    ## this guarantees that both are within thte
    # n_samples_single_proportion and n_samples_all_proportions

    # n_samples_all_proportions is used to estimate the population-level proportions, so should be divided up over the n incoming edges (i.e., strata)

    # "worst case" approach where we assume we don't know anything about the underlying distribution and p = 0.5
    n_samples_per_incoming_edge_single_prob = (
        get_sample_size_to_estimate_single_proportion(
            success_prob_alpha, success_prob_d
        )
    )

    # a "worst case variance" approach, where we assume we have an estimate of the variance that we can use to estimate a lower number of samples compared to the worst case approach without a variance estimate
    ## here, we can take advantage of the fact that we know the true mean falls within the range [0, 1] to set the variance so it covers
    ## actually, instead of using a normal distribution, we could use a truncated once since we know the true mean falls within the range of [0, 1]
    ### idk this is raising more questions than I want to deal with. I just need a preliminary number of samples.
    # curr_sample_var_estimate = 1/6
    # n_samples_success_prob_with_var = get_sample_size_from_variance(curr_sample_var_estimate, success_prob_alpha, success_prob_d)

    n_samples_total_all_probs = get_sample_size_to_estimate_all_proportions(
        joint_state_prob_alpha, joint_state_prob_d
    )
    n_samples_per_incoming_edge_all_probs = int(
        np.ceil(n_samples_total_all_probs / n_incoming_edges)
    )
    n_samples_per_incoming_edge = max(
        n_samples_per_incoming_edge_single_prob, n_samples_per_incoming_edge_all_probs
    )

    print(f"{n_samples_per_incoming_edge_single_prob} samples needed to estimate success_rate with\nalpha (confidence level) = {success_prob_alpha}\nCI width = {success_prob_d}\n(assuming the worst case of success_rate=0.5)")

    # print("Worst case joint state n samples:", n_samples_per_incoming_edge_all_probs)

    n_agent_positions = 6

    # since we assume each of the strata have an equal number of samples, we can simply append all
    # of the sampled state lists to get a population-level proportion. This acts as if each stratum
    # contributes equally.
    sampled_states_list = []
    sampled_success_bools = []
    estimated_success_probs = np.zeros(n_incoming_edges)
    estimated_success_prob_conf_ints = np.zeros((n_incoming_edges, 2))

    for i in range(n_incoming_edges):
        # sample the success probs
        sampled_success_bools.append(sample_success_dist(n_samples_per_incoming_edge))

        # sample the final states
        positions = []
        for j in range(n_agent_positions):
            positions.append(
                np.expand_dims(
                    sample_final_state_dist(n_samples_per_incoming_edge), axis=1
                )
            )
        sampled_states_list.append(np.concatenate(positions, axis=1))

    # estimate the success probs for each incoming edge
    for i in range(n_incoming_edges):
        estimated_success_probs[i], estimated_success_prob_conf_ints[i, :] = (
            get_mean_confidence_interval(sampled_success_bools[i], success_prob_alpha)
        )

    # aggregate the estimated final states distributions into a single distribution
    all_sampled_states = np.concatenate(sampled_states_list, axis=0)
    unique_sampled_states, state_counts = np.unique(
        all_sampled_states, axis=0, return_counts=True
    )
    final_state_probs = state_counts / np.sum(state_counts)
    estimated_final_state_dist = [
        [prob, list(unique_sampled_states[i])]
        for i, prob in enumerate(final_state_probs)
    ]

    return (
        estimated_success_probs,
        estimated_success_prob_conf_ints,
        estimated_final_state_dist,
    )


def get_sample_size_to_estimate_single_proportion(alpha: float = 0.05, d: float = 0.05) -> int:
    """based on equation 5.2 of "Sampling" (Thompson, 2012)
    gives the samples size needed to obtain an estimator \hat{theta} which has probability of at least 1-alpha of being no farther than d from the true theta
    """
    # assume we have no prior knowledge of how well an agent will perform
    ## this isn't a bad assumption because MARL algorithms can be very unstable in their performance during training
    worst_case_p = 0.5

    # z is the upper alpha / 2 point of the standard normal distribution
    ## for alpha = 0.05 and large samples, this is approximately 1.96
    n_samples_z = 100000
    z = students_t.ppf(1 - alpha / 2, n_samples_z)

    n_samples = (z**2 * worst_case_p * (1 - worst_case_p)) / (d**2)
    n_samples = int(np.ceil(n_samples))
    return n_samples


def get_sample_size_to_estimate_all_proportions(alpha: float = 0.05, d: float = 0.05) -> int:
    """
    To get the worst-case n which assumes no knowledge of the probabilities, I need to solve the following optimization problem from equation 1 of "Sample Size for Estimating Multinomial Proportions" (Thompson, 1987)

    n = max_{m} z^2 (1/m) (1 - 1/m) / d^2

    where z is the 1 - (alpha / (2*m)) point of the standard normal distribution, and m is an integer
    """

    n_samples_list = []
    # from Table 1 of (Thompson, 1987), m as an integer [1, 4] covers the cases of alpha a real number in [0.0001, 0.5], so the procedure below should be sufficient to find the max number of samples for any reasonable values of alpha

    for m in range(1, 5):
        n_samples_z = 100000
        z = students_t.ppf(1 - alpha / (2 * m), n_samples_z)

        n = z**2 * (1 / m) * (1 - 1 / m) / d**2
        n_samples_list.append(n)

    m, n_samples_worst_case = np.argmax(n_samples_list), int(
        np.ceil(np.max(n_samples_list))
    )
    # print("m: ", m)
    # print("N Samples:", n_samples_worst_case)

    return n_samples_worst_case


def main() -> None:
    # mean, conf_int = get_estimated_mean_return()
    # print(mean, conf_int)
    # run_mean_return_experiment()

    # (
    #     estimated_success_probs,
    #     estimated_success_prob_conf_ints,
    #     estimated_final_state_dist,
    # ) = get_estimated_final_joint_state_distribution()
    # print(estimated_success_probs)
    # print(estimated_success_prob_conf_ints)

    # equivalent thing already implement in scipy.stats
    # import scipy.stats as st

    # p_success_true = 0.75
    # these things like to cluster up so much, you need a ton of samples to get a small confidence interval
    # this amount of samples lets you say "with confidence 0.05, the agent that got an 80% success rate has statitically"

    # actually, you need a 2-sample  test to tell if two values are statistically different, that's a totally different thing than doing two confidence intervals and checking if they overlap
    # n_samples = 1500
    # n_success = int(p_success_true * n_samples)

    # result = st.binomtest(k=n_success, n=n_samples)
    # ci = result.proportion_ci(confidence_level=0.95)
    # ci_width = ci.high - ci.low

    # # confidence intervals can overlap while the values themselves are statistically significantly different
    # # to check that, you instead need to check if the difference between the two success rates has a confidence interval that includes 0 or not
    # print(f"p_success_true: {p_success_true}")
    # # print(result)
    # print(f"CI (%): ({ci.low * 100:.2f}, {ci.high * 100:.2f})")
    # print(f"CI Width (%): {ci_width * 100:.2f}")

    # print(result)
    # print(result.confidence_interval(confidence_level=0.95))

    # you do need the raw outcomes, but if you have estimated success rate, you can just generate a binary array based on that, and then that is input to the thing
    import statsmodels.api as sm
    # the number of required samples to ensure you have high resolution increases as you get closer to 0.5

    # if you go w/ a sampling of 200, you can distinguish 70 from 80%
    # if you go w/ 300, you can distinguish numbers that are 8 apart
    # that's about as good as I need
    delta = 0.05
    n_samples = 800

    p_vals = []
    step=0.05
    lower_vals = np.arange(start=0, stop=1+step, step=step)
    for lower in lower_vals:
        success_rates = np.array([lower, lower + delta])
        samples_arr = n_samples * np.ones_like(success_rates)
        n_success = samples_arr * success_rates * np.ones_like(success_rates)
        _, p_val = sm.stats.proportions_ztest(count=n_success, nobs=samples_arr)
        p_vals.append(p_val)
        # confidence_level = 0.05
        # print(f"p_value: {p_val}, {p_val < confidence_level}")

    # to ensure you can distinguish values across the entire range of possible values, you need the max value of p_vals < confidence_level (e.g., 0.05)
    # if not, there may be some values you cannot distinguish given your number of samples
    # a given number of samples gives different p values for different lower and delta

    import matplotlib.pyplot as plt
    print(max(p_vals))
    plt.plot(lower_vals, p_vals)
    plt.savefig("stats.png")




if __name__ == "__main__":
    main()

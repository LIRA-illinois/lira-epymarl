import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt


def credible_interval_analysis():
    # setting a, b = 1 is the same as assuming a uniform prior distribution over the interval [0, 1]
    a_prior = 1
    b_prior = 1

    true_success_rate = 0.8
    n_samples = 600
    n_success = n_samples * true_success_rate

    # the posterior distribution for a Bernoulli likelihood is a Beta distribution
    # once we observe n_samples and n_success, we have the parameters of the Beta distribution
    a_posterior = a_prior + n_success
    b_posterior = b_prior + n_samples - n_success

    # Define the posterior distribution object for a Bernoulli prior
    posterior = st.beta(a_posterior, b_posterior)

    # once we have the posterior distribution, we can get the value of the success rate associated with the lower and upper-bounds of the probability that we want
    # given this we, can say "there is a 95% probability the true success rate is in this interval" (assuming a uniform prior distribution, which is reasonable b/c of noise in the MARL training process that makes it hard to actually guess what the true success rate will be)
    credible_interval_size = 0.99

    # amount of probability outside the credible interval
    tail_prob = 1.0 - credible_interval_size

    # equal-tailed probs
    # you can also do versions of this where the distribution has a heavier tail in one direction or the other, but we're not gonna deal with that here
    lower_tail_prob = tail_prob / 2
    upper_tail_prob = 1.0 - lower_tail_prob

    # probabilistic lower and upper bounds on the value of the success rate
    lower_bound = posterior.ppf(lower_tail_prob)
    upper_bound = posterior.ppf(upper_tail_prob)

    print(f"n_samples: {n_samples}")
    print(f"Credible interval size: {100 * (upper_bound - lower_bound):.2f}%")
    print(f"True Success Rate: {true_success_rate:.4f}")
    print(
        f"{credible_interval_size * 100}%  Credible Interval for success rate: [{lower_bound:.4f}, {upper_bound:.4f}]"
    )


def credible_interval_width(p_success=0.5, credible_interval_size=0.95):
    """Estimate expected posterior credible interval width for Bernoulli with Beta(1,1) prior.

    Returns arrays (ns, widths).
    """
    # prior distribution before sampling data
    a_prior = 1
    b_prior = 1

    # relevant data for updating the prior to get the posterior
    n_samples = np.unique(np.round(np.logspace(0, 4, 50)).astype(int))
    n_success = n_samples * p_success

    # posterior distribution after sampling data
    a_posterior = a_prior + n_success
    b_posterior = b_prior + n_samples - n_success
    posterior = st.beta(a_posterior, b_posterior)

    tail_prob = 1.0 - credible_interval_size
    lower_tail_prob = tail_prob / 2
    upper_tail_prob = 1.0 - lower_tail_prob

    lower = posterior.ppf(lower_tail_prob)
    upper = posterior.ppf(upper_tail_prob)
    widths = upper - lower

    return n_samples, widths


def plot_credible_interval_vs_n_samples(p_true=0.5, save_path=None):
    n_samples, widths = credible_interval_width(p_success=p_true)

    plt.figure(figsize=(6, 4))
    plt.loglog(n_samples, widths, marker="o")
    # plt.loglog(ns, widths, marker="o")
    plt.xlabel("Number of samples (n)")
    plt.ylabel("Expected 95% credible interval width")
    plt.title(f"Expected posterior credible interval width vs n (p={p_true})")
    plt.grid(which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # example: plot and save to file
    plot_credible_interval_vs_n_samples(p_true=0.5, save_path="credible_interval_vs_n.png")

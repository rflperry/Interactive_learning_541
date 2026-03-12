import numpy as np


def ucb(T, outcomes):
    n = len(outcomes)
    pulls = np.ones(n)
    returns = np.asarray([outcomes[t, t] for t in range(n)])  # sample once

    for t in range(n, T):
        cbs = returns / pulls + np.sqrt(2 * np.log(2 * n * T**2) / pulls)
        arm = np.argmax(cbs)
        returns[arm] += outcomes[arm, t]
        pulls[arm] += 1

    return returns


def thompson_sampling(T, outcomes):
    n = len(outcomes)
    pulls = np.zeros(n)
    returns = np.zeros(n)

    # prior is N(0, 1)
    # likelihood is N(mu, 1)
    for t in range(T):
        samples = np.random.normal(returns / (1 + pulls), 1 / (1 + pulls))
        arm = np.argmax(samples)
        returns[arm] += outcomes[arm, t]
        pulls[arm] += 1

    return returns


def etc(T, outcomes, m=1):
    n = len(outcomes)
    pulls = np.zeros(n)
    returns = np.zeros(n)

    for t in range(T):
        if np.sum(pulls) <= n * m:
            arm = t % m
        else:
            arm = np.argmax(returns / pulls)
        returns[arm] += outcomes[arm, t]
        pulls[arm] += 1

    return returns


def exp3(T, outcomes, eta=1, gamma=0):
    n = len(outcomes)
    returns = np.zeros(n)
    p = np.ones(n) / n
    weights = np.ones(n)

    for t in range(T):
        qt = (1 - gamma) * p + gamma / n
        noise = np.random.gumbel(0, 1, size=n)
        # gumbel trick, plus mix in uniform noise
        arm = np.argmax(np.log(qt) + noise)
        returns[arm] += outcomes[arm, t]
        weights[arm] *= np.exp(-eta * outcomes[arm, t] / qt[arm])
        p = weights / np.sum(weights)

    return returns
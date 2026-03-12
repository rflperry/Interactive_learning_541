import numpy as np
from sklearn.linear_model import LogisticRegression

def etc_linear(X, reward_func, action_set, tau):
    T = len(X)
    n = len(action_set)
    rewards = np.zeros(T)
    # actions = np.zeros(n)
    features = []

    for t in range(tau):
        k = np.random.choice(n)
        action = action_set[k]
        rewards[t] = reward_func(t, action)
        features.append(np.outer(X[t], action).flatten())

    # model world
    phi = np.vstack(features)
    theta = np.linalg.inv(phi.T @ phi) @ phi.T @ rewards[:tau]

    for t in range(tau, T):
        k = np.argmax([
            np.dot(theta, np.outer(X[t], act).flatten()) for act in action_set
        ])
        rewards[t] = reward_func(t, action_set[k])

    return rewards


def etc_bias_linear(X, reward_func, action_set, tau):
    T = len(X)
    n = len(action_set)
    rewards = np.zeros(T)
    actions = np.zeros(tau)

    for t in range(tau):
        k = np.random.choice(n)
        action = action_set[k]
        actions[t] = k
        rewards[t] = reward_func(t, action)

    # model bias
    idx = np.where(rewards == 1)[0]
    logreg = LogisticRegression().fit(X[:tau][idx], actions[idx])

    for t in range(tau, T):
        k = int(logreg.predict(X[t].reshape(1, -1))[0])
        rewards[t] = reward_func(t, action_set[k])

    return rewards

def ftl_linear(X, reward_func, action_set, tau):
    T = len(X)
    n = len(action_set)
    rewards = np.zeros(T)
    features = []

    for t in range(tau):
        k = np.random.choice(n)
        action = action_set[k]
        rewards[t] = reward_func(t, action)
        features.append(np.outer(X[t], action).flatten())

    # model world
    phi = np.vstack(features)
    P = np.linalg.inv(phi.T @ phi)
    theta = P @ phi.T @ rewards[:tau]

    for t in range(tau, T):
        k = np.argmax([
            np.dot(theta, np.outer(X[t], act).flatten()) for act in action_set
        ])
        action = action_set[k]
        rewards[t] = reward_func(t, action)

        x = np.outer(X[t], action).flatten()
        Px = P @ x
        P = P - np.outer(Px, Px) / (1 + x @ Px)
        theta = theta + P @ x * (rewards[t] - x @ theta)

        features.append(x)

    return rewards

def linUCB(X, reward_func, action_set, gamma=1):
    T = len(X)
    d = X.shape[1] * action_set.shape[1]

    V_inv = np.eye(d) / gamma
    theta = np.zeros(d)
    rewards = np.zeros(T)
    features = []
    theta = np.zeros(d)

    for t in range(T):
        ucb_scores = []
        for act in action_set:
            phi = np.outer(X[t], act).flatten()
            alpha = np.sqrt(gamma) + np.sqrt(2 * np.log(1 / (t + 1) + d * np.log(1 + (t + 1) / d / gamma)))
            bonus = alpha * np.sqrt(phi @ V_inv @ phi)
            ucb_scores.append(theta @ phi + bonus)
        k = np.argmax(ucb_scores)
        action = action_set[k]
        rewards[t] = reward_func(t, action)

        x = np.outer(X[t], action).flatten()
        Vx = V_inv @ x
        V_inv = V_inv - np.outer(Vx, Vx) / (1 + x @ Vx)
        theta = theta + V_inv @ x * (rewards[t] - x @ theta)

        features.append(x)

    return rewards

def ts_linear(X, reward_func, action_set, gamma=1):
    T = len(X)
    n = len(action_set)
    d = X.shape[1] * action_set.shape[1]
    rewards = np.zeros(T)
    V_inv = np.eye(d) / gamma
    theta = np.zeros(d)
    
    for t in range(T):
        theta_t = np.random.multivariate_normal(theta, V_inv)
        k = np.argmax([
            np.dot(theta_t, np.outer(X[t], act).flatten()) for act in action_set
        ])

        action = action_set[k]
        rewards[t] = reward_func(t, action)
        x = np.outer(X[t], action).flatten()
        Vx = V_inv @ x
        V_inv = V_inv - np.outer(Vx, Vx) / (1 + x @ Vx)
        theta = theta + V_inv @ x * (rewards[t] - x @ theta)

    return rewards
import time

import jax
from jax.experimental import jet

import jax.numpy as jnp
import matplotlib.pyplot as plt

def simple_polynomial(x):
    return jnp.sum(x * x * x)


def trigonometric_function(x):
    return jnp.sum(jnp.sin(jnp.pi * x))


def activation_function(x):
    return jnp.tanh(trigonometric_function(x))


def hessian_diag(fn, x, batch_size):
    key = jax.random.key(0)
    dim = x.shape[0]

    idx_set = jax.random.choice(
        key, dim, shape=(batch_size,), replace=False
    )

    rand_jet = jax.vmap(lambda i: jnp.eye(dim)[i])(idx_set)
    pushfwd_2_fn = lambda v: jet.jet(
        fun=fn, primals=(x,), series=((v, jnp.zeros(dim)),)
    )  # pushforward of the 2-jet (x, v, 0), i.e. \dd^2 f(x, v, 0)
    f_vals, (_, vhv) = jax.vmap(pushfwd_2_fn)(rand_jet)
    hess_diag_val = dim / batch_size * vhv

    return hess_diag_val, idx_set


def hessian_trace_mean(fn, x, dim, batch_size, n_rounds):
    diagonals = []

    for i in range(n_rounds):
        key = jax.random.key(i)

        idx_set = jax.random.choice(
            key, dim, shape=(batch_size,), replace=False
        )
        rand_jet = jax.vmap(lambda i: jnp.eye(dim)[i])(idx_set)
        pushfwd_2_fn = lambda v: jet.jet(
            fun=fn, primals=(x,), series=((v, jnp.zeros(dim)),)
        )  # pushforward of the 2-jet (x, v, 0), i.e. \dd^2 f(x, v, 0)
        f_vals, (_, vhv) = jax.vmap(pushfwd_2_fn)(rand_jet)
        hess_diag_val = dim / batch_size * vhv
        diagonals.append(hess_diag_val)

    diagonals = jnp.stack(diagonals)
    traces = jnp.sum(diagonals, axis=1)
    avg_trace = jnp.mean(traces)
    variance = jnp.var(traces)

    check = jax.hessian(fn, 0)
    check_hessian = check(x)
    check_trace = jnp.trace(check_hessian)

    return avg_trace, variance, check_trace


# each run gets the same compute limit cl=n_iter*batch_size
# then compare the reached accuracies
# just using plain abs(x - s) is too influenced by luck and how slowly means change
def hessian_trace_with_compute_limit(fn, x, dim, batch_size, compute_limit):
    diagonals = []
    n_iter = compute_limit // batch_size

    pushfwd_2_fn = lambda v: jet.jet(
        fun=fn, primals=(x,), series=((v, jnp.zeros(dim)),)
    )  # pushforward of the 2-jet (x, v, 0), i.e. \dd^2 f(x, v, 0)

    for i in range(n_iter):
        key = jax.random.key(i)

        idx_set = jax.random.choice(
            key, dim, shape=(batch_size,), replace=False
        )
        rand_jet = jax.vmap(lambda i: jnp.eye(dim)[i])(idx_set)

        f_vals, (_, vhv) = jax.vmap(pushfwd_2_fn)(rand_jet)
        hess_diag_val = dim / batch_size * vhv

        diagonals.append(hess_diag_val)

    check = jax.hessian(fn, 0)
    check_hessian = check(x)
    exact_trace = jnp.trace(check_hessian)

    diagonals = jnp.stack(diagonals)
    traces = jnp.sum(diagonals, axis=1)

    mse = jnp.mean((exact_trace - traces) ** 2)

    variance = jnp.var(traces)

    return mse, variance


def measure_variance(dim, fn, x, n_rounds):
    variances = []
    for batch_size in range(1, dim + 1):
        _, variance, _ = hessian_trace_mean(fn, x, dim, batch_size, n_rounds)
        variances.append(variance)

    plt.plot(range(1, dim + 1), variances)
    plt.show()


def benchmark_batch_size(dim, fn, x, compute_limit):
    errors = []
    variances = []

    for batch_size in range(3, dim + 1):
        print(f"running batch_size {batch_size}")
        mse, var = hessian_trace_with_compute_limit(fn, x, dim, batch_size, compute_limit)
        errors.append(mse)
        variances.append(var)

    plt.plot(range(3, dim + 1), errors, marker='o')
    plt.title(f"Hessian trace approximation with compute_limit={compute_limit}")
    plt.xlabel("batch size")
    plt.ylabel("mean squared error")
    # plt.plot(range(3, dim + 1), variances)

    plt.show()


def newton_raphson(fn, x_init, batch_size, tol=1e-6):
    eta = 0.1
    eps = 1e-10
    x = x_init

    grad = jax.grad(fn)
    g = grad(x)

    while jnp.linalg.norm(g) > tol:
        print(jnp.linalg.norm(g))
        g = grad(x)
        diag, idx_set = hessian_diag(fn, x, batch_size)

        step = -eta * g[idx_set] / (jnp.abs(diag) + eps)

        x = x.at[idx_set].add(step)


def newton_raphson_exact(fn, x_init, tol=1e-6):
    eta = 1
    eps = 1e-10
    x = x_init

    grad = jax.grad(fn)
    g = grad(x)
    hessian = jax.hessian(fn, 0)

    while jnp.linalg.norm(g) > tol:
        print(jnp.linalg.norm(g))
        g = grad(x)
        step = -eta * jnp.linalg.solve(hessian(x), g)

        x = x + step


def main():
    jnp.set_printoptions(linewidth=160)

    dim = 10

    uniform_x = jnp.ones((dim,))
    key = jax.random.key(0)
    random_x = jax.random.uniform(key, (dim,))
    range_x = jnp.linspace(-1.0, 1.0, dim)


    benchmark_batch_size(dim, trigonometric_function, random_x, 300)


main()
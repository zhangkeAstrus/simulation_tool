from scipy.optimize import brentq


def calculate_limited_mean(dist, params, limit):
    """Return E[min(X, limit)] for a fitted scipy.stats distribution."""
    if limit <= 0:
        raise ValueError("Claim limit must be greater than zero.")

    params = tuple(float(p) for p in params)
    shape_args = params[:-2]
    loc = params[-2]
    scale = params[-1]

    expected_below_limit = dist.expect(
        lambda x: x,
        args=shape_args,
        loc=loc,
        scale=scale,
        lb=0,
        ub=limit,
        conditional=False,
    )

    expected_above_limit = limit * dist.sf(
        limit,
        *shape_args,
        loc=loc,
        scale=scale,
    )

    return float(expected_below_limit + expected_above_limit)


def solve_scale_for_limited_mean(dist, params, limit, target_limited_mean):
    """
    Keep every fitted parameter except scale unchanged and solve for the
    scale that produces the requested E[min(X, limit)].
    """
    if target_limited_mean <= 0:
        raise ValueError("Target limited severity must be greater than zero.")
    if target_limited_mean >= limit:
        raise ValueError("Target limited severity must be below the claim limit.")

    params = tuple(float(p) for p in params)
    current_scale = params[-1]

    def objective(scale):
        test_params = params[:-1] + (float(scale),)
        return calculate_limited_mean(dist, test_params, limit) - target_limited_mean

    lower = max(current_scale * 1e-8, 1e-10)
    upper = max(current_scale, 1.0)

    # Increase the upper bound until the target is bracketed.
    while objective(upper) < 0:
        upper *= 2.0
        if upper > 1e15:
            raise ValueError("Could not find a reasonable scale parameter for this target.")

    return float(brentq(objective, lower, upper, xtol=1e-8, rtol=1e-10))

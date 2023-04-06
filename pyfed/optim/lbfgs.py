from torch.optim import LBFGS


class LBFGSB(LBFGS):
    r"""Implements L-BFGS-B algorithm.
    The L-BFGS-B algorithm extends L-BFGS to handle simple box constraints (aka bound constraints) on variables;
    that is, constraints of the form li ≤ xi ≤ ui where li and ui are per-variable constant lower and upper bounds,
    respectively (for each xi, either or both bounds may be omitted).
    The method works by identifying fixed and free variables at every step (using a simple gradient method),
    and then using the L-BFGS method on the free variables only to get higher accuracy, and then repeating the process.

    Arguments:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25)
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5)
        tolerance_change (float): termination tolerance on function/parameter
            changes (default: 1e-9)
        history_size (int): update history size (default: 100)
        line_search_fn (str): either 'strong_wolfe' or 'backtracking'. Note that
            the latter is not as robust as the former.
        stochastic (bool): Whether to run stochastic or full batch version.
    """
    def __init__(self, params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-5,
                 tolerance_change=1e-9, history_size=100, line_search_fn=None,
                 stochastic=False):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(lr=lr, max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad,
                        tolerance_change=tolerance_change,
                        history_size=history_size,
                        line_search_fn=line_search_fn,
                        stochastic=stochastic)
        super(LBFGSB, self).__init__(params, defaults)
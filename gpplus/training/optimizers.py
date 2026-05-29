from functools import reduce
from typing import Callable, Optional

import numpy as np
import torch
from scipy.optimize import fmin_l_bfgs_b


class LBFGSScipy(torch.optim.Optimizer):
    """Wrap L-BFGS algorithm, using scipy routines.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now CPU only

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Callback contract:
        Supports an iteration callback hook via ``set_iteration_callback`` for
        integration with training callbacks.

    Arguments:
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
    """

    def __init__(
        self, params, max_iter=2000, max_eval=5000, tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10
    ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
        )
        super(LBFGSScipy, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None
        self._n_iter = 0
        self._last_loss = None
        self.iteration_callback: Optional[Callable[[int, float], None]] = None

        # Numerical epsilon for scipy
        self.eps = np.finfo("double").eps

    def set_iteration_callback(self, callback: Optional[Callable[[int, float], None]]) -> None:
        """Register a callback called after each scipy LBFGS iteration."""
        self.iteration_callback = callback

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _distribute_flat_params(self, params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = params[offset : offset + numel].view_as(p.data)
            offset += numel

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """

        if closure is None:
            raise RuntimeError("LBFGSScipy requires a closure.")

        group = self.param_groups[0]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        history_size = group["history_size"]

        def wrapped_closure(flat_params):
            """closure must call zero_grad() and backward()"""
            flat_params = torch.from_numpy(flat_params).to(self._params[0].device)
            self._distribute_flat_params(flat_params)
            loss = closure()
            self._last_loss = loss
            loss_value = loss.item()
            flat_grad = self._gather_flat_grad().cpu().numpy()
            return loss_value, flat_grad

        def callback(_flat_params):
            self._n_iter += 1
            if self.iteration_callback is not None and self._last_loss is not None:
                self.iteration_callback(self._n_iter, float(self._last_loss.item()))
            # Optional: print progress (can be disabled)
            # print('Iter %i Loss %.5f' % (self._n_iter, self._last_loss.item()))

        initial_params = self._gather_flat_params().cpu().numpy()

        # Run scipy L-BFGS-B optimization
        result = fmin_l_bfgs_b(
            wrapped_closure,
            initial_params,
            maxiter=max_iter,
            maxfun=max_eval,
            factr=tolerance_change / self.eps,
            pgtol=tolerance_grad,
            epsilon=0,
            m=history_size,
            callback=callback,
        )

        # Update parameters with final result
        target_device = self._params[0].device
        final_params = torch.from_numpy(result[0]).to(target_device)
        self._distribute_flat_params(final_params)

        return self._last_loss

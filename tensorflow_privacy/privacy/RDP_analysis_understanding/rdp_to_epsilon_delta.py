from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import numpy as np
from scipy import special
import six

########################
# LOG-SPACE ARITHMETIC #
########################

def get_privacy_spent(orders, rdp, target_eps=None, target_delta=None):
  """Compute delta (or eps) for given eps (or delta) from RDP values.

  Args:
    orders: An array (or a scalar) of RDP orders.
    rdp: An array of RDP values. Must be of the same length as the orders list.
    target_eps: If not None, the epsilon for which we compute the corresponding
              delta.
    target_delta: If not None, the delta for which we compute the corresponding
              epsilon. Exactly one of target_eps and target_delta must be None.

  Returns:
    eps, delta, opt_order.

  Raises:
    ValueError: If target_eps and target_delta are messed up.
  """
  if target_eps is None and target_delta is None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (Both are).")

  if target_eps is not None and target_delta is not None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (None is).")

  if target_eps is not None:
    delta, opt_order = _compute_delta(orders, rdp, target_eps)
    return target_eps, delta, opt_order
  else:
    eps, opt_order = _compute_eps(orders, rdp, target_delta)
    return eps, target_delta, opt_order

def _compute_delta(orders, rdp, eps):
  """Compute delta given a list of RDP values and target epsilon.

  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    eps: The target epsilon.

  Returns:
    Pair of (delta, optimal_order).

  Raises:
    ValueError: If input is malformed.

  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

    ##----------------------EQ: 7------------------------
  deltas = np.exp((rdp_vec - eps) * (orders_vec - 1))
  idx_opt = np.argmin(deltas)
  return min(deltas[idx_opt], 1.), orders_vec[idx_opt]


def _compute_eps(orders, rdp, delta):
  """Compute epsilon given a list of RDP values and target delta.

  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.

  Returns:
    Pair of (eps, optimal_order).

  Raises:
    ValueError: If input is malformed.

  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")
##----------------------EQ: 6------------------------
  eps = rdp_vec - math.log(delta) / (orders_vec - 1)

  idx_opt = np.nanargmin(eps)  # Ignore NaNs
  return eps[idx_opt], orders_vec[idx_opt]

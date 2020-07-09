"""RDP analysis of the Sampled Gaussian Mechanism.

Functionality for computing Renyi differential privacy (RDP) of an additive
Sampled Gaussian Mechanism (SGM). Its public interface consists of two methods:
  compute_rdp(q, noise_multiplier, T, orders) computes RDP for SGM iterated
                                   T times.
  get_privacy_spent(orders, rdp, target_eps, target_delta) computes delta
                                   (or eps) given RDP at multiple orders and
                                   a target value for eps (or delta).

Example use:

Suppose that we have run an SGM applied to a function with l2-sensitivity 1.
Its parameters are given as a list of tuples (q1, sigma1, T1), ...,
(qk, sigma_k, Tk), and we wish to compute eps for a given delta.
The example code would be:

  max_order = 32
  orders = range(2, max_order + 1)
  rdp = np.zeros_like(orders, dtype=float)
  for q, sigma, T in parameters:
   rdp += rdp_accountant.compute_rdp(q, sigma, T, orders)
  eps, _, opt_order = rdp_accountant.get_privacy_spent(rdp, target_delta=delta)
"""

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

def compute_rdp(q, noise_multiplier, steps, orders):
  """Compute RDP of the Sampled Gaussian Mechanism.

  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
        to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.

  Returns:
    The RDPs at all orders, can be np.inf.
  """
  if np.isscalar(orders):
    rdp = _compute_rdp(q, noise_multiplier, orders) # Call-1
  else:
    rdp = np.array([_compute_rdp(q, noise_multiplier, order)
                    for order in orders])

  return rdp * steps

def _compute_rdp(q, sigma, alpha): # Called-1
  """Compute RDP of the Sampled Gaussian mechanism at order alpha.
  Args:
    q: The sampling rate.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.
  Returns:
    RDP at alpha, can be np.inf.
  """
  if q == 0:
    return 0

  if q == 1.:
    return alpha / (2 * sigma**2)

  if np.isinf(alpha):
    return np.inf

  return _compute_log_a(q, sigma, alpha) / (alpha - 1) # Call-2

def _compute_log_a(q, sigma, alpha):  # Called-2
  """Compute log(A_alpha) for any positive finite alpha."""
  if float(alpha).is_integer():
    return _compute_log_a_int(q, sigma, int(alpha)) # Call-3
  
def _compute_log_a_int(q, sigma, alpha): # Called-3
  """Compute log(A_alpha) for integer alpha. 0 < q < 1."""
  assert isinstance(alpha, six.integer_types)

  # Initialize with 0 in the log space.
  log_a = -np.inf
##--------- EQ:1 from notes summed from k=0 to alpha----------
  for i in range(alpha + 1):
    ##-------EQ:4-----
    log_coef_i = (
        math.log(special.binom(alpha, i)) + i * math.log(q) +
        (alpha - i) * math.log(1 - q))
	##-------EQ:5-----
    s = log_coef_i + (i * i - i) / (2 * (sigma**2))
    log_a = _log_add(log_a, s)

 ##--------- EQ:1 summed from k=0 to alpha DONE----------
  return float(log_a)

def _log_add(logx, logy):
  """Add two numbers in the log space."""
  a, b = min(logx, logy), max(logx, logy)
  if a == -np.inf:  # adding 0
    return b
  # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
  return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)
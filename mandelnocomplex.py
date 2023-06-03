from utils import messure, progress_bar
from mandelbase import MandelBase
import numpy as np
import tensorflow as tf

class MandelNoComplex(MandelBase):
  def __init__(self) -> None:
    super().__init__()

  def recalculate_c(self) -> None:
    im = np.repeat(np.linspace(self._y_min, self._y_max, self._height).reshape((1, -1)), self._width, axis=0).T
    re = np.repeat(np.linspace(self._x_min, self._x_max, self._width).reshape((1, -1)), self._height, axis=0)
    self._c = np.stack((re, im), axis=2)

  @messure
  def calculate_mandelbrot(self) -> np.ndarray:
    negate = tf.constant([[1.,-1.]], dtype=tf.float64)
    C = tf.constant(self._c)
    Z = tf.Variable(np.zeros_like(self._c))
    N = tf.Variable(np.ones((self._height, self._width)), dtype=tf.float64) # use ones because of log

    progress_bar(0, self._iterations)

    if self._inplace:
      if self._masking:
        mask = tf.Variable(tf.ones(N.shape, dtype=tf.bool))
        for idx in range(self._iterations):
          step_inplace_masking(Z, C, N, negate, mask, self._threshold)
          progress_bar(idx, self._iterations)
      else:
        for idx in range(self._iterations):
          step_inplace(Z, C, N, negate, self._threshold)
          progress_bar(idx, self._iterations)
    else:
      if self._masking:
        mask = tf.ones(N.shape, dtype=tf.bool)
        for idx in range(self._iterations):
          Z, N, mask = step_masking(Z, C, N, negate, mask, self._threshold)
          progress_bar(idx, self._iterations)
      else:
        for idx in range(self._iterations):
          Z, N = step(Z, C, N, negate, self._threshold)
          progress_bar(idx, self._iterations)

    progress_bar(self._iterations, self._iterations, True)

    return N.numpy()

@tf.function
def step(Z: tf.Tensor, C: tf.Tensor, N: tf.Tensor, negate: tf.Tensor, threshold: float) -> tuple[tf.Tensor, tf.Tensor]:
  re = tf.reduce_sum(Z ** 2 * negate, axis=2)
  im = tf.reduce_prod(Z, axis=2) * 2
  Zn = tf.stack((re, im), axis=2) + C
  conv = tf.sqrt(tf.reduce_sum(Zn ** 2, axis=2)) < threshold
  Nn = tf.add(N, tf.cast(conv, tf.float64))
  return Zn, Nn

@tf.function
def step_inplace(Z: tf.Variable, C: tf.Tensor, N: tf.Variable, negate: tf.Tensor, threshold: float) -> None:
  re = tf.reduce_sum(Z ** 2 * negate, axis=2)
  im = tf.reduce_prod(Z, axis=2) * 2
  Z.assign(tf.stack((re, im), axis=2) + C)
  conv = tf.sqrt(tf.reduce_sum(Z ** 2, axis=2)) < threshold
  N.assign_add(tf.cast(conv, tf.float64))

@tf.function
def step_masking(Z: tf.Tensor, C: tf.Tensor, N: tf.Tensor, negate: tf.Tensor, mask: tf.Tensor, threshold: float) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  indices = tf.where(mask)
  Zm = tf.gather_nd(Z, indices)
  Cm = tf.gather_nd(C, indices)
  re = tf.reduce_sum(Zm ** 2 * negate, axis=1)
  im = tf.reduce_prod(Zm, axis=1) * 2
  Zn = tf.tensor_scatter_nd_update(Z, indices, tf.stack((re, im), axis=1) + Cm)
  maskn = tf.sqrt(tf.reduce_sum(Zn ** 2, axis=2)) < threshold
  Nn = tf.add(N, tf.cast(maskn, tf.float64))
  return Zn, Nn, maskn

@tf.function
def step_inplace_masking(Z: tf.Variable, C: tf.Tensor, N: tf.Variable, negate: tf.Tensor, mask: tf.Variable, threshold: float) -> None:
  indices = tf.where(mask)
  Zm = tf.gather_nd(Z, indices)
  Cm = tf.gather_nd(C, indices)
  re = tf.reduce_sum(Zm ** 2 * negate, axis=1)
  im = tf.reduce_prod(Zm, axis=1) * 2
  Z.assign(tf.tensor_scatter_nd_update(Z, indices, tf.stack((re, im), axis=1) + Cm))
  mask.assign(tf.sqrt(tf.reduce_sum(Z ** 2, axis=2)) < threshold)
  N.assign_add(tf.cast(mask, tf.float64))

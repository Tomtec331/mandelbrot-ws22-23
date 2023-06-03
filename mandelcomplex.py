from utils import messure, progress_bar
from mandelbase import MandelBase
import numpy as np
import tensorflow as tf

class MandelComplex(MandelBase):
  def __init__(self) -> None:
    super().__init__()

  def recalculate_c(self) -> None:
    y = np.repeat(np.linspace(self._y_min, self._y_max, self._width, False, dtype=np.float64), self._height, axis=0).reshape((self._width, self._height))
    x = np.repeat([np.linspace(self._x_min, self._x_max, self._height, False, dtype=np.float64)], self._width, axis=0)
    self._c = x + 1j * y

  @messure
  def calculate_mandelbrot(self) -> np.ndarray:
    C:  tf.Tensor = tf.constant(self._c)
    Z = tf.Variable(tf.zeros_like(self._c))
    N = tf.Variable(tf.ones((self._width, self._height), dtype=tf.float64)) # use ones because of log

    progress_bar(0, self._iterations)

    if self._inplace:
      if self._masking:
        mask = tf.Variable(tf.ones(N.shape, dtype=tf.bool))
        for idx in range(self._iterations):
          Zw = tf.constant(Z) # workaround for gather_nd
          step_inplace_masking(Z, Zw, C, N, mask, self._threshold)
          progress_bar(idx, self._iterations)
      else:
        for idx in range(self._iterations):
          step_inplace(Z, C, N, self._threshold)
          progress_bar(idx, self._iterations)
    else:
      if self._masking:
        Z = tf.constant(Z) # workaround for gather_nd
        mask = tf.ones(N.shape, dtype=tf.bool)
        for idx in range(self._iterations):
          Z, N, mask = step_masking(Z, C, N, mask, self._threshold)
          progress_bar(idx, self._iterations)
      else:
        for idx in range(self._iterations):
          Z, N = step(Z, C, N, self._threshold)
          progress_bar(idx, self._iterations)

    progress_bar(self._iterations, self._iterations, True)

    return N.numpy()

@tf.function
def step(Z: tf.Tensor, C: tf.Tensor, N: tf.Tensor, threshold: float) -> tuple[tf.Tensor, tf.Tensor]:
  Zn: tf.Tensor = Z ** 2 + C
  conv = tf.abs(Zn) < threshold
  Nn = tf.add(N, tf.cast(conv, tf.float64))
  return Zn, Nn

@tf.function
def step_inplace(Z: tf.Variable, C: tf.Tensor, N: tf.Variable, threshold: float) -> None:
  Z.assign(Z ** 2 + C)
  conv = tf.abs(Z) < threshold
  N.assign_add(tf.cast(conv, tf.float64))

@tf.function
def step_masking(Z: tf.Tensor, C: tf.Tensor, N: tf.Tensor, mask: tf.Tensor, threshold: float) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  indices = tf.where(mask)
  Zm = tf.gather_nd(Z, indices)
  Cm = tf.gather_nd(C, indices)
  Zn = tf.tensor_scatter_nd_update(Z, indices, Zm ** 2 + Cm)
  maskn = tf.abs(Zn) < threshold
  Nn = tf.add(N, tf.cast(maskn, tf.float64))
  return Zn, Nn, maskn

@tf.function # Zw is a workaround as gather_nd has problems with tf.Variable with dtype=complex
def step_inplace_masking(Z: tf.Variable, Zw: tf.Tensor, C: tf.Tensor, N: tf.Variable, mask: tf.Variable, threshold: float) -> None:
  indices = tf.where(mask)
  Zm = tf.gather_nd(Zw, indices)
  Cm = tf.gather_nd(C, indices)
  Z.assign(tf.tensor_scatter_nd_update(Z, indices, Zm ** 2 + Cm))
  mask.assign(tf.abs(Z) < threshold)
  N.assign_add(tf.cast(mask, tf.float64))

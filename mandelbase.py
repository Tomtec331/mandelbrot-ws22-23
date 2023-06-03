from abc import ABC, abstractmethod
import numpy as np

class MandelBase(ABC):
  def __init__(self) -> None:
    self._inplace = False
    self._masking = False

    self._x_min = -2.0
    self._x_max =  2.0
    self._y_min = -2.0
    self._y_max =  2.0

    self._threshold = 2.0
    self._iterations = int(100)

    self._width  = int(800)
    self._height = int(800)

    self.recalculate_c()
    super().__init__()

  def set_functionality(self, inplace: bool, masking: bool) -> None:
    """Set which functionality should be used for calculation"""
    if isinstance(inplace, bool):
      self._inplace = inplace
    else:
      print("Warning: Wrong type for set_functionality (inplace). Value not changed.")

    if isinstance(masking, bool):
      self._masking = masking
    else:
      print("Warning: Wrong type for set_functionality (masking). Value not changed.")

  def set_threshold(self, value: float) -> None:
    """Set the threshold used while calculating the mandelbrot image"""
    if isinstance(value, float) or isinstance(value, int):
      self._threshold = float(value)
    else:
      print("Warning: Wrong type for set_threshold. Defaulting to 2.0.")
      self._threshold = 2.0

  def set_iterations(self, value: int) -> None:
    """Set the iteration count for calulating the mandelbrot image"""
    if isinstance(value, int) or isinstance(value, float):
      self._iterations = int(value)
    else:
      print("Warning: Wrong type for set_iterations. Defaulting to 100.")
      self._iterations = int(100)

  def set_resolution(self, width: int, height: int) -> None:
    """Set the resolution of the image"""
    old_width = self._width
    old_height = self._height

    if isinstance(width, int) or isinstance(width, float):
      self._width  = int(width)
    else:
      print("Warning: Wrong type for set_resolution (width). Defaulting to 800.")
      self._width  = int(800)

    if isinstance(height, int) or isinstance(height, float):
      self._height  = int(height)
    else:
      print("Warning: Wrong type for set_resolution (height). Defaulting to 800.")
      self._height  = int(800)
    
    if self._height != old_height or self._width != old_width:
      self.recalculate_c()

  def set_coordinates(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    """Set the coordinates of the image section"""
    old_x_min = self._x_min
    old_x_max = self._x_max
    old_y_min = self._y_min
    old_y_max = self._y_max

    if (isinstance(x_min, float) or isinstance(x_min, int)) and \
       (isinstance(x_max, float) or isinstance(x_max, int)) and \
       (isinstance(y_min, float) or isinstance(y_min, int)) and \
       (isinstance(y_max, float) or isinstance(y_max, int)):
      self._x_min = float(x_min)
      self._x_max = float(x_max)
      self._y_min = float(y_min)
      self._y_max = float(y_max)

      if self._x_min != old_x_min or \
         self._x_max != old_x_max or \
         self._y_min != old_y_min or \
         self._y_max != old_y_max:
        self.recalculate_c()
    else:
      print("Warning: Wrong type for set_coordinates. Nothing will be changed.")

  @abstractmethod
  def recalculate_c(self) -> None:
    """Recalculates the C matrix. Called by constructor, set_resolution and set_coordinates!"""
    raise NotImplementedError("Abstract function called")

  @abstractmethod
  def calculate_mandelbrot(self) -> np.ndarray:
    """Calculate the mandelbrot image"""
    raise NotImplementedError("Abstract function called")

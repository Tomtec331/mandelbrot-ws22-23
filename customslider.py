from typing import Sequence, Callable
from matplotlib.widgets import Slider
from matplotlib.figure import Figure
from matplotlib.axes import Axes

class CustomSlider():
  def __init__(self, figure: Figure, position: Sequence[float], label: str, min: float, max: float, init_value: float, handler: Callable[[str], None]) -> None:
    self._min: float = None
    self._max: float = None
    self._init_value: float = None
    self._axis: Axes = None
    self._handler_id: int = None
    self.raw: Slider = None

    self._label = label
    self._figure = figure
    self._position = position
    self._handler = handler

    self.set_limits(min, max, init_value)

  @property
  def val(self):
    return self.raw.val

  def set_val(self, value: float, no_callback = False) -> None:
    """Set the current value of the slider"""
    if no_callback:
      self.raw.disconnect(self._handler_id)

    self.raw.set_val(value)

    if no_callback:
      self._handler_id = self.raw.on_changed(self._handler)

  def set_limits(self, min: float, max: float, init_value: float) -> None:
    """Change the limits of the slider"""
    if self._min != min or self._max != max or self._init_value != init_value:
      self._min = min
      self._max = max
      self._init_value = init_value

      if self._axis != None:
        self._axis.remove()

      self._axis = self._figure.add_axes(self._position)
      self.raw = Slider(self._axis, self._label, self._min, self._max, self._init_value, orientation="vertical")
      self._handler_id = self.raw.on_changed(self._handler)
      self._figure.canvas.draw_idle()

  def get_limits(self) -> tuple[float, float, float]:
    """Gets the currently set limits (min, max, init)"""
    return self._min, self._max, self._init_value

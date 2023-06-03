import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from customslider import CustomSlider
from matplotlib.widgets import CheckButtons, RadioButtons, Button, TextBox, RectangleSelector
from utils import debounce, my_cmap
from mandelcomplex import MandelComplex
from mandelnocomplex import MandelNoComplex

# default values
default_iterations = int(100)
default_resolution = int(800)
default_threshold = 2.0
default_max: float = np.log(default_iterations)
default_max_interval = (0.0, default_max * 2)
default_min: float = 0.0
default_min_interval = (-default_max, default_max)
default_x_min = -2.0
default_y_min = -2.0
default_x_max =  2.0
default_y_max =  2.0

# current values
active_renderer = MandelComplex()
current_fractal = np.ones((1, 1))
current_fractal_modified = None
current_x_min = default_x_min
current_x_max = default_x_max
current_y_min = default_y_min
current_y_max = default_y_max

# set default coords
active_renderer.set_coordinates(current_x_min, current_x_max, current_y_min, current_y_max)

fig, ax = plt.subplots()
fig.subplots_adjust(left=0.2, bottom=0.25) # make room for controls

# mouse handler -------------------------------------------
def handler_click(event):
  """Handler for mouse clicks"""
  global current_x_max, current_x_min, current_y_max, current_y_min

  if event.button == 3:
    current_x_max = default_x_max
    current_x_min = default_x_min
    current_y_max = default_y_max
    current_y_min = default_y_min
    active_renderer.set_coordinates(current_x_min, current_x_max, current_y_min, current_y_max)
    recalculate_image()

fig.canvas.mpl_connect('button_press_event', handler_click)

# interactive selector ------------------------------------
def handler_selector(eclick, erelease) -> None:
  """Handler for rectangle selector"""
  global current_x_max, current_x_min, current_y_max, current_y_min

  x1, y1 = eclick.xdata, eclick.ydata
  x2, y2 = erelease.xdata, erelease.ydata

  width = get_current_width()
  height = get_current_height()

  # map to 0..1
  x1_a: float = x1 / width
  y1_a: float = y1 / height
  x2_a: float = x2 / width
  y2_a: float = y2 / height

  # map to complex plane corner points
  x_min = x1_a * (current_x_max - current_x_min) + current_x_min
  y_min = y1_a * (current_y_max - current_y_min) + current_y_min
  x_max = x2_a * (current_x_max - current_x_min) + current_x_min
  y_max = y2_a * (current_y_max - current_y_min) + current_y_min

  current_x_min = x_min
  current_x_max = x_max
  current_y_min = y_min
  current_y_max = y_max

  # set coordinates
  active_renderer.set_coordinates(current_x_min, current_x_max, current_y_min, current_y_max)
  recalculate_image()

selector = RectangleSelector(
  ax,
  handler_selector,
  useblit=True,
  button=[1], 
  minspanx=5,
  minspany=5,
  spancoords="pixels",
  interactive=False
)

# postprocessing ------------------------------------------
def handler_postprocessing(label) -> None:
  """Handler for postprocessing checkboxes"""
  modify_fractal()
  set_slider_limits()
  if get_current_automax_state():
    set_image_maximum()
  set_image()

position = fig.add_axes([0.05, 0.85, 0.15, 0.1])
position.text(0.0, 1.05, "Post-Processing")
labels = ["Use Logarithm", "Automatic maximum"]
visibility = [True, False]
post = CheckButtons(position, labels, visibility)
post_id = post.on_clicked(handler_postprocessing)

def get_current_log_state() -> bool:
  """Get the current log checkbox state"""
  log, auto = post.get_status()
  return log

def get_current_automax_state() -> bool:
  """Get the current automatic maximum checkbox state"""
  log, auto = post.get_status()
  return auto

# min, max sliders ----------------------------------------
def set_slider_limits(reset = False) -> None:
  """Sets slider limits according the iteration count"""
  old_min, old_max, old_init = max_slider.get_limits()
  iter = get_current_iterations()
  iter_modified = modify_data(iter)

  if old_init != iter_modified:
    max_slider.set_limits(0.0, iter_modified * 2, iter_modified)
    min_slider.set_limits(-iter_modified, iter_modified, 0.0)
  else:
    if reset:
      max_slider.set_val(iter_modified, True)
      min_slider.set_val(0.0, True)

@debounce(0.5)
def handler_max_slider(value) -> None:
  """Handler for max slider"""
  current_min = min_slider.val
  if value < current_min:
    max_slider.set_val(current_min)
  else:
    set_image()

@debounce(0.5)
def handler_min_slider(value) -> None:
  """Handler for min slider"""
  current_max = max_slider.val
  if value > current_max:
    min_slider.set_val(current_max)
  else:
    set_image()

min_slider = CustomSlider(
  fig,
  [0.08, 0.28, 0.02, 0.50],
  "Min",
  default_min_interval[0],
  default_min_interval[1],
  default_min,
  handler_min_slider
)

max_slider = CustomSlider(
  fig,
  [0.13, 0.28, 0.02, 0.50],
  "Max",
  default_max_interval[0],
  default_max_interval[1],
  default_max,
  handler_max_slider
)

# enable complex ------------------------------------------
def handler_complex(label) -> None:
  """Handler for complex radio buttons"""
  global active_renderer
  
  if label == "No":
    print ("Set new renderer -> No Complex")
    active_renderer = MandelNoComplex()
  else:
    print ("Set new renderer -> Complex")
    active_renderer = MandelComplex()

  active_renderer.set_functionality(
    get_current_inplace_state(),
    get_current_masking_state()
  )
  active_renderer.set_iterations(get_current_iterations())
  active_renderer.set_threshold(get_current_threshold())
  active_renderer.set_resolution(get_current_width(), get_current_height())
  active_renderer.set_coordinates(current_x_min, current_x_max, current_y_min, current_y_max)

position = fig.add_axes([0.05, 0.05, 0.15, 0.1])
position.text(0.0, 1.05, "Use complex")
radio = RadioButtons(position, ("Yes", "No"))
radio.on_clicked(handler_complex)

# functionalities -----------------------------------------
def handler_functionalities(label) -> None:
  """Handler for functionalities checkboxes"""
  inplace, masking = check.get_status()
  active_renderer.set_functionality(inplace, masking)

position = fig.add_axes([0.21, 0.05, 0.15, 0.1])
position.text(0.0, 1.05, "Functionality")
labels = ["Inplace", "Masking"]
visibility = [False, False]
check = CheckButtons(position, labels, visibility)
check.on_clicked(handler_functionalities)

def get_current_inplace_state() -> bool:
  """Get the current inplace checkbox state"""
  inplace, masking = check.get_status()
  return inplace

def get_current_masking_state() -> bool:
  """Get the current masking checkbox state"""
  inplace, masking = check.get_status()
  return masking

# resolution ----------------------------------------------
position = fig.add_axes([0.37, 0.105, 0.15, 0.045])
position.text(0.0, 1.1, "Resolution")
res_height = TextBox(position, "H", default_resolution, label_pad=-0.15, textalignment="center")
position = fig.add_axes([0.37, 0.05, 0.15, 0.045])
res_width = TextBox(position, "W", default_resolution, label_pad=-0.15, textalignment="center")

def get_current_height() -> int:
  """Get the current image height (user input)"""
  try:
    return int(res_height.text)
  except ValueError:
    print(f"Invalid user input (Resolution->Height). Defaulting to {default_resolution}.")
    return default_resolution

def get_current_width() -> int:
  """Get the current image width (user input)"""
  try:
    return int(res_width.text)
  except ValueError:
    print(f"Invalid user input (Resolution->Width). Defaulting to {default_resolution}.")
    return default_resolution

# threshold -----------------------------------------------
position = fig.add_axes([0.53, 0.105, 0.15, 0.045])
position.text(0.0, 1.1, "Threshold")
threshold = TextBox(position, "", default_threshold, textalignment="center")

def get_current_threshold() -> float:
  """Get the current mandelbrot threshold (user input)"""
  try:
    return float(threshold.text)
  except ValueError:
    print(f"Invalid user input (Threshold). Defaulting to {default_threshold}.")
    return default_threshold

# reset ---------------------------------------------------
def handler_reset(event) -> None:
  """Handler for reset button"""
  global current_x_max, current_x_min, current_y_max, current_y_min, post_id

  current_x_max = default_x_max
  current_x_min = default_x_min
  current_y_max = default_y_max
  current_y_min = default_y_min

  if get_current_inplace_state():
    check.set_active(0)
  if get_current_masking_state():
    check.set_active(1)

  res_height.set_val(default_resolution)
  res_width.set_val(default_resolution)
  threshold.set_val(default_threshold)
  iterations.set_val(default_iterations)
  radio.set_active(0)
  active_renderer.set_coordinates(current_x_min, current_x_max, current_y_min, current_y_max)

  post.disconnect(post_id)
  if not get_current_log_state():
    post.set_active(0)
  if get_current_automax_state():
    post.set_active(1)
  post_id = post.on_clicked(handler_postprocessing)

  set_slider_limits(True)
  recalculate_image()

position = fig.add_axes([0.53, 0.05, 0.15, 0.045])
reset = Button(position, "Reset")
reset.on_clicked(handler_reset)

# iterations ----------------------------------------------
position = fig.add_axes([0.69, 0.105, 0.15, 0.045])
position.text(0.0, 1.1, "Iterations")
iterations = TextBox(position, "", default_iterations, textalignment="center")

def get_current_iterations() -> int:
  """Get the current iteration count (user input)"""
  try:
    return int(iterations.text)
  except ValueError:
    print(f"Invalid user input (Iterations). Defaulting to {default_iterations}.")
    return default_iterations

# redraw button -------------------------------------------
def handler_draw(event) -> None:
  """Handler for draw button"""
  active_renderer.set_resolution(get_current_width(), get_current_height())
  active_renderer.set_threshold(get_current_threshold())
  active_renderer.set_iterations(get_current_iterations())
  set_slider_limits(True)
  recalculate_image()

position = fig.add_axes([0.69, 0.05, 0.15, 0.045])
draw = Button(position, "Draw")
draw.on_clicked(handler_draw)

# main code -----------------------------------------------
def recalculate_image() -> None:
  """Recalculates and displays the image"""
  global current_fractal
  current_fractal = active_renderer.calculate_mandelbrot()
  modify_fractal()
  if get_current_automax_state():
    set_image_maximum()
  set_image()

def modify_data(data: Any) -> Any:
  """Modifies data with log if log state is set"""
  use_log = get_current_log_state()
  return np.log(data) if use_log else data

def modify_fractal() -> None:
  """Modifies the fractal array if log state is set"""
  global current_fractal_modified
  current_fractal_modified = modify_data(current_fractal)
  print("Variance:", np.var(current_fractal_modified / np.max(current_fractal_modified)))

def set_image_maximum():
  """Sets the image maximum value as slider values"""
  current_max = np.max(current_fractal_modified)
  max_slider.set_val(current_max, True)
  min_slider.set_val(0.0, True)

def set_image() -> None:
  """Refreshes the mpl canvas with current values"""
  ax.clear()
  ax.axis("off")
  ax.imshow(current_fractal_modified, cmap=my_cmap, vmin=min_slider.val, vmax=max_slider.val)
  fig.canvas.draw_idle()

recalculate_image()
plt.show()

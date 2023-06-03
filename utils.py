import time
from threading import Timer
from matplotlib.colors import LinearSegmentedColormap

cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 0.0, 0.0),
                 (0.4, 0.2, 0.2),
                 (0.6, 0.0, 0.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
        'green':((0.0, 0.0, 0.0),
                 (0.1, 0.0, 0.0),
                 (0.2, 0.0, 0.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 1.0, 1.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 0.0, 0.0),
                 (0.8, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}

my_cmap = LinearSegmentedColormap('my_colormap', cdict, 256)

def messure(func):
  """Messures a functions run-time"""
  def decorator(*args, **kwargs):
    start = time.perf_counter()
    res = func(*args, **kwargs)
    stop = time.perf_counter()
    print (f"Calculation took {stop - start:0.4f} seconds to finish")
    return res

  return decorator

# Source: https://gist.github.com/walkermatt/2871026
def debounce(wait):
  """ Decorator that will postpone a functions
      execution until after wait seconds
      have elapsed since the last time it was invoked. """
  def decorator(fn):
    def debounced(*args, **kwargs):
      def call_it():
        fn(*args, **kwargs)
      try:
        debounced.t.cancel()
      except(AttributeError):
        pass
      debounced.t = Timer(wait, call_it)
      debounced.t.start()
    return debounced
  return decorator

def progress_bar(current, total, remove = False):
  """Mimics a progress bar in console"""
  barLength = 40
  percent = float(current) * 100 / total
  done = "█" * int(percent/100 * barLength)
  free = "░" * (barLength - len(done))
  mesg = "> Calculating image: [%s%s] %d %%" % (done, free, percent)

  if remove: print(" " * len(mesg), end="\r")
  else: print(mesg, end="\r")

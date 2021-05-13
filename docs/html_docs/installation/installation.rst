============
Installation
============

-------------------------
Installation From Release
-------------------------

pyNastran is an easy package to install once you have the required Python
modules.  It's a pure Python package so you shouldn't have too many problems.
Just type on the command line:

``pip install pyNastran``

That will install the minimum set of what you need to run pyNastran (so no GUI).
If you want GUI functionality, chances are you have PyQt5 or PySide2, but don't have vtk.
Vtk is a bit more challenging on Windows, but there is a website to help with that.

Additionally, the software can **optionally** use matplotlib, pandas, h5py, colorama,
but chances are you already have those.  If you don't, they're very easy to install.

Python
------
The software is tested against:

* Python 3.7 **(Windows/Linux)**
* Python 3.8 **(Windows/Linux)**
* Python 3.9 **(Windows/Linux)**  (available in pyNastran 1.4)

Packages
--------
pyNastran is tested against a range of package versions (lowest to highest
based on availability), so it should work.  The recommended set of packages are:

* **Required**:

  * numpy >= 1.14
  * scipy >= 1.0
  * cpylog >= 1.4.0
  * docopt-ng == 0.7.2   **(required for command line tools)**

* **Optional**:

  * colorama >= 0.3.9    **(colored logging)**
  * pandas >= 0.25
  * matplotlib >= 2.2.4  **(plotting)**
  * h5py >= 2.8.0        **(HDF5 support)**

* **GUI**:

  * vtk >= 7  (vtk==9 is somewhat buggy)
  * qtpy >= 1.4.0
  * Qt **(pick one)**

    * PyQt5 >= 5.9.2
    * PySide2 >= 5.11.2

  * QScintilla >= ??? **(optional for fancy scripting; PyQt5 only)**
  * pygments >= 2.2.0 **(optional for fancy scripting; PyQt5 only)**
  * imageio >= 2.4.1  **(optional for animation support)**

-----------------------------------------------------
Install Procedure - From Regular Python (recommended)
-----------------------------------------------------
Base functionality:

* `64-bit Python <https://www.python.org/downloads/>`_
* ``pip install numpy``
* ``pip install scipy``
* ``pip install pandas``     **(optional)**
* ``pip install h5py``       **(optional for HDF5 support)**
* ``pip install matplotlib`` **(optional for plotting)**
* ``pip install colorama``   **(optional for colored logging)**
* ``pip install docopt-ng``   **(required for command line tools)**
* ``pip install cpylog``
* ``pip install pyNastran``

For **optional** GUI support:

* On the command line:

  * ``pip install imageio`` **(optional for animation support)**
  * ``pip install pyside2``
  * ``pip install VTK*.whl``
  * ``pip install qtpy``

* Additional source for `Windows binaries <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_

----------------------------------------------------------------
Install Procedure - From Anaconda (not recommended and untested)
----------------------------------------------------------------

You've been warned, but in general Anaconda doesn't work well with pip.  You need to be very careful with using ``pip`` instead of ``conda``.  In general, it's best to always use conda first and pip only if conda fails.

* `64-bit Python <https://www.anaconda.com/products/individual>`_
* ``conda install numpy``
* ``conda install scipy``
* ``conda install pandas``   **(optional)**
* ``conda install h5py``       **(optional for HDF5 support)**
* ``conda install matplotlib`` **(optional for plotting)**
* ``conda install colorama``   **(optional for colored logging)**
* ``pip install docopt-ng``   **(required for command line tools)**
* ``pip install cpylog``
* ``pip install pyNastran``

Documentation
-------------
Two options for documentation exist.

* https://pynastran-git.readthedocs.io/en/latest/installation/building_docs.html

If you don't want to use build the docs, just use the docs on the web.

See `docs <https://pynastran-git.readthedocs.io/en/latest/>`_

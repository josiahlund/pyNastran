############
Installation
############

*******************************************
Install From Release - Python (recommended)
*******************************************

Installation from release is recommended for most users.

For Base functionality:

.. code-block:: console

  pip install pyNastran

For optional GUI support:

.. code-block:: console

  pip install PyQt5
  pip install tvk

For additional optional features:

.. code-block:: console

  pip install pandas     (for ?)
  pip install h5py       (for HDF5 support)
  pip install colorama   (for colored logging)

************************
Installation From Source
************************

Installing from source is only recommend for developers or users with air gapped machines.

1. Obtain source code by either cloning or download from Github
2. Use python to install
  .. code-block:: console

    python setup.py install

  **OR**

  .. code-block:: console

    python setup_no_gui.py install

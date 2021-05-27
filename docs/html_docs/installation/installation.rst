*******************************************
Install From Release - Python (recommended)
*******************************************

Installation from release is recommended for most users.

For Base functionality:

* ``pip install pyNastran``

For optional GUI support:

* ``pip install PyQt5``
* ``pip install tvk``

For additional optional features:

* ``pip install pandas``     **(What does this provide?)**
* ``pip install h5py``       **(HDF5 support)**
* ``pip install colorama``   **(colored logging)**

*****************************************
Installation From Source (Advanced Users)
*****************************************

Installing from source is recommend only for developers and users with air gapped machines which lack the ability to pip
install PyNastran.

* Clone or download pyNastran from Github
* ``python setup.py install`` **OR** ``python setup_no_gui.py install``

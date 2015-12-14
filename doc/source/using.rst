==============
Using the code
==============

The requirements are:

* `Python <http://www.python.org/>`_ 2.6 or higher
* `numpy <http://www.numpy.org/>`_ and `scipy <http://www.scipy.org/>`_
* `matplotlib <http://www.matplotlib.org/>`_ for plotting
* `gmpy2 <https://code.google.com/p/gmpy/>`_ the arbituary precision package
* `mpi4py <http://www.mpi4py.scipy.org/>`_ (optional)

These packages can be installed using a single command::

    $pip install -r requirements.txt

**requirements.txt** is contained in the source code.

**Before installing these packages, make sure you have lapack or mkl library on your host**.
However, it is recommended to use `Anaconda <https://www.continuum.io/downloads/>`_ to install these packages.

Download the code using the `Download ZIP
<https://github.com/GiggleLiu/nrg_mapping/archive/master.zip>`_
button on github, or run the following command from a terminal::

    $ wget -O nrg_mapping-master.zip https://github.com/GiggleLiu/nrg_mapping/archive/master.zip

Within a terminal, execute the following to unpack the code::

    $ unzip nrg_mapping-master.zip
    $ cd nrg_mapping-master/source/

Once the relevant software is installed, each program is contained
entirely in a single file.  The first program, for instance, can be
run by issuing::

    $ python sample_simple.py

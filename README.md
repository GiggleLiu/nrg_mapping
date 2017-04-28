#Map a channel mixing bath to a chain

This is a versaltile mapping scheme starting from hybridization function,
it maps a general multi-orbital non-interacting bath with channel mixing to a chain model.

It provides a solution to impurity solvers that requires a concrete physical model(chain Hamiltonian), like NRG/vMPS et. al.

###To use this program
Please install the following numerical packages for python(pip install -r requirements.txt)

* numpy
* scipy
* matplotlib
* gmpy2
* mpi4py(optional)

or *Anaconda* all in one pack: https://store.continuum.io/cshop/anaconda/

###To run the first example.
    ```
    $ cd source/
    $ python quickmap.py
    ```

Program will use the default configuration file 'config-sample.ini' as input.
Also, you can specify your own configuration file by `$ python quickmap.py <your-config>.ini`.
The input hybridization is specified by a data file('sample_hybridization_func.dat' above),
it will be interpolated as a continuous function, thus choosing a good omega mesh(like a logarithmic one) is important.

###Documentation
* Paper: 

        Quantum impurities in channel mixing baths
        Jin-Guo Liu, Da Wang, and Qiang-Hua Wang
        Phys. Rev. B 93, 035102 – Published 4 January 2016

        Its Arxiv version: **doc/1509.01461v2.pdf**, **doc/SupplMater.pdf**

* Technical Details: **doc/technical.pdf**
* Program documentation: **doc/program_manual.pdf**

###Information
* Author:  Jinguo Leo, NanJing University.
* Paper:  arXiv:1509.01461
* Date:  2015/09/07
* Contact: dg1422033@smail.nju.edu.cn



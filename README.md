# Mapping a channel mixing bath to a Wilson chain

A versaltile mapping scheme starting from hybridization function,
mapping a general multi-orbital non-interacting channel mixing bath to a Wilson chain model.
The motivation is to transform the impurity problems to a real space chain, thus can be solved using NRG and VMPS.

### To use this program
Please install the following numerical packages for python(pip install -r requirements.txt)

* numpy
* scipy
* matplotlib
* gmpy2(optional)

or *Anaconda* all in one pack: https://store.continuum.io/cshop/anaconda/

### To run an example.
    ```
    $ cd source/
    $ python quickmap.py
    ```

Program will use the default configuration file 'config-sample.ini' as input.
Also, you can specify your own configuration file by `$ python quickmap.py <your-config>.ini`.
The input hybridization is specified by a data file('sample_hybridization_func.dat' above),
it will be interpolated as a continuous function, thus choosing a good omega mesh(like a logarithmic one) is important.

### Documentation
* Paper: 

        Quantum impurities in channel mixing baths
        Jin-Guo Liu, Da Wang, and Qiang-Hua Wang
        Phys. Rev. B 93, 035102 – Published 4 January 2016

        Its Arxiv version: doc/1509.01461v2.pdf, doc/SupplMater.pdf

* Technical Details: *doc/technical.pdf*
* Program documentation: *doc/program_manual.pdf*

### Information
* Author:  Jinguo Leo, NanJing University.
* Date:  2015/09/07
* Contact: dg1422033@smail.nju.edu.cn



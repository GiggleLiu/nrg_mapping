===================
Tutorial
===================

Example
-----------------------
First, I will show a rather simple example, it looks like:

.. literalinclude:: ../../source/sample_simple.py
   :linenos:

We discretized a hybridization function :math:`\Delta(\omega)={\bf 1}+0.4\omega\sigma_x+0.5(0.1+\omega^2)\sigma_y` 
using the function *discretization.quick_map* for 10 twisting parameters z ranging from 0.05 to 0.95.
Then the discrete model is transform to a Wilson chain using *chaimmapper.map2chain*.
Finally, we checked the validity of the mapping scheme by recovering the hybridization function for both discrete model and Wilson chain.

In the following subsections, we will show how this example works.

Construct hybridization function
-------------------

The first step is to construct the hybridization function.
A hybridization function should be a function(any callable instance) with only one input variable :math:`\omega`
and ouput variable :math:`{\Delta(\omega)}` which could be either a number of a square matrix.

The choice of :math:`\omega` space is important, if a linear :math:`\omega`-mesh is choosen naively,
the precision near :math:`\omega=0` is not guranteened. To generate a logarithmic :math:`\omega`-list::

    >>> from discretization import get_wlist
    >>> wlist = get_wlist(w0=1e-8,Nw=5000,mesh_type='log',D=1,Gap=0)

Here, parameter  *Gap* is the gapped interval, and *w0* is the onset(the minimum energy scale) of logarithmic mesh.

Using the right discretization mesh points.
--------------------
The discretization mesh points(tick_params) are important for aquiring optimal scaling behavior.
Most of times, you need an 'adaptive' mesh in NRG mapping scheme.
The hybridization strength needed by 'adaptive' mesh points is then defined as the mean square of the eigenvalues of :math:`\Delta(\omega)`.

Other types of discretization meshes like *log*, *linear* et. al. which can be used in some special cases are also predefined.

Mapping procedure
---------------------
The simplist approach to discretize the continuous hybridization function is to use the *discretization.quick_map* function.
Tuple of (<Ticker>s, <DiscModel>) is returned by this function.

<Ticker>s is a list with <Ticker> instances for negative and positive branches. The discrete ticks can be generated using::

    >>> indices = arange(1,10)    #the decrete indices.
    >>> z = 1.0        #twisting parameter
    >>> tick_position = ticker(indices+z)   #tick position is always positive.

<DiscModel> is nicknamed "sun model" or "star model" with bath replaced by a set of sites directly coupled to the impurity.

The interface of *quick_map* looks like
.. autofunction:: discretization.quick_map

In this function, the discretization mesh used is specified by a dict *tick_params*,
parameters like *tick_type*, *Lambda* (scaling factor) and some other parameters are passed to the program.

Twisting parameter :math:`z` could be either a float or an 1D array, :math:`0<z\leq 1` is required.

If you're planning to use this model in other programs like Frotran/C++,
use <DiscModel>.save('xxx') method to save it to plain texts, see API of <DiscModel> for details of the data format.

Tridiagonalization towards a Wilson chain.
-------------------------
The method *chainmapper.map2chain* is used to cope with this problem, parameters of this function can be easily understood.
A <Chain> instance is returned by this function, to use it in other programs, call <Chain>.save('xxx') method to store chain datas.

This procedure is based on the Lanczos(*tridiagonalize.tridiagonalize*) Block Lanczos(*tridiagonalize.tridiagonalize_qr*) algorithm.

For 2 x 2 block size, the block lanczos has an optional implementation *tridiagonalize.tridiagonalize2*,
which maximally retains the channel symmetry(general multi-channel version is not implemented due to the difficulty of analitical eigenvalue decomposition).

The interface of function *map2chain* looks like
.. autofunction:: chainmapper.map2chain

Check for validity of mapping
---------------------------
Finally, we need a method to check the validity of results, 
we provided 2 functions to check the quality of the "sun model" and Wilson chain, they are *discretization.check_disc_pauli/discretization.check_disc_eval* and *chainmapper.check_spec*

Due to the difficulty of choosing the smearing factor in Green's function(essentially lorenzian smearing)
for logarithmic disctributed energies, the checking procedure itself brings lost of precision.
The smearing parameter should be choosen carefully to get a suitible result.

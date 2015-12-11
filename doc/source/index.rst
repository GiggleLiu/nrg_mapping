.. LegBuilder documentation master file, created by
   sphinx-quickstart on Mon Sep  7 00:20:20 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===========
Mapping A Impurity Problem to A Chain
===========

Source code: https://github.com/GiggleLiu/nrg_mapping

The goal of this tutorial

The four modules build up DMRG from its simplest implementation to
more complex implementations and optimizations.  Each file adds lines
of code and complexity compared with the previous version.

1. :doc:`Infinite system algorithm <01_infinite_system>`
   (~180 lines, including comments)
2. :doc:`Finite system algorithm <02_finite_system>`
   (~240 lines)

Authors
=======

- JinGuo Leo (NJU)

Licensed under the GNU license.  If you plan to publish work based on
this code, please contact us to find out how to cite us.


contents

.. toctree::
   :maxdepth: 3

   using
   tutorial
   discretization
   chainmapper
   tridiagonalize
   utils
   hybri_sc



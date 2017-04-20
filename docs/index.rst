.. ZhuSuan documentation master file, created by
   sphinx-quickstart on Wed Feb  8 15:01:57 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ZhuSuan
=======

ZhuSuan is a python	library	for	**Generative Models**, built upon
`Tensorflow <https://www.tensorflow.org>`_.
Unlike existing deep learning libraries, which are mainly designed for
supervised tasks, ZhuSuan is featured for its deep root into Bayesian
Inference, thus supporting various kinds of generative models: both the
traditional **hierarchical Bayesian models** and recent
**deep generative models**.

With ZhuSuan, users can enjoy powerful fitting and multi-GPU training of deep
learning, while at the same time they can use generative models to model the
complex world, exploit unlabeled data and deal with uncertainty by performing
principled Bayesian inference.

.. toctree::
   :maxdepth: 2


Getting Started
---------------

This version is for internal release. There are two ways to install.

Install built package
^^^^^^^^^^^^^^^^^^^^^
::

   pip install http://ml.cs.tsinghua.edu.cn/~jiaxin/ZhuSuan-0.3.0-py2.py3-none-any.whl

Install from source
^^^^^^^^^^^^^^^^^^^
Download the source package from::

   http://ml.cs.tsinghua.edu.cn/~jiaxin/ZhuSuan-0.3.0.tar.gz

then follow the README.md in the main directory.

After installation, open your python console and type::

   >>> import zhusuan as zs

If no error occurs, you've successfully installed ZhuSuan.

Tutorials
---------

.. toctree::
   :maxdepth: 2

   tutorials


API Docs
--------

Information on specific functions, classes, and methods.

.. toctree::
   :maxdepth: 2

   api


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

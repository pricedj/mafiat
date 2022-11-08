MAFIAT
======

The Magnetic Field Analysis Tools (MAFIAT) package facilitates the investigation of the magnetic twist of flux ropes in coronal simulation domains.

Installation
------------
Mafiat requires python >= 3.9.

1. To utilise the Jupyter notebooks you should clone this repository.
2. Next it is recommended to create an `Anaconda <https://www.anaconda.com/products/distribution>`_ environment as below:

.. code:: bash

    $ conda create --name mafiat python=3.9
    $ conda activate mafiat

3. Then you can install mafiat and its dependencies from inside the mafiat folder, as below:

.. code:: bash

    $ pip install .

FAQ / Troubleshooting
---------------------
1. If the k3d extension is not working inside of your notebooks you may need to run the following:

.. code:: bash

    $ jupyter nbextension install --py --user k3d
    $ jupyter nbextension enable --py --user k3d

..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _setup:

############
Setup Lorien
############
Lorien is a Python package, so you can simply install it by running the following command.

.. code-block:: bash

    python3 setup.py install

Note that Lorien installation does not include the dialect-specific packages such as TVM in TVM dialects, because every dialect is optional. When launching Lorien, it detects the system environment it is running and loads the supported dialects. In other words, if you want to use the TVM dialects in Lorien, for example, you have to manually install TVM before running Lorien. You can either build TVM from source, or install the nightly build by referring to `this website <https://tlcpack.ai/>`_:

.. code-block:: bash

    pip3 install tlcpack-nightly-cu102 -f https://tlcpack.ai/wheels

If you prefer not to set up an environment on your host machine, you can also consider to run it inside docker with Lorien prebuilt docker images. See :ref:`on-docker` for detail steps.


.. toctree::
   :maxdepth: 1

   on_docker

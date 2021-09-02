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

.. _on-docker:

#############
Docker Images
#############

We provide prebuilt docker images to let you 1) quickly try out Lorien, 2) refer the environment set up, and 3) use it for AWS batch containers. You can find them on `Docker Hub <https://hub.docker.com/r/comaniac0422/lorien>`_. We currently only provide prebuilt images for Ubuntu 18.04 on CPU and GPU platforms. The docker images include dependent packages and nightly built TVM. If you prefer other platforms or OS and is willing to make one, refer to :ref:`contribute` to file a pull request.

First of all, you need to set up `docker <https://docs.docker.com/engine/installation/>`_ (or `nvidia-docker <https://github.com/NVIDIA/nvidia-docker/>`_ if you want to use cuda). In the rest of this guide, we will focus on the CPU platform, but you should be able to get the GPU platform working with exactly the same steps with ``nvidia-docker``.

Let's first pull the docker image from Docker Hub. Note that you may need ``sudo`` for all docker comamnds in this guide if you do not configure your docker permission.

.. code-block:: bash

    docker pull comaniac0422/lorien:ubuntu-18.04-<the latest version>

Now the docker image is available on your machine. We then create a container and log in it using the provided script (run it in the Lorien root directory):

.. code-block:: bash

    cd docker; ./bash.sh comaniac0422/lorien:ubuntu-18.04-<the latest version>

You can get the script by cloning Lorien, or copy it from `bash.sh <https://github.com/comaniac/lorien/blob/master/docker/bash.sh>`_. The ``bash.sh`` script creates a container with the given docker image and execute a command. If no command is provided as shown in the above example, then it launches a bash shell for you to interact with. Note that your AWS credential on the host machine will not be available in a docker container, so you have to specify them in environment variables if you want Lorien to access AWS services (e.g., S3, DynamoDB, batch). ``bash.sh`` will set up an AWS credential in container according to the following environment variables.

.. code-block:: bash

    export AWS_REGION="us-west-1"
    export AWS_ACCESS_KEY="OOOOXXX"
    export AWS_SECRET_ACCESS_KEY="XXXYYY"
    cd docker; ./bash.sh comaniac0422/lorien:ubuntu-18.04-v0.01


After you login the container, you can clone and install Lorien (the prebuilt docker images do not have Lorien deployed as it is used for CI).

.. code-block:: bash

    git clone https://github.com/comaniac/lorien.git
    python3 setup.py install


That's all. You can now use Lorien! Try out some commands in Lorien README and have fun.

############
Docker Files
############
Check out `docker files <https://github.com/comaniac/lorien/tree/master/docker>`_ for Lorien docker files if you want to build your own docker images for your platform or newer dependencies.

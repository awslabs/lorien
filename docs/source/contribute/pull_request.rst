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

#####################
Submit a Pull Request
#####################

Assuming you have forked Lorien and made some nice features. Here is a guide to submit your changes as a pull request (PR).

- Add test-cases to cover the new features or bugfix the patch introduces.
- Document the code you wrote.
- Run unit tests locally and make sure your changes can pass all of them:

  .. code:: bash

    make unit_test

  If you only need to re-run a single test file:

  .. code:: bash

    python3 -m pytest tests/<test_file_name>

- Run code coverage test and make sure your changes will not decrease the code coverage rate:

  .. code:: bash

    make cov

  Note that since code coverage test has to run unit test as well, you can actually skip the previous step if you plan to get a coverage report.


- Rebase your branch on the head of master. Note that you may need to resolve conflicts during rebasing. Follow the prompt up git instructions to resolve them.

  .. code:: bash

    git remote add upstream git@github.com:comaniac/lorien.git
    git fetch upstream
    git rebase upstream/master your_branch
    git push origin -f

- Check the code format:

  .. code:: bash

    make check_format

  If you got any errors when checking the format, use the following command to auto-format the code:

  .. code:: bash

    make format

- Check lint. We use ``pylint`` to check if the coding style aligns to PEP8 standards. Run the following command for linting.

  .. code:: bash

    make lint

  You have to get a perfect score in order to pass the CI. If you believe that the ERROR/WARNING you got from linting does not make sense, you are also welcome to open an issue for discussion. Do NOT simply add ``pylint: disable`` to workaround it without reasons.

  ::

    Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)

- Run the following command to check type. We use ``mypy`` to check if the code has potential type errors.

  .. code:: bash

    make type

  You should see the following message (the number of source files may change over time).

  ::

    Success: no issues found in 29 source files

- Commit all changes you made during the above steps.
- Send the pull request and request code reviews from other contributors.

  - To get your code reviewed quickly, we encourage you to help review others' code so they can do the favor in return.
  - Code review is a shepherding process that helps to improve contributor's code quality.
    We should treat it proactively, to improve the code as much as possible before the review.
    We highly value patches that can get in without extensive reviews.
  - The detailed guidelines and summarizes useful lessons.

- The pull request can be merged after passing the CI and approved by at least one reviewer.


**************
CI Environment
**************
We use Github Action with the prebuilt docker image to run CI. You can find the prebuilt docker images at `Docker Hub <https://hub.docker.com/r/comaniac0422/lorien>`_ .

Since updating docker images may cause CI problems and need fixes to accommodate the new environment, here is the protocol to update the docker image for CI:

- Send PR to update build script in the repo and ask one of the code owners to perform the following steps.
- Build the new docker image: ``./build cpu``.
- Tag and publish the image with a new version: ``./publish.sh cpu comaniac0422/lorien:ubuntu-18.04-v<new_version>``.
- Update the version (most of the time increase the minor version) in ``./github/workflows/ubuntu-ci.yaml``, send a PR.
- The PR should now use the updated environment to run CI. Fix any issues wrt to the new image versions.
- Merge the PR and now we are in new version.

*************************
Code Coverage Enforcement
*************************
The CI will upload the code coverage report to codecov.io to keep track of code coverage changes. You can find a badge in the README showing the current code coverage of master branch. We enforce the code coverage to be higher than 90%.

After your PR has passed the CI, you should see a Codecov bot posts a comment in your PR for a code coverage change report. Your PR should guarantee that the code coverage does not drop before getting merged.


#!/usr/bin/env bash
#
# Start a bash for debugging docker image.
#
# Usage: docker/bash.sh <CONTAINER_NAME>
#     Starts an interactive session
#
# Usage2: docker/bash.sh <CONTAINER_NAME> [COMMAND]
#     Execute command in the docker image, non-interactive
#
if [ "$#" -lt 1 ]; then
    echo "Usage: docker/bash.sh <CONTAINER_NAME> [COMMAND]"
    exit 0
fi

# Setup AWS credentials.
# The default AWS access key can only be used to access
# local DynamoDB but not any other services (e.g., S3).
if [ ! ${AWS_REGION} ]; then
    AWS_REGION="us-west-2"
fi
if [ ! ${AWS_ACCESS_KEY_ID} ]; then
    AWS_ACCESS_KEY_ID="aaa"
fi
if [ ! ${AWS_SECRET_ACCESS_KEY} ]; then
    AWS_SECRET_ACCESS_KEY="bbb"
fi


DOCKER_IMAGE_NAME=("$1")

if [ "$#" -eq 1 ]; then
    COMMAND="bash"
    if [[ $(uname) == "Darwin" ]]; then
        # Docker's host networking driver isn't supported on macOS.
        # Use default bridge network and expose port for jupyter notebook.
        DOCKER_EXTRA_PARAMS=("-it -p 8888:8888")
    else
        DOCKER_EXTRA_PARAMS=("-it --net=host")
    fi
else
    shift 1
    COMMAND=("$@")
fi

# Use nvidia-docker if the container is GPU.
if [[ ! -z $CUDA_VISIBLE_DEVICES ]]; then
    CUDA_ENV="-e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    CUDA_ENV=""
fi

if [[ "${DOCKER_IMAGE_NAME}" == *"gpu"* ]]; then
    if ! type "nvidia-docker" 1> /dev/null 2> /dev/null
    then
        DOCKER_BINARY="docker"
        CUDA_ENV=" --gpus all "${CUDA_ENV}
    else
        DOCKER_BINARY="nvidia-docker"
    fi
else
    DOCKER_BINARY="docker"
fi

# Print arguments.
echo "Running '${COMMAND[@]}' inside ${DOCKER_IMAGE_NAME}..."

# By default we cleanup - remove the container once it finish running (--rm)
# and share the PID namespace (--pid=host) so the process inside does not have
# pid 1 and SIGKILL is propagated to the process inside.
${DOCKER_BINARY} run --rm --pid=host\
    -e "AWS_REGION=${AWS_REGION}" \
    -e "AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}" \
    -e "AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}" \
    ${CUDA_ENV}\
    ${DOCKER_EXTRA_PARAMS[@]} \
    ${DOCKER_IMAGE_NAME}\
    ${COMMAND[@]}


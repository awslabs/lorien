set -e

if [ "$#" -lt 2 ]; then
    echo "Usage ./build.sh <x86|arm|gpu> <add CI? 0|1>"
    exit 0
fi

PLATFORM=${1}
BUILD_ON_CI=${2}

# Build base
DOCKER_BUILDKIT=1 docker build -f Dockerfile.base -t lorien:${PLATFORM}-base-latest \
    --build-arg platform=${PLATFORM} .

# Build TVM on base
DOCKER_BUILDKIT=1 docker build -f Dockerfile.tvm -t lorien:${PLATFORM}-tvm-latest \
    --build-arg platform=${PLATFORM} .

# Determine the base image for Lorien to deploy on
if [ $BUILD_ON_CI -eq 1 ]; then
    # Build CI on TVM
    DOCKER_BUILDKIT=1 docker build -f Dockerfile.ci -t lorien:${PLATFORM}-ci-latest \
        --build-arg platform=${PLATFORM} .

    LORIEN_BASE=lorien:${PLATFORM}-ci-latest
else
    LORIEN_BASE=lorien:${PLATFORM}-tvm-latest
fi

# Deploy Lorien
DOCKER_BUILDKIT=1 docker build -f Dockerfile.lorien -t lorien:${PLATFORM}-latest \
    --build-arg base=${LORIEN_BASE} .


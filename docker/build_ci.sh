set -e

if [ "$#" -lt 1 ]; then
    echo "Usage ./build.sh <x86|arm|gpu>"
    exit 0
fi

PLATFORM=${1}

# Build base
DOCKER_BUILDKIT=1 docker build -f Dockerfile.base -t lorien:${PLATFORM}-base-latest \
    --build-arg platform=${PLATFORM} .

# Build TVM on base
DOCKER_BUILDKIT=1 docker build -f Dockerfile.tvm -t lorien:${PLATFORM}-tvm-latest \
    --build-arg platform=${PLATFORM} .

# Build CI on TVM
DOCKER_BUILDKIT=1 docker build -f Dockerfile.ci -t lorien:${PLATFORM}-ci-latest \
    --build-arg platform=${PLATFORM} .


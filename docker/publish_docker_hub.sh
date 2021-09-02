# Note that you have to `docker login` first to authorize the write permission.

if [ "$#" -lt 1 ]; then
    echo "Usage ./build.sh <PLATFORM x86|gpu>"
    exit 0
fi

IMAGE_VERSION="v0.06"
PLATFORM=${1}

if [ $PLATFORM = "x86" ]; then
    TARGET_REPO=comaniac0422/lorien:ubuntu-18.04-${IMAGE_VERSION}
else
    TARGET_REPO=comaniac0422/lorien:ubuntu-18.04-cuda-${IMAGE_VERSION}
fi

echo "Uploading image to ${TARGET_REPO}"
docker tag lorien:${PLATFORM}-ci-latest ${TARGET_REPO}
docker push ${TARGET_REPO}



# Repo path could be either AWS ECR repo or Docker hub repo.
# Note that you have to `docker login` first to authorize the write permission.
# Example: ./publish.sh lorien:cpu-latest <aws-account>.dkr.ecr.<region>.amazonaws.com/<ecr repo>/cpu-latest

if [ "$#" -lt 2 ]; then
    echo "Usage ./build.sh <IMAGE TAG> <REPO PATH>"
    exit 0
fi

IMAGE_TAG=${1}
TARGET_REPO=${2}

echo "Uploading ${IMAGE_TAG} to ${TARGET_REPO}"
docker tag ${IMAGE_TAG} ${TARGET_REPO}
docker push ${TARGET_REPO}


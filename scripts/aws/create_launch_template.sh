# The purpose of creating an AWS batch launch template is to configure
# the instance storage and docker image size limit (default 10G).
# See https://aws.amazon.com/premiumsupport/knowledge-center/batch-job-failure-disk-space/

REGION=us-west-2

python3 gen_launch_template.py
aws ec2 --region ${REGION} create-launch-template --cli-input-json file://aws_batch_launch_template.json
rm aws_batch_launch_template.json

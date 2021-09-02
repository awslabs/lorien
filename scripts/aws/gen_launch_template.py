import base64
import json

template = {
    "LaunchTemplateName": "batch-template-for-lorien",
    "LaunchTemplateData": {
        "BlockDeviceMappings": [
            {
                "Ebs": {"DeleteOnTermination": True, "VolumeSize": 100, "VolumeType": "gp2"},
                "DeviceName": "/dev/xvda",
            },
            {
                "Ebs": {"DeleteOnTermination": True, "VolumeSize": 100, "VolumeType": "gp2"},
                "DeviceName": "/dev/xvdcz",
            },
        ],
    },
}

# Be careful about the first empty line.
user_data = """Content-Type: multipart/mixed; boundary="==BOUNDARY==" 
MIME-Version: 1.0

--==BOUNDARY==
Content-Type: text/cloud-boothook; charset="us-ascii"
#cloud-boothook
#!/bin/bash
cloud-init-per once docker_options echo 'OPTIONS="${OPTIONS} --storage-opt dm.basesize=40G"' >> /etc/sysconfig/docker

--==BOUNDARY== 
"""
user_data_bytes = user_data.encode("ascii")

template["LaunchTemplateData"]["UserData"] = base64.b64encode(user_data_bytes).decode("ascii")
with open("aws_batch_launch_template.json", "w") as fp:
    json.dump(template, fp, indent=2)

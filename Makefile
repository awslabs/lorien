PRJ_NAME=lorien
PORT=18871

env:
	virtualenv -p python3 venv --system-site-packages
	venv/bin/pip3 install -r requirements.txt

lint:
	python3 -m pylint ${PRJ_NAME} --rcfile=tests/lint/pylintrc

type:
	# intall-types is the new feature since mypy 0.900 that installs missing stubs.
	python3 -m mypy ${PRJ_NAME} --ignore-missing-imports --install-types --non-interactive

format:
	black -l 100 `git diff --name-only --diff-filter=ACMRTUX origin/master -- "*.py" "*.pyi"`

check_format:
	black -l 100 --check `git diff --name-only --diff-filter=ACMRTUX origin/master -- "*.py" "*.pyi"`

local_db:
	mkdir $@
	cd $@; curl -O https://s3-us-west-2.amazonaws.com/dynamodb-local/dynamodb_local_latest.zip
	cd $@; unzip dynamodb_local_latest.zip; rm dynamodb_local_latest.zip

launch_local_db: local_db
	java -Djava.library.path=./local_db/DynamoDBLocal_lib \
		 -jar ./local_db/DynamoDBLocal.jar \
		 -sharedDb -port 10020

launch_rpc_server:
	# OBJC_DISABLE_INITIALIZE_FORK_SAFETY is a workaround to a MacOS 10.13 (High Sierra) issue with Python.
	# See http://sealiesoftware.com/blog/archive/2017/6/5/Objective-C_and_fork_in_macOS_1013.html.
	OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 -m lorien rpc-server --port ${PORT}

unit_test:
	python3 -m pytest --lf

cov:
	python3 -m pytest tests --cov-config=tests/lint/coveragerc --cov=${PRJ_NAME} --cov-report term

doc:
	make -C docs html

clean:
	rm -rf .coverage* *.xml *.log *.pyc *.egg-info tests/temp* test_* tests/*.pdf curr *.db
	find . -name "__pycache__" -type d -exec rm -r {} +
	find . -name ".pytest_cache" -type d -exec rm -r {} +
	find . -name ".pkl_memoize_py3" -type d -exec rm -r {} +


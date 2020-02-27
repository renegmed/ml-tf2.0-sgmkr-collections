create:
	virtualenv -p python3 ~/.virtualenvs/jsimon-mnist/
.PHONY: create

source:
	$(source) ~/.virtualenvs/jsimon-mnist/bin/activate
.PHONY: activate

install:
	pip install tensorflow-gpu==2.0.0 tensorflow-serving-api==2.0.0
	pip install sagemaker smdebug smdebug-rulesconfig==0.1.2 awscli boto3 keras numpy pandas --upgrade
.PHONY: install

exit:
	deactivate
.PHONY: exit


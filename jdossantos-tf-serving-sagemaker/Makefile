create:
	virtualenv -p python3 ~/.virtualenvs/jdossantos-tf-serving-sagemaker/
.PHONY: create

source:
	$(source) ~/.virtualenvs/jdossantos-tf-serving-sagemaker/bin/activate
.PHONY: activate

install:
	pip install tensorflow-gpu==2.1.0 tensorflow-serving-api==2.0.0
	pip install sagemaker smdebug smdebug-rulesconfig==0.1.2 awscli boto3 keras numpy pandas --upgrade
.PHONY: install

exit:
	deactivate
.PHONY: exit

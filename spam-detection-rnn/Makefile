create:
	virtualenv -p python3 ~/.virtualenvs/spam-detection-rnn/
.PHONY: create

source:
	$(source) ~/.virtualenvs/spam-detection-rnn/bin/activate
.PHONY: activate

install:
	pip install tensorflow-gpu==2.0.0 tensorflow-serving-api==2.0.0
	pip install sagemaker smdebug smdebug-rulesconfig==0.1.2 awscli boto3 keras numpy pandas sklearn --upgrade
	pip install jupyter
.PHONY: install

exit:
	deactivate
.PHONY: exit

jupyter:
	jupyter notebook --allow-root
.PHONY: jupyter



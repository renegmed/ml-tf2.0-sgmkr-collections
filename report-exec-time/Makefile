create:
	virtualenv -p python3 ~/.virtualenvs/report-exec-time/
.PHONY: create

source:
	$(source) ~/.virtualenvs/report-exec-time/bin/activate
.PHONY: activate

install:
	pip install tensorflow-gpu==2.0.0 tensorflow-serving-api==2.0.0
	pip install sagemaker smdebug smdebug-rulesconfig==0.1.2 awscli boto3 keras numpy pandas sklearn --upgrade
	pip install matplotlib --upgrade
	pip install jupyter --upgrade 
.PHONY: install

exit:
	deactivate
.PHONY: exit

jupyter:
	jupyter notebook --allow-root
.PHONY: jupyter



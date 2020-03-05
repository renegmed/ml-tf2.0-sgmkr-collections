create:
	virtualenv -p python3 ~/.virtualenvs/mnist-notebook-serving/
.PHONY: create

source:
	$(source) ~/.virtualenvs/mnist-notebook-serving/bin/activate
.PHONY: activate

install:
	pip install tensorflow-gpu==2.0.0 tensorflow-serving-api==2.0.0
	pip install sagemaker smdebug smdebug-rulesconfig==0.1.2 awscli keras numpy pandas sklearn matplotlib PILLOW --upgrade
	pip install jupyter
.PHONY: install

exit:
	deactivate
.PHONY: exit

jupyter:
	jupyter notebook --allow-root
.PHONY: jupyter



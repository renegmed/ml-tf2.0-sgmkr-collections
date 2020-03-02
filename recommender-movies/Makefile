create:
	virtualenv -p python3 ~/.virtualenvs/recommender-movies/
.PHONY: create

source:
	$(source) ~/.virtualenvs/recommender-movies/bin/activate
.PHONY: activate

install:
	pip install tensorflow-gpu==2.0.0 tensorflow-serving-api==2.0.0
	pip install sagemaker smdebug smdebug-rulesconfig==0.1.2 awscli boto3 keras numpy pandas --upgrade
	pip install pip install jupyter
.PHONY: install

jupyter:
	jupyter notebook --allow-root
.PHONY: jupyter

exit:
	deactivate
.PHONY: exit


 

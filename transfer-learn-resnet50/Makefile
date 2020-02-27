create:
	virtualenv -p python3 ~/.virtualenvs/transfer-learn-resnet50/
.PHONY: create

source:
	$(source) ~/.virtualenvs/transfer-learn-resnet50/bin/activate
.PHONY: activate

install:
	pip install tensorflow-gpu==2.0.0 tensorflow-serving-api==2.0.0
	pip install sagemaker smdebug smdebug-rulesconfig==0.1.2 awscli boto3 keras numpy pandas matplotlib Pillow --upgrade
.PHONY: install

exit:
	deactivate
.PHONY: exit


generate-data:
	python utils/generate_cifar10_tfrecords.py --data-dir=/tmp/cifar-10-data
.PHONY:generate-data


 

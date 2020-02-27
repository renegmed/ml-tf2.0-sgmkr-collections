copy-data:
	rm -rf test_dir/input/data
	mkdir -p test_dir/input/data
	aws s3 cp s3://sagemaker-us-east-1-731833107751/transfer-learning-data/train test_dir/input/data/training --recursive
.PHONY: copy-data

local-train:
	sh train_local.sh transfer-learn-resnet50-train:latest
.PHONY: local-train


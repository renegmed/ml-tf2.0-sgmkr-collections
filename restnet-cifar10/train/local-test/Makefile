copy-data:
	rm -rf test_dir/input/data
	mkdir -p test_dir/input/data
	aws s3 cp s3://sagemaker-us-east-1-731833107751/resnet-cirfar10/data/training --recursive
.PHONY: copy-data

local-train:
	sh train_local.sh resnet-cifar10-train:latest
.PHONY: local-train

copy-model:
	rm -rf model
	cp -rf ../train/local-test/test_dir/model/ .
.PHONY: copy-model

build:
	docker build -t transfer-learn-resnet50-serving .
.PHONY: build





local-run:
	docker run --rm -p 8080:8080 transfer-learn-resnet50-serving
.PHONY: local-run


local-query:
	curl -X POST http://localhost:8080/invocations -d '{"instances": ""}'
.PHONY: local-query



push:
	sh push.sh      
.PHONY: push

create-model:
	sh sagemaker_model.sh
.PHONY: sagemaker_model

create-endpoint:
	sh sagemaker_endpoint.sh
.PHONY: sagemaker_endpoint

endpoint-status:
	aws sagemaker describe-endpoint --endpoint-name fashion-mnist-v1
.PHONY: endpoint-status

predict1:
	sh predict.sh
.PHONY:predict1


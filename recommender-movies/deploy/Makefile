copy-model:
	rm -rf model
	cp -rf ../train/local-test/test_dir/model/ .
.PHONY: copy-model

build:
	docker build -t recommender-movies-serving .
.PHONY: build




local-run:
	docker run --rm -p 8080:8080 recommender-movies-serving
.PHONY: local-run


local-query:
	  curl -X POST http://localhost:8080/invocations -d '{"instances": [["Buy the best shoes in the market for 30.00 US Dollars."]]}'
.PHONY: local-query 


query-spam:
	curl -X POST http://localhost:8080/invocations -d @spam_0000.json
.PHONY: query-spam


query-spam-2:
	curl -X POST http://localhost:8080/invocations -d @spam_1003.json
.PHONY: query-spam-2


query-ham:
	curl -X POST http://localhost:8080/invocations -d @ham_0002.json
.PHONY: query-ham


tail:
	docker logs --tail reverent_elbakyan
.PHONY: tail

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

predict:
	sh predict.sh
.PHONY:predict



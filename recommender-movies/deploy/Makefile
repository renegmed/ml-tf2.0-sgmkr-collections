copy-model:
	rm -rf model
	cp -rf ../train/local-test/test_dir/model/ .
.PHONY: copy-model

build:
	#docker stop $(docker ps -a -q)
    #docker rm $(docker ps -a -q)
	#dsr
	docker build -t recommender-movies-serving .
.PHONY: build




local-run:
	docker run --rm -p 8080:8080 recommender-movies-serving
.PHONY: local-run


local-query:
	  curl -X POST http://localhost:8080/invocations -d '{"signature_name": "serving_default", "instances": [[99878], [3593]]}'
.PHONY: local-query 



local-status:
	  curl -X POST http://localhost:8080/invocations 
.PHONY: local-status 


local-ping:
	curl -X POST http://localhost:8080/ping
.PHONY: local-ping

  
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



copy-model:
	rm -rf model
	cp -rf ../train/local-test/test_dir/model/ .
.PHONY: copy-model

build:
	docker build -t report-exec-time-serving .
.PHONY: build




local-run:
	docker run --rm -p 8080:8080 report-exec-time-serving
.PHONY: local-run


local-ping:
	curl -X POST http://localhost:8080/ping
.PHONY: local-ping


local-metadata:
	curl -X GET http://localhost:8080/metadata
.PHONY: local-metadata


query:
	#curl -X POST http://localhost:8080/invocations -d @spam_0000.json
	curl -d '{"instances": [[ 3.19179609,  2.05277296, -0.51536518, -0.4880486,  -0.50239337, -0.50629114, -0.74968743, -0.68702182,  1.45992522]]}' \
     		-X POST http://localhost:8080/invocations
.PHONY: query





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
	aws sagemaker describe-endpoint --endpoint-name report-exec-time-v1
.PHONY: endpoint-status

predict1:
	sh predict.sh
.PHONY:predict1



run:
	docker run -d --name serving_base tensorflow/serving
.PHONY: run

copy:
	#docker cp model_report_exec_time serving_base:/models/report_exec_time
	docker cp model serving_base:/models/report_exec_time
.PHONY: copy

commit:
	docker commit --change "ENV MODEL_NAME report_exec_time" serving_base katanaml/core-serving:v19.8
.PHONY: commit

serve:
	docker run -d -p 8500:8500 -p 8501:8501 --name katana-ml-serving katanaml/core-serving:v19.8
.PHONY: serve

logs:
	docker logs -f katana-ml-serving
.PHONY: logs


logs:
	docker logs -f katana-ml-serving
.PHONY: logs

status:
	curl http://localhost:8501/v1/models/report_exec_time
.PHONY: status

post:
	curl -d '{"instances": [[ 3.19179609,  2.05277296, -0.51536518, -0.4880486,  -0.50239337, -0.50629114, -0.74968743, -0.68702182,  1.45992522]]}' \
     -X POST http://localhost:8501/v1/models/report_exec_time:predict
.PHONY: post



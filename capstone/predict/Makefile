predict-test:
	# Calling predict function from terminal. MLINPROD_MODEL_PATH at the beginning sets the enviromental variable
	MLINPROD_MODEL_PATH="../train/train/tests/models/" python predict/run.py --text="ruby on rails: how to change BG color of options in select list, ruby-on-rails"

flask-server-local:
	# starts flaks server
	MLINPROD_MODEL_PATH="../train/train/tests/models/" python predict/app.py

request:
	# make a request to flask server (needs to be running, for example having executes flask-server-local)
	curl -v http://0.0.0.0:5000/tag/predict/ -d text="ruby on rails: how to change BG color of options in select list, ruby-on-rails"

flask-server:
	docker run -p 5000:5000 predict

flask-server-debug:
	docker run -it -p 5000:5000 -v "$PWD":/libraries/app predict



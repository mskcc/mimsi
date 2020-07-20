install:
	pip install .

test: 
	cd tests; \
	nosetests;


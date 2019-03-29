install:
	pip --no-cache-dir install -r requirements.txt --quiet

clean:
	rm -f tests/microsatellites.list

extract-ms-list:
	gunzip < utils/microsatellites.list.gz > tests/microsatellites.list

test: extract-ms-list
	export PYTHONPATH=$$PYTHONPATH:$(shell pwd):$(shell pwd)/data:$(shell pwd)/model ; \
	cd tests; \
	nosetests; \
	rm -f microsatellites.list


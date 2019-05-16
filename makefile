install:
	pip install .

clean:
	rm -f tests/microsatellites.list

extract-ms-list:
	gunzip < utils/microsatellites.list.gz > tests/microsatellites.list

test: extract-ms-list
	cd tests; \
	nosetests; \
	rm -f microsatellites.list


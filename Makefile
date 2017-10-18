.PHONY: default
default:
	$(MAKE) build
	$(MAKE) check

.PHONY: build
build:
	$(MAKE) -C core
	sbt -batch assembly

.PHONY:
check:
	spark-submit --driver-memory 2g test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar 2>&1 | tee test.log
	@echo test complete

.PHONY: clean
clean:
	$(MAKE) -C core clean
	sbt -batch clean

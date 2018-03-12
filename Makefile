.PHONY: default
default:
	$(MAKE) build
	#$(MAKE) check # out of date

.PHONY: build
build:
	$(MAKE) -C core
	sbt -batch assembly

.PHONY:
check:
	spark-submit --driver-memory 5g --executor-memory 5g --num-executors 2 test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar 2>&1 | tee test.log
	@echo test complete

.PHONY: clean
clean:
	$(MAKE) -C core clean
	sbt -batch clean

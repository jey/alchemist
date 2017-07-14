* need a plugin architecture desperately for code maintenance reasons
* need a unified logging interface desperately for profiling and debugging (cf. spdlog w/ custom sink)
* need to separate the tests and have more coverage
* need a testing framework that lets you debug the CPP code w/o having to wait on the scala code to compile and run, and avoids mixing the output with scala messages

* add ability to store/receive/send local matrices from driver
* allow directly specifying how matrices should be laid out when passing them in to Alchemist
* allow replacing an Alchemist matrix w/ a relaid out version (e.g. if you're going to do a bunch of operations that require row layout, best to do it in one go)

* assume the Spark matrix is materialized (should be), and use the locality info to set up the layout matrix
* currently packing up each row and sending separately to Alchemist clients, would be better to send all at once to each client?

* need to extend truncatedSVD, currently assumes matrix is tall and skinny
* need to fix kMeans on Spark and Alchemist side to use all the options passed in
* need to fix kMeans on Alchemist side to use the seed appropriately on driver and workers
* need to go through and uniformatize the datatype used for row/col indices (e.g. am sure in kmeans code, am using uint32_t for sampling row indices, but the rows can be longs)

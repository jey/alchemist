* need a plugin architecture desperately for code maintenance reasons
* need a unified logging interface desperately for profiling and debugging
* need to separate the tests and have more coverage

* add ability to store/receive/send local matrices from driver
* allow directly specifying how matrices should be laid out when passing them in to Alchemist
* allow replacing an Alchemist matrix w/ a relaid out version (e.g. if you're going to do a bunch of operations that require row layout, best to do it in one go)

* assume the Spark matrix is materialized (should be), and use the locality info to set up the layout matrix
* currently packing up each row and sending separately to Alchemist clients, would be better to send all at once to each client?

* need to extend truncatedSVD, currently assumes matrix is tall and skinny
* need to fix kMeans on Spark and Alchemist side to use all the options passed in
* need to fix kMeans on Alchemist side to use the seed appropriately on driver and workers
* need to fix kMeans on Alchemist side to use a non-trivial default seed

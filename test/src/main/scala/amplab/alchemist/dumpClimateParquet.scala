package amplab.alchemist
// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.sql.Row
// spark-sql
import org.apache.spark.sql.SparkSession

object dumpClimateToParquet {
    def main(args: Array[String]): Unit = {
        val spark = (SparkSession
                      .builder()
                      .appName("Dump Climate to Parquet")
                      .getOrCreate())
        val sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        import spark.implicits._
        
        val al = new Alchemist(sc)
        val varname = "/rows"
        val infname = "/global/cscratch1/sd/gittens/large-datasets/ocean.h5"
        val outfname = "/global/cscratch1/sd/gittens/large-datasets/smallOcean.parquet"
        val alClimateMat = al.readHDF5(infname, varname)
        println(" ")
        println("FINISHED LOADING ON ALCHEMIST, NOW RETRIEVING")
        println(" ")
        val A = alClimateMat.getIndexedRowMatrix()
        A.rows.map( x => (x.index, x.vector)).toDF("index", "vector").write.parquet(outfname)
        println(" ")
        println("FINISHED WRITING TO PARQUET")
        println(" ")

        al.stop
        sc.stop
    }
}

package utils

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
// spark-sql
import org.apache.spark.sql.SparkSession
// mllib
import org.apache.spark.mllib.linalg.{Vector, Vectors, Matrix, Matrices}
// others
import scala.math


object Utils {
    /**
     * Load data from libsvm format file.
     *
     * @param spark SparkSession
     * @param filename a string of file name
     * @param numSplits if it is specificed, spark will change the number of partitions
     * @param isCoalesce if true, use coalesce(); otherwise use repartition()
     */
    def loadLibsvmData(spark: SparkSession, filename: String, numSplits: Int = -1, isCoalesce: Boolean = true): RDD[(Double, Array[Double])] = {
        // Loads data
        var rawdata = spark.read.format("libsvm")
                                .load(filename)
                                .rdd
        
        if (numSplits > 0) {
            // coalesce() avoids a full shuffle, so it is usually faster than repartition().
            // However, coalesce() may not return the desired number of partitions;
            // if the data is small, the resulting #partition can be smaller than numSplits.
            if (isCoalesce) {
                var t1 = System.nanoTime()
                rawdata = rawdata.coalesce(numSplits)
                var t2 = System.nanoTime()
                var tdiff = (t2 - t1) * 1E-9
                println("Time cost of coalesce is " + tdiff.toString)
            }
            // repartition() is slower, but it is guaranteed to return exactly numSplits partitions.
            else {
                var t1 = System.nanoTime()
                rawdata = rawdata.repartition(numSplits)
                var t2 = System.nanoTime()
                var tdiff = (t2 - t1) * 1E-9
                println("Time cost of repartition is " + tdiff.toString)
            }
            // note: coalesce can result in data being sent over the network. avoid this for large datasets
            
        }
        
        val labelVectorRdd: RDD[(Double, Array[Double])] = rawdata
                .map(pair => (pair(0).toString.toDouble, Vectors.parse(pair(1).toString).toArray))
                .persist()
        
        labelVectorRdd
    }
    
    def meanAndMax(labelVectorRdd: RDD[(Double, Array[Double])]): (Double, Array[Double]) = {
        val maxFeatures: Array[Double] = labelVectorRdd.map(pair => pair._2.map(math.abs))
                                .reduce((a, b) => (a zip b).map(pair => math.max(pair._1, pair._2)) )
        val meanLabel: Double = labelVectorRdd.map(pair => pair._1)
                                .mean
        (meanLabel, maxFeatures)
    }
    
    def normalize(sc: SparkContext, labelVectorRdd: RDD[(Double, Array[Double])], meanLabel: Double, maxFeatures: Array[Double]): RDD[(Double, Array[Double])] = {
        val maxFeaturesBc = sc.broadcast(maxFeatures)
        val meanLabelBc = sc.broadcast(meanLabel)
        
        val normalizedLabelVectorRdd: RDD[(Double, Array[Double])] = labelVectorRdd
            .map(pair => (pair._1-meanLabelBc.value, (pair._2 zip maxFeaturesBc.value).map(a => a._1 / a._2)))
        
        normalizedLabelVectorRdd
    }
    
    
    def parseLibsvm(str: String, d: Int): (Double, Vector) = {
        val strArray: Array[String] = str.split(" ")
        val label: Double = strArray(0).toDouble
        val elements: Array[(Int, Double)] = strArray.drop(1)
                                                .map(s => s.split(":"))
                                                .map(pair => (pair(0).toDouble.toInt, pair(1).toDouble))
        val feature: Vector = Vectors.sparse(d, elements)
        (label, feature)
    }
}
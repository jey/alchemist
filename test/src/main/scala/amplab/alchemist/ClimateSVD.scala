package org.apache.spark.mllib.linalg.distributed
// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
// spark-sql
import org.apache.spark.sql.SparkSession
// spark-mllib
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors, Matrix, Matrices}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.linalg.SingularValueDecomposition
//breeze
import breeze.linalg.{DenseVector => BDV, max, min, DenseMatrix => BDM, norm, diag, svd}
import breeze.numerics._
// others
import scala.math
import java.io._

//import org.nersc.io._
import amplab.alchemist.{Alchemist, AlMatrix}

object ClimateSVD {
  def ticks(): Long = {
    return System.currentTimeMillis();
  }

    def main(args: Array[String]): Unit = {
        // Parse parameters from command line arguments
        val k: Int = args(0).toInt 
        val fname: String = args(1).toString
        val useAlc: Int = args(2).toInt

        // Print Info
        println("Settings: ")
        println("Target dimension = " + k.toString)
        println("Input data file = " + fname)
        println(" ")

        // Launch Spark
        var t1 = System.nanoTime()
        val spark = (SparkSession
                      .builder()
                      .appName("Alchemist Climate SVD Test")
                      .getOrCreate())
        val sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        var t2 = System.nanoTime()
        println("Time cost of starting Spark session: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        if (useAlc == 1) {
            // Launch Alchemist
            t1 = System.nanoTime()
            val al = new Alchemist(sc)
            t2 = System.nanoTime()
            println("Time cost of starting Alchemist session: ")
            println((t2 - t1) * 1.0E-9)
            println(" ")

            // Load Data from File 
            val loadAlc: Int = args(3).toInt
            var alMatA : AlMatrix = null

            if (loadAlc == 1) {
                val varname = args(4)
                var loadStart = ticks()
                alMatA = al.readHDF5(fname, varname)
                var loadEnd = ticks()
                println("Time cost for loading dataset in Alchemist")
                println((loadEnd - loadStart)/1000.0)
                println(" ")
            } else {
                var loadStart = ticks()
                val df = spark.read.parquet(fname)
                val indexedRows = df.rdd.map(pair => new IndexedRow(pair(0).asInstanceOf[Long], pair(1).asInstanceOf[DenseVector]))
                val rdd = new IndexedRowMatrix(indexedRows)
                rdd.rows.cache
                rdd.rows.count
                var loadEnd = ticks()
                println("Time cost for creating and caching dataset for SVD test")
                println((loadEnd - loadStart)/1000.0)
                println(" ")

                var txStart = ticks()
                alMatA = AlMatrix(al, rdd)
                var txEnd = ticks()
                println("Cost of sending dataset for SVD test:")
                println((txEnd - txStart)/1000.0)
                println(" ")
            }

            var computeStart = ticks()
            val (alU, alS, alV) = al.truncatedSVD(alMatA, k) // returns sing vals in increas
            var computeEnd = ticks()

            var rcStart = ticks()
            val alUreturned = alU.getIndexedRowMatrix()
            val alSreturned = alS.getIndexedRowMatrix()
            val alVreturned = alV.getIndexedRowMatrix()
            var rcEnd = ticks()
            System.err.println(s"Alchemist timing: svd=${(computeEnd-computeStart)/1000.0}, receive=${(rcEnd - rcStart)/1000.0}")

            // TODO: report approximation error on Alchemist side 

            al.stop
        }
        else {
            // Load Data from File 
            var loadStart = ticks()
            val df = spark.read.parquet(fname)
            val indexedRows = df.rdd.map(pair => new IndexedRow(pair(0).asInstanceOf[Long], pair(1).asInstanceOf[DenseVector]))
            val rdd = new IndexedRowMatrix(indexedRows)
            rdd.rows.cache
            rdd.rows.count
            var loadEnd = ticks()
            println("Time cost for creating and caching dataset for SVD test")
            println((loadEnd - loadStart)/1000.0)
            println(" ")

            // Compute SVD using Spark
            var computeStart = ticks()
            val svd = rdd.toRowMatrix().computeSVD(k, computeU = true, 1e-9, 300, 1e-10, "auto") // default Spark computeSVD arguments
            svd.U.rows.count()
            var computeEnd = ticks()
            System.err.println(s"Spark timing: svd= ${(computeEnd-computeStart)/1000.0}")
        }

        spark.stop
    }
    
}

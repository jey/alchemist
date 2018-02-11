package amplab.alchemist
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

import org.nersc.io._

object ClimateSVD {
    def main(args: Array[String]): Unit = {
        // Parse parameters from command line arguments
        val k: Int = if(args.length > 0) args(0).toInt else 2
        val fname: String = if(args.length > 1) args(1).toString else "Error!"
        val varname: String = args(2)
        // Print Info
        println("Settings: ")
        println("Target dimension = " + k.toString)
        println("Input data file = " + fname)
        println("Varname = " + varname)
        println(" ")
        
        //// Test Spark
        println("\n#======== Testing Spark ========#")
        //testSpark(k, infile, partitions)

        
        //// Test Alchemist
        println("\n#======== Testing Alchemist ========#")
        testAlchemist(k, fname, varname)

    }
    
    def testSpark(k: Int, infile: String, partitions : Int): Unit = {  
        //// Launch Spark
        var t1 = System.nanoTime()
        val spark = (SparkSession
                      .builder()
                      .appName("Alchemist Test")
                      .getOrCreate())
        val sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        var t2 = System.nanoTime()
        println("Time cost of starting Spark session: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        //// Load Data from File
        println("Starting to load from file ")
        val rdd = read.h5read_irow(sc, infile, "rows", partitions=partitions)
        val labelVecRdd : RDD[(Int, Vector)] = rdd.map(ivec => (ivec.index.toInt, ivec.vector))
        println("Done loading")
        
        //// Compute the Squared Frobenius Norm
        val sqFroNorm: Double = labelVecRdd.map(pair => Vectors.norm(pair._2, 2))
                                        .map(norm => norm * norm)
                                        .reduce((a, b) => a + b)
        
        //// Spark Build-in Truncated SVD
        t1 = System.nanoTime()
        val mat: RowMatrix = new RowMatrix(labelVecRdd.map(pair => pair._2))
        val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(k, computeU=false)
        val v: Matrix = svd.V
        t2 = System.nanoTime()
        println("Time cost of Spark truncated SVD: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        //// Compute Approximation Error
        val vBroadcast = sc.broadcast(v)
        val err: Double = labelVecRdd
                .map(pair => (pair._2, vBroadcast.value.transpose.multiply(pair._2)))
                .map(pair => (pair._1, Vectors.dense(vBroadcast.value.multiply(pair._2).toArray)))
                .map(pair => Vectors.sqdist(pair._1, pair._2))
                .reduce((a, b) => a + b)
        val relativeError = err / sqFroNorm
        println("Squared Frobenius error of rank " + k.toString + " SVD is " + err.toString)
        println("Squared Frobenius norm of A is " + sqFroNorm.toString)
        println("Relative Error is " + relativeError.toString)
        
        spark.stop
    }
    
    def testAlchemist(k: Int, infile: String, varname: String): Unit = {  
        //// Launch Spark
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
        
        //// Launch Alchemist
        t1 = System.nanoTime()
        val al = new Alchemist(sc)
        t2 = System.nanoTime()
        println("Time cost of starting Alchemist session: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")

        //// Load Data from File and compute truncated SVD
        val alClimateMat = al.readHDF5(infile, varname);
        t1 = System.nanoTime()
        val (alU, alS, alV) = al.truncatedSVD(alClimateMat, k)
        t2 = System.nanoTime()
        println("Time cost of Alchemist truncated SVD: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        
        //// Alchemist Matrix to Local Matrix
        t1 = System.nanoTime()
        val sparkV = alV.getIndexedRowMatrix()
        t2 = System.nanoTime()
        val txtime1 = t2 - t1
        t1 = System.nanoTime()
        val sparkU = alU.getIndexedRowMatrix()
        t2 = System.nanoTime()
        val txtime2 = t2 - t1
        t1 = System.nanoTime()
        val sparkS = alS.getIndexedRowMatrix()
        t2 = System.nanoTime()
        val txtime3 = t2 - t1
        println("Time cost of getting SVD factors to Spark from Alchemist: ")
        println((txtime1 + txtime2 + txtime3) * 1.0E-9)
        println(" ")
        
        //// TODO: report approximation error on Alchemist side 
        al.stop
        spark.stop
    }
    
    def splitLabelVec(labelVecRdd: RDD[(Int, Vector)]): (Array[Int], IndexedRowMatrix) = {
        //// Convert Data to Indexed Vectors and Labels
        val indexLabelVecRdd = labelVecRdd.zipWithIndex.persist()
        val sortedLabels = indexLabelVecRdd.map(pair => (pair._2, pair._1._1))
                                        .collect
                                        .sortWith(_._1 < _._1)
                                        .map(pair => pair._2)
                                        
        //sortedLabels.take(20).foreach(println)

        val indexRows = indexLabelVecRdd.map(pair => new IndexedRow(pair._2, new DenseVector(pair._1._2.toArray)))
        //print(indexRows.take(10).map(pair => (pair._1, pair._2.mkString(", "))).mkString(";\n"))
        val indexedMat = new IndexedRowMatrix(indexRows)
        
        (sortedLabels, indexedMat)
    }
    
}

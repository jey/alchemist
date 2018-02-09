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
        val infile: String = if(args.length > 1) args(1).toString else "Error!"
        val partitions: Int = args(2).toInt
        // Print Info
        println("Settings: ")
        println("Target dimension = " + k.toString)
        println("Input data file = " + infile)
        println(" ")
        
        //// Test Spark
        println("\n#======== Testing Spark ========#")
        testSpark(k, infile, partitions)

        
        //// Test Alchemist
        println("\n#======== Testing Alchemist ========#")
        val outfile2 = System.getenv("OUTPUT_FILE") + "_alchemist.txt"
        testAlchemist(k, infile, partitions)

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
    
    def testAlchemist(k: Int, infile: String, partitions: Int): Unit = {  
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
        val rdd = read.h5read_imat(sc, infile, "rows", partitions=partitions)
        val labelVecRdd : RDD[(Int, Vector)] = rdd.rows.map(ivec => (ivec.index.toInt, ivec.vector))
        
        //// Compute the Squared Frobenius Norm
        val sqFroNorm: Double = labelVecRdd.map(pair => Vectors.norm(pair._2, 2))
                                        .map(norm => norm * norm)
                                        .reduce((a, b) => a + b)
        
        //// Launch Alchemist
        t1 = System.nanoTime()
        val al = new Alchemist(sc)
        t2 = System.nanoTime()
        println("Time cost of starting Alchemist session: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
             
        //// Convert Data to Indexed Vectors and Labels
        t1 = System.nanoTime()
        val (sortedLabels, indexedMat) = splitLabelVec(labelVecRdd)
        t2 = System.nanoTime()
        println("Time cost of creating indexed vectors and labels: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        //// Convert Spark IndexedRowMatrix to Alchemist AlMatrix
        t1 = System.nanoTime()
        val alMat = AlMatrix(al, indexedMat)
        t2 = System.nanoTime()
        println("Time cost of converting Spark matrix to Alchemist matrix: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        //// Alchemist Truncated SVD
        t1 = System.nanoTime()
        val (alU, alS, alV) = al.truncatedSVD(alMat, k)
        t2 = System.nanoTime()
        println("Time cost of Alchemist truncates SVD: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        
        //// Alchemist Matrix to Local Matrix
        t1 = System.nanoTime()
        val arrV: Array[Array[Double]] =  alV.getIndexedRowMatrix()
                    .rows
                    .map(row => row.vector.toArray)
                    .collect
        val d = arrV.size
        val matV: Matrix = Matrices.dense(k, d, arrV.flatten)
        t2 = System.nanoTime()
        println("Time cost of Alchemist matrix to local matrix: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        //println("Number of rows of V: " + matV.numRows.toString)
        //println("Number of columns of V: " + matV.numCols.toString)
        //println(" ")
        
        //// Compute Approximation Error
        val vBroadcast = sc.broadcast(matV)
        val err: Double = labelVecRdd
                .map(pair => (pair._2, vBroadcast.value.multiply(pair._2)))
                .map(pair => (pair._1, Vectors.dense(vBroadcast.value.transpose.multiply(pair._2).toArray)))
                .map(pair => Vectors.sqdist(pair._1, pair._2))
                .reduce((a, b) => a + b)
        val relativeError = err / sqFroNorm
        println("Squared Frobenius error of rank " + k.toString + " SVD is " + err.toString)
        println("Squared Frobenius norm of A is " + sqFroNorm.toString)
        println("Relative Error is " + relativeError.toString)
                    
        
        
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

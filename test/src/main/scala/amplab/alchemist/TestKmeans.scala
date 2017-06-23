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
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
//breeze
import breeze.linalg.{DenseVector => BDV, max, min, DenseMatrix => BDM, norm, diag, svd}
import breeze.numerics._
// others
import scala.math
import java.io._

object TestKmeans {
    def main(args: Array[String]): Unit = {
        // Parse parameters from command line arguments
        var k: Int = if(args.length > 0) args(0).toInt else 2
        var maxiters: Int = if(args.length > 1) args(1).toInt else 10
        // Print Info
        println("Settings: ")
        println("cluster number = " + k.toString)
        println(" ")
        
        val infile = System.getenv("DATA_FILE")

        
        //// Test Spark
        println("\n#======== Testing Spark ========#")
        val outfile1 = System.getenv("OUTPUT_FILE") + "_spark.txt"
        testSpark(k, maxiters, infile, outfile1)

        
        //// Test Alchemist
        println("\n#======== Testing Alchemist ========#")
        val outfile2 = System.getenv("OUTPUT_FILE") + "_alchemist.txt"
        testAlchemist(k, maxiters, infile, outfile2)

    }
    
    def testSpark(k: Int, maxiters: Int, infile: String, outfile: String): Unit = {  
        //// Launch Spark
        var t1 = System.nanoTime()
        val spark = (SparkSession
                      .builder()
                      .appName("Alchemist Test")
                      .getOrCreate())
        val sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        var t2 = System.nanoTime()
        // Print Info
        println("Time cost of starting Spark session: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        //// Load Data from File
        val labelVecRdd: RDD[(Int, Vector)] = loadData(spark, infile)

        //// Spark Build-in K-Means
        val labelStr: String = kmeansSpark(sc, labelVecRdd, k, maxiters)

        ///// Write (true_label, predicted_label) pairs to file outfile
        val writer = new PrintWriter(new File(outfile))
        writer.write(labelStr)
        writer.close()
        
        spark.stop
    }
    
    def testAlchemist(k: Int, maxiters: Int, infile: String, outfile: String): Unit = {  
        //// Launch Spark
        var t1 = System.nanoTime()
        val spark = (SparkSession
                      .builder()
                      .appName("Alchemist Test")
                      .getOrCreate())
        val sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        var t2 = System.nanoTime()
        // Print Info
        println("Time cost of starting Spark session: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        //// Load Data from File
        val labelVecRdd: RDD[(Int, Vector)] = loadData(spark, infile)

        
        //// Launch Alchemist
        t1 = System.nanoTime()
        val al = new Alchemist(sc)
        t2 = System.nanoTime()
        // Print Info
        println("Time cost of starting Alchemist session: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        
        //// Alchemist K-Means
        val labelStr = kmeansAlchemist(al, labelVecRdd, k, maxiters) 

        
        ///// Write (true_label, predicted_label) pairs to file outfile
        val writer = new PrintWriter(new File(outfile))
        writer.write(labelStr)
        writer.close()
        
        
        al.stop
        spark.stop
    }
    
    def loadData(spark: SparkSession, infile: String): RDD[(Int, Vector)] = {
        // Load and Parse Data
        val t1 = System.nanoTime()
        val df = spark.read.format("libsvm").load(infile)
        val label_vector_rdd: RDD[(Int, Vector)] = df.rdd
                .map(pair => (pair(0).toString.toFloat.toInt, Vectors.parse(pair(1).toString)))
                .persist()
        label_vector_rdd.count
        val t2 = System.nanoTime()
        
        // Print Info
        println("spark.conf.getAll:")
        spark.conf.getAll.foreach(println)
        println(" ")
        println("Number of partitions: ")
        println(label_vector_rdd.getNumPartitions)
        println(" ")
        println("getExecutorMemoryStatus:")
        val sc = spark.sparkContext
        println(sc.getExecutorMemoryStatus.toString())
        println(" ")
        println("Time cost of loading data: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")

        label_vector_rdd
    }
    
    
    def kmeansSpark(sc: SparkContext, label_vector_rdd: RDD[(Int, Vector)], k: Int, maxiter: Int): String = {
        // K-Means Clustering
        val t1 = System.nanoTime()
        val clusters = KMeans.train(label_vector_rdd.map(pair => pair._2), k, maxiter)
        val broadcast_clusters = sc.broadcast(clusters)
        val labels: Array[String] = label_vector_rdd
                .map(pair => (pair._1, broadcast_clusters.value.predict(pair._2)))
                .map(pair => pair._1.toString + " " + pair._2.toString)
                .collect()
        val t2 = System.nanoTime()
        
        // Print Info
        println("Time cost of Spark k-means clustering: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        val label_str: String = (labels mkString " ").trim
        label_str
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

    def kmeansAlchemist(al: Alchemist, labelVecRdd: RDD[(Int, Vector)], k: Int, maxiter: Int): String = {        
        //// Convert Data to Indexed Vectors and Labels
        var t1 = System.nanoTime()
        val (sortedLabels, indexedMat) = splitLabelVec(labelVecRdd)
        var t2 = System.nanoTime()
        // Print Info
        println("Time cost of creating indexed vectors and labels: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        //// Convert Spark IndexedRowMatrix to Alchemist AlMatrix
        t1 = System.nanoTime()
        val alMatkMeans = AlMatrix(al, indexedMat)
        t2 = System.nanoTime()
        // Print Info
        println("Time cost of converting Spark matrix to Alchemist matrix: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        //// K-Means Clustering by Alchemist
        t1 = System.nanoTime()
        val threshold: Double = 1e-4
        val (alCenters, alAssignments, numIters, percentageStable, restarts, totalIters) = al.kMeans(alMatkMeans, k, maxiter, threshold)
        t2 = System.nanoTime()
        // Print Info
        println("Time cost of Alchemist k-means clustering: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        //// Collect the clustering results
        t1 = System.nanoTime()
        val indexPredLabels = alAssignments.getIndexedRowMatrix()
                                        .rows.map(row => (row.index, row.vector.toArray(0)toInt))
                                        .collect
        t2 = System.nanoTime()
        // Print Info
        println("Time cost of sending alchemist cluster assignments back to local: ")
        println((t2 - t1) * 1.0E-9)
        println(" ")
        
        val sortedPredLabels = indexPredLabels.sortWith(_._1 < _._1)
                                        .map(pair => pair._2)
        val labels = (sortedLabels zip sortedPredLabels).map(pair => pair._1.toString + " " + pair._2.toString)
        val labelStr: String = (labels mkString " ").trim
        
        labelStr
    }
}

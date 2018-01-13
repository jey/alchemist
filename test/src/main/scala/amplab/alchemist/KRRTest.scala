package alchemist.test.regression

// Spark Core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._

// Spark SQL
import org.apache.spark.sql.SparkSession

// Spark MLLib
import org.apache.spark.mllib.linalg.{Vector, Vectors, Matrix, Matrices}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.distributed.{RowMatrix, IndexedRow, IndexedRowMatrix}

// Alchemist
import amplab.alchemist._

// Others
import scala.math
import java.io._
import utils._

object AlchemistRFMClassification {

    def main(args: Array[String]) : Unit = {
        // Parse parameters from command line
        val filepath: String = args(0)
        val format: String = args(1)
        val numFeatures: Int = args(2).toInt
        val gamma: Double = args(3).toDouble
        val numClass: Int = args(4).toInt 
        val whereRFM: String = args(5)
        val numSplits: Int = 100 
        val maxIter: Int = 500

        // Launch Spark
        var t1 = System.nanoTime()
        val spark = (SparkSession.builder()
                        .appName("Test Alchemist Random Feature Regression")
                        .getOrCreate())
        val sc: SparkContext = spark.sparkContext
        sc.setLogLevel("ERROR")
        var t2 = System.nanoTime()

        println(s"Time to start Spark session was ${(t2 - t1) * 1.0E-9} sec")
        println(" ")

        // Launch Alchemist
        t1 = System.nanoTime()
        val al = new Alchemist(sc)
        t2 = System.nanoTime()
        println(s"Took ${(t2 -t1)*1.0E-9} seconds to launch Alchemist")
        println(" ")

        // Load data 
        println("Starting to load the raw data")
        println(" ")
        var t3 = System.nanoTime()
        val rddRaw: RDD[(Int, (Int, Array[Double]))] = format.toUpperCase match {
            case "LIBSVM" => RDDToIndexedRDD(loadLibsvmData(spark, filepath, numSplits))
            case "CSV" => RDDToIndexedRDD(loadCsvData(spark, filepath, numSplits))
        }
        var t4 = System.nanoTime()
        println(s"Took ${(t4 - t3)*1.0E-9} seconds to load the raw data")
        println(" ")

        whereRFM.toUpperCase match {
            case "SPARK" => SparkRFMTest(al, rddRaw, numFeatures, numClass, gamma, maxIter)
            case "ALCHEMIST" => AlchemistRFMTest(al, rddRaw, numFeatures, numClass, gamma, maxIter)
        }
        

        al.stop
        sc.stop
    }

    def AlchemistRFMTest(al: Alchemist, rddRaw : RDD[(Int, (Int, Array[Double]))], numFeatures: Int, numClass: Int, gamma: Double, maxIter: Int) : Unit = {

       // Constructed IndexedRowMatrix containing the raw features and the encoded targets
       println("Extracting the raw features and the encoded targets")
       println(" ")
       val extractStart = System.nanoTime()

       val (rddA, sigma) = extractFeatures(rddRaw)
       val numPts = rddA.count()
       val numCols = rddA.take(1)(0).vector.size
       val matA : IndexedRowMatrix = new IndexedRowMatrix(rddA, numPts, numCols)


        val rddTargets : RDD[(Int, Int)] = rddRaw.mapValues(pair => pair._1)
        val rddB: RDD[IndexedRow] = rddTargets.map{ case (index, classindex) => 
                                                        new IndexedRow(index, new DenseVector(oneHotEncode(classindex, numClass)))
                                                  }.cache()
        rddB.count()
        val matB : IndexedRowMatrix = new IndexedRowMatrix(rddB, numPts, numClass)

        val extractEnd = System.nanoTime()

        // Train ridge regression via CG
        val txStart = System.nanoTime()
        val alMatA = AlMatrix(al, matA)
        val alMatB = AlMatrix(al, matB)
        val txEnd = System.nanoTime()
        
        val seed = 12453
        val rfComputeStart = System.nanoTime()
        val alMatF = al.RandomFourierFeatures(alMatA, numFeatures, sigma, seed)
        val rfComputeEnd = System.nanoTime()

        val cgComputeStart = System.nanoTime()
        val alMatX = al.factorizedCGKRR(alMatF, alMatB, gamma, maxIter)
        val cgComputeEnd = System.nanoTime()

        val rxStart = System.nanoTime()
        val rddX = alMatX.getIndexedRowMatrix()
        var rxEnd = System.nanoTime()

        // Compute misclassification rate 
        val alMatPredict = al.matMul(alMatF, alMatX)
        val predictedEncodings : IndexedRowMatrix = alMatPredict.getIndexedRowMatrix()
        val matPredictedEncodings : RDD[(Long, Array[Double])] = predictedEncodings.rows.map(row => (row.index, row.vector.toArray))
        val matTrueEncodings : RDD[(Long, Array[Double])]= rddB.map(row => (row.index, row.vector.toArray))
        val misClassified : Double = matPredictedEncodings.join(matTrueEncodings).mapValues{
          case (enc1, enc2) => {
            val pred : Int = enc1.indexOf(enc1.max)
            val label : Int  = enc2.indexOf(enc2.max)
            if (pred != label) 1 else 0
          }
        }.values.collect.sum.toDouble


        println("Alchemist timing (sec):")
        println(s"dataset creation: ${(extractEnd - extractStart)*1.0E-9}, transmit: ${(txEnd - txStart)*1.0E-9}")
        println(s"random features computation: ${(rfComputeEnd - rfComputeStart)*1.0E-9}, CG computation: ${(cgComputeEnd - cgComputeStart)*1.0E-9}")
        println(s"recieve: ${(rxEnd - rxStart)*1.0E-9}")
        println(s"Misclassification rate: ${misClassified/numPts}")
        println(" ")

    }

    def SparkRFMTest(al: Alchemist, rddRaw : RDD[(Int, (Int, Array[Double]))], numFeatures: Int, numClass: Int, gamma: Double, maxIter: Int) : Unit = {

        println("Computing the random Fourier Features and one-hot encodings")
        println(" ")

        var t3 = System.nanoTime()
        val (rddA, rddTargets) = randomFeatureMap(rddRaw, numFeatures)
        rddRaw.unpersist()
        val numPts = rddA.count()
        val matA : IndexedRowMatrix = new IndexedRowMatrix(rddA, numPts, numFeatures)
        rddTargets.count()
        rddRaw.unpersist()

        val rddB: RDD[IndexedRow] = rddTargets.map{ case (index, classindex) => 
                                                        new IndexedRow(index, new DenseVector(oneHotEncode(classindex, numClass)))
                                                  }.cache()
        rddB.count()
        rddTargets.unpersist()

        val matB : IndexedRowMatrix = new IndexedRowMatrix(rddB, numPts, numClass)
        var t4 = System.nanoTime()

        println("Finished computing the features and encodings")
        println(" ")

        // Train ridge regression via CG
        val alMatA = AlMatrix(al, matA)
        val alMatB = AlMatrix(al, matB)
        var t5 = System.nanoTime()
        val alMatX = al.factorizedCGKRR(alMatA, alMatB, gamma, maxIter)
        var t6 = System.nanoTime()
        val rddX = alMatX.getIndexedRowMatrix()
        var t7 = System.nanoTime()

        // Compute misclassification rate 
        val alMatPredict = al.matMul(alMatA, alMatX)
        val predictedEncodings : IndexedRowMatrix = alMatPredict.getIndexedRowMatrix()
        val matPredictedEncodings : RDD[(Long, Array[Double])] = predictedEncodings.rows.map(row => (row.index, row.vector.toArray))
        val matTrueEncodings : RDD[(Long, Array[Double])]= rddB.map(row => (row.index, row.vector.toArray))
        val misClassified : Double = matPredictedEncodings.join(matTrueEncodings).mapValues{
          case (enc1, enc2) => {
            val pred : Int = enc1.indexOf(enc1.max)
            val label : Int  = enc2.indexOf(enc2.max)
            if (pred != label) 1 else 0
          }
        }.values.collect.sum.toDouble


        println(s"Alchemist timing (sec); conversion: ${(t4 - t3)*1.0E-9}, send: ${(t5-t4)*1.0E-9}, compute: ${(t6-t5)*1.0E-9}, receive: ${(t7-t6)*1.0E-9}")
        println(s"Misclassification rate: ${misClassified/numPts}")
        println(" ")

    }

    def loadCsvData(spark: SparkSession, filepath: String, numSplits: Int) : RDD[(Int, Array[Double])] = {
        // Load data from file
        val t1 = System.nanoTime()
        val df = spark.read.format("csv").load(filepath)
        val rdd: RDD[Array[Double]] = df.rdd.map(vec => Vectors.parse(vec.toString).toArray).persist()
        val d: Int = rdd.take(1)(0).size
        val rdd2: RDD[(Int, Array[Double])] = rdd.map(arr => (arr(0).toInt-1, arr.slice(1, d))).persist()
        val n = rdd2.count()
        val t2 = System.nanoTime()
        println(s"Feature matrix is $n by ${d-1}")
        println(s"Loaded the data in ${(t2 - t1)*1.0E-9} secs")
        println(" ")
        rdd.unpersist()
        rdd2
    }

    def loadLibsvmData(spark: SparkSession, filepath: String, numSplits: Int) : RDD[(Int, Array[Double])] = {
      // Load data from file
      val t1 = System.nanoTime()
      val df = spark.read.format("libsvm").load(filepath)
      val rdd: RDD[(Int, Array[Double])] = df.rdd
                        .map(pair => (pair(0).toString.toFloat.toInt, Vectors.parse(pair(1).toString).toArray))
                        .persist()
      val n = rdd.count()
      val d = rdd.take(1)(0)._2.size
      val t2 = System.nanoTime()
      println(s"Feature matrix is $n by $d")
      println(s"Loaded the data in ${(t2 - t1)*1.0E-9} secs")
      println(" ")
      rdd
    }

    
    def randomFeatureMap(rdd: RDD[(Int, (Int, Array[Double]))], numFeatures: Int) : 
    Tuple2[RDD[IndexedRow], RDD[(Int, Int)]]= {

        val targetRdd: RDD[(Int, Int)] = rdd.mapValues(pair => pair._1).persist()

        // rdd2 contains just the features
        val rdd2: RDD[(Int, Array[Double])] = rdd.mapValues(pair => pair._2)

        // estimate the kernel parameter (if it is unknown)
        var sigma: Double = rdd2.map(pair => (pair._1.toDouble, pair._2)).glom.map(Kernel.estimateSigma).mean
        sigma = math.sqrt(sigma)
        println(s"Estimated sigma is ${sigma}")

        // random feature mapping
        val t1 = System.nanoTime()
        val rfmRdd : RDD[IndexedRow] = rdd2.map(pair => (pair._1.toDouble, pair._2))
                                                        .mapPartitions(Kernel.rbfRfm(_, numFeatures, sigma))
                                                        .map(pair => new IndexedRow(pair._1.toInt, new DenseVector(pair._2)))
                                                        .cache()
        rfmRdd.count()
        val numfeats = rfmRdd.take(1)(0).vector.size
        println(s"numfeats = ${numfeats}")
        var t2 = System.nanoTime()
        println(s"Computing random feature mapping took ${(t2-t1)*1.0E-9} seconds")
        println(" ")
        (rfmRdd, targetRdd)
    }

    def extractFeatures(rdd: RDD[(Int, (Int, Array[Double]))]) : Tuple2[RDD[IndexedRow], Double] = {
        
        val targetRdd: RDD[(Int, Int)] = rdd.mapValues(pair => pair._1).persist()

        // rdd2 contains just the features
        val rdd2: RDD[(Int, Array[Double])] = rdd.mapValues(pair => pair._2)

        // indexedrowmatrix containing just the features
        val rdd3: RDD[IndexedRow] = rdd2.map(pair => new IndexedRow(pair._1, new DenseVector(pair._2))).cache()

        // estimate the kernel parameter (if it is unknown)
        var sigma: Double = rdd2.map(pair => (pair._1.toDouble, pair._2)).glom.map(Kernel.estimateSigma).mean
        sigma = math.sqrt(sigma)
        println(s"Estimated sigma is ${sigma}")

        (rdd3, sigma)
    }

    def oneHotEncode(target: Int, numClass: Int) : Array[Double] = {
        val encoding : Array[Double] = new Array[Double](numClass)
        encoding(target) = 1
        encoding
    }

    // arbitrarily indices the rows in an rdd from zero
    def RDDToIndexedRDD[T]( inRdd: RDD[T] ) : RDD[(Int, T)] = {
        val t1 = System.nanoTime()
        val rand = new scala.util.Random()
        val seedBase = rand.nextInt()
        val prioritizedElements = inRdd.mapPartitionsWithIndex(
            (index, iterator) => {
                val rand = new scala.util.Random(index + seedBase)
                iterator.map(x => (rand.nextDouble, x))
            }, true
        ).persist
        val sortedPriorities = prioritizedElements.map(pair => pair._1).collect().sorted.zipWithIndex.toMap
        val indexedRdd = prioritizedElements.map(pair => (sortedPriorities(pair._1), pair._2))
        val t2 = System.nanoTime()
        println(s"Converting RDD to an Indexed RDD took ${(t2 - t1)*1.0E-9} seconds")
        println(" ")
        indexedRdd.cache()
    }

}




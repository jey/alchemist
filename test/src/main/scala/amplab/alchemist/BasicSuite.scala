package amplab.alchemist
import org.slf4j.LoggerFactory
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, SingularValueDecomposition, Matrix}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.random.{RandomRDDs}
import org.apache.spark.mllib.clustering.KMeans
import breeze.linalg.{DenseVector => BDV, max, min, DenseMatrix => BDM, norm, diag, svd}
import breeze.numerics._

// TODO: break into separate tests

object BasicSuite {
  import LossType._
  import RegularizerType._
  import KernelType._

  def ticks(): Long = {
    return System.currentTimeMillis();
  }

  def main(args: Array[String]): Unit = {
    args(0).toUpperCase match {
        case "SVD" => testSVD(args.tail)
        case "SPARK-SVD" => testSparkSVD(args.tail)
        case "KMEANS" => testKMeans(args.tail)
        case "MATMUL" => testMatMul(args.tail)
        case "LSQR" => testLSQR(args.tail)
        case "ADMMKRR" => testADMMKRR(args.tail)
    }
  }

  def testADMMKRR( args: Array[String] ): Unit = {
    val conf = new SparkConf().setAppName("Alchemist ADMM KRR Test")
    val sc = new SparkContext(conf)
  
    System.err.println("test: creating alchemist")
    val al = new Alchemist(sc)
    System.err.println("test: done creating alchemist")

    // Only do regression w/ default params
    var n = args(0).toInt // rows
    var p = args(1).toInt // columns
    var m = args(2).toInt // number of targets
    var sigma = args(3).toDouble // std of noise added to each target
    var partitions = if (args(4).toInt > 0) args(4).toInt else sc.defaultParallelism
    System.err.println(s"using ${partitions} parallelism for the rdds") 

    // arguments to skylark's solver
    val regression = true
    val lossfunction = SQUARED
    val regularizer = NOREG
    val kernel = K_LINEAR
    val kernelparam = 1.0
    val kernelparam2 = 0.1
    val kernelparam3 = 0.1
    val lambda = 1.0
    val maxiter = 20
    val tolerance = 0.001
    val rho = 1.0
    val seed = 12345
    val randomfeatures = 0
    val numfeaturepartitions = 1

    // generate system matrix
    val rows = RandomRDDs.uniformVectorRDD(sc, n, p, partitions)
    rows.cache()
    val Ardd = new IndexedRowMatrix(rows.zipWithIndex.map(x => new IndexedRow(x._2, x._1)))

    // create right hand sides
    m = if (regression) 1 else m;
    val dm = BDM.rand[Double](p, m) 
    val Xstar = new DenseMatrix(dm.rows, dm.cols, dm.data, dm.isTranspose)
    val Brdd = Ardd.multiply(Xstar)

    var sendStart = ticks()
    val alMatA = AlMatrix(al, Ardd)
    val alMatB = AlMatrix(al, Brdd)
    var sendEnd = ticks()
    var computeStart = sendEnd
    val alMatX = al.SkylarkADMMKRR(alMatA, alMatB, regression, lossfunction, regularizer,
      kernel, kernelparam, kernelparam2, kernelparam3, lambda, maxiter, tolerance,
      rho, seed, randomfeatures, numfeaturepartitions)
    var computeEnd = ticks()
    var receiveStart = computeEnd
    var solXrdd = alMatX.getIndexedRowMatrix()
    var receiveEnd = ticks()

    val solXmat = BDM(solXrdd.rows.collect().sortBy(_.index).map(_.vector.toArray):_*)
    val solXrddnew = new DenseMatrix(dm.rows, dm.cols, dm.data, dm.isTranspose)
    val Brddnew = Ardd.multiply(solXrddnew)
    val truerows = Brdd.rows.map(pair => (pair.index, new BDV(pair.vector.toArray)))
    val predictedrows = Brddnew.rows.map(pair => (pair.index, new BDV(pair.vector.toArray)))
    val frobnorm = truerows.union(predictedrows).reduceByKey(_ - _).map(pair => scala.math.pow(norm(pair._2, 2), 2)).collect().sum

    System.err.println(s"Alchemist timing: send=${(sendEnd-sendStart)/1000.0}, compute=${(computeEnd-computeStart)/1000.0}, receive=${(receiveEnd - receiveStart)/1000.0}")
    System.err.println(s"Residual error: ${frobnorm}")

    al.stop
    sc.stop
  }

  def testLSQR( args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Alchemist LSQR Test")
    val sc = new SparkContext(conf)
  
    System.err.println("test: creating alchemist")
    val al = new Alchemist(sc)
    System.err.println("test: done creating alchemist")

    var n = args(0).toInt // rows
    var p = args(1).toInt // columns
    var m = args(2).toInt // number of rhs
    var maxIters = 100
    var threshold = 1e-14
    var partitions = if (args(3).toInt > 0) args(3).toInt else sc.defaultParallelism
    System.err.println(s"using ${partitions} parallelism for the rdds") 

    // generate system matrix
    val rows = RandomRDDs.uniformVectorRDD(sc, n, p, partitions);
    rows.cache()
    val Ardd = new IndexedRowMatrix(rows.zipWithIndex.map(x => new IndexedRow(x._2, x._1)))

    // create right hand sides
    val dm = BDM.rand[Double](p, m) 
    val Xstar = new DenseMatrix(dm.rows, dm.cols, dm.data, dm.isTranspose)
    val Brdd = Ardd.multiply(Xstar)

    var sendStart = ticks()
    val alMatA = AlMatrix(al, Ardd)
    val alMatB = AlMatrix(al, Brdd)
    var sendEnd = ticks()
    var computeStart = sendEnd
    val alMatX = al.LSQR(alMatA, alMatB)
    var computeEnd = ticks()
    var receiveStart = computeEnd
    var solXrdd = alMatX.getIndexedRowMatrix()
    var receiveEnd = ticks()

    val solXmat = BDM(solXrdd.rows.collect().sortBy(_.index).map(_.vector.toArray):_*)
    val solXrddnew = new DenseMatrix(dm.rows, dm.cols, dm.data, dm.isTranspose)
    val Brddnew = Ardd.multiply(solXrddnew)
    val truerows = Brdd.rows.map(pair => (pair.index, new BDV(pair.vector.toArray)))
    val predictedrows = Brddnew.rows.map(pair => (pair.index, new BDV(pair.vector.toArray)))
    val frobnorm = truerows.union(predictedrows).reduceByKey(_ - _).map(pair => scala.math.pow(norm(pair._2, 2), 2)).collect().sum

    System.err.println(s"Alchemist timing: send=${(sendEnd-sendStart)/1000.0}, compute=${(computeEnd-computeStart)/1000.0}, receive=${(receiveEnd - receiveStart)/1000.0}")
    System.err.println(s"Residual error: ${frobnorm}")

    al.stop
    sc.stop
  }

    def testKMeans(args: Array[String]): Unit = {
        val conf = new SparkConf().setAppName("Alchemist Kmeans Test")
        val sc = new SparkContext(conf)

        System.err.println("test: creating alchemist")
        val al = new Alchemist(sc)
        System.err.println("test: done creating alchemist")

        var m = args(0).toInt
        var n = args(1).toInt
        var k = args(2).toInt
        var maxIters = 100
        var threshold = 1e-4
        var partitions = if (args(3).toInt > 0) args(3).toInt else sc.defaultParallelism
        System.err.println(s"using ${partitions} parallelism for the rdds") 

        val rows = RandomRDDs.uniformVectorRDD(sc, m, n, partitions);
        rows.cache()
        val rdd = new IndexedRowMatrix(rows.zipWithIndex.map(x => new IndexedRow(x._2, x._1)))

        var txStart = ticks()
        val alMatA = AlMatrix(al, rdd)
        var txEnd = ticks()
        var computeStart = txEnd
        val (alCenters, alAssignments, numIters) = al.kMeans(alMatA, k, maxIters, threshold)
        var computeEnd = ticks()
        System.err.println(s"Alchemist timing: send=${(txEnd-txStart)/1000.0}, computeandreceive=${(computeEnd-computeStart)/1000.0}")

        computeStart = ticks()
        val clusters = KMeans.train(rows, k, maxIters, "kmeans||")
        computeEnd = ticks()
        System.err.println(s"Spark timing: svd= ${(computeEnd-computeStart)/1000.0}")

        al.stop
        sc.stop
    }

    def testSparkSVD(args: Array[String]): Unit = {
        val conf = new SparkConf().setAppName("Alchemist SVD Test")
        val sc = new SparkContext(conf)

        var m = args(0).toInt
        var n = args(1).toInt
        var k = args(2).toInt
        var partitions = if (args(3).toInt > 0) args(3).toInt else sc.defaultParallelism
        System.err.println(s"using ${partitions} parallelism for the rdds") 

        val rows = RandomRDDs.normalVectorRDD(sc, m, n, partitions).zipWithIndex
        val rdd = new IndexedRowMatrix(rows.map(x => new IndexedRow(x._2, x._1)))
        rdd.rows.cache
        rdd.rows.count
        System.err.println("done creating and caching dataset for SVD test")

        val computeStart = ticks()
        val svd = rdd.computeSVD(k, computeU = true) 
        svd.U.rows.count()
        val computeEnd = ticks()
        System.err.println(s"Spark timing: svd= ${(computeEnd-computeStart)/1000.0}")

        sc.stop
    }

    def testSVD(args: Array[String]): Unit = {
        val conf = new SparkConf().setAppName("Alchemist SVD Test")
        val sc = new SparkContext(conf)

        System.err.println("test: creating alchemist")
        val al = new Alchemist(sc)
        System.err.println("test: done creating alchemist")

        var m = args(0).toInt
        var n = args(1).toInt
        var k = args(2).toInt
        var partitions = if (args(3).toInt > 0) args(3).toInt else sc.defaultParallelism
        System.err.println(s"using ${partitions} parallelism for the rdds") 

        val rows = RandomRDDs.normalVectorRDD(sc, m, n, partitions).zipWithIndex
        val rdd = new IndexedRowMatrix(rows.map(x => new IndexedRow(x._2, x._1)))
        rdd.rows.cache
        rdd.rows.count
        System.err.println("done creating and caching dataset for SVD test")

        var txStart = ticks()
        val alMatA = AlMatrix(al, rdd)
        var txEnd = ticks()
        System.err.println("done sending dataset for SVD test")
        var computeStart = txEnd
        val (alU, alS, alV) = al.truncatedSVD(alMatA, k) // returns sing vals in increas
        var computeEnd = ticks()
        var rcStart = ticks()
        val alUreturned = alU.getIndexedRowMatrix()
        val alSreturned = alS.getIndexedRowMatrix()
        val alVreturned = alV.getIndexedRowMatrix()
        var rcEnd = ticks()
        System.err.println(s"Alchemist timing: send=${(txEnd-txStart)/1000.0}, svd=${(computeEnd-computeStart)/1000.0}, receive=${(rcEnd - rcStart)/1000.0}")

/*
        computeStart = ticks()
        val svd = rdd.computeSVD(k, computeU = true) 
        svd.U.rows.count()
        computeEnd = ticks()
        System.err.println(s"Spark timing: svd= ${(computeEnd-computeStart)/1000.0}")
        */

        al.stop
        sc.stop
    }

    def testMatMul(args: Array[String]): Unit = {
        val conf = new SparkConf().setAppName("Alchemist MatMul Test")
        val sc = new SparkContext(conf)

        System.err.println("test: creating alchemist")
        val al = new Alchemist(sc)
        System.err.println("test: done creating alchemist")

        // multiply m-by-n and n-by-k
        val m = args(0).toInt
        val n = args(1).toInt
        val k = args(2).toInt
        val partitions = args(3).toInt

        val rowsA = RandomRDDs.uniformVectorRDD(sc, m, n, partitions).zipWithIndex
        val sparkMatA = new IndexedRowMatrix(rowsA.map(x => new IndexedRow(x._2, x._1)))
        val rowsB = RandomRDDs.uniformVectorRDD(sc, n, k, partitions).zipWithIndex
        val sparkMatB = new IndexedRowMatrix(rowsB.map(x => new IndexedRow(x._2, x._1)))

        var txStart = ticks()
        val alMatA = AlMatrix(al, sparkMatA)
        val alMatB = AlMatrix(al, sparkMatB)
        var txEnd = ticks()
        var computeStart = txEnd
        val alMatC = al.matMul(alMatA, alMatB)
        var computeEnd = ticks()
        var rcStart = ticks()
        val alRes = alMatC.getIndexedRowMatrix()
        var rcEnd = ticks()
        System.err.println(s"Alchemist timing: send=${(txEnd-txStart)/1000.0}, mul=${(computeEnd-computeStart)/1000.0}, receive=${(rcEnd - rcStart)/1000.0}")

        // // Spark matrix multiply
        val smulStart = ticks()
        val sparkMatC = sparkMatA.toBlockMatrix(sparkMatA.numRows.toInt, sparkMatA.numCols.toInt).
                      multiply(sparkMatB.toBlockMatrix(sparkMatB.numRows.toInt, sparkMatB.numCols.toInt)).toIndexedRowMatrix
        val smulEnd = ticks()
        System.err.println(s"Spark matrix multiplication time(s): ${(smulEnd - smulStart)/1000.0}")

        al.stop
        sc.stop
    }

/*
	def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Alchemist Test")
    val sc = new SparkContext(conf)

    val sparkMatA = deterministicMatrix(sc, 5000, 2000, 1)
    sparkMatA.rows.cache
    val sparkMatB = deterministicMatrix(sc, 2000, 5000, 10)
    sparkMatB.rows.cache

    System.err.println("test: creating alchemist")
    val al = new Alchemist(sc)
    System.err.println("test: done creating alchemist")


    /**
    // TEST: Alchemist matrix multiply
    var txStart = ticks()
    val alMatA = AlMatrix(al, sparkMatA)
    val alMatB = AlMatrix(al, sparkMatB)
    var txEnd = ticks()
    var computeStart = txEnd
    val alMatC = al.matMul(alMatA, alMatB)
    var computeEnd = ticks()
    var rcStart = ticks()
    val alRes = alMatC.getIndexedRowMatrix()
    var rcEnd = ticks()
    println(s"Alchemist timing: send=${(txEnd-txStart)/1000.0}, mul=${(computeEnd-computeStart)/1000.0}, receive=${(rcEnd - rcStart)/1000.0}")

    // // Spark matrix multiply
    val smulStart = ticks()
    val sparkMatC = sparkMatA.toBlockMatrix(sparkMatA.numRows.toInt, sparkMatA.numCols.toInt).
                  multiply(sparkMatB.toBlockMatrix(sparkMatB.numRows.toInt, sparkMatB.numCols.toInt)).toIndexedRowMatrix
    val smulEnd = ticks()
    println(s"Spark matrix multiplication time(s): ${(smulEnd - smulStart)/1000.0}")
    val alResLocalMat = toLocalMatrix(alRes)
    val sparkLocalMat = toLocalMatrix(sparkMatC)
    val diff = norm(alResLocalMat.toDenseVector - sparkLocalMat.toDenseVector)
    println(s"The frobenius norm difference between Spark and Alchemist's results is ${diff}")

    **/
    al.stop
    sc.stop
  }
  */

  def notCurrentlyMain() = {
    //// TEST: check SVD
    //// TODO: check U,V orthonormal
    //// check U*S*V is original matrix
    //val (alU, alS, alV) = al.thinSVD(alMatA)
    //val alULocalMat = toLocalMatrix(alU.getIndexedRowMatrix())
    //val alSLocalMat = toLocalMatrix(alS.getIndexedRowMatrix())
    //val alVLocalMat = toLocalMatrix(alV.getIndexedRowMatrix())
    //println(norm((alULocalMat*diag(alSLocalMat(::,0))*alVLocalMat.t - toLocalMatrix(alMatA.getIndexedRowMatrix())).toDenseVector))

    //// TEST: check matrix transpose
    //val alMatATranspose = alMatA.transpose()
    //val alMatATransposeLocalMat = toLocalMatrix(alMatATranspose.getIndexedRowMatrix())
    //println(norm((alMatATransposeLocalMat - toLocalMatrix(alMatA.getIndexedRowMatrix).t).toDenseVector))

    /*
    // TEST: basic check k-means
    val n : Int = 30;
    val d : Int = 20;
    val k : Int = 5
    val maxIters : Int = 10
    val threshold : Double = 0.01
    val (rowAssignments, matKMeans) = kmeansTestMatrix(sc, n, d, k) 
    val alMatkMeans = AlMatrix(al, matKMeans)
    val (alCenters, alAssignments, numIters) = al.kMeans(alMatkMeans, k, maxIters, threshold)

    val alCentersLocalMat = toLocalMatrix(alCenters.getIndexedRowMatrix())
    val alAssignmentsLocalMat = toLocalMatrix(alAssignments.getIndexedRowMatrix())
    displayBDM(alCenters.getIndexedRowMatrix())
    println(rowAssignments.groupBy(_ + 0).mapValues(_.length).toList)
    println(alAssignmentsLocalMat.data.groupBy(_ + 0).mapValues(_.length).toList)
    */

    /*
    // TEST: larger k-means
    val n : Int = 10000;
    val d : Int = 780;
    val k : Int = 10;
    val maxIters : Int = 100
    val threshold : Double = 1e-4
    val noiseLevel : Double = 2.0*d
    val (rowAssignments, matKMeans) = kmeansTestMatrix(sc, n, d, k, noiseLevel)
    val alMatkMeans = AlMatrix(al, matKMeans)
    val kmeansStart = System.nanoTime
    val (alCenters, alAssignments, numIters) = al.kMeans(alMatkMeans, k, maxIters, threshold)
    val kmeansDuration = (System.nanoTime - kmeansStart)/1e9d

    
    val sparkKmeansStart = System.nanoTime
    val clusters = KMeans.train(matKMeans.rows.map(pair => pair.vector), k, maxIters)
    val sparkKmeansDuration = (System.nanoTime - sparkKmeansStart)/1e9d

    println(s"Alchemist kmeans: ${kmeansDuration}")
    println(s"Alchemist iters: ${numIters}")
    println(s"Spark kmeans: ${sparkKmeansDuration}")
    */

    /*
    // TEST truncatedSVD
    val k : Int = 10;
    // 10000 rows
    val sparkMatSVD = randomMatrix(sc, 10000, 780)
    val transferStart = System.nanoTime
    val alMatA = AlMatrix(al, sparkMatSVD)
    val transferDuration = (System.nanoTime - transferStart)/1e9d
    val svdStart = System.nanoTime
    val (alU, alS, alV) = al.truncatedSVD(alMatA, k) // returns sing vals in increas
    val svdDuration = (System.nanoTime - svdStart)/1e9d
    val alULocalMat = toLocalMatrix(alU.getIndexedRowMatrix())
    val alSLocalVec = toLocalMatrix(alS.getIndexedRowMatrix())
    val alVLocalMat = toLocalMatrix(alV.getIndexedRowMatrix())

    val sparkSVDStart = System.nanoTime
    val sparkSVD: SingularValueDecomposition[IndexedRowMatrix, Matrix] = sparkMatSVD.computeSVD(k, computeU=true)
    val sparkSVDDuration = (System.nanoTime - sparkSVDStart)/1e9d
    val sparkV = new BDM(sparkSVD.V.numRows, sparkSVD.V.numCols, sparkSVD.V.toArray) // both spark and breeze store in column-major format
    val sparkS = new BDV(sparkSVD.s.toArray)

    // causes a memory error because u*u^t is too large
    //println(norm(udiff.toDenseVector))
    //val udiff = u(::, 0 until k)*u(::, 0 until k).t - alULocalMat*alULocalMat.t
    val vdiff = sparkV*sparkV.t - alVLocalMat*alVLocalMat.t
    println(norm(vdiff.toDenseVector))
    println(alSLocalVec)
    println(sparkS(0 until k))

    println(s"Alchemist transfer: ${transferDuration}, svd: ${svdDuration}")
    println(s"Spark svd: ${sparkSVDDuration}")
    */
  }

  implicit def arrayofIntsToLocalMatrix(arr: Array[Int]) : BDM[Double] = {
    new BDM(arr.length, 1, arr.toList.map(_.toDouble).toArray)
  }

  implicit def indexedRowMatrixToLocalMatrix(mat: IndexedRowMatrix) : BDM[Double] = {
    toLocalMatrix(mat)
  }

  def toLocalMatrix(mat: IndexedRowMatrix) : BDM[Double] = {
    val numRows = mat.numRows.toInt
    val numCols = mat.numCols.toInt
    val res = BDM.zeros[Double](numRows, numCols)
    mat.rows.collect.foreach{ indexedRow => res(indexedRow.index.toInt, ::) := BDV(indexedRow.vector.toArray).t }
    res
  }

  def displayBDM(mat : BDM[Double], truncationLevel : Double = 1e-10) = {
    println(s"${mat.rows} x ${mat.cols}:")
    (0 until mat.rows).foreach{ i => println(mat(i, ::).t.toArray.map( x => if (x <= truncationLevel) 0 else x).mkString(" ")) }
  }

  def deterministicMatrix(sc: SparkContext, numRows: Int, numCols: Int, scale: Double): IndexedRowMatrix = {
    val mat = BDM.zeros[Double](numRows, numCols) 
    (0 until min(numRows, numCols)).foreach { i : Int => mat(i, i) = scale * (i + 1)}
    val rows = sc.parallelize( (0 until numRows).map( i => mat(i, ::).t.toArray )).zipWithIndex
    new IndexedRowMatrix(rows.map(x => new IndexedRow(x._2, new DenseVector(x._1))))
  }

  def randomMatrix(sc: SparkContext, numRows: Int, numCols: Int): IndexedRowMatrix = {
    val rows = RandomRDDs.uniformVectorRDD(sc, numRows, numCols, 128).zipWithIndex
    new IndexedRowMatrix(rows.map(x => new IndexedRow(x._2, x._1)))
  }

  def kmeansTestMatrix(sc : SparkContext, numRows : Int, numCols: Int, numCenters: Int, noiseLevel : Double) : Tuple2[Array[Int], IndexedRowMatrix] = {
    assert(numCols >= numCenters)
    val rowAssignments = Array.fill(numRows)(scala.util.Random.nextInt(numCenters))
    val trueMat = BDM.zeros[Double](numRows, numCols)
    (0 until numRows).foreach{ i : Int => trueMat(i, rowAssignments(i)) = numCols }
    val noiseMat = 2.0*(BDM.rand[Double](numRows, numCols) - .5)*noiseLevel
    val totalMat = trueMat + noiseMat
    val rows = sc.parallelize( (0 until numRows).map( i => totalMat(i, ::).t.toArray )).zipWithIndex
    val indexedMat = new IndexedRowMatrix(rows.map(x => new IndexedRow(x._2, new DenseVector(x._1))))
    (rowAssignments, indexedMat)
  }
}

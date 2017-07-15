package amplab.alchemist
//import org.scalatest.FunSuite
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, SingularValueDecomposition, Matrix}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.random.{RandomRDDs}
import org.apache.spark.mllib.clustering.KMeans
import breeze.linalg.{DenseVector => BDV, max, min, DenseMatrix => BDM, norm, diag, svd}
import breeze.numerics._

object BasicSuite {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Alchemist Test")
    val sc = new SparkContext(conf)
    System.err.println("test: creating alchemist")
    val al = new Alchemist(sc)
    System.err.println("test: done creating alchemist")
    val sparkMatA = deterministicMatrix(sc, 30, 50, 1)
    sparkMatA.rows.cache
    val sparkMatB = deterministicMatrix(sc, 50, 20, 10)
    sparkMatB.rows.cache

    // // Spark matrix multiply
    // val sparkMatC = sparkMatA.toBlockMatrix(sparkMatA.numRows.toInt, sparkMatA.numCols.toInt).
    //               multiply(sparkMatB.toBlockMatrix(sparkMatB.numRows.toInt, sparkMatB.numCols.toInt)).toIndexedRowMatrix

    // TEST: check that sending/receiving matrices works
    //println(norm((toLocalMatrix(alMatB.getIndexedRowMatrix()) - toLocalMatrix(sparkMatB)).toDenseVector))
    //println(norm((toLocalMatrix(alMatA.getIndexedRowMatrix()) - toLocalMatrix(sparkMatA)).toDenseVector))
    //println("alResLocalMat:")
    //displayBDM(alResLocalMat)
    //println("sparkLocalMat:")
    //displayBDM(sparkLocalMat)
    
    // TEST: Alchemist matrix multiply
    //val alMatA = AlMatrix(al, sparkMatA)
    //val alMatB = AlMatrix(al, sparkMatB)
    //val alMatC = al.matMul(alMatA, alMatB)
    //val alRes = alMatC.getIndexedRowMatrix()
    //assert(alRes.numRows == sparkMatA.numRows)
    //assert(alRes.numCols == sparkMatB.numCols)

    //val alResLocalMat = toLocalMatrix(alRes)
    //val sparkLocalMat = toLocalMatrix(sparkMatC)
    //val diff = norm(alResLocalMat.toDenseVector - sparkLocalMat.toDenseVector)
    //println(s"The frobenius norm difference between Spark and Alchemist's results is ${diff}")


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
    al.stop
    sc.stop
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
    val rows = RandomRDDs.normalVectorRDD(sc, numRows, numCols, 128).zipWithIndex
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

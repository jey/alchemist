package amplab.alchemist
//import org.scalatest.FunSuite
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.random.{RandomRDDs}
import breeze.linalg.{DenseVector => BDV, max, min, DenseMatrix => BDM, norm, diag}
import breeze.numerics._
import breeze.linalg.svd

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

    //// TEST: check k-means
    //val n : Int = 30;
    //val d : Int = 20;
    //val k : Int = 5
    //val maxIters : Int = 10
    //val threshold : Double = 0.2
    //val (rowAssignments, matKMeans) = kmeansTestMatrix(sc, n, d, k) 
    //val alMatkMeans = AlMatrix(al, matKMeans)
    //val (alCenters, alAssignments, numIters, percentageStable, restarts, totalIters) = al.kMeans(alMatkMeans, k, maxIters, threshold)

    //val alCentersLocalMat = toLocalMatrix(alCenters.getIndexedRowMatrix())
    //val alAssignmentsLocalMat = toLocalMatrix(alAssignments.getIndexedRowMatrix())
    //displayBDM(alCenters.getIndexedRowMatrix())
    //println(rowAssignments.groupBy(_ + 0).mapValues(_.length).toList)
    //println(alAssignmentsLocalMat.data.groupBy(_ + 0).mapValues(_.length).toList)
    
    // TEST truncatedSVD
    val k : Int = 5;
    val alMatA = AlMatrix(al, sparkMatA)
    val (alU, alS, alV) = al.truncatedSVD(alMatA, k)
    val alULocalMat = toLocalMatrix(alU.getIndexedRowMatrix())
    val alSLocalVec = toLocalMatrix(alS.getIndexedRowMatrix())
    val alVLocalMat = toLocalMatrix(alV.getIndexedRowMatrix())

    displayBDM(sparkMatA)
    displayBDM(alSLocalVec)

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

  def kmeansTestMatrix(sc : SparkContext, numRows : Int, numCols: Int, numCenters: Int) : Tuple2[Array[Int], IndexedRowMatrix] = {
    assert(numCols >= numCenters)
    val rowAssignments = Array.fill(numRows)(scala.util.Random.nextInt(numCenters))
    val mat = BDM.zeros[Double](numRows, numCols)
    (0 until numRows).foreach{ i : Int => mat(i, rowAssignments(i)) = numCols }
    val rows = sc.parallelize( (0 until numRows).map( i => mat(i, ::).t.toArray )).zipWithIndex
    val indexedMat = new IndexedRowMatrix(rows.map(x => new IndexedRow(x._2, new DenseVector(x._1))))
    (rowAssignments, indexedMat)
  }
}

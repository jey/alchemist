package amplab.alchemist
//import org.scalatest.FunSuite
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.random.{RandomRDDs}
import breeze.linalg.{DenseVector => BDV, max, min, DenseMatrix => BDM, norm}
import breeze.numerics._

object BasicSuite {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Alchemist Test")
    val sc = new SparkContext(conf)
    System.err.println("test: creating alchemist")
    val al = new Alchemist(sc)
    System.err.println("test: done creating alchemist")
//    val sparkMatA = randomMatrix(sc, 300, 500)
    val sparkMatA = deterministicMatrix(sc, 30, 50)
    sparkMatA.rows.cache
//    val sparkMatB = randomMatrix(sc, 500, 200)
    val sparkMatB = deterministicMatrix(sc, 50, 20)
    sparkMatB.rows.cache

    // Spark matrix multiply
    val sparkMatC = sparkMatA.toBlockMatrix(sparkMatA.numRows.toInt, sparkMatA.numCols.toInt).
                  multiply(sparkMatB.toBlockMatrix(sparkMatB.numRows.toInt, sparkMatB.numCols.toInt)).toIndexedRowMatrix

    // Alchemist matrix multiply
    val alMatA = AlMatrix(al, sparkMatA)
    val alMatB = AlMatrix(al, sparkMatB)
    val alMatC = al.matMul(alMatA, alMatB)
    val alRes = alMatC.getIndexedRowMatrix()
    assert(alRes.numRows == sparkMatA.numRows)
    assert(alRes.numCols == sparkMatB.numCols)

    val alResLocalMat = toLocalMatrix(alRes)
    val sparkLocalMat = toLocalMatrix(sparkMatC)
    val diff = norm(alResLocalMat.toDenseVector - sparkLocalMat.toDenseVector)
    println(s"The frobenius norm difference between Spark and Alchemist's results is ${diff}")

    // check that sending/receiving matrices works
    println(norm((toLocalMatrix(alMatB.getIndexedRowMatrix()) - toLocalMatrix(sparkMatB)).toDenseVector))
    println(norm((toLocalMatrix(alMatA.getIndexedRowMatrix()) - toLocalMatrix(sparkMatA)).toDenseVector))
    displayBDM(alResLocalMat)
    displayBDM(sparkLocalMat)
    al.stop
    sc.stop
  }

  def toLocalMatrix(mat: IndexedRowMatrix) : BDM[Double] = {
    val numRows = mat.numRows.toInt
    val numCols = mat.numCols.toInt
    val res = BDM.zeros[Double](numRows, numCols)
    mat.rows.collect.foreach{ indexedRow => res(indexedRow.index.toInt - 1, ::) := BDV(indexedRow.vector.toArray).t }
    res
  }

  def displayBDM(mat : BDM[Double]) = {
    println(s"${mat.rows} x ${mat.cols}:")
    (0 until mat.rows).foreach{ i => println(mat(i, ::).t.toArray.mkString(" ")) }
  }

  def deterministicMatrix(sc: SparkContext, numRows: Int, numCols: Int): IndexedRowMatrix = {
    val mat = BDM.zeros[Double](numRows, numCols) 
    (0 until min(numRows, numCols)).foreach { i : Int => mat(i, i) = i }
    val rows = sc.parallelize( (0 until numRows).map( i => mat(i, ::).t.toArray )).zipWithIndex
    new IndexedRowMatrix(rows.map(x => new IndexedRow(x._2, new DenseVector(x._1))))
  }

  def randomMatrix(sc: SparkContext, numRows: Int, numCols: Int): IndexedRowMatrix = {
    val rows = RandomRDDs.normalVectorRDD(sc, numRows, numCols, 128).zipWithIndex
    new IndexedRowMatrix(rows.map(x => new IndexedRow(x._2, x._1)))
  }
}

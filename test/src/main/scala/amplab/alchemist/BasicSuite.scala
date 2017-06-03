package amplab.alchemist
//import org.scalatest.FunSuite
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseMatrix}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.random.{RandomRDDs}
import breeze.linalg.{DenseVector => BDV, max}
import breeze.numerics._

object BasicSuite {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Alchemist Test")
    val sc = new SparkContext(conf)
    System.err.println("test: creating alchemist")
    val al = new Alchemist(sc)
    System.err.println("test: done creating alchemist")
    val rddA = randomMatrix(sc, 300, 500)
    rddA.rows.cache
    val rddB = randomMatrix(sc, 500, 200)
    rddB.rows.cache
    val alA = AlMatrix(al, rddA)
    val alB = AlMatrix(al, rddB)
    val alC = al.matMul(alA, alB)
    val res = alC.getIndexedRowMatrix()
    al.stop
    sc.stop
  }

  def randomMatrix(sc: SparkContext, numRows: Int, numCols: Int): IndexedRowMatrix = {
    val rows = RandomRDDs.normalVectorRDD(sc, numRows, numCols, 128).zipWithIndex
    new IndexedRowMatrix(rows.map(x => new IndexedRow(x._2, x._1)))
  }
}

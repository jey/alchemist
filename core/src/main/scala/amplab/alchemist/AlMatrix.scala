package amplab.alchemist
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import scala.math.max

class AlMatrix(val mal: Alchemist, mhandle: MatrixHandle) {
  val handle : MatrixHandle = mhandle
  val al : Alchemist = mal

  def getDimensions() : Tuple2[Long, Int] = {
    return al.client.getMatrixDimensions(handle)
  }

  def getIndexedRowMatrix() : IndexedRowMatrix = {
    val (numRows, numCols) = getDimensions()
    // TODO:
    // should map the rows back to the executors using locality information if possible
    // otherwise shuffle the rows on the MPI side before sending them back to SPARK
    val numPartitions = max(sc.defaultParallelism, al.client.workerCount)
    val sacrificialRDD = sc.parallelize(1 to numRows, numPartitions)
    val layout : Array[WorkerId] = (1 to sacrificialRDD.partitions.count).map(x => new WorkerId(x % al.client.workerCount))

    al.client.getIndexedRowMatrixStart(handle, layout)
    val rows = sacrificialRDD.mapPartitionsWithIndex( (idx, part) => {
      val worker = ctx.connectWorker(layout(idx))
      val result = part.foreach { rowIndex => 
        new IndexedRow(rowIndex, worker.getIndexedRowMatrix_getRow(handle, rowIndex))
      }
      worker.getIndexedRowMatrix_partitionComplete(handle)
      worker.close()
      Iterator.single(result)}, 
      preservesPartitioning=true)
    al.client.getIndexedRowMatrixFinish(handle)
    new IndexedRowMatrix(rows, numRows, numCols)
  } 

}

object AlMatrix {
  def apply(al: Alchemist, mat: IndexedRowMatrix): AlMatrix = {
    val ctx = al.context
    val workerIds = ctx.workerIds
    val layout = mat.rows.partitions.zipWithIndex.map {
      case (part, idx) => workerIds(idx % workerIds.length)
    }.toArray
    val handle = al.client.newMatrixStart(mat.numRows, mat.numCols, layout)
    mat.rows.mapPartitionsWithIndex { (idx, part) =>
      val client = ctx.connectWorker(layout(idx))
      part.foreach { row =>
        client.newMatrix_addRow(handle, row.index, row.vector.toArray)
      }
      client.newMatrix_partitionComplete(handle)
      client.close()
      Iterator.single(true)
    }.count
    al.client.newMatrixFinish(handle)
    result = new AlMatrix(al, handle)
    return result
  }
}

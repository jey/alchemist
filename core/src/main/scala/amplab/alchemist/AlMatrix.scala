package amplab.alchemist
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}

class AlMatrix(val al: Alchemist, handle: MatrixHandle) {
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
    new AlMatrix(al, handle)
  }
}

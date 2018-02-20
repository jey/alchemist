package amplab.alchemist
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import scala.math.max
import java.nio.ByteBuffer

class AlMatrix(val al: Alchemist, val handle: MatrixHandle) {
  def getDimensions() : Tuple2[Long, Int] = {
    return al.client.getMatrixDimensions(handle)
  }

  def transpose() : AlMatrix = {
    new AlMatrix(al, al.client.getTranspose(handle))
  }

  // Caches result by default, because may not want to recreate (e.g. if delete referenced matrix on Alchemist side to save memory)
  def getIndexedRowMatrix() : IndexedRowMatrix = {
    val (numRows, numCols) = getDimensions()
    // TODO:
    // should map the rows back to the executors using locality information if possible
    // otherwise shuffle the rows on the MPI side before sending them back to SPARK

    val numPartitions = max(al.sc.defaultParallelism, al.client.workerCount)
    val sacrificialRDD = al.sc.parallelize(0 until numRows.toInt, numPartitions)
    val layout : Array[WorkerId] = (0 until sacrificialRDD.partitions.size).map(x => new WorkerId(x % al.client.workerCount)).toArray
    val full_layout : Array[WorkerId] = (layout zip sacrificialRDD.mapPartitions(iter => Iterator.single(iter.size), true).collect()).
                                          flatMap{ case (workerid, partitionSize) => Array.fill(partitionSize)(workerid) }
    // capture references needed by the closure without capturing `this.al`
    val ctx = al.context
    val handle = this.handle

    al.client.getIndexedRowMatrixStart(handle, full_layout)
    val rows = sacrificialRDD.mapPartitionsWithIndex( (idx, rowindices) => {
      var result : Iterator[IndexedRow] = Iterator.empty // it's possible not every partition will have data in it when the matrix being returned is small
      if (!rowindices.toList.isEmpty) {
          val worker = ctx.connectWorker(layout(idx))
          result = rowindices.toList.map { rowIndex =>
            new IndexedRow(rowIndex, worker.getIndexedRowMatrix_getRow(handle, rowIndex, numCols))
          }.iterator
          worker.close()
      }
      result
    }, preservesPartitioning=true)
    val result = new IndexedRowMatrix(rows, numRows, numCols)
    result.rows.cache()
    result.rows.count
    al.client.getIndexedRowMatrixFinish(handle)
    result
  }

}

object AlMatrix {
  def apply(al: Alchemist, mat: IndexedRowMatrix): AlMatrix = {
    // Sends the IndexedRowMatrix over to Alchemist as MD, STAR Elemental matrix
    
    val ctx = al.context
    val workerIds = ctx.workerIds
    // rowWorkerAssignments is an array of WorkerIds whose ith entry is the world rank of the alchemist worker
    // that will take the ith row (ranging from 0 to numworkers-1). Note 0 is an executor, not the driver
    val (handle, rowWorkerAssignments) = al.client.newMatrixStart(mat.numRows, mat.numCols)
    mat.rows.mapPartitionsWithIndex { (idx, part) =>
      val rows = part.toArray
      val relevantWorkers = rows.map(row => rowWorkerAssignments(row.index.toInt).id).distinct.map(id => new WorkerId(id))
      println("Sending data to following workers: ")
      println(relevantWorkers.map(node => node.id.toString).mkString(" "))
      val maxWorkerId = relevantWorkers.map(node => node.id).max
      var nodeClients = Array.fill(maxWorkerId+1)(None: Option[WorkerClient])
      System.err.println(s"Connecting to ${relevantWorkers.length} workers")
      relevantWorkers.foreach(node => nodeClients(node.id) = Some(ctx.connectWorker(node)))
      System.err.println(s"Successfully connected to all workers; have ${rows.length} rows to send")

      // TODO: randomize the order the rows are sent in to avoid queuing issues?
      var count = 0
      val buflen = 4 + 4 + 8 + 8 + 8 * rows(0).vector.toArray.length
      val reuseBuf = ByteBuffer.allocateDirect(Math.min(buflen, 16*1024*1024))
      rows.foreach{ row =>
        count += 1
//        System.err.println(s"Sending row ${row.index.toInt}, ${count} of ${rows.length}")
        nodeClients(rowWorkerAssignments(row.index.toInt).id).get.
          newMatrix_addRow(handle, row.index, row.vector.toArray, reuseBuf)
      }
      System.err.println("Finished sending rows")
      nodeClients.foreach(client => 
          if (client.isDefined) {
            client.get.newMatrix_partitionComplete(handle)
            client.get.close()
          })
      Iterator.single(true)
    }.count
    al.client.newMatrixFinish(handle)
    new AlMatrix(al, handle)
  }
}

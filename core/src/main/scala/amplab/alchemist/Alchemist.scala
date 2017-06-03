package amplab.alchemist
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import scala.util.Random
import java.io.{
    InputStream, OutputStream,
    DataInputStream => JDataInputStream,
    DataOutputStream
}
import java.nio.{
    DoubleBuffer, ByteBuffer
}
import java.nio.charset.StandardCharsets

class DataInputStream(istream: InputStream) extends JDataInputStream(istream) {
  def readBuffer(): Array[Byte] = {
    val buf = new Array[Byte](readInt())
    read(buf)
    buf
  }

  def readDoubleArray(): Array[Double] = {
    // http://www.scala-lang.org/old/node/11002
    // maybe this code is inefficient; can make more efficient using
    // platform dependent (endianness) code?
    // is there not a fast readDouble() primitive in JDataInputStream?
    val buf = readBuffer()
    val dbuf = DoubleBuffer.allocate(buf.length/8)
    dbuf.put(ByteBuffer.wrap(buf).asDoubleBuffer)
    dbuf.array
  }

  def readString(): String = {
    new String(readBuffer(), StandardCharsets.US_ASCII)
  }
}

class ProtocolError extends Exception {
}

class WorkerId(val id: Int) extends Serializable {
}

class MatrixHandle(val id: Int) extends Serializable {
}

class WorkerClient(val hostname: String, val port: Int) {
  val sock = new java.net.Socket(hostname, port)
  val output = new DataOutputStream(sock.getOutputStream)
  val input = new DataInputSream(sock.getInputStream)

  def newMatrix_addRow(handle: MatrixHandle, rowIdx: Long, vals: Array[Double]) = {
    output.writeInt(0x1)  // typeCode = addRow
    output.writeInt(handle.id)
    output.writeLong(rowIdx)
    output.writeLong(vals.length)
    for(v <- vals) {
      output.writeDouble(v)
    }
  }

  def newMatrix_partitionComplete(handle: MatrixHandle) = {
    output.writeInt(0x2)  // typeCode = partitionComplete
  }

  // difficulty? multiple spark workers can be trying to get rows from the 
  // same alchemist worker at the same time
  def getIndexedRowMatrix_getRow(handle: MatrixHandle, rowIndex : Long) : DenseVector = {
    output.writeInt(0x3) // typeCode = getRow
    output.writeInt(handle.id)
    output.writeLong(rowIndex)
    new DenseVector(input.readDoubleArray())
  }

  def getIndexedRowMatrix_partitionComplete(handle: MatrixHandle) = {
    output.writeInt(0x4) // typeCode = doneGettingRows
  }

  def close() = {
    output.close()
    sock.close()
  }
}

class WorkerInfo(val id: WorkerId, val hostname: String, val port: Int) extends Serializable {
  def newClient(): WorkerClient = {
    System.err.println(s"connecting to $hostname:$port")
    new WorkerClient(hostname, port)
  }
}

class DriverClient(val istream: InputStream, val ostream: OutputStream) {
  val input = new DataInputStream(istream)
  val output = new DataOutputStream(ostream)

  // handshake
  output.writeInt(0xABCD)
  output.writeInt(0x1)
  output.flush()
  if(input.readInt() != 0xDCBA || input.readInt() != 0x1) {
    throw new ProtocolError()
  }

  val workerCount = input.readInt()

  val workerIds: Array[WorkerId] =
    (0 until workerCount).map(new WorkerId(_)).toArray

  val workerInfo: Array[WorkerInfo] =
    workerIds.map(id => new WorkerInfo(id, input.readString, input.readInt))

  def shutdown(): Unit = {
    output.writeInt(0xFFFFFFFF)
    output.flush()
    if(input.readInt() != 0x1) {
      throw new ProtocolError()
    }
  }

  // layout contains one worker ID per matrix partition
  def newMatrixStart(rows: Long, cols: Long, layout: Array[WorkerId]): MatrixHandle = {
    output.writeInt(0x1)
    output.writeLong(rows)
    output.writeLong(cols)
    output.writeLong(layout.length)
    layout.map(w => output.writeInt(w.id))
    output.flush()
    if(input.readInt() != 0x1) {
      throw new ProtocolError()
    }
    val handle = new MatrixHandle(input.readInt())
    System.err.println(s"got handle: ${handle.id}")
    return handle
  }

  def matrixMulStart(matA: MatrixHandle, matB: MatrixHandle) : MatrixHandle = {
    output.writeInt(0x2)
    output.writeInt(matA.id)
    output.writeInt(matB.id)
    output.flush()
    if(input.readInt() != 0x1) {
      throw new ProtocolError()
    }
    val handle = new MatrixHandle(input.readInt())
    System.err.println(s"got handle: ${handle.id}")
    return handle
  }

  def matrixMulFinish(mat: MatrixHandle) = {
    if(input.readInt() != 0x1) {
      throw new ProtocolError() 
    } 
  }

  def newMatrixFinish(mat: MatrixHandle) = {
    if(input.readInt() != 0x1) {
      throw new ProtocolError()
    }
  }

  def getMatrixDimensions(mat: MatrixHandle) : Tuple2[Long, Int] = {
    output.writeInt(0x3)
    output.writeInt(mat.id)
    output.flush()
    if(input.readInt() != 0x1) {
      throw new ProtocolError()
    }
    return (input.readLong(), input.readLong().toInt)
  }

  // layout maps each partition to a worker id (so has length number of partitions in the spark matrix being retrieved)
  def getIndexedRowMatrixStart(mat: MatrixHandle, layout: Array[WorkerId]) = {
    output.writeInt(0x4)
    output.writeInt(mat.id)
    output.writeLong(layout.length)
    layout.map(w => output.writeInt(w.id))
    output.flush()
    if(input.readInt() != 0x1) {
      throw new ProtocolError()
    }
  }

  def getIndexedRowMatrixFinish(mat: MatrixHandle) = {
    if(input.readInt() != 0x1) {
      throw new ProtocolError()
    }
  }

}

class Driver {
  val driverProc: Process = {
    val pb = {
      if(System.getenv("NERSC_HOST") != null) {
        throw new NotImplementedError()
        val hostfilePath = s"${System.getenv("SPARK_WORKER_DIR")}/slaves.alchemist"
        new ProcessBuilder("srun", "-O", "-I30", "-N", "2", "-w", hostfilePath, "core/target/alchemist")
      } else {
        new ProcessBuilder("mpirun", "-q", "-np", "4", "core/target/alchemist")
      }
    }
    pb.redirectError(ProcessBuilder.Redirect.INHERIT).start
  }

  val client = new DriverClient(driverProc.getInputStream, driverProc.getOutputStream)

  def stop(): Unit = {
    client.shutdown
    driverProc.waitFor
  }
}

class AlContext(driver: DriverClient) extends Serializable {
  val workerIds: Array[WorkerId] = driver.workerIds
  val workerInfo: Array[WorkerInfo] = driver.workerInfo

  def connectWorker(worker: WorkerId): WorkerClient = {
    workerInfo(worker.id).newClient
  }
}

class Alchemist(val mysc: SparkContext) {
  System.err.println("launching alchemist")
  val sc : SparkContext = mysc

  val driver = new Driver()
  val client = driver.client
  val context = new AlContext(client)

  def stop(): Unit = {
    driver.stop
  }

  def matMul(matA: AlMatrix, matB: AlMatrix) : AlMatrix = {
    val handle = client.matrixMulStart(matA.handle, matB.handle)
    client.matrixMulFinish(handle)
    new AlMatrix(this, handle)
  }
}

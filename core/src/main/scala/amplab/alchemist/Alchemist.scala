package amplab.alchemist
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import scala.util.Random

class AlContext extends Serializable {
}

class Alchemist(val sc: SparkContext) {
  System.err.println("launching alchemist")

  val driverProc: Process = {
    val pb = {
      if(System.getenv("NERSC_HOST") != null) {
        val hostfilePath = s"${System.getenv("SPARK_WORKER_DIR")}/slaves.alchemist"
        new ProcessBuilder("srun", "-O", "-I30", "-N", "2", "-w", hostfilePath, "core/target/alchemist")
      } else {
        new ProcessBuilder("mpirun", "-np", "4", "core/target/alchemist")
      }
    }
    pb.redirectError(ProcessBuilder.Redirect.INHERIT).start
  }

  val driverEndpoint: (String, Int) = {
    val input = new java.io.DataInputStream(driverProc.getInputStream)
    System.err.println("reading alchemist port")
    val hostnameLen = input.readInt
    val hostnameBuf = new Array[Byte](hostnameLen)
    input.readFully(hostnameBuf)
    val hostname = new String(hostnameBuf, "US-ASCII")
    val port = input.readInt
    (hostname, port)
  }

  def stop(): Unit = {
    //driver.exit
    driverProc.waitFor
  }
}

package utils

// breeze
import breeze.linalg._
import breeze.numerics._
// MLlib
import org.apache.spark.mllib.linalg.SparseVector
// others
import scala.math


/*
def randFeature(matX, s, sigma=None):
    n, d = matX.shape
    if sigma is None:
        idx = numpy.random.choice(n, 500, replace=False)
        sigma = estimate_param(matX[idx, :], matX[idx, :])
    #k = int(numpy.ceil(s * 0.8)) # can be tuned
    matW = numpy.random.standard_normal((d, s)) / sigma
    vecV = numpy.random.rand(1, s) * 2 * numpy.pi
    matL = numpy.dot(matX, matW) + vecV
    del matW
    matL = numpy.cos(matL) * numpy.sqrt(2/s)
    return matL

*/
object Kernel {
    def rbfRfm(yx: Iterator[(Double, Array[Double])], numFeature: Int, sigma: Double): Iterator[(Double, Array[Double])] = {
        val yxArray: Array[(Double, Array[Double])] = yx.toArray
        val n: Int = yxArray.length
        val d: Int = yxArray(0)._2.length
        
        // z: random Gaussian matrix
        val seed1: Int = 1234
        val rand1: scala.util.Random = new scala.util.Random(seed1: Int)
        val z: DenseMatrix[Double] = DenseMatrix.zeros[Double](numFeature, d)
        for (j <- 0 until d){
            for (i <- 0 until numFeature) {
                z(i, j) = rand1.nextGaussian() / sigma
            }
        }
        
        // v: random vector
        val seed2: Int = 4321
        val rand2: scala.util.Random = new scala.util.Random(seed2: Int)
        val v: DenseVector[Double] = DenseVector.zeros[Double](numFeature)
        for (j <- 0 until numFeature) v(j) = rand2.nextDouble() * 6.283185307179586
        
        // random feature mapping
        val const: Double = math.sqrt(2.0 / numFeature.toDouble)
        val buffer: Array[(Double, Array[Double])] = new Array[(Double, Array[Double])](n)
        for (j <- 0 until n) {
            val pair: (Double, Array[Double]) = yxArray(j)
            val x: DenseVector[Double] = new DenseVector(pair._2)
            val feat: Array[Double] = (z * x + v).toArray.map((a: Double) => math.cos(a) * const)
            buffer(j) = (pair._1, feat)
        }
        
        buffer.toIterator
    }

    def estimateSigma(yxArray: Array[(Double, Array[Double])]): Double = {
        val n: Int = if(yxArray.length < 500) yxArray.length else 500 
        val d: Int = yxArray(0)._2.length
        val x: Array[DenseVector[Double]] = new Array[DenseVector[Double]](n)
        var a: Double = 0.0
        var b: Double = 0.0
        val n2inv: Double = 1.0 / (n * n).toDouble
        for (j <- 0 until n) {
            x(j) = new DenseVector(yxArray(j)._2)
        }
        for (j <- 0 until n) {
            val xj: DenseVector[Double] = x(j)
            for (i <- 0 until n) {
                b = (xj - x(i)).toArray.map(a => a*a).sum
                a += b * n2inv
            }
        }
        a
    }

    
    
    def hashRfm(yx: Iterator[(Double, SparseVector)], numFeature: Int): Iterator[(Double, Array[Double])] = {
        val yxArray: Array[(Double, SparseVector)] = yx.toArray
        val n: Int = yxArray.size
        val d: Int = yxArray(0)._2.size
        
        // h: d-dim vector with entries drawn from {0,...,numFeature}
        val seed1: Int = 1234
        val rand1: scala.util.Random = new scala.util.Random(seed1: Int)
        val h: Array[Int] = new Array[Int](d)
        for (j <- 0 until d){
            h(j) = rand1.nextInt(numFeature)
        }
        val s: Array[Int] = new Array[Int](d)
        for (j <- 0 until d){
            s(j) = rand1.nextInt(2) * 2 - 1
        }
        
        def randhash(arr: SparseVector): Array[Double] = {
            val idxOld: Array[Int] = arr.indices
            val valOld: Array[Double] = arr.values
            val result: Array[Double] = new Array[Double](numFeature)
            val nnz: Int = idxOld.size
            for (l <- 0 until nnz) {
                val j: Int = idxOld(l)
                val i: Int = h(j)
                result(i) += valOld(l) * s(j)
            }
            result
        }
        
        val buffer: Array[(Double, Array[Double])] = yxArray.map(pair => (pair._1, randhash(pair._2)))
        
        buffer.toIterator
    }
    
}
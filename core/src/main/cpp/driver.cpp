#include "alchemist.h"
#include "data_stream.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sys/stat.h>
#include <map>
#include <random>
#include <sstream>
#include <boost/asio.hpp>
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <boost/tokenizer.hpp>
#include "arpackpp/arrssym.h"
#include "spdlog/spdlog.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace alchemist {

using namespace El;

struct Driver {
  mpi::communicator world;
  DataInputStream input;
  DataOutputStream output;
  std::vector<WorkerInfo> workers;
  std::map<MatrixHandle, MatrixDescriptor> matrices;
  uint32_t nextMatrixId;
  std::shared_ptr<spdlog::logger> log;

  Driver(const mpi::communicator &world, std::istream &is, std::ostream &os, std::shared_ptr<spdlog::logger> log);
  void issue(const Command &cmd);
  MatrixHandle registerMatrix(size_t numRows, size_t numCols);
  int main();

  void handle_newMatrix();
  void handle_matrixMul();
  void handle_matrixDims();
  void handle_computeThinSVD();
  void handle_getMatrixRows();
  void handle_getTranspose();
  void handle_kmeansClustering();
  void handle_truncatedSVD();
  void handle_readHDF5();
};

Driver::Driver(const mpi::communicator &world, std::istream &is, std::ostream &os, std::shared_ptr<spdlog::logger> log) :
    world(world), input(is), output(os), log(log), nextMatrixId(42) {
}

void Driver::issue(const Command &cmd) {
  const Command *cmdptr = &cmd;
  mpi::broadcast(world, cmdptr, 0);
}

MatrixHandle Driver::registerMatrix(size_t numRows, size_t numCols) {
  MatrixHandle handle{nextMatrixId++};
  MatrixDescriptor info(handle, numRows, numCols);
  matrices.insert(std::make_pair(handle, info));
  return handle;
}

int Driver::main() {
  //log to console as well as file (single-threaded logging)
  //TODO: allow to specify log directory, log level, etc.
  std::vector<spdlog::sink_ptr> sinks;
  sinks.push_back(std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>());
  sinks.push_back(std::make_shared<spdlog::sinks::simple_file_sink_st>("driver.log"));
  log = std::make_shared<spdlog::logger>("driver", std::begin(sinks), std::end(sinks));
  log->flush_on(spdlog::level::info); // flush whenever warning or more critical message is logged
  log->set_level(spdlog::level::info); // only log stuff at or above info level, for production
  log->info("Started Driver");

  // get WorkerInfo
  auto numWorkers = world.size() - 1;
  workers.resize(numWorkers);
  for(auto id = 0; id < numWorkers; ++id) {
    world.recv(id + 1, 0, workers[id]);
  }
  log->info("{} workers ready, sending hostnames and ports to Spark", numWorkers);

  // handshake
  ENSURE(input.readInt() == 0xABCD);
  ENSURE(input.readInt() == 0x1);
  output.writeInt(0xDCBA);
  output.writeInt(0x1);
  output.writeInt(numWorkers);
  for(auto id = 0; id < numWorkers; ++id) {
    output.writeString(workers[id].hostname);
    output.writeInt(workers[id].port);
  }
  output.flush();

  bool shouldExit = false;
  while(!shouldExit) {
    uint32_t typeCode = input.readInt();
    log->info("Received code {:#x}", typeCode);

    switch(typeCode) {
      // shutdown
      case 0xFFFFFFFF:
        shouldExit = true;
        issue(HaltCommand());
        output.writeInt(0x1);
        output.flush();
        break;

      // new matrix
      case 0x1:
        handle_newMatrix();
        break;

      // matrix multiplication
      case 0x2:
        handle_matrixMul();
        break;

      // get matrix dimensions
      case 0x3:
        handle_matrixDims();
        break;

      // return matrix to Spark
      case 0x4:
        handle_getMatrixRows();
        break;

      case 0x5:
        handle_computeThinSVD();
        break;

      case 0x6:
        handle_getTranspose();
        break;

      case 0x7:
        handle_kmeansClustering();
        break;

      case 0x8:
        handle_truncatedSVD();
        break;

      case 0x9:
        handle_readHDF5();
        break;

      default:
        log->error("Unknown typeCode {#x}", typeCode);
        abort();
    }
    log->info("Waiting on next command");
  }

  // wait for workers to reach exit
  world.barrier();
  return EXIT_SUCCESS;
}

// A = USV
/*
void Driver::handle_computeLowrankSVD() {
  uint32_t inputMat = input.readInt();
  uint32_t kRank = input.readInt();
  uint32_t whichFactors = input.readInt(); // 0 = U, 1 = U and S, 2 = V and S, 3/default = U, S, and V

  MatrixHandle U;
  MatrixHandle S;
  MatrixHandle V;

  // TODO: register the U, S, V factors with false newmatrixcommands to track them
  switch(whichFactors) {
    case 0: U = MatrixHandle{nextMatrixId++};
            break;
    case 1: U = MatrixHandle{nextMatrixId++};
            S = MatrixHandle{nextMatrixId++};
            break;
    case 2: V = MatrixHandle{nextMatrixId++};
            S = MatrixHandle{nextMatrixId++};
            break;
    default: U = MatrixHandle{nextMatrixId++};
             S = MatrixHandle{nextMatrixId++};
             V = MatrixHandle{nextMatrixId++};
             break;
  }

  LowrankSVDCommand cmd(MatrixHandle{inputMat}, whichFactors, krank, U, S, V);
  issue(cmd);

  output.writeInt(0x1); // statusCode
  switch(whichFactors) {
    case 0: output.writeInt(U.id);
            break;
    case 1: output.writeInt(U.id);
            output.writeInt(S.id);
            break;
    case 2: output.writeInt(V.id);
            output.writeInt(S.id);
            break;
    default: output.writeInt(U.id);
             output.writeInt(S.id);
             output.writeInt(V.id);
             break;
  }
  output.flush();

  // wait for command to finish
  world.barrier();
  output.writeInt(0x1);
  output.flush();
}
*/

// TODO: the cluster centers should be stored locally on driver and reduced/broadcasted. the current
// way of updating kmeans centers is ridiculous
// TODO: currently only implements kmeans||
void Driver::handle_kmeansClustering() {
  MatrixHandle inputMat{input.readInt()};
  uint32_t numCenters = input.readInt();
  uint32_t maxnumIters = input.readInt(); // how many iteration of Lloyd's algorithm to use
  uint32_t initSteps = input.readInt(); // number of initialization steps to use in kmeans||
  double changeThreshold = input.readDouble(); // if all the centers change by Euclidean distance less than changeThreshold, then we stop the iterations
  uint32_t method = input.readInt(); // which initialization method to use to choose initial cluster center guesses
  uint64_t seed = input.readLong(); // randomness seed used in driver and workers

  log->info("Starting K-means on matrix {}", inputMat);
  log->info("numCenters = {}, maxnumIters = {}, initSteps = {}, changeThreshold = {}, method = {}, seed = {}",
      numCenters, maxnumIters, initSteps, changeThreshold, method, seed);

  if (changeThreshold < 0.0 || changeThreshold > 1.0) {
    log->error("Unreasonable change threshold in k-means: {}", changeThreshold);
    abort();
  }
  if (method != 1) {
    log->warn("Sorry, only k-means|| initialization has been implemented, so ignoring your choice of method {}", method);
  }

  auto n = matrices[inputMat].numRows;
  auto d = matrices[inputMat].numCols;
  MatrixHandle centersHandle = this->registerMatrix(numCenters, d);
  MatrixHandle assignmentsHandle = this->registerMatrix(n, 1);
  KMeansCommand cmd(inputMat, numCenters, method, initSteps, changeThreshold, seed, centersHandle, assignmentsHandle);
  issue(cmd); // initial call initializes stuff and waits for next command

  /******** START of kmeans|| initialization ********/
  std::mt19937 gen(seed);
  std::uniform_int_distribution<unsigned long> dis(0, n-1);
  uint32_t rowidx = dis(gen);
  std::vector<double> initialCenter(d);

  mpi::broadcast(world, rowidx, 0); // tell the workers which row to use as initialization in kmeans||
  world.barrier(); // wait for workers to return oversampled cluster centers and sizes

  std::vector<uint32_t> clusterSizes;
  std::vector<MatrixXd> initClusterCenters;
  world.recv(1, mpi::any_tag, clusterSizes);
  world.recv(1, mpi::any_tag, initClusterCenters);
  world.barrier();

  log->info("Retrieved the k-means|| oversized set of potential cluster centers");
  log->debug("{}", initClusterCenters);

  // use kmeans++ locally to find the initial cluster centers
  std::vector<double> weights;
  weights.reserve(clusterSizes.size());
  std::for_each(clusterSizes.begin(), clusterSizes.end(), [&weights](const uint32_t & cnt){ weights.push_back((double) cnt); });
  MatrixXd clusterCenters(numCenters, d);

  kmeansPP(gen(), initClusterCenters, weights, clusterCenters, 30); // same number of maxIters as spark kmeans

  log->info("Ran local k-means on the driver to determine starting cluster centers");
  log->debug("{}", clusterCenters);

  mpi::broadcast(world, clusterCenters.data(), numCenters*d, 0);
  /******** END of kMeans|| initialization ********/

  /******** START of Lloyd's algorithm iterations ********/
  double percentAssignmentsChanged = 1.0;
  bool centersMovedQ = true;
  uint32_t numChanged = 0;
  uint32_t numIters = 0;
  std::vector<uint32_t> parClusterSizes(numCenters);
  std::vector<uint32_t> zerosVector(numCenters);

  for(uint32_t clusterIdx = 0; clusterIdx < numCenters; clusterIdx++)
    zerosVector[clusterIdx] = 0;

  uint32_t command = 1; // do another iteration
  while (centersMovedQ && numIters++ < maxnumIters)  {
    log->info("Starting iteration {} of Lloyd's algorithm, {} percentage changed in last iter",
        numIters, percentAssignmentsChanged*100);
    numChanged = 0;
    for(uint32_t clusterIdx = 0; clusterIdx < numCenters; clusterIdx++)
      parClusterSizes[clusterIdx] = 0;
    command = 1; // do a basic iteration
    mpi::broadcast(world, command, 0);
    mpi::reduce(world, (uint32_t) 0, numChanged, std::plus<int>(), 0);
    mpi::reduce(world, zerosVector.data(), numCenters, parClusterSizes.data(), std::plus<uint32_t>(), 0);
    world.recv(1, mpi::any_tag, centersMovedQ);
    percentAssignmentsChanged = ((double) numChanged)/n;

    for(uint32_t clusterIdx = 0; clusterIdx < numCenters; clusterIdx++) {
      if (parClusterSizes[clusterIdx] == 0) {
        // this is an empty cluster, so randomly pick a point in the dataset
        // as that cluster's centroid
        centersMovedQ = true;
        command = 2; // reinitialize this cluster center
        uint32_t rowIdx = dis(gen);
        mpi::broadcast(world, command, 0);
        mpi::broadcast(world, clusterIdx, 0);
        mpi::broadcast(world, rowIdx, 0);
        world.barrier();
      }
    }

  }
  command = 0xf; // terminate and finalize the k-means centers and assignments as distributed matrices
  mpi::broadcast(world, command, 0);
  double objVal = 0.0;
  mpi::reduce(world, 0.0, objVal, std::plus<double>(), 0);
  world.barrier();

  /******** END of Lloyd's iterations ********/

  log->info("Finished Lloyd's algorithm: took {} iterations, final objective value {}", numIters, objVal);
  output.writeInt(0x1);
  output.writeInt(assignmentsHandle.id);
  output.writeInt(centersHandle.id);
  output.writeInt(numIters);
  output.flush();
}

void Driver::handle_getTranspose() {
  MatrixHandle inputMat{input.readInt()};
  log->info("Constructing the transpose of matrix {}", inputMat);

  auto numRows = matrices[inputMat].numCols;
  auto numCols = matrices[inputMat].numRows;
  MatrixHandle transposeHandle = registerMatrix(numRows, numCols);
  TransposeCommand cmd(inputMat, transposeHandle);
  issue(cmd);

  world.barrier(); // wait for command to finish
  output.writeInt(0x1);
  output.writeInt(transposeHandle.id);
  log->info("Wrote handle for transpose");
  output.writeInt(0x1);
  output.flush();
}

// CAVEAT: Assumes tall-and-skinny for now, doesn't allow many options for controlling
// LIMITATIONS: assumes V small enough to fit on one machine (so can use ARPACK instead of PARPACK), but still distributes U,S,V and does distributed computations not needed
void Driver::handle_truncatedSVD() {
  log->info("Starting truncated SVD computation");
  MatrixHandle inputMat{input.readInt()};
  uint32_t k = input.readInt();

  MatrixHandle UHandle{nextMatrixId++};
  MatrixHandle SHandle{nextMatrixId++};
  MatrixHandle VHandle{nextMatrixId++};

  auto m = matrices[inputMat].numRows;
  auto n = matrices[inputMat].numCols;
  TruncatedSVDCommand cmd(inputMat, UHandle, SHandle, VHandle, k);
  issue(cmd);

  ARrcSymStdEig<double> prob(n, k, "LM");
  uint32_t command;
  std::vector<double> zerosVector(n);
  for(uint32_t idx = 0; idx < n; idx++)
    zerosVector[idx] = 0;

  int iterNum = 0;

  while (!prob.ArnoldiBasisFound()) {
    prob.TakeStep();
    ++iterNum;
    if (prob.GetIdo() == 1 || prob.GetIdo() == -1) {
      command = 1;
      mpi::broadcast(world, command, 0);
      mpi::broadcast(world, prob.GetVector(), n, 0);
      mpi::reduce(world, zerosVector.data(), n, prob.PutVector(), std::plus<double>(), 0);
    }
  }

  prob.FindEigenvectors();
  uint32_t nconv = prob.ConvergedEigenvalues();
  uint32_t niters = prob.GetIter();
  log->info("Done after {} Arnoldi iterations", niters);

  // assuming tall and skinny A for now
  MatrixXd rightVecs(n, nconv);
  // Eigen uses column-major layout by default!
  for(uint32_t idx = 0; idx < nconv; idx++)
    std::memcpy(rightVecs.col(idx).data(), prob.RawEigenvector(idx), n*sizeof(double));

  // Populate U, V, S
  command = 2;
  mpi::broadcast(world, command, 0);
  mpi::broadcast(world, nconv, 0);
  mpi::broadcast(world, rightVecs.data(), n*nconv, 0);
  mpi::broadcast(world, prob.RawEigenvalues(), nconv, 0);

  MatrixDescriptor Uinfo(UHandle, m, nconv);
  MatrixDescriptor Sinfo(SHandle, nconv, 1);
  MatrixDescriptor Vinfo(VHandle, n, nconv);
  ENSURE(matrices.insert(std::make_pair(UHandle, Uinfo)).second);
  ENSURE(matrices.insert(std::make_pair(SHandle, Sinfo)).second);
  ENSURE(matrices.insert(std::make_pair(VHandle, Vinfo)).second);

  world.barrier();
  output.writeInt(0x1);
  output.writeInt(UHandle.id);
  output.writeInt(SHandle.id);
  output.writeInt(VHandle.id);
  output.flush();
}

void Driver::handle_computeThinSVD() {
  MatrixHandle inputMat{input.readInt()};

  // this needs to be done automatically rather than hand-coded. e.g. what if
  // we switch to determining rank by sing-val thresholding instead of doing thin SVD?
  auto m = matrices[inputMat].numRows;
  auto n = matrices[inputMat].numCols;
  auto k = std::min(m,n);
  MatrixHandle Uhandle = registerMatrix(m, k);
  MatrixHandle Shandle = registerMatrix(k, 1);
  MatrixHandle Vhandle = registerMatrix(n, k);
  ThinSVDCommand cmd(inputMat, Uhandle, Shandle, Vhandle);
  issue(cmd);

  output.writeInt(0x1); // statusCode
  output.writeInt(Uhandle.id);
  output.writeInt(Shandle.id);
  output.writeInt(Vhandle.id);
  output.flush();

  // wait for command to finish
  world.barrier();
  log->info("Done with SVD computation");
  output.writeInt(0x1);
  output.flush();
}

void Driver::handle_matrixMul() {
  MatrixHandle matA{input.readInt()};
  MatrixHandle matB{input.readInt()};
  log->info("Multiplying matrices {} and {}", matA, matB);

  auto numRows = matrices[matA].numRows;
  auto numCols = matrices[matB].numCols;
  MatrixHandle destHandle = registerMatrix(numRows, numCols);
  MatrixMulCommand cmd(destHandle, matA, matB);
  issue(cmd);

  // tell spark id of resulting matrix
  output.writeInt(0x1); // statusCode
  output.writeInt(destHandle.id);
  output.flush();

  // wait for it to finish
  world.barrier();
  output.writeInt(0x1);
  output.flush();
}

void Driver::handle_matrixDims() {
  MatrixHandle matrixHandle{input.readInt()};
  auto info = matrices[matrixHandle];
  output.writeInt(0x1);
  output.writeLong(info.numRows);
  output.writeLong(info.numCols);
  output.flush();

}

void Driver::handle_getMatrixRows() {
  MatrixHandle handle{input.readInt()};
  uint64_t layoutLen = input.readLong();
  std::vector<uint32_t> layout;
  layout.reserve(layoutLen);
  for(uint64_t part = 0; part < layoutLen; ++part) {
    layout.push_back(input.readInt());
  }
  log->info("Returning matrix {} to Spark", handle);

  MatrixGetRowsCommand cmd(handle, layout);
  issue(cmd);

  // tell Spark to start asking for rows
  output.writeInt(0x1);
  output.flush();

  // wait for it to finish
  world.barrier();
  output.writeInt(0x1);
  output.flush();
}

// TODO: handle exceptions!
void Driver::handle_readHDF5() {
  // read args
  std::string fname = input.readString();
  std::string varname = input.readString();

  log->info("Attempting to read variable \"{}\" from file \"{}\"", varname, fname);
  ReadHDF5Command cmd(fname, varname);

  uint64_t m, n;

  issue(cmd);
	mpi::broadcast(world, m, 1);
	mpi::broadcast(world, n, 1);
  world.barrier();

  MatrixHandle matHandle{nextMatrixId++};
  MatrixDescriptor matInfo(matHandle, m, n);
	mpi::broadcast(world, matHandle, 0);
  ENSURE(matrices.insert(std::make_pair(matHandle, matInfo)).second);

  output.writeInt(0x1); // success signal
  output.writeInt(matHandle.id);
  output.flush();
}

void Driver::handle_newMatrix() {
  // read args
  uint64_t numRows = input.readLong();
  uint64_t numCols = input.readLong();

  // assign id and notify workers
  MatrixHandle handle = registerMatrix(numRows, numCols);
  NewMatrixCommand cmd(matrices[handle]);
  log->info("Recieving new matrix {}, with dimensions {}x{}", handle, numRows, numCols);
  issue(cmd);

  output.writeInt(0x1);
  output.writeInt(handle.id);
  output.flush();

  // tell spark which worker expects each row
  std::vector<int> rowWorkerAssignments(numRows, 0);
  std::vector<uint64_t> rowsOnWorker;
  for(int workerIdx = 1; workerIdx < world.size(); workerIdx++) {
    world.recv(workerIdx, 0, rowsOnWorker);
    world.barrier();
    for(auto rowIdx: rowsOnWorker) {
      rowWorkerAssignments[rowIdx] = workerIdx;
    }
  }
  log->info("Sending list of which worker each row should go to");
  output.writeInt(0x1); // statusCode
  for(auto workerIdx: rowWorkerAssignments)
    output.writeInt(workerIdx);
  output.flush();

  // wait for spark to finish sending the data over to the alchemist executors ...
  world.barrier();
  output.writeInt(0x1);  // statusCode
  output.flush();
  log->info("Entire matrix has been received");
}

inline bool exist_test (const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0); 
}

int driverMain(const mpi::communicator &world, int argc, char *argv[]) {
  //log to console as well as file (single-threaded logging)
  //TODO: allow to specify log directory, log level, etc.
  std::shared_ptr<spdlog::logger> log;
  std::vector<spdlog::sink_ptr> sinks;
  sinks.push_back(std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>());
  sinks.push_back(std::make_shared<spdlog::sinks::simple_file_sink_st>("driver.log"));
  log = std::make_shared<spdlog::logger>("driver", std::begin(sinks), std::end(sinks));
  //log->flush_on(spdlog::level::warn); // flush whenever warning or more critical message is logged
  //log->set_level(spdlog::level::info); // only log stuff at or above info level, for production
  log->flush_on(spdlog::level::info); // flush always, for debugging
  log->info("Started Driver");

  char machine[255];
  char port[255];

  if (argc == 3) { // we are on a non-NERSC system, so passed in Spark driver machine name and port
      log->info("Non-NERSC system assumed");
      log->info("Connecting to Spark executor at {}:{}", argv[1], argv[2]);
      std::strcpy(machine, argv[1]);
      std::strcpy(port, argv[2]);
  } else { // assume we are on NERSC, so look in a specific location for a file containing the machine name and port
      char const* tmp = std::getenv("SPARK_WORKER_DIR");
      std::string sockPath;
      if (tmp == NULL) {
          log->info("Couldn't find the SPARK_WORKER_DIR variable");
          world.abort(1);
      } else {
        sockPath = std::string(tmp) + "/connection.info";
      }
      log->info("NERSC system assumed");
      log->info("Searching for connection information in file {}", sockPath);

      while(!exist_test(sockPath)) {
          boost::this_thread::sleep_for(boost::chrono::milliseconds(50));
      }
      // now wait for a while for the connection file to be completely written, hopefully is enough time
      // TODO: need a more robust way of ensuring this is the case
      boost::this_thread::sleep_for(boost::chrono::milliseconds(500));

      std::string sockSpec;
      std::ifstream infile(sockPath);
      std::getline(infile, sockSpec);
      infile.close();
      boost::tokenizer<> tok(sockSpec);
      boost::tokenizer<>::iterator iter=tok.begin();
      std::string machineName = *iter;
      std::string portName = *(++iter);

      log->info("Connecting to Spark executor at {}:{}", machineName, portName);
      strcpy(machine, machineName.c_str());
      strcpy(port, portName.c_str());
  }

  using boost::asio::ip::tcp;
  boost::asio::ip::tcp::iostream stream(machine, port);
  ENSURE(stream);
  stream.rdbuf()->non_blocking(false);
  auto result = Driver(world, stream, stream, log).main();
  return result;
}

} // namespace alchemist

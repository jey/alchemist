#include "alchemist.h"
#include "data_stream.h"
#include <iostream>
#include <fstream>
#include <map>
#include <ext/stdio_filebuf.h>
#include "arrssym.h"
#include <random>
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
  std::map<MatrixHandle, NewMatrixCommand> matrices; // need to account for other commands that generate (multiple) matrices 
  uint32_t nextMatrixId;
  std::shared_ptr<spdlog::logger> log;

  Driver(const mpi::communicator &world, std::istream &is, std::ostream &os);
  void issue(const Command &cmd);
  int main();

  void handle_newMatrix();
  void handle_matrixMul();
  void handle_matrixDims();
  void handle_computeThinSVD();
  void handle_getMatrixRows();
  void handle_getTranspose();
  void handle_kmeansClustering();
  void handle_truncatedSVD();
};

Driver::Driver(const mpi::communicator &world, std::istream &is, std::ostream &os) :
    world(world), input(is), output(os), nextMatrixId(42) {
}

void Driver::issue(const Command &cmd) {
  const Command *cmdptr = &cmd;
  mpi::broadcast(world, cmdptr, 0);
}

int Driver::main() {
  //log to console as well as file (single-threaded logging)
  //TODO: allow to specify log directory, log level, etc.
  std::vector<spdlog::sink_ptr> sinks;
  sinks.push_back(std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>());
  sinks.push_back(std::make_shared<spdlog::sinks::simple_file_sink_st>("driver.log"));
  log = std::make_shared<spdlog::logger>("driver", std::begin(sinks), std::end(sinks));
  log->flush_on(spdlog::level::warn); // flush whenever warning or more critical message is logged
  log->set_level(spdlog::level::info); // only log stuff at or above info level, for production
  log->info("Started Driver");

  // get WorkerInfo
  auto numWorkers = world.size() - 1;
  workers.resize(numWorkers);
  for(auto id = 0; id < numWorkers; ++id) {
    world.recv(id + 1, 0, workers[id]);
  }
  log->info("{} workers ready", numWorkers);

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
    log->info("Received code {#x}", typeCode);

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
  uint32_t inputMat = input.readInt();
  uint32_t numCenters = input.readInt();
  uint32_t maxnumIters = input.readInt(); // how many iteration of Lloyd's algorithm to use
  uint32_t initSteps = input.readInt(); // number of initialization steps to use in kmeans||
  double changeThreshold = input.readDouble(); // if all the centers change by Euclidean distance less than changeThreshold, then we stop the iterations
  uint32_t method = input.readInt(); // which initialization method to use to choose initial cluster center guesses
  uint64_t seed = input.readLong(); // randomness seed used in driver and workers

  log->info("Starting K-means on matrix {}", MatrixHandle{inputMat});
  log->info("numCenters = {}, maxnumIters = {}, initSteps = {}, changeThreshold = {}, method = {}, seed = {}",
      numCenters, maxnumIters, initSteps, changeThreshold, method, seed);

  MatrixHandle centersHandle{nextMatrixId++};
  MatrixHandle assignmentsHandle{nextMatrixId++};

  if (changeThreshold < 0.0 || changeThreshold > 1.0) {
    log->error("Unreasonable change threshold in k-means: {}", changeThreshold);
    abort();
  }
  if (method != 1) {
    log->warn("Sorry, only k-means|| initialization has been implemented, so ignoring your choice of method {}", method);
  }

  KMeansCommand cmd(MatrixHandle{inputMat}, numCenters, method, initSteps, changeThreshold, seed, centersHandle, assignmentsHandle);
  std::vector<uint32_t> dummy_layout(1);
  auto n = matrices[MatrixHandle{inputMat}].numRows;
  auto d = matrices[MatrixHandle{inputMat}].numCols;
  NewMatrixCommand centersDummyCmd(centersHandle, numCenters, d, dummy_layout);
  NewMatrixCommand assignmentDummyCmd(assignmentsHandle, n, 1, dummy_layout);
  ENSURE(matrices.insert(std::make_pair(centersHandle, centersDummyCmd)).second);
  ENSURE(matrices.insert(std::make_pair(assignmentsHandle, assignmentDummyCmd)).second);

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
  uint32_t inputMat = input.readInt();
  MatrixHandle transposeHandle{nextMatrixId++};
  log->info("Constructing the transpose of matrix {}", MatrixHandle{inputMat});

  TransposeCommand cmd(MatrixHandle{inputMat}, transposeHandle);
  std::vector<uint32_t> dummy_layout(1);
  auto m = matrices[MatrixHandle{inputMat}].numRows;
  auto n = matrices[MatrixHandle{inputMat}].numCols;
  NewMatrixCommand dummycmd(transposeHandle, n, m, dummy_layout);
  ENSURE(matrices.insert(std::make_pair(transposeHandle, dummycmd)).second);

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
  uint32_t inputMat = input.readInt();
  uint32_t k = input.readInt();

  MatrixHandle UHandle{nextMatrixId++};
  MatrixHandle SHandle{nextMatrixId++};
  MatrixHandle VHandle{nextMatrixId++};

  auto m = matrices[MatrixHandle{inputMat}].numRows;
  auto n = matrices[MatrixHandle{inputMat}].numCols;
  TruncatedSVDCommand cmd(MatrixHandle{inputMat}, UHandle, SHandle, VHandle, k);

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
  
  std::vector<uint32_t> dummy_layout(1);
  NewMatrixCommand dummyUcmd(UHandle, m, nconv, dummy_layout);
  NewMatrixCommand dummyScmd(SHandle, nconv, 1, dummy_layout);
  NewMatrixCommand dummyVcmd(VHandle, n, nconv, dummy_layout);
  ENSURE(matrices.insert(std::make_pair(UHandle, dummyUcmd)).second);
  ENSURE(matrices.insert(std::make_pair(SHandle, dummyScmd)).second);
  ENSURE(matrices.insert(std::make_pair(VHandle, dummyVcmd)).second);

  world.barrier();
  output.writeInt(0x1);
  output.writeInt(UHandle.id);
  output.writeInt(SHandle.id);
  output.writeInt(VHandle.id);
  output.flush();
}

void Driver::handle_computeThinSVD() {
  uint32_t inputMat = input.readInt();

  MatrixHandle Uhandle{nextMatrixId++};
  MatrixHandle Shandle{nextMatrixId++};
  MatrixHandle Vhandle{nextMatrixId++};

  ThinSVDCommand cmd(MatrixHandle{inputMat}, Uhandle, Shandle, Vhandle);
  // this needs to be done automatically rather than hand-coded. e.g. what if
  // we switch to determining rank by sing-val thresholding instead of doing thin SVD?
  std::vector<uint32_t> dummy_layout(1);
  auto m = matrices[MatrixHandle{inputMat}].numRows;
  auto n = matrices[MatrixHandle{inputMat}].numCols;
  auto k = std::min(m,n);
  NewMatrixCommand dummycmdU(Uhandle, m, k, dummy_layout);
  NewMatrixCommand dummycmdS(Shandle, k, 1, dummy_layout);
  NewMatrixCommand dummycmdV(Uhandle, n, k, dummy_layout);
  ENSURE(matrices.insert(std::make_pair(Uhandle, dummycmdU)).second);
  ENSURE(matrices.insert(std::make_pair(Shandle, dummycmdS)).second);
  ENSURE(matrices.insert(std::make_pair(Vhandle, dummycmdV)).second);

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
  uint32_t handleA = input.readInt();
  uint32_t handleB = input.readInt();

  MatrixHandle destHandle{nextMatrixId++};
  MatrixMulCommand cmd(destHandle, MatrixHandle{handleA}, MatrixHandle{handleB});

  // add a dummy newmatrixcommand to track this matrix
  std::vector<uint32_t> dummy_layout(1);
  auto numRows = matrices[MatrixHandle{handleA}].numRows;
  auto numCols = matrices[MatrixHandle{handleB}].numCols;
  NewMatrixCommand dummycmd(destHandle, numRows, numCols, dummy_layout);
  ENSURE(matrices.insert(std::make_pair(destHandle, dummycmd)).second);

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
  uint32_t matrixHandle = input.readInt();
  auto matrixCmd = matrices[MatrixHandle{matrixHandle}];

  output.writeInt(0x1);
  output.writeLong(matrixCmd.numRows);
  output.writeLong(matrixCmd.numCols);
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

  MatrixGetRowsCommand cmd(handle, layout);
  issue(cmd);

//  std::cerr << "Layout for returning matrix: " << std::endl;
//  for (auto i = layout.begin(); i != layout.end(); ++i)
//    std::cerr << *i << " ";
//  std::cerr << std::endl;

  // tell Spark to start asking for rows
  output.writeInt(0x1);
  output.flush();

  // wait for it to finish
  world.barrier();
  output.writeInt(0x1);
  output.flush();
}

void Driver::handle_newMatrix() {
  // read args
  uint64_t numRows = input.readLong();
  uint64_t numCols = input.readLong();
  uint64_t layoutLen = input.readLong();
  std::vector<uint32_t> layout;
  layout.reserve(layoutLen);
  for(uint64_t part = 0; part < layoutLen; ++part) {
    layout.push_back(input.readInt());
  }

  // assign id and notify workers
  MatrixHandle handle{nextMatrixId++};
  NewMatrixCommand cmd(handle, numRows, numCols, layout);
  ENSURE(matrices.insert(std::make_pair(handle, cmd)).second);
  issue(cmd);

  // tell spark to start loading
  output.writeInt(0x1);  // statusCode
  output.writeInt(handle.id);
  output.flush();

  // wait for it to finish...
  world.barrier();
  output.writeInt(0x1);  // statusCode
  output.flush();
}

int driverMain(const mpi::communicator &world) {
  int outfd = ::dup(1);
  ENSURE(::dup2(2, 1) == 1);
  __gnu_cxx::stdio_filebuf<char> outbuf(outfd, std::ios::out);
  std::ostream output(&outbuf);
  auto result = Driver(world, std::cin, output).main();
  output.flush();
  ::close(outfd);
  return result;
}

} // namespace alchemist

#include "alchemist.h"
#include "data_stream.h"
#include <iostream>
#include <fstream>
#include <map>
#include <ext/stdio_filebuf.h>

namespace alchemist {

using namespace El;

struct Driver {
  mpi::communicator world;
  DataInputStream input;
  DataOutputStream output;
  std::vector<WorkerInfo> workers;
  std::map<MatrixHandle, NewMatrixCommand> matrices; // need to account for other commands that generate (multiple) matrices 
  uint32_t nextMatrixId;

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
};

Driver::Driver(const mpi::communicator &world, std::istream &is, std::ostream &os) :
    world(world), input(is), output(os), nextMatrixId(42) {
}

void Driver::issue(const Command &cmd) {
  const Command *cmdptr = &cmd;
  mpi::broadcast(world, cmdptr, 0);
}

int Driver::main() {
  // get WorkerInfo
  auto numWorkers = world.size() - 1;
  workers.resize(numWorkers);
  for(auto id = 0; id < numWorkers; ++id) {
    world.recv(id + 1, 0, workers[id]);
  }
  std::cerr << "AlDriver: workers ready" << std::endl;

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

      default:
        std::cerr << "Unknown typeCode: " << std::hex << typeCode << std::endl;
        abort();
    }
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
void Driver::handle_kmeansClustering() {
  uint32_t inputMat = input.readInt();
  uint32_t numCenters = input.readInt();
  uint32_t maxnumIters = input.readInt();
  uint32_t changeThreshold = input.readDouble(); // terminate if no more than changeThreshold percentage of the points change membership in an iteration
  MatrixHandle centersHandle{nextMatrixId++};
  MatrixHandle assignmentsHandle{nextMatrixId++};

  if (changeThreshold < 0.0 || changeThreshold > 1.0) {
      std::cerr << "unreasonable change threshold in k-means: " << changeThreshold << std::endl;
      abort();
  }

  KMeansCommand cmd(MatrixHandle{inputMat}, numCenters, 0,
      centersHandle, assignmentsHandle);
  std::vector<uint32_t> dummy_layout(1);
  auto n = matrices[MatrixHandle{inputMat}].numRows;
  auto d = matrices[MatrixHandle{inputMat}].numCols;
  NewMatrixCommand centersDummyCmd(centersHandle, numCenters, d, dummy_layout);
  NewMatrixCommand assignmentDummyCmd(assignmentsHandle, n, 1, dummy_layout);
  ENSURE(matrices.insert(std::make_pair(centersHandle, centersDummyCmd)).second);
  ENSURE(matrices.insert(std::make_pair(assignmentsHandle, assignmentDummyCmd)).second);

  issue(cmd); // initial call initializes stuff and waits for next command
  double percentMoved = 1.0;
  uint32_t numChanged = 0;
  uint32_t numIters = 0;

  uint32_t command = 1; // do another iteration
  while (percentMoved > changeThreshold && numIters++ < maxnumIters)  {
    numChanged = 0;
    mpi::broadcast(world, command, 0);
    std::cerr << "about to reduce\n";
    uint32_t mychanged = 0;
    mpi::reduce(world, mychanged, numChanged, std::plus<int>(), 0);
    percentMoved = ((double) numChanged)/n;
    std::cerr << format("driver: on iteration %d of Lloyd's algorithm, %f percentage changed\n") % numIters % percentMoved;
  }
  command = 0xf; // terminate and finalize the k-means centers and assignments as distributed matrices
  mpi::broadcast(world, command, 0);
  world.barrier();

  output.writeInt(0x1);
  output.writeInt(assignmentsHandle.id);
  output.writeInt(centersHandle.id);
  output.writeInt(numIters);
  output.writeDouble(percentMoved);
  output.flush();
}

void Driver::handle_getTranspose() {
  uint32_t inputMat = input.readInt();
  MatrixHandle transposeHandle{nextMatrixId++};

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
  std::cerr << "wrote handle" << std::endl;
  output.writeInt(0x1);
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
  std::cerr << "Done with SVD computation" << std::endl;
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

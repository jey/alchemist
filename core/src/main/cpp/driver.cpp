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
  std::map<MatrixHandle, NewMatrixCommand> matrices;
  uint32_t nextMatrixId;

  Driver(const mpi::communicator &world, std::istream &is, std::ostream &os);
  void issue(const Command &cmd);
  int main();

  void handle_newMatrix();
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

      default:
        std::cerr << "Unknown typeCode: " << std::hex << typeCode << std::endl;
        abort();
    }
  }

  // wait for workers to reach exit
  world.barrier();
  return EXIT_SUCCESS;
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

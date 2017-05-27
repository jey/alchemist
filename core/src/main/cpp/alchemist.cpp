#include "alchemist.h"

int main(int argc, char *argv[]) {
  using namespace alchemist;
  mpi::environment env(argc, argv);
  mpi::communicator world;
  El::Initialize();
  bool isDriver = world.rank() == 0;
  mpi::communicator peers = world.split(isDriver ? 0 : 1);
  auto status = isDriver ? driverMain(world) : workerMain(world, peers);
  El::Finalize();
  return status;
}

BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::Command);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::HaltCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::NewMatrixCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::MatrixMulCommand)

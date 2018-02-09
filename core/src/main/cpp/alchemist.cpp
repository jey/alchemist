#include "alchemist.h"

int main(int argc, char *argv[]) {
  using namespace alchemist;
  mpi::environment env(argc, argv);
  mpi::communicator world;
  El::Initialize();
  bool isDriver = world.rank() == 0;
  mpi::communicator peers = world.split(isDriver ? 0 : 1);
  auto status = isDriver ? driverMain(world, argc, argv) : workerMain(world, peers);
  El::Finalize();
  return status;
}

BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::MatrixDescriptor);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::Command);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::HaltCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::NewMatrixCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::MatrixMulCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::MatrixGetRowsCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::ThinSVDCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::TransposeCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::KMeansCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::TruncatedSVDCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::SkylarkKernelSolverCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::SkylarkLSQRSolverCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::FactorizedCGSolverCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::RandomFourierFeaturesCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::ReadHDF5Command);


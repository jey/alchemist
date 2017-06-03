#ifndef ALCHEMIST__ALCHEMIST_H
#define ALCHEMIST__ALCHEMIST_H

#include <El.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/format.hpp>
#include <boost/mpi.hpp>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <unistd.h>

#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#ifndef NDEBUG
#define ENSURE(x) assert(x)
#else
#define ENSURE(x) do { if(!(x)) abort(); } while(0)
#endif

namespace alchemist {

namespace mpi = boost::mpi;
namespace serialization = boost::serialization;
using boost::format;

typedef El::Matrix<double> Matrix;
typedef El::AbstractDistMatrix<double> DistMatrix;
typedef uint32_t WorkerId;

struct Worker;

struct MatrixHandle {
  uint32_t id;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & id;
  }
};

inline bool operator < (const MatrixHandle &lhs, const MatrixHandle &rhs) {
  return lhs.id < rhs.id;
}

struct WorkerInfo {
  std::string hostname;
  uint32_t port;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & hostname;
    ar & port;
  }
};

struct Command {
  virtual ~Command() {
  }

  virtual void run(Worker *self) const = 0;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
  }
};

struct HaltCommand : Command {
  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
  }
};

struct MatrixMulCommand : Command {
  MatrixHandle handle;
  MatrixHandle inputA;
  MatrixHandle inputB;

  explicit MatrixMulCommand() {}

  MatrixMulCommand(MatrixHandle dest, MatrixHandle A, MatrixHandle B) :
    handle(dest), inputA(A), inputB(B) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & handle;
    ar & inputA;
    ar & inputB;
  }
};

struct MatrixGetRowsCommand : Command {
  MatrixHandle handle;
  std::vector<WorkerId> layout;

  explicit MatrixGetRowsCommand() : {}

  MatrixGetRowsCommand(MatrixHandle handle, std::vector<WorkerId> layout) : 
    handle(handle), layout(layout) {}

  virtual void run(Worker * self) const;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & handle;
    ar & layout;
  }
};

struct NewMatrixCommand : Command {
  MatrixHandle handle;
  size_t numRows;
  size_t numCols;
  std::vector<WorkerId> layout;

  explicit NewMatrixCommand() :
      numRows(0), numCols(0) {
  }

  NewMatrixCommand(MatrixHandle handle, size_t numRows, size_t numCols,
        const std::vector<WorkerId> &layout) :
      handle(handle), numRows(numRows), numCols(numCols),
      layout(layout) {
  }

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & handle;
    ar & numRows;
    ar & numCols;
    ar & layout;
  }
};

int driverMain(const mpi::communicator &world);
int workerMain(const mpi::communicator &world, const mpi::communicator &peers);

} // namespace alchemist

BOOST_CLASS_EXPORT_KEY(alchemist::Command);
BOOST_CLASS_EXPORT_KEY(alchemist::HaltCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::NewMatrixCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::MatrixMulCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::MatrixGetRowsCommand);

#endif

#ifndef ALCHEMIST__ALCHEMIST_H
#define ALCHEMIST__ALCHEMIST_H

#include <El.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/format.hpp>
#include <boost/mpi.hpp>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <unistd.h>
#include <arpa/inet.h>
#include <eigen3/Eigen/Dense>
#include "spdlog/fmt/fmt.h"
// #include "spdlog/fmt/ostr.h"
#include "endian.h"

#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#ifndef NDEBUG
#define ENSURE(x) assert(x)
#else
#define ENSURE(x) do { if(!(x)) { \
  fprintf(stderr, "FATAL: invariant violated: %s:%d: %s\n", __FILE__, __LINE__, #x); fflush(stderr); abort(); } while(0)
#endif

namespace alchemist {

namespace serialization = boost::serialization;
namespace mpi = boost::mpi;
using boost::format;

typedef El::Matrix<double> Matrix;
typedef El::AbstractDistMatrix<double> DistMatrix;
typedef uint32_t WorkerId;


void kmeansPP(uint32_t seed, std::vector<Eigen::MatrixXd> points, std::vector<double> weights, 
    Eigen::MatrixXd & fitCenters, uint32_t maxIters);

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


/*
struct ThinSVDCommand : Command {
  MatrixHandle mat;
  uint32_t whichFactors;
  uint32_t krank;
  MatrixHandle U;
  MatrixHandle S;
  MatrixHandle V;

  explicit ThinSVDCommand() {}

  ThinSVDCommand(MatrixHandle mat, uint32_t whichFactors, uint32_t krank,
      MatrixHandle U, MatrixHandle S, MatrixHandle V) :
    mat(mat), whichFactors(whichFactors), krank(krank), U(U), S(S), V(V) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_oject<Command>(*this);
    ar & mat;
    ar & whichFactors;
    ar & krank;
    ar & U;
    ar & S;
    ar & V;
  }
}
*/

struct TransposeCommand : Command {
  MatrixHandle origMat;
  MatrixHandle transposeMat;

  explicit TransposeCommand() {}

  TransposeCommand(MatrixHandle origMat, MatrixHandle transposeMat) :
    origMat(origMat), transposeMat(transposeMat) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & origMat;
    ar & transposeMat;
  }
};

struct KMeansCommand : Command {
  MatrixHandle origMat;
  uint32_t numCenters;
  uint32_t initSteps; // relevant in k-means|| only
  double changeThreshold; // stop when all centers change by Euclidean distance less than changeThreshold
  uint32_t method;
  uint64_t seed;
  MatrixHandle centersHandle;
  MatrixHandle assignmentsHandle;

  explicit KMeansCommand() {}

  KMeansCommand(MatrixHandle origMat, uint32_t numCenters, uint32_t method,
      uint32_t initSteps, double changeThreshold, uint64_t seed, 
      MatrixHandle centersHandle, MatrixHandle assignmentsHandle) :
    origMat(origMat), numCenters(numCenters), method(method),
    initSteps(initSteps), changeThreshold(changeThreshold), 
    seed(seed), centersHandle(centersHandle), assignmentsHandle(assignmentsHandle) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & origMat;
    ar & numCenters;
    ar & initSteps;
    ar & changeThreshold;
    ar & method;
    ar & seed,
    ar & centersHandle;
    ar & assignmentsHandle;
  }
};

struct TruncatedSVDCommand : Command {
  MatrixHandle mat;
  MatrixHandle UHandle;
  MatrixHandle SHandle;
  MatrixHandle VHandle;
  uint32_t k;

  explicit TruncatedSVDCommand() {}

  TruncatedSVDCommand(MatrixHandle mat, MatrixHandle UHandle, 
      MatrixHandle SHandle, MatrixHandle VHandle, uint32_t k) :
    mat(mat), UHandle(UHandle), SHandle(SHandle), VHandle(VHandle),
    k(k) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & mat;
    ar & UHandle;
    ar & SHandle;
    ar & VHandle;
    ar & k;
  }
};

struct ThinSVDCommand : Command {
  MatrixHandle mat;
  MatrixHandle Uhandle;
  MatrixHandle Shandle;
  MatrixHandle Vhandle;

  explicit ThinSVDCommand() {}

  ThinSVDCommand(MatrixHandle mat, MatrixHandle Uhandle, 
      MatrixHandle Shandle, MatrixHandle Vhandle) :
    mat(mat), Uhandle(Uhandle), Shandle(Shandle), Vhandle(Vhandle) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & mat;
    ar & Uhandle;
    ar & Shandle;
    ar & Vhandle;
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

  explicit MatrixGetRowsCommand() {}

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

namespace fmt {
  // for displaying Eigen expressions. Note, if you include spdlog/fmt/ostr.h, this will be 
  // hidden by the ostream<< function for Eigen objects
  template <typename Formatter, typename Derived>
  inline void format_arg(Formatter &f,
      const char *&format_str, const Eigen::MatrixBase<Derived> &exp) {
    std::stringstream buf;
    buf << "Eigen matrix " << std::endl << exp;
    f.writer().write("{}", buf.str()); 
  }

  template <typename Formatter> 
  inline void format_arg(Formatter &f,
      const char *&format_str, const Eigen::Matrix<double, -1, -1> &exp) {
    std::stringstream buf;
    buf << "Eigen matrix " << std::endl << exp;
    f.writer().write("{}", buf.str()); 
  }

  // for displaying vectors
  template <typename T, typename A>
  inline void format_arg(BasicFormatter<char> &f, 
      const char *&format_str, const std::vector<T,A> &vec) {
    std::stringstream buf;
    buf << "Vector of length " << vec.size() << std::endl << "{";
    for(typename std::vector<T>::size_type pos=0; pos < vec.size()-1; ++pos) {
      buf << vec[pos] << "," << std::endl;
    }
    buf << vec[vec.size()-1] << "}";
    f.writer().write("{}", buf.str());
  }

  inline void format_arg(BasicFormatter<char> &f,
      const char *&format_str, const alchemist::MatrixHandle &handle) {
    f.writer().write("[{}]", handle.id);
  }
}

namespace boost { namespace serialization {
  // to serialize Eigen Matrix objects
	template< class Archive,
						class S,
						int Rows_,
						int Cols_,
						int Ops_,
						int MaxRows_,
						int MaxCols_>
	inline void serialize(Archive & ar, 
		Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> & matrix, 
		const unsigned int version)
	{
		int rows = matrix.rows();
		int cols = matrix.cols();
		ar & make_nvp("rows", rows);
		ar & make_nvp("cols", cols);    
		matrix.resize(rows, cols); // no-op if size does not change!

		// always save/load col-major
		for(int c = 0; c < cols; ++c)
			for(int r = 0; r < rows; ++r)
				ar & make_nvp("val", matrix(r,c));
	}
}} // namespace boost::serialization

BOOST_CLASS_EXPORT_KEY(alchemist::Command);
BOOST_CLASS_EXPORT_KEY(alchemist::HaltCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::NewMatrixCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::MatrixMulCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::MatrixGetRowsCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::ThinSVDCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::TransposeCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::KMeansCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::TruncatedSVDCommand);

#endif

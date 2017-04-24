#ifndef ALCHEMIST__ALCHEMIST_H
#define ALCHEMIST__ALCHEMIST_H

#include <El.hpp>

#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#ifndef NDEBUG
#define ENSURE(x) assert(x)
#else
#define ENSURE(x) do { if(!(x)) abort(); } while(0)
#endif

namespace alchemist {

typedef El::Matrix<double> Matrix;
typedef El::AbstractDistMatrix<double> DistMatrix;

} // namespace alchemist

#endif

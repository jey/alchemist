#ifndef ALCHEMIST__DATA_STREAM_H
#define ALCHEMIST__DATA_STREAM_H

#include "alchemist.h"
#include <iostream>

namespace alchemist {

union DoubleBytes {
  uint64_t qword;
  double value;
};

inline double ntohd(uint64_t qword) {
  DoubleBytes d;
  d.qword = ntohll(qword);
  return d.value;
}

inline uint64_t htond(double value) {
  DoubleBytes d;
  d.value = value;
  return htonll(d.qword);
}

struct DataInputStream {
  std::istream &is;

  struct IOError : std::runtime_error {
    IOError() :
      std::runtime_error("DataInputStream::IOError") {
    }
  };

  DataInputStream(std::istream &is) :
      is(is) {
  }

  uint32_t readInt() {
    uint32_t val;
    is.read(reinterpret_cast<char*>(&val), sizeof(val));
    if(!is) throw IOError();
    val = ntohl(val);
    return val;
  }

  uint64_t readLong() {
    uint64_t val;
    is.read(reinterpret_cast<char*>(&val), sizeof(val));
    if(!is) throw IOError();
    val = ntohll(val);
    return val;
  }
};

struct DataOutputStream {
  std::ostream &os;

  struct IOError : std::runtime_error {
    IOError() :
      std::runtime_error("DataOutputStream::IOError") {
    }
  };

  DataOutputStream(std::ostream &os) :
      os(os) {
  }

  void writeInt(uint32_t val) {
    val = htonl(val);
    os.write(reinterpret_cast<const char*>(&val), sizeof(val));
    if(!os) throw IOError();
  }

  void writeLong(uint64_t val) {
    val = htonll(val);
    os.write(reinterpret_cast<const char*>(&val), sizeof(val));
    if(!os) throw IOError();
  }

  void writeString(const std::string &s) {
    writeInt(s.size());
    os.write(&s[0], s.size());
    if(!os) throw IOError();
  }

  void flush() {
    os.flush();
    if(!os) throw IOError();
  }
};

} // namespace alchemist

#endif

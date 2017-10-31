#ifndef ALCHEMIST__DATA_STREAM_H
#define ALCHEMIST__DATA_STREAM_H

#include "alchemist.h"
#include <iostream>

namespace alchemist {

union DoubleBytes {
  uint64_t qword;
  double value;
};

inline uint64_t htond(double value) {
  DoubleBytes d;
  d.value = value;
  return be64toh(d.qword);
}

inline double ntohd(uint64_t qword) {
  DoubleBytes d;
  d.qword = htobe64(qword);
  return d.value;
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
    val = htobe64(val);
    return val;
  }

  double readDouble() {
    uint64_t val;
    is.read(reinterpret_cast<char*>(&val), sizeof(val));
    if(!is) throw IOError();
    return ntohd(val);
  }

  std::string readString() {
    uint64_t stringLen = readLong();
    char * strin = new char [stringLen + 1];
    is.read(strin, stringLen);
    if(!is) throw IOError();
    strin[stringLen]='\0';
    std::string result = std::string(strin);
    delete[] strin;
    return result;
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
    val = be64toh(val);
    os.write(reinterpret_cast<const char*>(&val), sizeof(val));
    if(!os) throw IOError();
  }

  void writeDouble(double val) {
    uint64_t word = htond(val);
    os.write(reinterpret_cast<const char*>(&word), sizeof(word));
    if(!os) throw IOError();
  }

  void writeString(const std::string &s) {
    writeLong(s.size());
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

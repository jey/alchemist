#include "alchemist.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <poll.h>
#include "data_stream.h"
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace alchemist {

struct Worker {
  WorkerId id;
  mpi::communicator world;
  mpi::communicator peers;
  El::Grid grid;
  bool shouldExit;
  int listenSock;
  std::map<MatrixHandle, std::unique_ptr<DistMatrix>> matrices;

  Worker(const mpi::communicator &world, const mpi::communicator &peers) :
      id(world.rank() - 1), world(world), peers(peers), grid(El::mpi::Comm(peers)),
      shouldExit(false), listenSock(-1) {
    ENSURE(peers.rank() == world.rank() - 1);
  }

  void receiveMatrixBlocks(MatrixHandle handle, const std::vector<WorkerId> &layout);
  void sendMatrixRows(MatrixHandle handle, size_t numCols, const std::vector<WorkerId> &layout,
      const std::vector<uint64_t> &localRowIndices, const std::vector<double> &localData);
  int main();
};

uint32_t updateAssignmentsAndCounts(MatrixXd const & dataMat, MatrixXd const & centers, 
    uint32_t * clusterSizes, std::vector<uint32_t> & rowAssignments) {
  uint32_t numCenters = centers.rows();
  VectorXd distanceSq(numCenters);
  El::Int newAssignment;
  uint32_t numChanged = 0;

  for(uint32_t idx = 0; idx < numCenters; ++idx) 
    clusterSizes[idx] = 0; 
  
  for(El::Int rowIdx = 0; rowIdx < dataMat.rows(); ++rowIdx) {
    for(uint32_t centerIdx = 0; centerIdx < numCenters; ++centerIdx) 
      distanceSq[centerIdx] = (dataMat.row(rowIdx) - centers.row(centerIdx)).squaredNorm();
    distanceSq.minCoeff(&newAssignment);
    if (rowAssignments[rowIdx] != newAssignment) 
      numChanged += 1;
    rowAssignments[rowIdx] = newAssignment;
    clusterSizes[rowAssignments[rowIdx]] += 1;
  }

  return numChanged;
}

void KMeansCommand::run(Worker *self) const {
  auto origDataMat = self->matrices[origMat].get();
  auto n = origDataMat->Height();
  auto d = origDataMat->Width();

  // TODO: look into using Elemental's read proxies for potentially faster transparent relayouts
  // btw, cf http://libelemental.org/pub/slides/ICS13.pdf slide 19 for the cost of redistribution
  // TODO: dataMat would be deleted once out of scope anyhow, so remove the unique_ptr wrapper
  std::unique_ptr<DistMatrix> dataMat_uniqptr{new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(n, d, self->grid)}; // so will delete this relayed out matrix once kmeans goes out of scope
  DistMatrix * dataMat = dataMat_uniqptr.get();
  El::Copy(*origDataMat, *dataMat); // relayout data so it is row-wise partitioned

  DistMatrix * centers = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(numCenters, d, self->grid);
  DistMatrix * assignments = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(n, 1, self->grid);
  ENSURE(self->matrices.insert(std::make_pair(centersHandle, std::unique_ptr<DistMatrix>(centers))).second);
  ENSURE(self->matrices.insert(std::make_pair(assignmentsHandle, std::unique_ptr<DistMatrix>(assignments))).second);

  MatrixXd localData(dataMat->LocalHeight(), d);
  MatrixXd localCenters = MatrixXd::Random(numCenters, d);

  // compute the map from local row indices to the row indices in the global matrix
  // and populate the local data matrix

  std::vector<El::Int> rowMap(localData.rows());
  for(El::Int rowIdx = 0; rowIdx < n; ++rowIdx) 
    if (dataMat->IsLocalRow(rowIdx)) {
      auto localRowIdx = dataMat->LocalRow(rowIdx);
      rowMap[localRowIdx] = rowIdx;
      for(El::Int colIdx = 0; colIdx < d; ++colIdx) 
        localData(localRowIdx, colIdx) = dataMat->GetLocal(localRowIdx, colIdx);
    }

  // compute the local cluster assignments
  std::unique_ptr<uint32_t[]> localCounts{new uint32_t[numCenters]};
  std::vector<uint32_t> rowAssignments(localData.rows());
  VectorXd distanceSq(numCenters);
  updateAssignmentsAndCounts(localData, localCenters, localCounts.get(), rowAssignments);

  MatrixXd centersBuf(numCenters, d);
  std::unique_ptr<uint32_t[]> countsBuf{new uint32_t[numCenters]};
  uint32_t numChanged = 0;

  while(true) {
    uint32_t nextCommand;
    mpi::broadcast(self->world, nextCommand, 0);

    if (nextCommand == 0xf)  // finished iterating
      break;
    else if (nextCommand == 2) { // reinitialize cluster centers 
      localCenters = MatrixXd::Random(numCenters, d);
      updateAssignmentsAndCounts(localData, localCenters, localCounts.get(), rowAssignments);
    }

    // update the centers
    // TODO: locally compute cluster sums and place in localCenters
    localCenters.setZero();
    for(uint32_t rowIdx = 0; rowIdx < localData.rows(); ++rowIdx) 
      localCenters.row(rowAssignments[rowIdx]) += localData.row(rowIdx);

    mpi::all_reduce(self->peers, localCenters.data(), numCenters*d, centersBuf.data(), std::plus<double>());
    std::memcpy(localCenters.data(), centersBuf.data(), numCenters*d*sizeof(double));
    mpi::all_reduce(self->peers, localCounts.get(), numCenters, countsBuf.get(), std::plus<uint32_t>());
    std::memcpy(localCounts.get(), countsBuf.get(), numCenters*sizeof(uint32_t));

    for(uint32_t rowIdx = 0; rowIdx < numCenters; ++rowIdx)
      if( localCounts[rowIdx] > 0)
        localCenters.row(rowIdx) /= localCounts[rowIdx];

    // compute new local assignments
    numChanged = updateAssignmentsAndCounts(localData, localCenters, localCounts.get(), rowAssignments);

    // return the number of changed assignments
    mpi::reduce(self->world, numChanged, std::plus<int>(), 0);
    // return the cluster counts
    mpi::reduce(self->world, localCounts.get(), numCenters, std::plus<uint32_t>(), 0);
  }

  // write the final k-means centers and assignments
  El::Zero(*assignments);
  assignments->Reserve(localData.rows());
  for(El::Int rowIdx = 0; rowIdx < localData.rows(); ++rowIdx)
    assignments->QueueUpdate(rowMap[rowIdx], 0, rowAssignments[rowIdx]);
  assignments->ProcessQueues();

  El::Zero(*centers);
  centers->Reserve(centers->LocalHeight()*d);
  for(uint32_t clusterIdx = 0; clusterIdx < numCenters; ++clusterIdx)
    if (centers->IsLocalRow(clusterIdx)) {
      for(El::Int colIdx = 0; colIdx < d; ++colIdx)
        centers->QueueUpdate(clusterIdx, colIdx, localCenters(clusterIdx, colIdx));
    }
  centers->ProcessQueues();

  self->world.barrier();
}

void TruncatedSVDCommand::run(Worker *self) const {
  auto m = self->matrices[mat]->Height();
  auto n = self->matrices[mat]->Width();
  auto A = self->matrices[mat].get();

  uint32_t command;
  std::unique_ptr<double[]> vecIn{new double[n]};
  std::unique_ptr<double[]> vecOut{new double[n]};

  for(El::Int idx = 0; idx < n; idx++)
    vecOut[idx] = 0.0;

  DistMatrix * x = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(n, 1, self->grid);
  DistMatrix * yintermed = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(m, 1, self->grid);
  DistMatrix * yfinal = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(n, 1, self->grid);


  std::cerr << format("%s: finished inits \n") % self->world.rank();

  while(true) {
    mpi::broadcast(self->world, command, 0); 
    if (command == 1) {
      mpi::broadcast(self->world, vecIn.get(), n, 0);
      for(El::Int idx=0; idx < n; idx++)
        if(x->IsLocal(idx,0)) {
         x->SetLocal(x->LocalRow(idx), 0, vecIn[idx]);
        }

      El::Gemv(El::NORMAL, 1.0, *A, *x, 0.0, *yintermed);
      El::Gemv(El::TRANSPOSE, 1.0, *A, *yintermed, 0.0, *yfinal);

      for(El::Int idx=0; idx < n; idx++)
        if(yfinal->IsLocal(idx,0))
          vecOut[idx] = yfinal->GetLocal(yfinal->LocalRow(idx),0);
      mpi::reduce(self->world, vecOut.get(), n, std::plus<double>(), 0);
      for(El::Int idx = 0; idx < n; idx++)
        vecOut[idx] = 0.0;
    }
    if (command == 2) {
      uint32_t nconv;
      mpi::broadcast(self->world, nconv, 0);

      MatrixXd rightEigs(n, nconv);
      mpi::broadcast(self->world, rightEigs.data(), n*nconv, 0);
      VectorXd singValsSq(nconv);
      mpi::broadcast(self->world, singValsSq.data(), nconv, 0);

      DistMatrix * U = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(m, nconv, self->grid);
      DistMatrix * S = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(nconv, 1, self->grid);
      DistMatrix * Sinv = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(nconv, 1, self->grid);
      DistMatrix * V = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(n, nconv, self->grid);

      ENSURE(self->matrices.insert(std::make_pair(UHandle, std::unique_ptr<DistMatrix>(U))).second);
      ENSURE(self->matrices.insert(std::make_pair(SHandle, std::unique_ptr<DistMatrix>(S))).second);
      ENSURE(self->matrices.insert(std::make_pair(VHandle, std::unique_ptr<DistMatrix>(V))).second);

      // populate V
      for(El::Int rowIdx=0; rowIdx < n; rowIdx++)
        for(El::Int colIdx=0; colIdx < (El::Int) nconv; colIdx++) 
          if(V->IsLocal(rowIdx, colIdx)) 
            V->SetLocal(V->LocalRow(rowIdx), V->LocalCol(colIdx), rightEigs(rowIdx,colIdx));
      // populate S, Sinv
      for(El::Int idx=0; idx < (El::Int) nconv; idx++) {
        if(S->IsLocal(idx, 0)) 
          S->SetLocal(S->LocalRow(idx), 0, std::sqrt(singValsSq(idx)));
        if(Sinv->IsLocal(idx, 0)) 
          Sinv->SetLocal(Sinv->LocalRow(idx), 0, 1/std::sqrt(singValsSq(idx)));
      }

      // form U
      El::Gemm(El::NORMAL, El::NORMAL, 1.0, *A, *V, 0.0, *U);
      // TODO: do a QR instead, but does column pivoting so would require postprocessing S,V to stay consistent
      El::DiagonalScale(El::RIGHT, El::NORMAL, *Sinv, *U);

      break;
    }
  }

  self->world.barrier();
}

void TransposeCommand::run(Worker *self) const {
  auto m = self->matrices[origMat]->Height();
  auto n = self->matrices[origMat]->Width();
  DistMatrix * transposeA = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(n, m, self->grid);
  El::Zero(*transposeA);
  
  ENSURE(self->matrices.insert(std::make_pair(transposeMat, std::unique_ptr<DistMatrix>(transposeA))).second);

  El::Transpose(*self->matrices[origMat], *transposeA);
  std::cerr << format("%s: finished transpose call\n") % self->world.rank();
  self->world.barrier();
}

void ThinSVDCommand::run(Worker *self) const {
  auto m = self->matrices[mat]->Height();
  auto n = self->matrices[mat]->Width();
  auto k = std::min(m, n);
  DistMatrix * U = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(m, k, self->grid);
  DistMatrix * singvals = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(k, k, self->grid);
  DistMatrix * V = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(n, k, self->grid);
  El::Zero(*U);
  El::Zero(*V);
  El::Zero(*singvals);

  ENSURE(self->matrices.insert(std::make_pair(Uhandle, std::unique_ptr<DistMatrix>(U))).second);
  ENSURE(self->matrices.insert(std::make_pair(Shandle, std::unique_ptr<DistMatrix>(singvals))).second);
  ENSURE(self->matrices.insert(std::make_pair(Vhandle, std::unique_ptr<DistMatrix>(V))).second);

  DistMatrix * Acopy = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(m, n, self->grid); // looking at source code for SVD, seems that DistMatrix Acopy(A) might generate copy rather than just copy metadata and risk clobbering
  El::Copy(*self->matrices[mat], *Acopy);
  El::SVD(*Acopy, *U, *singvals, *V);
  std::cerr << format("%s: singvals is %s by %s\n") % self->world.rank() % singvals->Height() % singvals->Width();
  self->world.barrier();
}

void MatrixMulCommand::run(Worker *self) const {
  auto m = self->matrices[inputA]->Height();
  auto n = self->matrices[inputB]->Width();
  DistMatrix * matrix = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(m, n, self->grid);
  ENSURE(self->matrices.insert(std::make_pair(handle, std::unique_ptr<DistMatrix>(matrix))).second);
  El::Gemm(El::NORMAL, El::NORMAL, 1.0, *self->matrices[inputA], *self->matrices[inputB], 0.0, *matrix);
  El::Display(*self->matrices[inputA], "A:");
  El::Display(*self->matrices[inputB], "B:");
  El::Display(*matrix, "A*B:");
  self->world.barrier();
}

// TODO: should send back blocks of rows instead of rows? maybe conversion on other side is cheaper?
void  MatrixGetRowsCommand::run(Worker * self) const {
  uint64_t numRowsFromMe = std::count(layout.begin(), layout.end(), self->id);
  auto matrix = self->matrices[handle].get();
  uint64_t numCols = matrix->Width();

  std::vector<uint64_t> localRowIndices; // maps rows in the matrix to rows in the local storage
  std::vector<double> localData(numCols * numRowsFromMe);

  localRowIndices.reserve(numRowsFromMe);
  matrix->ReservePulls(numCols * numRowsFromMe);
  for(uint64_t curRowIdx = 0; localRowIndices.size() < numRowsFromMe; curRowIdx++) {
    if( layout[curRowIdx] == self->id ) {
      localRowIndices.push_back(curRowIdx);
      for(uint64_t col = 0; col < numCols; col++) {
        matrix->QueuePull(curRowIdx, col);
      }
    }
  } 
  matrix->ProcessPullQueue(&localData[0]);

  self->sendMatrixRows(handle, matrix->Width(), layout, localRowIndices, localData);
  self->world.barrier();
}

void NewMatrixCommand::run(Worker *self) const {
  DistMatrix *matrix = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(numRows, numCols, self->grid);
  Zero(*matrix);
  ENSURE(self->matrices.insert(std::make_pair(handle, std::unique_ptr<DistMatrix>(matrix))).second);
  self->receiveMatrixBlocks(handle, layout);
  self->peers.barrier();
  matrix->ProcessQueues();
  self->world.barrier();
}

void HaltCommand::run(Worker *self) const {
  self->shouldExit = true;
}

struct WorkerClientSendHandler {
  int sock;
  short pollEvents;
  std::vector<char> inbuf;
  std::vector<char> outbuf;
  size_t inpos;
  size_t outpos;
  const std::vector<uint64_t> &localRowIndices;
  const std::vector<double> &localData;
  MatrixHandle handle;
  const size_t numCols;

  // only set POLLOUT when have data to send
  // sends 0x3 code (uint32), then matrix handle (uint32), then row index (long = uint64_t)
  // localData contains the rows of localRowIndices in order
  WorkerClientSendHandler(int sock, MatrixHandle handle, size_t numCols, const std::vector<uint64_t> &localRowIndices, const std::vector<double> &localData) :
    sock(sock), pollEvents(POLLIN), inbuf(16), outbuf(8 + numCols * 8), inpos(0), outpos(0),
    localRowIndices(localRowIndices), localData(localData), handle(handle), numCols(numCols) {
  }

  ~WorkerClientSendHandler() {
    close();
  }

  // note this is never used! (it should be, to remove the client from the set of clients being polled once the operation on that client is done 
  bool isClosed() const {
    return sock == -1;
  }

  void close() {
    if(sock != -1) ::close(sock);
    sock = -1;
    pollEvents = 0;
  }

  int handleEvent(short revents) {
    mpi::communicator world;
    int rowsCompleted = 0;

    // handle reads
    if(revents & POLLIN && pollEvents & POLLIN) {
      while(!isClosed()) {
        int count = recv(sock, &inbuf[inpos], inbuf.size() - inpos, 0);
        //std::cerr << format("%s: read: sock=%s, inbuf=%s, inpos=%s, count=%s\n")
        //    % world.rank() % sock % inbuf.size() % inpos % count;
        if (count == 0) {
          break;
        } else if( count == -1) {
          if(errno == EAGAIN) {
            // no more input available until next POLLIN
            break;
          } else if(errno == EINTR) {
            // interrupted (e.g. by signal), so try again
            continue;
          } else if(errno == ECONNRESET) {
            close();
            break;
          } else {
            // TODO
            abort();
          }
        } else {
          ENSURE(count > 0);
          inpos += count;
          ENSURE(inpos <= inbuf.size());
          if(inpos >= 4) {
            char *dataPtr = &inbuf[0];
            uint32_t typeCode = ntohl(*(uint32_t*)dataPtr);
            dataPtr += 4;
            if(typeCode == 0x3 && inpos == inbuf.size()) {
              // sendRow
              ENSURE(ntohl(*(uint32_t*)dataPtr) == handle.id);
              dataPtr += 4;
              uint64_t rowIdx = ntohll(*(uint64_t*)dataPtr);
              dataPtr += 8;
              auto localRowOffsetIter = std::find(localRowIndices.begin(), localRowIndices.end(), rowIdx);
              ENSURE(localRowOffsetIter != localRowIndices.end());
              auto localRowOffset = localRowOffsetIter - localRowIndices.begin();
              *reinterpret_cast<uint64_t*>(&outbuf[0]) = htonll(numCols * 8);
              // treat the output as uint64_t[] instead of double[] to avoid type punning issues with htonll
              auto invals = reinterpret_cast<const uint64_t*>(&localData[numCols * localRowOffset]);
              auto outvals = reinterpret_cast<uint64_t*>(&outbuf[8]);
              for(uint64_t idx = 0; idx < numCols; ++idx) {
                // can't use std::transform since htonll is a macro (on macOS)
                outvals[idx] = htonll(invals[idx]);
              }
              inpos = 0;
              pollEvents = POLLOUT; // after parsing the request, send the data
              break;
            } 
          }
        }
      }
    }

    // handle writes
    if(revents & POLLOUT && pollEvents & POLLOUT) {
      // a la https://stackoverflow.com/questions/12170037/when-to-use-the-pollout-event-of-the-poll-c-function
      // and http://www.kegel.com/dkftpbench/nonblocking.html
      while(!isClosed()) {
        int count = write(sock, &outbuf[outpos], outbuf.size() - outpos);
        //std::cerr << format("%s: write: sock=%s, outbuf=%s, outpos=%s, count=%s\n")
        //    % world.rank() % sock % outbuf.size() % outpos % count;
        if (count == 0) {
          break; 
        } else if(count == -1) {
          if(errno == EAGAIN) {
            // out buffer is full for now, wait for next POLLOUT
            break;
          } else if(errno == EINTR) {
            // interrupted (e.g. by signal), so try again
            continue;
          } else if(errno == ECONNRESET) {
            close();
            break;
          } else {
            // TODO
            abort();
          } 
        } else {
          ENSURE(count > 0);
          outpos += count;
          ENSURE(outpos <= outbuf.size());
          if (outpos == outbuf.size()) { // after sending the row, wait for the next request
            rowsCompleted += 1;
            outpos = 0;
            pollEvents = POLLIN;
            break;
          }
        }
      }
    }

    return rowsCompleted;
  }
};

struct WorkerClientRecieveHandler {
  int sock;
  short pollEvents;
  std::vector<char> inbuf;
  size_t pos;
  DistMatrix *matrix;
  MatrixHandle handle;

  WorkerClientRecieveHandler(int sock, MatrixHandle handle, DistMatrix *matrix) :
      sock(sock), pollEvents(POLLIN), inbuf(matrix->Width() * 8 + 24),
      pos(0), matrix(matrix), handle(handle) {
  }

  ~WorkerClientRecieveHandler() {
    close();
  }

  bool isClosed() const {
    return sock == -1;
  }

  void close() {
    if(sock != -1) ::close(sock);
    sock = -1;
    pollEvents = 0;
  }

  int handleEvent(short revents) {
    mpi::communicator world;
    int partitionsCompleted = 0;
    if(revents & POLLIN && pollEvents & POLLIN) {
      while(!isClosed()) {
        int count = recv(sock, &inbuf[pos], inbuf.size() - pos, 0);
        if(count == 0) {
          break;
        } else if(count == -1) {
          if(errno == EAGAIN) {
            // no more input available until next POLLIN
            break;
          } else if(errno == EINTR) {
            continue;
          } else if(errno == ECONNRESET) {
            close();
            break;
          } else {
            // TODO
            abort();
          }
        } else {
          ENSURE(count > 0);
          pos += count;
          ENSURE(pos <= inbuf.size());
          if(pos >= 4) {
            char *dataPtr = &inbuf[0];
            uint32_t typeCode = ntohl(*(uint32_t*)dataPtr);
            dataPtr += 4;
            if(typeCode == 0x1 && pos == inbuf.size()) {
              // addRow
              size_t numCols = matrix->Width();
              ENSURE(ntohl(*(uint32_t*)dataPtr) == handle.id);
              dataPtr += 4;
              uint64_t rowIdx = ntohll(*(uint64_t*)dataPtr);
              dataPtr += 8;
              ENSURE(rowIdx < (size_t)matrix->Height());
              ENSURE(ntohll(*(uint64_t*)dataPtr) == numCols * 8);
              dataPtr += 8;
              matrix->Reserve(numCols);
              for(size_t colIdx = 0; colIdx < numCols; ++colIdx) {
                double value = ntohd(*(uint64_t*)dataPtr);
                matrix->QueueUpdate(rowIdx, colIdx, value);
                dataPtr += 8;
              }
              ENSURE(dataPtr == &inbuf[inbuf.size()]);
              pos = 0;
            } else if(typeCode == 0x2) {
              // partitionComplete
              partitionsCompleted++;
              pos = 0;
            }
          }
        }
      }
    }
    return partitionsCompleted;
  }
};

void Worker::sendMatrixRows(MatrixHandle handle, size_t numCols, const std::vector<WorkerId> &layout,
    const std::vector<uint64_t> &localRowIndices, const std::vector<double> &localData) {
  auto numRowsFromMe = std::count(layout.begin(), layout.end(), this->id);
  std::vector<std::unique_ptr<WorkerClientSendHandler>> clients;
  std::vector<pollfd> pfds;
  while(numRowsFromMe > 0) {
    pfds.clear();
    for(auto it = clients.begin(); it != clients.end();) {
      const auto &client = *it;
      if(client->isClosed()) {
        it = clients.erase(it);
      } else {
        pfds.push_back(pollfd{client->sock, client->pollEvents});
        it++;
      }
    }
    pfds.push_back(pollfd{listenSock, POLLIN}); // must be last entry 
    int count = poll(&pfds[0], pfds.size(), -1);
    if(count == -1 && (errno == EAGAIN || errno == EINTR)) continue;
    ENSURE(count != -1);
    for(size_t idx=0; idx < pfds.size() && count > 0; ++idx) {
      auto curSock = pfds[idx].fd;
      auto revents = pfds[idx].revents;
      if(revents != 0) {
        count--;
        if(curSock == listenSock) {
          ENSURE(revents == POLLIN);
          sockaddr_in addr;
          socklen_t addrlen = sizeof(addr);
          int clientSock = accept(listenSock, reinterpret_cast<sockaddr*>(&addr), &addrlen);
          ENSURE(addrlen == sizeof(addr));
          ENSURE(fcntl(clientSock, F_SETFD, O_NONBLOCK) != -1);
          std::unique_ptr<WorkerClientSendHandler> client(new WorkerClientSendHandler(clientSock, handle, numCols, localRowIndices, localData));
          clients.push_back(std::move(client));
        } else {
          ENSURE(clients[idx]->sock == curSock);
          numRowsFromMe -= clients[idx]->handleEvent(revents);
        }
      }
    }
  }
  std::cerr << format("%s: finished sending rows\n") % world.rank();
}

void Worker::receiveMatrixBlocks(MatrixHandle handle, const std::vector<WorkerId> &layout) {
  auto numPartsForMe = std::count(layout.begin(), layout.end(), this->id);
  std::vector<std::unique_ptr<WorkerClientRecieveHandler>> clients;
  std::vector<pollfd> pfds;
  while(numPartsForMe > 0) {
    pfds.clear();
    for(auto it = clients.begin(); it != clients.end();) {
      const auto &client = *it;
      if(client->isClosed()) {
        it = clients.erase(it);
      } else {
        pfds.push_back(pollfd{client->sock, client->pollEvents});
        it++;
      }
    }
    pfds.push_back(pollfd{listenSock, POLLIN});  // must be last entry
    int count = poll(&pfds[0], pfds.size(), -1);
    if(count == -1 && (errno == EAGAIN || errno == EINTR)) continue;
    ENSURE(count != -1);
    for(size_t idx = 0; idx < pfds.size() && count > 0; ++idx) {
      auto curSock = pfds[idx].fd;
      auto revents = pfds[idx].revents;
      if(revents != 0) {
        count--;
        if(curSock == listenSock) {
          ENSURE(revents == POLLIN);
          sockaddr_in addr;
          socklen_t addrlen = sizeof(addr);
          int clientSock = accept(listenSock, reinterpret_cast<sockaddr*>(&addr), &addrlen);
          ENSURE(addrlen == sizeof(addr));
          ENSURE(fcntl(clientSock, F_SETFD, O_NONBLOCK) != -1);
          std::unique_ptr<WorkerClientRecieveHandler> client(new WorkerClientRecieveHandler(clientSock, handle, matrices[handle].get()));
          clients.push_back(std::move(client));
        } else {
          ENSURE(clients[idx]->sock == curSock);
          numPartsForMe -= clients[idx]->handleEvent(revents);
        }
      }
    }
  }
  std::cerr << format("%s: finished receiving blocks\n") % world.rank();
}

int Worker::main() {
  // create listening socket, bind to an available port, and get the port number
  ENSURE((listenSock = socket(AF_INET, SOCK_STREAM, 0)) != -1);
  sockaddr_in addr = {AF_INET};
  ENSURE(bind(listenSock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
  ENSURE(listen(listenSock, 1024) == 0);
  ENSURE(fcntl(listenSock, F_SETFD, O_NONBLOCK) != -1);
  socklen_t addrlen = sizeof(addr);
  ENSURE(getsockname(listenSock, reinterpret_cast<sockaddr*>(&addr), &addrlen) == 0);
  ENSURE(addrlen == sizeof(addr));
  uint16_t port = ntohs(addr.sin_port);

  // transmit WorkerInfo to driver
  char hostname[256];
  ENSURE(gethostname(hostname, sizeof(hostname)) == 0);
  WorkerInfo info{hostname, port};
  world.send(0, 0, info);

  // handle commands until done
  while(!shouldExit) {
    const Command *cmd = nullptr;
    mpi::broadcast(world, cmd, 0);
    cmd->run(this);
    delete cmd;
  }

  // synchronized exit
  world.barrier();
  return EXIT_SUCCESS;
}

int workerMain(const mpi::communicator &world, const mpi::communicator &peers) {
  ENSURE(::dup2(2, 1) == 1); // replaces stdout w/ stderr?
  return Worker(world, peers).main();
}

} // namespace alchemist

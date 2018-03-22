#include "alchemist.h"
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <poll.h>
#include "data_stream.h"
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include "spdlog/spdlog.h"
#include <skylark.hpp>
#include "hilbert.hpp"
#include "utils.hpp"
#include "factorizedCG.hpp"
#include "alchemistreadhdf5.hpp"
#include <time.h>

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
  std::shared_ptr<spdlog::logger> log;

  Worker(const mpi::communicator &world, const mpi::communicator &peers) :
      id(world.rank() - 1), world(world), peers(peers), grid(El::mpi::Comm(peers)),
      shouldExit(false), listenSock(-1) {
    ENSURE(peers.rank() == world.rank() - 1);
  }

  void receiveMatrixBlocks(MatrixHandle handle);
  void sendMatrixRows(MatrixHandle handle, size_t numCols, const std::vector<WorkerId> &layout,
      const std::vector<uint64_t> &localRowIndices, const std::vector<double> &localData);
  int main();
};

// custom GEMM for multiplying [VR,STAR] matrices in normal orientation and storing as [VR, STAR]
// overhead: additional maxPanelSize GB (1 for now) per rank => at most numcores * maxPanelGB per machine
void Gemm(double alpha, const El::DistMatrix<double, El::VR, El::STAR> & A, const El::DistMatrix<double, El::VR, El::STAR> & B,
    double beta, El::DistMatrix<double, El::VR, El::STAR> & C, std::shared_ptr<spdlog::logger> & log) {
  auto m = A.Height();
  auto n = A.Width();
  auto k = B.Width();
  assert(n == B.Height());

  El::Int maxPanelSize = 1; // maximum panel size (will be stored on each process) in GB
  El::Int maxPanelWidth = std::max( (int) std::floor( (maxPanelSize*1000*1000*1000)/((double)8*n) ), 1);
  El::Int numPanels = (int) std::ceil(k/(double)maxPanelWidth);
  log->info("Using {} panels", numPanels);

  El::DistMatrix<double, El::STAR, El::STAR> curBPanel(B.Grid());
  El::Matrix<double> curCPanel;
  for(int curPanelNum = 0; curPanelNum < numPanels; ++curPanelNum) {
    El::Int startCol = curPanelNum*maxPanelWidth;
    El::Int lastCol = std::min(startCol + maxPanelWidth - 1, k - 1);
    El::Int curPanelWidth = lastCol - startCol + 1;

    log->info("Creating next B panel");
    El::Zeros(curBPanel, n, curPanelWidth);
    curBPanel.Reserve(curBPanel.LocalHeight()*curPanelWidth);
    for(El::Int row = 0; row < B.LocalHeight(); ++row)
      for(El::Int col = startCol; col <= lastCol; ++col)
        curBPanel.QueueUpdate(B.GlobalRow(row), col, B.LockedMatrix().Get(row, col));
    curBPanel.ProcessQueues();
    log->info("Finished creating current B panel");

    log->info("Creating next C panel");
    El::View(curCPanel, C.Matrix(), El::Range<El::Int>(0, C.LocalHeight()), El::Range<El::Int>(startCol, lastCol+1));
    log->info("Finished creating current C panel");
    log->info("Multiplying A by current B panel, storing into current C panel");
    El::Gemm(El::NORMAL, El::NORMAL, alpha, A.LockedMatrix(), curBPanel.LockedMatrix(), beta, curCPanel);
    log->info("Done storing current C panel");
  }
}

uint32_t updateAssignmentsAndCounts(MatrixXd const & dataMat, MatrixXd const & centers,
    uint32_t * clusterSizes, std::vector<uint32_t> & rowAssignments, double & objVal) {
  uint32_t numCenters = centers.rows();
  VectorXd distanceSq(numCenters);
  El::Int newAssignment;
  uint32_t numChanged = 0;
  objVal = 0.0;

  for(uint32_t idx = 0; idx < numCenters; ++idx)
    clusterSizes[idx] = 0;

  for(El::Int rowIdx = 0; rowIdx < dataMat.rows(); ++rowIdx) {
    for(uint32_t centerIdx = 0; centerIdx < numCenters; ++centerIdx)
      distanceSq[centerIdx] = (dataMat.row(rowIdx) - centers.row(centerIdx)).squaredNorm();
    objVal += distanceSq.minCoeff(&newAssignment);
    if (rowAssignments[rowIdx] != newAssignment)
      numChanged += 1;
    rowAssignments[rowIdx] = newAssignment;
    clusterSizes[rowAssignments[rowIdx]] += 1;
  }

  return numChanged;
}

// TODO: add seed as argument (make sure different workers do different things)
void kmeansParallelInit(Worker * self, DistMatrix const * dataMat,
    MatrixXd const & localData, uint32_t scale, uint32_t initSteps, MatrixXd & clusterCenters, uint64_t seed) {

  auto d = localData.cols();

  // if you have the initial cluster seed, send it to everyone
  uint32_t rowIdx;
  mpi::broadcast(self->world, rowIdx, 0);
  MatrixXd initialCenter;

  if (dataMat->IsLocalRow(rowIdx)) {
    auto localRowIdx = dataMat->LocalRow(rowIdx);
    initialCenter = localData.row(localRowIdx);
    int maybe_root = 1;
    int rootProcess;
    mpi::all_reduce(self->peers, self->peers.rank(), rootProcess, std::plus<int>());
    mpi::broadcast(self->peers, initialCenter, self->peers.rank());
  }
  else {
    int maybe_root = 0;
    int rootProcess;
    mpi::all_reduce(self->peers, 0, rootProcess, std::plus<int>());
    mpi::broadcast(self->peers, initialCenter, rootProcess);
  }

  //in each step, sample 2*k points on average (totalled across the partitions)
  // with probability proportional to their squared distance from the current
  // cluster centers and add the sampled points to the set of cluster centers
  std::vector<double> distSqToCenters(localData.rows());
  double Z; // normalization constant
  std::mt19937 gen(seed + self->world.rank());
  std::uniform_real_distribution<double> dis(0, 1);

  std::vector<MatrixXd> initCenters;
  initCenters.push_back(initialCenter);

  for(int steps = 0; steps < initSteps; ++steps) {
    // 1) compute the distance of your points from the set of centers and all_reduce
    // to get the normalization for the sampling probability
    VectorXd distSq(initCenters.size());
    Z = 0;
    for(uint32_t pointIdx = 0; pointIdx < localData.rows(); ++pointIdx) {
      for(uint32_t centerIdx = 0; centerIdx < initCenters.size(); ++centerIdx)
        distSq[centerIdx] = (localData.row(pointIdx) - initCenters[centerIdx]).squaredNorm();
      distSqToCenters[pointIdx] = distSq.minCoeff();
      Z += distSqToCenters[pointIdx];
    }
    mpi::all_reduce(self->peers, mpi::inplace_t<double>(Z), std::plus<double>());

    // 2) sample your points accordingly
    std::vector<MatrixXd> localNewCenters;
    for(uint32_t pointIdx = 0; pointIdx < localData.rows(); ++pointIdx) {
      bool sampledQ = ( dis(gen) < ((double)scale) * distSqToCenters[pointIdx]/Z ) ? true : false;
      if (sampledQ) {
        localNewCenters.push_back(localData.row(pointIdx));
      }
    };

    // 3) have each worker broadcast out their sampled points to all other workers,
    // to update each worker's set of centers
    for(uint32_t root= 0; root < self->peers.size(); ++root) {
      if (root == self->peers.rank()) {
        mpi::broadcast(self->peers, localNewCenters, root);
        initCenters.insert(initCenters.end(), localNewCenters.begin(), localNewCenters.end());
      } else {
        std::vector<MatrixXd> remoteNewCenters;
        mpi::broadcast(self->peers, remoteNewCenters, root);
        initCenters.insert(initCenters.end(), remoteNewCenters.begin(), remoteNewCenters.end());
      }
    }
  } // end for

  // figure out the number of points closest to each cluster center
  std::vector<uint32_t> clusterSizes(initCenters.size(), 0);
  std::vector<uint32_t> localClusterSizes(initCenters.size(), 0);
  VectorXd distSq(initCenters.size());
  for(uint32_t pointIdx = 0; pointIdx < localData.rows(); ++pointIdx) {
    for(uint32_t centerIdx = 0; centerIdx < initCenters.size(); ++centerIdx)
      distSq[centerIdx] = (localData.row(pointIdx) - initCenters[centerIdx]).squaredNorm();
    uint32_t clusterIdx;
    distSq.minCoeff(&clusterIdx);
    localClusterSizes[clusterIdx] += 1;
  }

  mpi::all_reduce(self->peers, localClusterSizes.data(), localClusterSizes.size(),
      clusterSizes.data(), std::plus<uint32_t>());

  // after centers have been sampled, sync back up with the driver,
  // and send them there for local clustering
  self->world.barrier();
  if(self->world.rank() == 1) {
    self->world.send(0, 0, clusterSizes);
    self->world.send(0, 0, initCenters);
  }
  self->world.barrier();

  clusterCenters.setZero();
  mpi::broadcast(self->world, clusterCenters.data(), clusterCenters.rows()*d, 0);
}

El::DistMatrix<double> * relayout(const El::DistMatrix<double, El::VR, El::STAR> & matIn, const El::Grid & grid) {
  El::DistMatrix<double> * matOut = new El::DistMatrix<double>(matIn.Height(), matIn.Width(), grid);
  El::Copy(matIn, *matOut);
  return matOut;
}

El::DistMatrix<double, El::VR, El::STAR> * delayout(const El::DistMatrix<double> & matIn, const El::Grid & grid) {
  auto matOut = new El::DistMatrix<double, El::VR, El::STAR>(matIn.Height(), matIn.Width(), grid);
  El::Copy(matIn, *matOut);
  return matOut;
}

// NB: it seems that Skylark's LSQR routine cannot work with VR, STAR matrices (get templating errors from their GEMM routine),
// so need to relayout the input and output matrices
void SkylarkLSQRSolverCommand::run(Worker *self) const {
  auto log = self->log;
  El::DistMatrix<double, El::VR, El::STAR> * Amat = (El::DistMatrix<double, El::VR, El::STAR> *) self->matrices[A].get();
  El::DistMatrix<double, El::VR, El::STAR> * Bmat = (El::DistMatrix<double, El::VR, El::STAR> *) self->matrices[B].get();
  El::DistMatrix<double> * Xmat = new El::DistMatrix<double>(Amat->Width(), Bmat->Width(), self->grid);

  log->info("Relaying out lhs and rhs for LSQR");
  auto startRelayout = std::chrono::system_clock::now();
  El::DistMatrix<double> * Arelayedout = relayout(*Amat, self->grid);
  El::DistMatrix<double> * Brelayedout = relayout(*Bmat, self->grid);
  std::chrono::duration<double, std::milli> relayoutDuration(std::chrono::system_clock::now() - startRelayout);
  log->info("Relayout took {} seconds", relayoutDuration.count()/1000.0);

  auto params = skylark::algorithms::krylov_iter_params_t(tolerance, iter_lim);
  skylark::algorithms::LSQR(*Arelayedout, *Brelayedout, *Xmat, params);
  Arelayedout->EmptyData();
  Brelayedout->EmptyData();
  El::DistMatrix<double, El::VR, El::STAR> * Xrelayedout = delayout(*Xmat, self->grid);
  Xmat->EmptyData();

  log->info("LSQR result has dimension {}-by-{}", Xrelayedout->Height(), Xrelayedout->Width());
  ENSURE(self->matrices.insert(std::make_pair(X, std::unique_ptr<DistMatrix>(Xrelayedout))).second);
  self->world.barrier();
}

// Note if we do regression, we can only have one rhs
// The call to the ADMM solver follows  the template of the LargeScaleKernelLearning function in hilbert.hpp
void SkylarkKernelSolverCommand::run(Worker *self) const {

  self->log->info("Setting up solver options");

  // set some options that aren't currently passed in as arguments
  bool usefast = false;
  SequenceType seqtype = LEAPED_HALTON;
  int numthreads = 1;
  bool cachetransforms = true;

  skylark::base::context_t context(seed);
  typedef El::Matrix<double> InputType;

  El::Matrix<double> localX;
  El::Matrix<double> localY;

  El::Transpose(self->matrices[features].get()->LockedMatrix(), localX);
  El::Transpose(self->matrices[targets].get()->LockedMatrix(), localY);

  // validation data
  El::Matrix<double> localXv;
  El::Matrix<double> localYv;

  // don't know what the variable targets is used for
  // shift indicates whether validation data should also be shifted
  auto dimensions = skylark::base::Height(localX);
  auto targets = regression ? 1 : GetNumTargets(self->peers, localY);
  bool shift = false;

  if (!regression && lossfunction == LOGISTIC && targets == 1) {
    ShiftForLogistic(localY);
    targets = 2;
    shift = true;
  }

  self->log->info("Setting up Skylark ADMM solver");

  skylark::algorithms::loss_t<double> *loss = NULL;
  switch(lossfunction) {
    case SQUARED:
        loss = new skylark::algorithms::squared_loss_t<double>();
        break;
    case HINGE:
        loss = new skylark::algorithms::hinge_loss_t<double>();
        break;
    case LOGISTIC:
        loss = new skylark::algorithms::logistic_loss_t<double>();
        break;
    case LAD:
        loss = new skylark::algorithms::lad_loss_t<double>();
        break;
		default:
				break;
  }

	skylark::algorithms::regularizer_t<double> *reg = NULL;
	if (lambda == 0 || regularizer == NOREG)
			reg = new skylark::algorithms::empty_regularizer_t<double>();
	else
			switch(regularizer) {
				case L2:
						reg = new skylark::algorithms::l2_regularizer_t<double>();
						break;
				case L1:
						reg = new skylark::algorithms::l1_regularizer_t<double>();
						break;
				default:
						break;
			}

  self->log->info("Initializing solver");

  BlockADMMSolver<InputType> *Solver = NULL;
    int features = 0;
    switch(kernel) {
    case K_LINEAR:
        features =
            (randomfeatures == 0 ? dimensions : randomfeatures);
        if (randomfeatures == 0) {
            Solver =
                new BlockADMMSolver<InputType>(loss,
                    reg,
                    lambda,
                    dimensions,
                    numfeaturepartitions);
          }
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    reg,
                    lambda,
                    features,
                    skylark::ml::linear_t(dimensions),
                    skylark::ml::sparse_feature_transform_tag(),
                    numfeaturepartitions);
        break;

    case K_GAUSSIAN:
        features = randomfeatures;
        if (!usefast)
            if (seqtype == LEAPED_HALTON)
                Solver =
                    new BlockADMMSolver<InputType>(context,
                        loss,
                        reg,
                        lambda,
                        features,
                        skylark::ml::gaussian_t(dimensions,
                            kernelparam),
                        skylark::ml::quasi_feature_transform_tag(),
                        numfeaturepartitions);
            else
                Solver =
                    new BlockADMMSolver<InputType>(context,
                        loss,
                        reg,
                        lambda,
                        features,
                        skylark::ml::gaussian_t(dimensions,
                            kernelparam),
                        skylark::ml::regular_feature_transform_tag(),
                        numfeaturepartitions);
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    reg,
                    lambda,
                    features,
                    skylark::ml::gaussian_t(dimensions,
                        kernelparam),
                    skylark::ml::fast_feature_transform_tag(),
                    numfeaturepartitions);
        break;

    case K_POLYNOMIAL:
        features = randomfeatures;
        Solver =
            new BlockADMMSolver<InputType>(context,
                loss,
                reg,
                lambda,
                features,
                skylark::ml::polynomial_t(dimensions,
                    kernelparam, kernelparam2, kernelparam3),
                skylark::ml::regular_feature_transform_tag(),
                numfeaturepartitions);
        break;

    case K_MATERN:
        features = randomfeatures;
        if (!usefast)
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    reg,
                    lambda,
                    features,
                    skylark::ml::matern_t(dimensions,
                        kernelparam, kernelparam2),
                    skylark::ml::regular_feature_transform_tag(),
                    numfeaturepartitions);
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    reg,
                    lambda,
                    features,
                    skylark::ml::matern_t(dimensions,
                        kernelparam, kernelparam2),
                    skylark::ml::fast_feature_transform_tag(),
                    numfeaturepartitions);
        break;

    case K_LAPLACIAN:
        features = randomfeatures;
        if (seqtype == LEAPED_HALTON)
            new BlockADMMSolver<InputType>(context,
                loss,
                reg,
                lambda,
                features,
                skylark::ml::laplacian_t(dimensions,
                    kernelparam),
                skylark::ml::quasi_feature_transform_tag(),
                numfeaturepartitions);
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    reg,
                    lambda,
                    features,
                    skylark::ml::laplacian_t(dimensions,
                        kernelparam),
                    skylark::ml::regular_feature_transform_tag(),
                    numfeaturepartitions);

        break;

    case K_EXPSEMIGROUP:
        features = randomfeatures;
        if (seqtype == LEAPED_HALTON)
            new BlockADMMSolver<InputType>(context,
                loss,
                reg,
                lambda,
                features,
                skylark::ml::expsemigroup_t(dimensions,
                    kernelparam),
                skylark::ml::quasi_feature_transform_tag(),
                numfeaturepartitions);
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    reg,
                    lambda,
                    features,
                    skylark::ml::expsemigroup_t(dimensions,
                        kernelparam),
                    skylark::ml::regular_feature_transform_tag(),
                    numfeaturepartitions);
        break;

    default:
        // TODO!
        break;

    }

    self->log->info("Solver initialized");

    // Set parameters
    Solver->set_rho(rho);
    Solver->set_maxiter(maxiter);
    Solver->set_tol(tolerance);
    Solver->set_nthreads(numthreads);
    Solver->set_cache_transform(cachetransforms);

  self->log->info("Training solver");
  skylark::ml::hilbert_model_t * model = Solver->train(localX, localY, localXv, localYv, regression, self->peers);
  self->log->info("Finished training, now retreiving coefficients");
  El::Matrix<double> X = model->get_coef();

  // Convert the model coefficients to a distributed matrix and store in the matrix table
  DistMatrix * Xdist = new El::DistMatrix<double, El::VR, El::STAR>(X.Height(), X.Width(), self->grid);
  for(uint32_t row = 0; row < Xdist->Height(); row++) 
    if (Xdist->IsLocalRow(row)) 
      for (uint32_t col = 0; col < Xdist->Width(); col++)
        Xdist->Set(row, col, X.Get(row,col));

  ENSURE(self->matrices.insert(std::make_pair(coefs, std::unique_ptr<DistMatrix>(Xdist))).second);
  self->log->info("Stored the coefficient matrix as matrix {}", coefs);

  self->world.barrier();
}

void RandomFourierFeaturesCommand::run(Worker *self) const {
    auto log = self->log;
    typedef El::DistMatrix<double, El::VR, El::STAR> DistMatrixType;
    namespace skys = skylark::sketch;

    DistMatrixType * Amat = (DistMatrixType *) self->matrices[A].get();
    DistMatrixType * Fmat = new DistMatrixType(Amat->Height(), numRandFeatures, self->grid);

    skylark::base::context_t context(seed);
    skys::GaussianRFT_t<DistMatrixType, DistMatrixType> RFFSketcher(Amat->Width(), numRandFeatures, sigma, context);

    log->info("Computing the Gaussian Random Features");
    RFFSketcher.apply(*Amat, *Fmat, skys::rowwise_tag());
    log->info("Finished computing");
    ENSURE(self->matrices.insert(std::make_pair(X, std::unique_ptr<DistMatrix>(Fmat))).second);

    self->world.barrier();
}

void ReadHDF5Command::run(Worker * self) const {
    auto log = self->log;
    typedef El::DistMatrix<double, El::VR, El::STAR> DistMatrixType;

    DistMatrixType *AMat = new DistMatrixType(1, 1, self->grid);

    alchemistReadHDF5(fname, varname, *AMat, log, colreplicas);

    ENSURE(self->matrices.insert(std::make_pair(A, std::unique_ptr<DistMatrix>(AMat))).second);
    log->info("Read matrix has Frobenius norm {}", El::FrobeniusNorm(*AMat));

    if (self->world.rank() == 1) {
        self->world.send(0, 0, AMat->Height());
        self->world.send(0, 0, AMat->Width());
    }
    self->world.barrier();
}

void FactorizedCGSolverCommand::run(Worker *self) const {
  auto log = self->log;
  El::DistMatrix<double, El::VR, El::STAR> * Amat = (El::DistMatrix<double, El::VR, El::STAR> *) self->matrices[A].get();
  El::DistMatrix<double, El::VR, El::STAR> * Bmat = (El::DistMatrix<double, El::VR, El::STAR> *) self->matrices[B].get();
  El::DistMatrix<double> * Xmat = new El::DistMatrix<double>(Amat->Width(), Bmat->Width(), self->grid);
  El::Bernoulli(*Xmat, Xmat->Height(), Xmat->Width());

  log->info("Relaying out lhs and rhs for CG solver");
  auto startRelayout = std::chrono::system_clock::now();
  El::DistMatrix<double> * Arelayedout = relayout(*Amat, self->grid);
  El::DistMatrix<double> * Brelayedout = relayout(*Bmat, self->grid);
  std::chrono::duration<double, std::milli> relayoutDuration(std::chrono::system_clock::now() - startRelayout);
  log->info("Relayout took {} seconds", relayoutDuration.count()/1000.0);

  auto params = skylark::algorithms::krylov_iter_params_t();
  params.iter_lim = maxIters;
  params.am_i_printing = true;
  params.log_level = 2;
  params.res_print = 20;
  log->info("Calling the CG solver");
  skylark::algorithms::factorizedCG(*Arelayedout, *Brelayedout, *Xmat, lambda, log, params);
  Arelayedout->EmptyData();
  Brelayedout->EmptyData();
  El::DistMatrix<double, El::VR, El::STAR> * Xrelayedout = delayout(*Xmat, self->grid);
  Xmat->EmptyData();

  log->info("CG results has dimensions {}-by-{}", Xrelayedout->Height(), Xrelayedout->Width());
  ENSURE(self->matrices.insert(std::make_pair(X, std::unique_ptr<DistMatrix>(Xrelayedout))).second);
  self->world.barrier();
  
  /*
  El::DistMatrix<double, El::VR, El::STAR> * Xmat = new El::DistMatrix<double, El::VR, El::STAR>(Amat->Width(), Bmat->Width(), self->grid);
  El::Bernoulli(*Xmat, Xmat->Height(), Xmat->Width());

  auto params = skylark::algorithms::krylov_iter_params_t();
  params.iter_lim = maxIters;
  params.am_i_printing = true;
  params.log_level = 2;
  params.res_print = 20;
  log->info("Calling the CG solver");
  skylark::algorithms::factorizedCG(*Amat, *Bmat, *Xmat, lambda, log, params);

  log->info("CG results has dimensions {}-by-{}", Xmat->Height(), Xmat->Width());
  ENSURE(self->matrices.insert(std::make_pair(X, std::unique_ptr<DistMatrix>(Xmat))).second);
  self->world.barrier();
  */
}

// TODO: add seed as argument (make sure different workers do different things)
void KMeansCommand::run(Worker *self) const {
  auto log = self->log;
  log->info("Started kmeans");
  auto origDataMat = self->matrices[origMat].get();
  auto n = origDataMat->Height();
  auto d = origDataMat->Width();

  // relayout matrix if needed so that it is in row-partitioned format
  // cf http://libelemental.org/pub/slides/ICS13.pdf slide 19 for the cost of redistribution
  auto distData = origDataMat->DistData();
  DistMatrix * dataMat = new El::DistMatrix<double, El::VR, El::STAR>(n, d, self->grid);
  if (distData.colDist == El::VR && distData.rowDist == El::STAR) {
   dataMat = origDataMat;
  } else {
    auto relayoutStart = std::chrono::system_clock::now();
    El::Copy(*origDataMat, *dataMat); // relayouts data so it is row-wise partitioned
    std::chrono::duration<double, std::milli> relayoutDuration(std::chrono::system_clock::now() - relayoutStart);
    log->info("Detected matrix is not row-partitioned, so relayouted to row-partitioned; took {} ms ", relayoutDuration.count());
  }

  // TODO: store these as local matrices on the driver
  DistMatrix * centers = new El::DistMatrix<double, El::VR, El::STAR>(numCenters, d, self->grid);
  DistMatrix * assignments = new El::DistMatrix<double, El::VR, El::STAR>(n, 1, self->grid);
  ENSURE(self->matrices.insert(std::make_pair(centersHandle, std::unique_ptr<DistMatrix>(centers))).second);
  ENSURE(self->matrices.insert(std::make_pair(assignmentsHandle, std::unique_ptr<DistMatrix>(assignments))).second);

  MatrixXd localData(dataMat->LocalHeight(), d);

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

  MatrixXd clusterCenters(numCenters, d);
  MatrixXd oldClusterCenters(numCenters, d);

  // initialize centers using kMeans||
  uint32_t scale = 2*numCenters;
  clusterCenters.setZero();
  kmeansParallelInit(self, dataMat, localData, scale, initSteps, clusterCenters, seed);

  // TODO: allow to initialize k-means randomly
  //MatrixXd clusterCenters = MatrixXd::Random(numCenters, d);

  /******** START Lloyd's iterations ********/
  // compute the local cluster assignments
  std::unique_ptr<uint32_t[]> counts{new uint32_t[numCenters]};
  std::vector<uint32_t> rowAssignments(localData.rows());
  VectorXd distanceSq(numCenters);
  double objVal;

  updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);

  MatrixXd centersBuf(numCenters, d);
  std::unique_ptr<uint32_t[]> countsBuf{new uint32_t[numCenters]};
  uint32_t numChanged = 0;
  oldClusterCenters = clusterCenters;

  while(true) {
    uint32_t nextCommand;
    mpi::broadcast(self->world, nextCommand, 0);

    if (nextCommand == 0xf)  // finished iterating
      break;
    else if (nextCommand == 2) { // encountered an empty cluster, so randomly pick a point in the dataset as that cluster's centroid
      uint32_t clusterIdx, rowIdx;
      mpi::broadcast(self->world, clusterIdx, 0);
      mpi::broadcast(self->world, rowIdx, 0);
      if (dataMat->IsLocalRow(rowIdx)) {
        auto localRowIdx = dataMat->LocalRow(rowIdx);
        clusterCenters.row(clusterIdx) = localData.row(localRowIdx);
      }
      mpi::broadcast(self->peers, clusterCenters, self->peers.rank());
      updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);
      self->world.barrier();
      continue;
    }

    /******** do a regular Lloyd's iteration ********/

    // update the centers
    // TODO: locally compute cluster sums and place in clusterCenters
    oldClusterCenters = clusterCenters;
    clusterCenters.setZero();
    for(uint32_t rowIdx = 0; rowIdx < localData.rows(); ++rowIdx)
      clusterCenters.row(rowAssignments[rowIdx]) += localData.row(rowIdx);

    mpi::all_reduce(self->peers, clusterCenters.data(), numCenters*d, centersBuf.data(), std::plus<double>());
    std::memcpy(clusterCenters.data(), centersBuf.data(), numCenters*d*sizeof(double));
    mpi::all_reduce(self->peers, counts.get(), numCenters, countsBuf.get(), std::plus<uint32_t>());
    std::memcpy(counts.get(), countsBuf.get(), numCenters*sizeof(uint32_t));

    for(uint32_t rowIdx = 0; rowIdx < numCenters; ++rowIdx)
      if( counts[rowIdx] > 0)
        clusterCenters.row(rowIdx) /= counts[rowIdx];

    // compute new local assignments
    numChanged = updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);
    std::cerr << "computed Updated assingments\n" << std::flush;

    // return the number of changed assignments
    mpi::reduce(self->world, numChanged, std::plus<int>(), 0);
    // return the cluster counts
    mpi::reduce(self->world, counts.get(), numCenters, std::plus<uint32_t>(), 0);
    std::cerr << "returned cluster counts\n" << std::flush;
    if (self->world.rank() == 1) {
      bool movedQ = (clusterCenters - oldClusterCenters).rowwise().norm().minCoeff() > changeThreshold;
      self->world.send(0, 0, movedQ);
    }
  }

  // write the final k-means centers and assignments
  auto startKMeansWrite = std::chrono::system_clock::now();
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
        centers->QueueUpdate(clusterIdx, colIdx, clusterCenters(clusterIdx, colIdx));
    }
  centers->ProcessQueues();
  std::chrono::duration<double, std::milli> kMeansWrite_duration(std::chrono::system_clock::now() - startKMeansWrite);
  std::cerr << self->world.rank() << ": writing the k-means centers and assignments took " << kMeansWrite_duration.count() << "ms\n";

  mpi::reduce(self->world, objVal, std::plus<double>(), 0);
  self->world.barrier();
}

void TruncatedSVDCommand::run(Worker *self) const {
  auto m = self->matrices[mat]->Height();
  auto n = self->matrices[mat]->Width();
  auto workingMat = self->matrices[mat].get();

  int LOCALEIGS = 0; // TODO: make these an enumeration, and global to Alchemist
  int LOCALEIGSPRECOMPUTE = 1;
  int DISTEIGS = 2; 

  // Assume matrix is row-partitioned b/c relaying it out doubles memory requirements

  //NB: sometimes it makes sense to precompute the gramMat (when it's cheap (we have a lot of cores and enough memory), sometimes
  // it makes more sense to compute A'*(A*x) separately each time (when we don't have enough memory for gramMat, or its too expensive
  // time-wise to precompute GramMat). trade-off depends on k (through the number of Arnoldi iterations we'll end up needing), the
  // amount of memory we have free to store GramMat, and the number of cores we have available
  El::Matrix<double> localGramChunk;

  if (method == LOCALEIGSPRECOMPUTE) {
      localGramChunk.Resize(n, n);
      self->log->info("Computing the local contribution to A'*A");
      self->log->info("Local matrix's dimensions are {}, {}", workingMat->LockedMatrix().Height(), workingMat->LockedMatrix().Width());
      self->log->info("Storing A'*A in {},{} matrix", n, n);
      auto startFillLocalMat = std::chrono::system_clock::now();
      if (workingMat->LockedMatrix().Height() > 0)
        El::Gemm(El::TRANSPOSE, El::NORMAL, 1.0, workingMat->LockedMatrix(), workingMat->LockedMatrix(), 0.0, localGramChunk);
      else
        El::Zeros(localGramChunk, n, n);
      std::chrono::duration<double, std::milli> fillLocalMat_duration(std::chrono::system_clock::now() - startFillLocalMat);
      self->log->info("Took {} ms to compute local contribution to A'*A", fillLocalMat_duration.count());
  }

  uint32_t command;
  std::unique_ptr<double[]> vecIn{new double[n]};
  El::Matrix<double> localx(n, 1);
  El::Matrix<double> localintermed(workingMat->LocalHeight(), 1);
  El::Matrix<double> localy(n, 1);
  localx.LockedAttach(n, 1, vecIn.get(), 1);
  auto distx = El::DistMatrix<double, El::STAR, El::STAR>(n, 1, self->grid);
  auto distintermed = El::DistMatrix<double, El::STAR, El::STAR>(m, 1, self->grid);

  self->log->info("finished initialization for truncated SVD");

  while(true) {
    mpi::broadcast(self->world, command, 0);
    if (command == 1 && method == LOCALEIGS) {
        mpi::broadcast(self->world, vecIn.get(), n, 0);
        El::Gemv(El::NORMAL, 1.0, workingMat->LockedMatrix(), localx, 0.0, localintermed);
        El::Gemv(El::TRANSPOSE, 1.0, workingMat->LockedMatrix(), localintermed, 0.0, localy);
        mpi::reduce(self->world, localy.LockedBuffer(), n, std::plus<double>(), 0);
    }
    if (command == 1 && method == LOCALEIGSPRECOMPUTE) {
      mpi::broadcast(self->world, vecIn.get(), n, 0);
      El::Gemv(El::NORMAL, 1.0, localGramChunk, localx, 0.0, localy);
      mpi::reduce(self->world, localy.LockedBuffer(), n, std::plus<double>(), 0);
    } 
    if (command == 1 && method == DISTEIGS) {
      El::Zeros(distx, n, 1);
      self->log->info("Computing a mat-vec prod against A^TA");
      if(self->world.rank() == 1) {
          self->world.recv(0, 0, vecIn.get(), n);
          distx.Reserve(n);
          for(El::Int row=0; row < n; row++)
              distx.QueueUpdate(row, 0, vecIn[row]);
      }
      else {
          distx.Reserve(0);
      }
      distx.ProcessQueues();
      self->log->info("Retrieved x, computing A^TAx");
      El::Gemv(El::NORMAL, 1.0, *workingMat, distx, 0.0, distintermed);
      self->log->info("Computed y = A*x");
      El::Gemv(El::TRANSPOSE, 1.0, *workingMat, distintermed, 0.0, distx);
      self->log->info("Computed x = A^T*y");
      if(self->world.rank() == 1) {
          self->world.send(0, 0, distx.LockedBuffer(), n);
      }
    }
    if (command == 2) {
      uint32_t nconv;
      mpi::broadcast(self->world, nconv, 0);

      MatrixXd rightEigs(n, nconv);
      mpi::broadcast(self->world, rightEigs.data(), n*nconv, 0);
      VectorXd singValsSq(nconv);
      mpi::broadcast(self->world, singValsSq.data(), nconv, 0);
      self->log->info("Received the right eigenvectors and the eigenvalues");

      auto U = new El::DistMatrix<double, El::VR, El::STAR>(m, nconv, self->grid);
      DistMatrix * S = new El::DistMatrix<double, El::VR, El::STAR>(nconv, 1, self->grid);
      DistMatrix * Sinv = new El::DistMatrix<double, El::VR, El::STAR>(nconv, 1, self->grid);
      DistMatrix * V = new El::DistMatrix<double, El::VR, El::STAR>(n, nconv, self->grid);

      ENSURE(self->matrices.insert(std::make_pair(UHandle, std::unique_ptr<DistMatrix>(U))).second);
      ENSURE(self->matrices.insert(std::make_pair(SHandle, std::unique_ptr<DistMatrix>(S))).second);
      ENSURE(self->matrices.insert(std::make_pair(VHandle, std::unique_ptr<DistMatrix>(V))).second);
      self->log->info("Created new matrix objects to hold U,S,V");

      // populate V
      for(El::Int rowIdx=0; rowIdx < n; rowIdx++)
        for(El::Int colIdx=0; colIdx < (El::Int) nconv; colIdx++)
          if(V->IsLocal(rowIdx, colIdx))
            V->SetLocal(V->LocalRow(rowIdx), V->LocalCol(colIdx), rightEigs(rowIdx,colIdx));
      rightEigs.resize(0,0); // clear any memory this temporary variable used (a lot, since it's on every rank)

      // populate S, Sinv
      for(El::Int idx=0; idx < (El::Int) nconv; idx++) {
        if(S->IsLocal(idx, 0))
          S->SetLocal(S->LocalRow(idx), 0, std::sqrt(singValsSq(idx)));
        if(Sinv->IsLocal(idx, 0))
          Sinv->SetLocal(Sinv->LocalRow(idx), 0, 1/std::sqrt(singValsSq(idx)));
      }
      self->log->info("Stored V and S");

      // form U
      self->log->info("computing A*V = U*Sigma");
      self->log->info("A is {}-by-{}, V is {}-by-{}, the resulting matrix should be {}-by-{}", workingMat->Height(), workingMat->Width(), V->Height(), V->Width(), U->Height(), U->Width());
      //Gemm(1.0, *workingMat, *V, 0.0, *U, self->log);
      El::Gemm(El::NORMAL, El::NORMAL, 1.0, *workingMat, *V, 0.0, *U);
      self->log->info("done computing A*V, rescaling to get U");
      // TODO: do a QR instead to ensure stability, but does column pivoting so would require postprocessing S,V to stay consistent
      El::DiagonalScale(El::RIGHT, El::NORMAL, *Sinv, *U);
      self->log->info("Computed and stored U");

      break;
    }
  }

  self->world.barrier();
}

void NormalizeMatInPlaceCommand::run(Worker *self) const {
    auto m = self->matrices[A]->Height();
    auto n = self->matrices[A]->Width();
    auto matA = self->matrices[A].get();
    auto localRows = matA->LockedMatrix();

    auto rowMeans = El::DistMatrix<double, El::STAR, El::STAR>(n, 1, self->grid);
    auto distOnes = El::DistMatrix<double, El::STAR, El::STAR>(m, 1, self->grid);
    El::Matrix<double>localRowMeans;
    El::Matrix<double> localOnes;

    // compute the rowMeans
    // explicitly compute matrix vector products to avoid relayouts from distributed Gemv!
    El::Ones(localOnes, localRows.Height(), 1);
    El::Gemv(El::TRANSPOSE, 1.0, localRows, localOnes, 0.0, localRowMeans);
    El::Zeros(rowMeans, n, 1);
    rowMeans.Reserve(n);
    for(El::Int col=0; col < n; col++)
        rowMeans.QueueUpdate(col, 1, localRowMeans.Get(col, 1));
    rowMeans.ProcessQueues();

    // subtract off the row means
    El::Ones(distOnes, m, 1);
    El::Ger(-1.0, *matA, distOnes, rowMeans); 

    // compute the column variances
    auto colVariances = El::DistMatrix<double, El::STAR, El::STAR>(n, 1, self->grid);
    auto localSquaredEntries = El::Matrix<double>(localRows.Height(), n);
    auto localColVariances = El::Matrix<double>(n, 1);

    El::Hadamard(localRows, localRows, localSquaredEntries);
    El::Gemv(El::TRANSPOSE, 1.0, localSquaredEntries, localOnes, 0.0, localColVariances);
    El::Zeros(colVariances, n, 1);
    colVariances.Reserve(n);
    for(El::Int col=0; col < n; col++)
        colVariances.QueueUpdate(col, 1, localColVariances.Get(col, 1));
    colVariances.ProcessQueues();

    // rescale by the inv col stdevs
    auto invColStdevs = El::DistMatrix<double, El::STAR, El::STAR>(n ,1, self->grid);
    El::Zeros(invColStdevs, n, 1);
    if(invColStdevs.DistRank() == 0) {
        invColStdevs.Reserve(n);
        for(El::Int col = 0; col < n; ++col) {
            auto curInvStdev = colVariances.Get(col, 1);
            if (curInvStdev < 1e-5) {
                curInvStdev = 1.0;
            } else {
                curInvStdev = 1/sqrt(curInvStdev);
            }
            invColStdevs.QueueUpdate(col, 1, curInvStdev);
        }
    }
    else {
        invColStdevs.Reserve(0);
    }
    invColStdevs.ProcessQueues();

    El::DiagonalScale(El::RIGHT, El::NORMAL, invColStdevs, *matA);
}

void TransposeCommand::run(Worker *self) const {
  auto m = self->matrices[origMat]->Height();
  auto n = self->matrices[origMat]->Width();
  DistMatrix * transposeA = new El::DistMatrix<double, El::VR, El::STAR>(n, m, self->grid);
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
  DistMatrix * U = new El::DistMatrix<double, El::VR, El::STAR>(m, k, self->grid);
  DistMatrix * singvals = new El::DistMatrix<double, El::VR, El::STAR>(k, k, self->grid);
  DistMatrix * V = new El::DistMatrix<double, El::VR, El::STAR>(n, k, self->grid);
  El::Zero(*U);
  El::Zero(*V);
  El::Zero(*singvals);

  ENSURE(self->matrices.insert(std::make_pair(Uhandle, std::unique_ptr<DistMatrix>(U))).second);
  ENSURE(self->matrices.insert(std::make_pair(Shandle, std::unique_ptr<DistMatrix>(singvals))).second);
  ENSURE(self->matrices.insert(std::make_pair(Vhandle, std::unique_ptr<DistMatrix>(V))).second);

  DistMatrix * Acopy = new El::DistMatrix<double, El::VR, El::STAR>(m, n, self->grid); // looking at source code for SVD, seems that DistMatrix Acopy(A) might generate copy rather than just copy metadata and risk clobbering
  El::Copy(*self->matrices[mat], *Acopy);
  El::SVD(*Acopy, *U, *singvals, *V);
  std::cerr << format("%s: singvals is %s by %s\n") % self->world.rank() % singvals->Height() % singvals->Width();
  self->world.barrier();
}

void MatrixMulCommand::run(Worker *self) const {
  auto m = self->matrices[inputA]->Height();
  auto n = self->matrices[inputB]->Width();
  self->log->info("Arrived in matrix multiplication code");
  auto A = dynamic_cast<El::DistMatrix<double, El::VR, El::STAR>*>(self->matrices[inputA].get());
  auto B = dynamic_cast<El::DistMatrix<double, El::VR, El::STAR>*>(self->matrices[inputB].get());
  auto C = new El::DistMatrix<double, El::VR, El::STAR>(m, n, self->grid);
  ENSURE(self->matrices.insert(std::make_pair(handle, std::unique_ptr<DistMatrix>(C))).second);
  self->log->info("Starting multiplication");
  //Gemm(1.0, *A, *B, 0.0, *C, self->log);
  El::Gemm(El::NORMAL, El::NORMAL, 1.0, *A, *B, 0.0, *C);
  self->log->info("Done with multiplication");
  self->world.barrier();
}

// TODO: should send back blocks of rows instead of rows? maybe conversion on other side is cheaper?
void  MatrixGetRowsCommand::run(Worker * self) const {
  uint64_t numRowsFromMe = std::count(layout.begin(), layout.end(), self->id);
  self->log->info("Sending over {} rows from matrix {}", numRowsFromMe, handle);
  auto search = self->matrices.find(handle);
  if (search != self->matrices.end()) {
    self->log->info("Found it!");
  } else {
    self->log->info("Not found it!");
  }
  auto matrix = self->matrices[handle].get();
  uint64_t numCols = matrix->Width();

  self->log->info("sending over {} rows with {} cols", numRowsFromMe, numCols);

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
  self->log->info("pulled the {} rows needed to this rank, now sending them over", numRowsFromMe);

  self->sendMatrixRows(handle, matrix->Width(), layout, localRowIndices, localData);
  self->world.barrier();
}

void NewMatrixCommand::run(Worker *self) const {
  auto handle = info.handle;
  self->log->info("Creating new distributed matrix");
  DistMatrix *matrix = new El::DistMatrix<double, El::VR, El::STAR>(info.numRows, info.numCols, self->grid);
  Zero(*matrix);
  ENSURE(self->matrices.insert(std::make_pair(handle, std::unique_ptr<DistMatrix>(matrix))).second);
  self->log->info("Created new distributed matrix");

  std::vector<uint64_t> rowsOnWorker;
  self->log->info("Creating vector of local rows");
  rowsOnWorker.reserve(info.numRows);
  std::stringstream ss;
  ALCHEMIST_TRACE(ss << "Local rows: ");
  for(El::Int rowIdx = 0; rowIdx < info.numRows; ++rowIdx)
    if (matrix->IsLocalRow(rowIdx)) {
      rowsOnWorker.push_back(rowIdx);
      ALCHEMIST_TRACE(ss << rowIdx << ' ');
    }
  ALCHEMIST_TRACE(self->log->info(ss.str()));

  for(int workerIdx = 1; workerIdx < self->world.size(); workerIdx++) {
    if( self->world.rank() == workerIdx ) {
      self->world.send(0, 0, rowsOnWorker);
    }
    self->world.barrier();
  }

  self->log->info("Starting to recieve my rows");
  self->receiveMatrixBlocks(handle);
  self->log->info("Received all my matrix rows");
  self->world.barrier();
}

void HaltCommand::run(Worker *self) const {
  self->shouldExit = true;
}

struct WorkerClientSendHandler {
  int sock;
  std::shared_ptr<spdlog::logger> log;
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
  WorkerClientSendHandler(int sock, std::shared_ptr<spdlog::logger> log, MatrixHandle handle, size_t numCols, const std::vector<uint64_t> &localRowIndices, const std::vector<double> &localData) :
    sock(sock), log(log), pollEvents(POLLIN), inbuf(16), outbuf(8 + numCols * 8), inpos(0), outpos(0),
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
          // means the other side has closed the socket
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
            uint32_t typeCode = be32toh(*(uint32_t*)dataPtr);
            dataPtr += 4;
            if(typeCode == 0x3 && inpos == inbuf.size()) {
              // sendRow
              ENSURE(be32toh(*(uint32_t*)dataPtr) == handle.id);
              dataPtr += 4;
              uint64_t rowIdx = htobe64(*(uint64_t*)dataPtr);
              dataPtr += 8;
              auto localRowOffsetIter = std::find(localRowIndices.begin(), localRowIndices.end(), rowIdx);
              ENSURE(localRowOffsetIter != localRowIndices.end());
              auto localRowOffset = localRowOffsetIter - localRowIndices.begin();
              *reinterpret_cast<uint64_t*>(&outbuf[0]) = be64toh(numCols * 8);
              // treat the output as uint64_t[] instead of double[] to avoid type punning issues with be64toh
              auto invals = reinterpret_cast<const uint64_t*>(&localData[numCols * localRowOffset]);
              auto outvals = reinterpret_cast<uint64_t*>(&outbuf[8]);
              for(uint64_t idx = 0; idx < numCols; ++idx) {
                outvals[idx] = be64toh(invals[idx]);
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

struct WorkerClientReceiveHandler {
  int sock;
  short pollEvents;
  std::vector<char> inbuf;
  size_t pos;
  DistMatrix *matrix;
  MatrixHandle handle;
  std::shared_ptr<spdlog::logger> log;

  WorkerClientReceiveHandler(int sock, std::shared_ptr<spdlog::logger> log, MatrixHandle handle, DistMatrix *matrix) :
      sock(sock), log(log), pollEvents(POLLIN), inbuf(matrix->Width() * 8 + 24),
      pos(0), matrix(matrix), handle(handle) {
  }

  ~WorkerClientReceiveHandler() {
    close();
  }

  bool isClosed() const {
    return sock == -1;
  }

  void close() {
    if(sock != -1) ::close(sock);
    //log->warn("Closed socket");
    sock = -1;
    pollEvents = 0;
  }

  int handleEvent(short revents) {
    mpi::communicator world;
    int rowsCompleted = 0;
    if(revents & POLLIN && pollEvents & POLLIN) {
      while(!isClosed()) {
        //log->info("waiting on socket");
        int count = recv(sock, &inbuf[pos], inbuf.size() - pos, 0);
        //log->info("count of received bytes {}", count);
        if(count == 0) {
          break;
        } else if(count == -1) {
          if(errno == EAGAIN) {
            // no more input available until next POLLIN
            //log->warn("EAGAIN encountered");
            break;
          } else if(errno == EINTR) {
            //log->warn("Connection interrupted");
            continue;
          } else if(errno == ECONNRESET) {
            //log->warn("Connection reset");
            close();
            break;
          } else {
            log->warn("Something else happened to the connection");
            // TODO
            abort();
          }
        } else {
          ENSURE(count > 0);
          pos += count;
          ENSURE(pos <= inbuf.size());
          if(pos >= 4) {
            char *dataPtr = &inbuf[0];
            uint32_t typeCode = be32toh(*(uint32_t*)dataPtr);
            dataPtr += 4;
            if(typeCode == 0x1 && pos == inbuf.size()) {
              // addRow
              size_t numCols = matrix->Width();
              ENSURE(be32toh(*(uint32_t*)dataPtr) == handle.id);
              dataPtr += 4;
              uint64_t rowIdx = htobe64(*(uint64_t*)dataPtr);
              dataPtr += 8;
              ENSURE(rowIdx < (size_t)matrix->Height());
              ENSURE(matrix->IsLocalRow(rowIdx));
              ENSURE(htobe64(*(uint64_t*)dataPtr) == numCols * 8);
              dataPtr += 8;
              auto localRowIdx = matrix->LocalRow(rowIdx);
              //log->info("Received row {} of matrix {}, writing to local row {}", rowIdx, handle.id, localRowIdx);
              for (size_t colIdx = 0; colIdx < numCols; ++colIdx) {
                double value = ntohd(*(uint64_t*)dataPtr);
                matrix->SetLocal(localRowIdx, matrix->LocalCol(colIdx), value); //LocalCal call should be unnecessary
                dataPtr += 8;
              }
              ENSURE(dataPtr == &inbuf[inbuf.size()]);
              //log->info("Successfully received row {} of matrix {}", rowIdx, handle.id);
              rowsCompleted++;
              pos = 0;
            } else if(typeCode == 0x2) {
              //log->info("All the rows coming to me from one Spark executor have been received");
              /**struct sockaddr_storage addr;
              socklen_t len;
              char peername[255];
              int result = getpeername(sock, (struct sockaddr*)&addr, &len);
              ENSURE(result == 0);
              getnameinfo((struct sockaddr*)&addr, len, peername, 255, NULL, 0, 0);
              log->info("Received {} rows from {}", rowsCompleted, peername);
              **/
              pos = 0;
            }
          }
        }
      }
    }
    //log->info("returning from handling events");
    return rowsCompleted;
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
    //log->info("Monitoring {} sockets (one is the listening socket)", pfds.size());
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
          ENSURE(fcntl(clientSock, F_SETFL, O_NONBLOCK) != -1);
          std::unique_ptr<WorkerClientSendHandler> client(new WorkerClientSendHandler(clientSock, log, handle, numCols, localRowIndices, localData));
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

void Worker::receiveMatrixBlocks(MatrixHandle handle) {
  std::vector<std::unique_ptr<WorkerClientReceiveHandler>> clients;
  std::vector<pollfd> pfds;
  uint64_t rowsLeft = matrices[handle].get()->LocalHeight(); 
  while(rowsLeft > 0) {
    //log->info("{} rows remaining", rowsLeft);
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
    //log->info("Pushed active clients to the polling list and added listening socket");
    int count = poll(&pfds[0], pfds.size(), -1);
    if(count == -1 && (errno == EAGAIN || errno == EINTR)) continue;
    ENSURE(count != -1);
    //log->info("Polled, now handling events");
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
          ENSURE(fcntl(clientSock, F_SETFL, O_NONBLOCK) != -1);
          std::unique_ptr<WorkerClientReceiveHandler> client(new WorkerClientReceiveHandler(clientSock, log, handle, matrices[handle].get()));
          clients.push_back(std::move(client));
          //log->info("Added new client");
        } else {
          ENSURE(clients[idx]->sock == curSock);
          //log->info("Handling a client's events");
          rowsLeft -= clients[idx]->handleEvent(revents);
        }
      }
    }
  }
}

int Worker::main() {
  // log to console as well as file (single-threaded logging)
  // TODO: allow to specify log directory, log level, etc.
  // TODO: make thread-safe
  // TODO: currently both stderr and logfile share the same report levels (can't have two sinks on same log with different level); use a split sink
  //  a la https://github.com/gabime/spdlog/issues/345 to allow stderr to print error messages only
  time_t rawtime = time(0);
  struct tm * timeinfo = localtime(&rawtime);
  char buf[80];
  strftime(buf, 80, "%F-%T", timeinfo);
  std::vector<spdlog::sink_ptr> sinks;
  auto stderr_sink = std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>(); // with ANSI color
  auto logfile_sink = std::make_shared<spdlog::sinks::simple_file_sink_st>(str(format("rank-%d-%s.log") % world.rank() % buf));

  stderr_sink->set_level(spdlog::level::err);
  logfile_sink->set_level(spdlog::level::info); // change to warn for production
  sinks.push_back(stderr_sink);
  sinks.push_back(logfile_sink);
  log = std::make_shared<spdlog::logger>( str(format("worker-%d") % world.rank()),
      std::begin(sinks), std::end(sinks));
  log->flush_on(spdlog::level::info);

  log->info("Started worker");
  log->info("Max number of OpenMP threads: {}", omp_get_max_threads());

  // create listening socket, bind to an available port, and get the port number
  ENSURE((listenSock = socket(AF_INET, SOCK_STREAM, 0)) != -1);
  sockaddr_in addr = {AF_INET};
  ENSURE(bind(listenSock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
  ENSURE(listen(listenSock, 1024) == 0);
  ENSURE(fcntl(listenSock, F_SETFL, O_NONBLOCK) != -1);
  socklen_t addrlen = sizeof(addr);
  ENSURE(getsockname(listenSock, reinterpret_cast<sockaddr*>(&addr), &addrlen) == 0);
  ENSURE(addrlen == sizeof(addr));
  uint16_t port = be16toh(addr.sin_port);

  // transmit WorkerInfo to driver
  char hostname[256];
  ENSURE(gethostname(hostname, sizeof(hostname)) == 0);
  WorkerInfo info{hostname, port};
  world.send(0, 0, info);
  log->info("Listening for a connection at {}:{}", hostname, port);

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
  return Worker(world, peers).main();
}

} // namespace alchemist

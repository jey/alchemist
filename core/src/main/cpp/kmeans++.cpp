#include <iostream>
#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

namespace alchemist{

// only returns the cluster centers, in fitCenters
void kmeansPP(uint32_t seed, std::vector<MatrixXd> points, std::vector<double> weights, MatrixXd & fitCenters, uint32_t maxIters = 30) {
  std::default_random_engine randGen(seed);
  std::uniform_real_distribution<double> unifReal(0.0, 1.0);
  uint32_t n = points.size();
  uint32_t d = points[0].cols();
  uint32_t k = fitCenters.rows();
  std::vector<uint32_t> pointIndices(n);
  std::vector<uint32_t> centerIndices(k);
  std::iota(pointIndices.begin(), pointIndices.end(), 0);
  std::iota(centerIndices.begin(), centerIndices.end(), 0);

  // pick initial cluster center using weighted sampling
  double stopSum = unifReal(randGen)*std::accumulate(weights.begin(), weights.end(), 0.0);
  double curSum = 0.0;
  uint32_t searchIdx = 0;
  while(searchIdx < n && curSum < stopSum) {
    curSum += weights[searchIdx];
    searchIdx += 1;
  }
  fitCenters.row(0) = points[searchIdx - 1];

  // iteratively select next cluster centers with 
  // probability proportional to the squared distance from the previous centers
  // recall we are doing weighted k-means so min sum(w_i*d(x_i,C)) rather than sum(d(x_i,C))


  auto start = std::chrono::system_clock::now();
  std::vector<double> samplingDensity(n);
  for(auto pointIdx : pointIndices) {
    samplingDensity[pointIdx] = weights[pointIdx]*(points[pointIdx] - fitCenters.row(0)).squaredNorm();
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> elapsed_ms(end - start);
  std::cerr<< "Took " << elapsed_ms.count()<< "ms to form the first sampling density" << std::endl;

  for(uint32_t centerSelectionIdx = 1; centerSelectionIdx < k; centerSelectionIdx++) {
    stopSum = unifReal(randGen)*std::accumulate(samplingDensity.begin(), samplingDensity.end(), 0.0);
    curSum = 0.0; 
    searchIdx = 0;
    while(searchIdx < n && curSum < stopSum) {
      curSum += samplingDensity[searchIdx];
      searchIdx += 1;
    }
    // if less than k initial points explain all the data, set remaining centers to the first point
    fitCenters.row(centerSelectionIdx) = searchIdx > 0 ? points[searchIdx - 1] : points[0]; 
    for(auto pointIdx : pointIndices)
      samplingDensity[pointIdx] = std::min(samplingDensity[pointIdx], 
          weights[pointIdx]*(points[pointIdx] - fitCenters.row(centerSelectionIdx)).squaredNorm());

//    std::cerr << Eigen::Map<Eigen::RowVectorXd>(samplingDensity.data(), n) << std::endl;
  }

  // run Lloyd's algorithm stop when reached max iterations or points stop changing cluster assignments
  bool movedQ;
  std::vector<double> clusterSizes(k, 0.0);
  std::vector<uint32_t> clusterAssignments(n, 0);
  MatrixXd clusterPointSums(k, d);
  VectorXd distanceSqToCenters(k);
  uint32_t newClusterAssignment;
  double sqDist;

  uint32_t iter = 0;
  for(; iter < maxIters; iter++) {
    std::cerr << iter << std::endl;
    movedQ = false;
    clusterPointSums.setZero();
    std::fill(clusterSizes.begin(), clusterSizes.end(), 0);

    // assign each point to nearest cluster and count number of points in each cluster
    for(auto pointIdx : pointIndices) {
      for(auto centerIdx : centerIndices)
        distanceSqToCenters(centerIdx) = (points[pointIdx] - fitCenters.row(centerIdx)).squaredNorm();
      sqDist = distanceSqToCenters.minCoeff(&newClusterAssignment);
      if (newClusterAssignment != clusterAssignments[pointIdx])
        movedQ = true;
      clusterAssignments[pointIdx] = newClusterAssignment;
      clusterPointSums.row(newClusterAssignment) += weights[pointIdx]*points[pointIdx];
      clusterSizes[newClusterAssignment] += weights[pointIdx];
    }

    // stop iterations if cluster assignments have not changed
    if(!movedQ) 
      break;

    // update cluster centers
    for(auto centerIdx : centerIndices) {
      if ( clusterSizes[centerIdx] > 0 ) {
        fitCenters.row(centerIdx) = clusterPointSums.row(centerIdx) / clusterSizes[centerIdx];
      } else {
        uint32_t randPtIdx = (uint32_t) std::round(unifReal(randGen)*n);
        fitCenters.row(centerIdx) = points[randPtIdx];
      }
    }
  }

  std::cerr << "finished after " << iter << " iterations" << std::endl;

  // seems necessary to force eigen to return the centers as an actual usable matrix
  for(uint32_t rowidx = 0; rowidx < k; rowidx++)
    for(uint32_t colidx = 0; colidx < k; colidx++)
      fitCenters(rowidx, colidx) = fitCenters(rowidx, colidx) + 0.0;
}

} // end namespace alchemist

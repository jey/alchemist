#ifndef SKYLARK_READHDF5_HPP
#define SKYLARK_READHDF5_HPP

#include "alchemist.h"
#include "hdf5.h"
#include <cmath>
#include <algorithm>
#include "spdlog/spdlog.h"

typedef El::DistMatrix<double, El::VR, El::STAR> DistMatrixType;
typedef El::DistMatrix<int, El::VR, El::STAR> PermVecType;

// TODO: manage case where reading matrix too small to have info on each executor, general error handling
// NB: each HDF5 read can only ask for 2GB, according to https://support.hdfgroup.org/HDF5/faq/limits.html (bottom of page)

/**
 * Reads a double-precision matrix from an HDF5 file, returns a row-partitioned Elemental matrix
 * Using collective calls b/c serial calls too slow even to read just one of the rows from each process
 * Merging the hyperslabs is too slow, so each process loads a chunk of rows, then they are permuted to 
 * get them into the right locations
 * CAVEAT: this procedure requires (1 + replicas) as much memory as the matrix itself takes up
 * CAVEAT: this assumes all the processes participate in reading (i.e., the matrix is not too small!)
 *
 * @param fnameIn HDF5 file
 * @param varName name of the variable to be read (including path)
 * @param Y matrix to be written into
 * @param log pointer to the spd logger object to use for output
 * @param replicas number of times to replicate the input matrix (replicates column-wise)
 * @param maxMB maximum number of megabytes to read in each rank in a single hdf5 read
 */
void alchemistReadHDF5(std::string fnameIn, std::string varName, DistMatrixType & Y, 
    const std::shared_ptr<spdlog::logger> log, int replicas = 1, int maxMB = 2000) {
    hid_t file_id, dataset_id, space_id, memspace_id;
    herr_t status;
    int ndims;
    hsize_t dims[2];
    hsize_t memDims[2];
    hsize_t offset[2];
    hid_t plist_id;
    hsize_t count[2];

    El::mpi::Comm Comm= Y.Grid().Comm();
    int myRank = El::mpi::Rank(Comm);
    int numProcesses = El::mpi::Size(Comm);

    // open file for collective reading
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, Comm.comm, MPI_INFO_NULL);
    file_id = H5Fopen(fnameIn.c_str(), H5F_ACC_RDONLY, plist_id);
    status = H5Pclose(plist_id);

    // open dataset
    dataset_id = H5Dopen2(file_id, varName.c_str(), H5P_DEFAULT);
    space_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_ndims(space_id);
    if (ndims != 2) {
        log->error("Only support reading matrices");
    }

    H5Sget_simple_extent_dims(space_id, dims, NULL);
    log->info("{} rows, {} columns", dims[0], dims[1]);

    DistMatrixType X;
    X.SetGrid(Y.Grid());
    X.Resize(dims[0], dims[1]); // HDF5 returns data in row-major format, while Elemental stores it in column-major, so we'll have to transpose each local block in place

    // figure out which rows to load
    El::Int myNumRows = X.LocalHeight();
    El::Int * allRanksRows = new El::Int[numProcesses];
    El::Int myRows[1];
    myRows[0] = myNumRows;
    int sendCount = 1;
    int recvCount = 1;
    El::mpi::AllGather(myRows, sendCount, allRanksRows, recvCount, Comm);

    // Read in chunks of rows of at most maxMB megabytes assuming double precision data
    int maxRowChunk = (maxMB*1024*1024) / (dims[1]*8);
    log->info("Will read at most {} rows at a time to keep to under {} MB per read per process", maxRowChunk, maxMB);

    // figure out how many times we will have to read chunks of that size in order to get all the data in
    int maxRowsPerProcess = myNumRows;
    for(int curRank = 0; curRank < numProcesses; curRank++)
        maxRowsPerProcess = maxRowsPerProcess < allRanksRows[curRank] ? allRanksRows[curRank] : maxRowsPerProcess;
    int numReadChunks = std::max((int)std::round((maxRowsPerProcess + 0.0) / maxRowChunk), 1);
    log->info("Will need to use {} separate reads", numReadChunks);

    // figure out the first row to read for this rank
    offset[0] = 0;
    for(int preceedingRank = 0; preceedingRank < myRank; preceedingRank++) 
        offset[0] += allRanksRows[preceedingRank];
    hsize_t startRow = offset[0];
    hsize_t endRow = offset[0] + myNumRows - 1;

    // read the rows for this rank
    memDims[0] = myNumRows;
    memDims[1] = dims[1];
    memspace_id = H5Screate_simple(2, memDims, NULL);
    hsize_t memoffset[2];
    memoffset[0] = 0;
    memoffset[1] = 0;
    offset[1] = 0;
    count[1] = dims[1];
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    // collective io seems slow for this
    //status = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

    double * tempTarget = new double[myNumRows*dims[1]];

    for(int curReadNum = 0; curReadNum < numReadChunks; curReadNum++) {
        if (offset[0] == endRow) {
            // This rank is done reading its data, but we have to participate in the collective call, so just read the last row again
            count[0] = 1;
            break;
        } else {
            // This rank is still reading its data, read the next chunk of the data in
            count[0] = std::min((int)endRow - (int)offset[0] + 1, maxRowChunk);
        }
        status = H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, NULL , count, NULL);
        status = H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET, memoffset, NULL, count, NULL);

        log->info("On read {} of {}, selected and reading rows {}--{}", curReadNum + 1, numReadChunks, offset[0], offset[0] + count[0] - 1);
        //status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, space_id, plist_id, X.Buffer() + memoffset[0]*dims[1]);
        status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, space_id, plist_id, tempTarget);
        if (status < 0) {
            log->info("Error reading from file");
            MPI_Abort(Comm.comm, -1);
        }
        
        if(offset[0] != endRow) {
            offset[0] += count[0];
            memoffset[0] += count[0];
        }
        if(offset[0] == (endRow + 1)) {
            offset[0] = endRow;
            memoffset[0] = myNumRows - 1;
            log->info("Finished reading my data");
        }
    }
    status = H5Pclose(plist_id);

    // HDF5 reads in row major order, Elemental expects local matrix in column major order, so convert
    log->info("Transposing the data from HDF5 format to Elemental format");
    double * Xbuffer = X.Buffer();
    El::Int Xldim = X.LDim();
    for(El::Int j=0; j < X.LocalWidth(); ++j)
        for(El::Int i=0; i < X.LocalHeight(); ++i)
            Xbuffer[i + j*Xldim] = tempTarget[j + i*dims[1]];
    delete[] tempTarget;

    // Output raw read matrix from HDF5 before permutation for error checking
    /*
    log->info("Dumping raw matrix to file for debugging");
    std::ofstream dumpfout;
    dumpfout.open("matdump.out", std::ios::trunc);
    El::Print(X, "raw read matrix from HDF5 (before row permutation)", dumpfout);
    dumpfout.close();
    */

    // Permute the rows so they are on the right processes
    // this current rank contains rows startRow : endRow
    // row i of this chunk has the Elemental index X.GlobalRow(i),
    // and we want it to map to the actual Elemental row i
    log->info("Now computing row permutation");
    El::DistPermutation perm;
    perm.MakeIdentity(dims[0]);
    /** this should work, but does not. I think setImage's behavior does depend on the 
     * previous permutation rather than explicitly setting the image and having it never change
    for(El::Int myCurRow = 0; myCurRow < X.LocalHeight(); ++myCurRow){
      auto origin = X.GlobalRow(myCurRow);
      auto dest = startRow + myCurRow;
      perm.SetImage(origin, dest);
    }
    */

    /**
     * More complicated: create look-up tables to map between the row indices in the true matrix and the row indices in the Elemental matrix
     * create the permutation that reorders the Elemental matrix's rows so that the row indices in the Elemental matrix match those in the original matrix by
     * iterating over row indices, and for a given row index j, swapping the current elemental row j with the elemental row that contains the jth row of the
     * original matrix. update the look-up tables as we swap. in one pass over the rows, this makes the permutation we need
     **/
    El::Int * trueToEl = new El::Int[dims[0]]; // maps from row index in true matrix to current row index in Elemental matrix
    El::Int * ElToTrue = new El::Int[dims[0]]; // maps from elemental row index to current row index in true matrix
    El::Int * temp = new El::Int[dims[0]]; 

    for(El::Int curRow = 0; curRow < X.Height(); ++curRow) {
      if (curRow >= startRow && curRow <= endRow) 
        temp[curRow] = X.GlobalRow(curRow - startRow);
      else
        temp[curRow] = 0;
    }
    El::mpi::AllReduce(temp, trueToEl, dims[0], El::mpi::SUM, Comm.comm);

    for(El::Int curRow = 0; curRow < X.Height(); ++curRow) {
      if (X.IsLocalRow(curRow))
        temp[curRow] = X.LocalRow(curRow) + startRow;
      else
        temp[curRow] = 0;
    }
    El::mpi::AllReduce(temp, ElToTrue, dims[0], El::mpi::SUM, Comm.comm);

    for(El::Int curRow = 0; curRow < X.Height(); ++curRow) {
      auto origin = trueToEl[curRow];
      perm.Swap(origin, curRow);
      trueToEl[ElToTrue[curRow]] = origin;
      ElToTrue[origin] = ElToTrue[curRow];
      trueToEl[curRow] = curRow;
      ElToTrue[curRow] = curRow;
    }
    log->info("Now permuting the rows");
    perm.PermuteRows(X);
    //El::Display(X);

    delete[] trueToEl;
    delete[] ElToTrue;
    delete[] temp;

    // now replicate the cols of X to form the cols of Y
    log->info("Replicating the rows locally");
    Y.Resize(dims[0], replicas*dims[1]);
    double * Ybuffer = Y.Buffer();
    El::Int Yldim = Y.LDim();
    for(El::Int j=0; j < Y.LocalWidth(); ++j)
        for(El::Int i=0; i < Y.LocalHeight(); ++i)
            Ybuffer[i + j*Yldim] = Xbuffer[i + (j % dims[1])*Xldim];

    status = H5Sclose(memspace_id);
    status = H5Sclose(space_id);
    status = H5Dclose(dataset_id);
    status = H5Fclose(file_id);

    delete allRanksRows;
}

#endif

#ifndef SKYLARK_READHDF5_HPP
#define SKYLARK_READHDF5_HPP

#include "alchemist.h"
#include "hdf5.h"
#include <cmath>
#include <algorithm>
#include "spdlog/spdlog.h"

typedef El::DistMatrix<double, El::VR, El::STAR> DistMatrixType;
typedef El::DistMatrix<int, El::VR, El::STAR> PermVecType;


// TODO: use logger, exit with error codes, use format lib
// NB: each HDF5 read can only ask for 2GB, according to https://support.hdfgroup.org/HDF5/faq/limits.html (bottom of page)

/**
 * Reads a double-precision matrix from an HDF5 file, returns a row-partitioned Elemental matrix
 * Using collective calls b/c serial calls too slow even to read just one of the rows from each process
 * Merging the hyperslabs is too slow, so each process loads a chunk of rows, then they are permuted to 
 * get them into the right locations
 * caveat: this procedure requires twice as much memory as the matrix itself takes up
 *
 * @param fnameIn HDF5 file
 * @param varName name of the variable to be read (including path)
 * @param log pointer to the spd logger object to use for output
 * @param maxMB maximum number of megabytes to read in each rank in a single hdf5 read
 */
void alchemistReadHDF5(std::string fnameIn, std::string varName, DistMatrixType & X, 
    const std::shared_ptr<spdlog::logger> log, int maxMB = 2000) {
    hid_t file_id, dataset_id, space_id, memspace_id;
    herr_t status;
    int ndims;
    hsize_t dims[2];
    hsize_t memDims[2];
    hsize_t offset[2];
    hid_t plist_id;
    hsize_t count[2];

    El::mpi::Comm Comm= X.Grid().Comm();
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
        htri_t file_validity = H5Sselect_valid(space_id);
        htri_t mem_validity = H5Sselect_valid(memspace_id);
        if ( (file_validity  > 0) && (mem_validity > 0)) {
            log->info("The file and memory selections are valid");
        }
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
    double * buffer = X.Buffer();
    El::Int ldim = X.LDim();
    for(El::Int j=0; j < X.LocalWidth(); ++j)
        for(El::Int i=0; i < X.LocalHeight(); ++i)
            buffer[i + j*ldim] = tempTarget[j + i*dims[1]];
    delete tempTarget;

    // TODO: permute the rows so they are on the right processes
    // if entry i of the vector is j, that indicates that row j should become row i after the permutation
    //PermVecType permVec(X.Height(), 1, X.Grid);
    //El::PermuteRows(X, permVec);

    status = H5Sclose(memspace_id);
    status = H5Sclose(space_id);
    status = H5Dclose(dataset_id);
    status = H5Fclose(file_id);

    delete allRanksRows;
}
#endif

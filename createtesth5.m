% for creating sample hdf5 for testing SVD/PCA code
% about 600 MB

n = 150000;
d = 500;
fname = 'testclimatesvd.h5';
dsname = '/rows';

A = (1:n)'*ones(1,d);
h5create(fname, dsname, size(A'));
h5write(fname, dsname, A');

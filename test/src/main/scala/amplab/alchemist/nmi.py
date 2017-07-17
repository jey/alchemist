import numpy
import optparse
from sklearn.metrics.cluster import normalized_mutual_info_score


def main():
    # Parse command line options
    parser = optparse.OptionParser()
    parser.add_option('-f',
                  dest="file",
                  type="str",
                  default="kernel_kmeans_result.txt",
                  help="file path"
                  )
    options, reminder = parser.parse_args()
    FILE_PATH = options.file

    result = numpy.loadtxt(FILE_PATH, dtype='int', delimiter=' ')
    n = int(len(result) / 2)
    label_pred = result.reshape(n, 2)
    label = label_pred[:, 0]
    pred = label_pred[:, 1]
    nmi = normalized_mutual_info_score(label, pred)
    
    print('#####################################')
    print("File: " + FILE_PATH)
    print("NMI = " + str(nmi))
    print('#####################################')


if __name__ == "__main__":
    main()

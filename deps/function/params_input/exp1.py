import sys

import numpy as np


class QParams:
    def setParams():
        return 0;


    m_params = []


def main(argc, argv):

    print("argc = ", argc)

    for i in range(argc):
        print(argv[i])


if __name__ == '__main__':
    argc = len(sys.argv)
    argv = sys.argv

    main(argc, argv)

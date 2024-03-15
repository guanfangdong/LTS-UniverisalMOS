import sys

import numpy as np



def numcheck(input):
    try:
        val = int(input)

        return True
    except ValueError:
        try:
            val = float(input)
            return True

        except ValueError:
            return False


class QParams:

    def setParams(self, argc, argv):

        cnt = 0

        while cnt < argc:
            if argv[cnt] == '-pa_im':
                self.m_params['pa_im'] = argv[cnt + 1]

            if argv[cnt] == '-pa_gt':
                self.m_params['pa_gt'] = argv[cnt + 1]

            if argv[cnt] == '-pa_fg':
                self.m_params['pa_fg'] = argv[cnt + 1]

            if argv[cnt] == '-pa_net':
                self.m_params['pa_net'] = argv[cnt + 1]

            if argv[cnt] == '-pa_out':
                self.m_params['pa_out'] = argv[cnt + 1]

            if argv[cnt] == '-gpuid':
                self.m_params['gpuid'] = argv[cnt + 1]

            if argv[cnt] == '-version':
                self.m_params['version'] = argv[cnt + 1]

            if argv[cnt] == '-imgs_idx':
                self.m_params['imgs_idx'] = [argv[cnt + 1]]

                tempcnt = cnt + 1

                flag = 0
                if tempcnt == argc - 1:
                    flag = 1

                while flag == 0:
                    if numcheck(argv[tempcnt + 1]):
                        self.m_params['imgs_idx'].append(argv[tempcnt + 1])
                    else:
                        flag = 1

                    tempcnt = tempcnt + 1

                    if tempcnt == argc - 1:
                        flag = 1

            cnt = cnt + 1

        return 0;

    def getParams():
        return 0


    def __getitem__(self, idx):

        return self.m_params[idx]

    def __setitem__(self, k, v):
        self.m_params[k] = v



    m_params = {}







def main(argc, argv):

    print("argc = ", argc)

    for i in range(argc):
        print(argv[i])

    print("borderlin ========================================")
    strlist = {'m_fg': 'this is fg path'}

    print(strlist)
    print(strlist['m_fg'])
    strlist['m_gt'] = 'this is gt path'

    strlist['frameidx'] = [1, 2, 3, 4, 5]

    print(strlist['m_gt'])
    print(strlist['frameidx'])


    params = QParams()
    params['t'] = 'test'
    params['value'] = 3
    params[6] = 'test str'


    print(params['t'])
    print(params['value'])
    print(params[6])

    inputparams = QParams()
    inputparams.setParams(argc, argv)


    print(inputparams['pa_im'])

    print(numcheck(4))

    print(inputparams['imgs_idx'])


    print("borderline ==================================")
    print(inputparams['pa_im'])
    print(inputparams['pa_gt'])
    print(inputparams['pa_fg'])
    print(inputparams['pa_net'])
    print(inputparams['pa_out'])
    print(inputparams['gpuid'])
    print(inputparams['version'])
    print(inputparams['imgs_idx'])

if __name__ == '__main__':
    argc = len(sys.argv)
    argv = sys.argv

    main(argc, argv)

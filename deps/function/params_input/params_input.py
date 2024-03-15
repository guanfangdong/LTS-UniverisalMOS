import sys

import numpy as np
import random



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


        self.m_params['ft_im'] = 'jpg'
        self.m_params['ft_gt'] = 'png'
        self.m_params['preload'] = -1

        while cnt < argc:
            if argv[cnt] == '-pa_im':
                self.m_params['pa_im'] = argv[cnt + 1]

            if argv[cnt] == '-ft_im':
                self.m_params['ft_im'] = argv[cnt + 1]

            if argv[cnt] == '-ft_gt':
                self.m_params['ft_gt'] = argv[cnt + 1]

            if argv[cnt] == '-pa_gt':
                self.m_params['pa_gt'] = argv[cnt + 1]

            if argv[cnt] == '-pa_fg':
                self.m_params['pa_fg'] = argv[cnt + 1]

            if argv[cnt] == '-pa_net':
                self.m_params['pa_net'] = argv[cnt + 1]

            if argv[cnt] == '-pa_out':
                self.m_params['pa_out'] = argv[cnt + 1]

            if argv[cnt] == '-check_data':
                self.m_params['check_data'] = argv[cnt + 1]

            if argv[cnt] == '-gpuid':
                self.m_params['gpuid'] = argv[cnt + 1]

            if argv[cnt] == '-version':
                self.m_params['version'] = argv[cnt + 1]

            if argv[cnt] == '-idx_net':
                self.m_params['idx_net'] = argv[cnt + 1]

            if argv[cnt] == '-check_data':
                self.m_params['check_data'] = argv[cnt + 1]

            if argv[cnt] == '-list_pa_im':
                self.m_params['list_pa_im'] = [argv[cnt + 1]]

                tempcnt = cnt + 1

                flag = 0
                if tempcnt == argc - 1:
                    flag = 1

                while flag == 0:
                    if argv[tempcnt + 1][0] != '-':
                        self.m_params['list_pa_im'].append(argv[tempcnt + 1])
                    else:
                        flag = 1

                    tempcnt = tempcnt + 1

                    if tempcnt == argc - 1:
                        flag = 1


            if argv[cnt] == '-list_pa_fg':
                self.m_params['list_pa_fg'] = [argv[cnt + 1]]

                tempcnt = cnt + 1

                flag = 0
                if tempcnt == argc - 1:
                    flag = 1

                while flag == 0:
                    if argv[tempcnt + 1][0] != '-':
                        self.m_params['list_pa_fg'].append(argv[tempcnt + 1])
                    else:
                        flag = 1

                    tempcnt = tempcnt + 1

                    if tempcnt == argc - 1:
                        flag = 1


            if argv[cnt] == '-list_pa_gt':
                self.m_params['list_pa_gt'] = [argv[cnt + 1]]

                tempcnt = cnt + 1

                flag = 0
                if tempcnt == argc - 1:
                    flag = 1

                while flag == 0:
                    if argv[tempcnt + 1][0] != '-':
                        self.m_params['list_pa_gt'].append(argv[tempcnt + 1])
                    else:
                        flag = 1

                    tempcnt = tempcnt + 1

                    if tempcnt == argc - 1:
                        flag = 1

            if argv[cnt] == '-idx_list':
                self.m_params['idx_list'] = []
                tempidxes = []
                tempcnt = cnt + 1

                flag = 0

                while flag == 0:
                    if numcheck(argv[tempcnt]):
                        if int(argv[tempcnt]) == -1:
                            self.m_params['idx_list'].append(tempidxes)
                            tempidxes = []
                        else:
                            tempidxes.append(int(argv[tempcnt]))
                    else:
                        flag = 1

                    tempcnt = tempcnt + 1


            if argv[cnt] == '-imgs_idx':
                self.m_params['imgs_idx'] = [int(argv[cnt + 1])]

                tempcnt = cnt + 1

                flag = 0
                if tempcnt == argc - 1:
                    flag = 1

                while flag == 0:
                    if numcheck(argv[tempcnt + 1]):
                        self.m_params['imgs_idx'].append(int(argv[tempcnt + 1]))
                    else:
                        flag = 1

                    tempcnt = tempcnt + 1

                    if tempcnt == argc - 1:
                        flag = 1

            if argv[cnt] == '-train_data':
                self.m_params['train_data'] = []
                num = int(argv[cnt + 1])

                for i in range(num):
                    self.m_params['train_data'].append(argv[cnt + i + 2])
   

            if argv[cnt] == '-bk_rate':
                self.m_params['bk_rate'] = float(argv[cnt + 1])

            if argv[cnt] == '-fg_rate':
                self.m_params['fg_rate'] = float(argv[cnt + 1])

            if argv[cnt] == '-vid_rate':
                self.m_params['vid_rate'] = float(argv[cnt + 1])

            if argv[cnt] == '-check_rate':
                self.m_params['check_rate'] = float(argv[cnt + 1])

            if argv[cnt] == '-epochnum':
                self.m_params['epochnum'] = int(argv[cnt + 1])

            if argv[cnt] == '-preload':
                self.m_params['preload'] = int(argv[cnt + 1])


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

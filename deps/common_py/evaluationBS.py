import numpy as np

import torch



def evaluation_numpy_entry_torch(prelabs, trulabs):

    TP = torch.sum((prelabs == 255) & (trulabs == 255))
    FP = torch.sum((prelabs == 255) & (trulabs == 0))
    TN = torch.sum((prelabs == 0) & (trulabs == 0))
    FN = torch.sum((prelabs == 0) & (trulabs == 255))

    return TP.numpy(), FP.numpy(), TN.numpy(), FN.numpy()



def evaluation_numpy_entry(prelabs, trulabs):
    prelabs = np.reshape(prelabs, (np.size(prelabs), 1))
    trulabs = np.reshape(trulabs, (np.size(trulabs), 1))

#     TP = np.sum( (prelabs == 1) & (trulabs == 1) )
#     FP = np.sum( (prelabs == 1) & (trulabs == 0) )
#     TN = np.sum( (prelabs == 0) & (trulabs == 0) )
#     FN = np.sum( (prelabs == 0) & (trulabs == 0) )
#
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    count = np.size(prelabs)

    for i in range(count):
        prolab = int(prelabs[i])
        trulab = int(trulabs[i])

#        print("prolab:", prolab, "trulab:", trulab)

        if prolab == 255 and trulab == 255:
            TP += 1
#            print("TP")

        if prolab == 255 and trulab == 0:
            FP += 1
#            print("FP")

        if prolab == 0 and trulab == 0:
            TN += 1
#            print("TN")

        if prolab == 0 and trulab == 255:
            FN += 1
#            print("FN")





#     Re = TP/max((TP + FN), 1)
#     Pr = TP/max((TP + FP), 1)
#
#     Fm = (2*Pr*Re)/max((Pr + Re), 1)


    return TP, FP, TN, FN






def evaluation_numpy(prelabs, trulabs):
    prelabs = np.reshape(prelabs, (np.size(prelabs), 1))
    trulabs = np.reshape(trulabs, (np.size(trulabs), 1))

#     TP = np.sum( (prelabs == 1) & (trulabs == 1) )
#     FP = np.sum( (prelabs == 1) & (trulabs == 0) )
#     TN = np.sum( (prelabs == 0) & (trulabs == 0) )
#     FN = np.sum( (prelabs == 0) & (trulabs == 0) )
#
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    count = np.size(prelabs)

    for i in range(count):
        prolab = int(prelabs[i])
        trulab = int(trulabs[i])

#        print("prolab:", prolab, "trulab:", trulab)

        if prolab == 255 and trulab == 255:
            TP += 1
#            print("TP")

        if prolab == 255 and trulab == 0:
            FP += 1
#            print("FP")

        if prolab == 0 and trulab == 0:
            TN += 1
#            print("TN")

        if prolab == 0 and trulab == 255:
            FN += 1
#            print("FN")





    Re = TP/max((TP + FN), 1)
    Pr = TP/max((TP + FP), 1)

    Fm = (2*Pr*Re)/max((Pr + Re), 0.0001)


    return Re, Pr, Fm






def evaluation_BS(prelabs, trulabs):
    prelabs = np.reshape(prelabs, (np.size(prelabs), 1))
    trulabs = np.reshape(trulabs, (np.size(trulabs), 1))

#     TP = np.sum( (prelabs == 1) & (trulabs == 1) )
#     FP = np.sum( (prelabs == 1) & (trulabs == 0) )
#     TN = np.sum( (prelabs == 0) & (trulabs == 0) )
#     FN = np.sum( (prelabs == 0) & (trulabs == 0) )
#

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    count = np.size(prelabs)

    for i in range(count):
        prolab = int(prelabs[i])
        trulab = int(trulabs[i])

#        print("prolab:", prolab, "trulab:", trulab)

        if prolab == 255 and trulab == 255:
            TP += 1
#            print("TP")

        if prolab == 255 and trulab == 0:
            FP += 1
#            print("FP")

        if prolab == 0 and trulab == 0:
            TN += 1
#            print("TN")

        if prolab == 0 and trulab == 255:
            FN += 1
#            print("FN")


    Re = TP/max((TP + FN), 1)
    Pr = TP/max((TP + FP), 1)

    Fm = (2*Pr*Re)/max((Pr + Re), 0.0001)


    return Re, Pr, Fm, TP, FP, TN, FN

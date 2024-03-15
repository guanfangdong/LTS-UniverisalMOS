import torch

from torch.autograd import Function

import time


def getHist(X, delta = 0.1):
    with torch.no_grad():

        num = torch.numel(X)



        min_X = torch.round(torch.min(X).mul(1.0/delta)).mul(delta)
        max_X = torch.round(torch.max(X).mul(1.0/delta)).mul(delta) + delta


        bins = ((max_X - min_X)/delta).detach().int()


        f_X = torch.histc(X, bins, min_X, max_X)/num
        c_X = torch.linspace(min_X, max_X - delta, bins)

        return c_X, f_X


def getHist_plus(X, delta = 0.1, min_X = None, max_X = None):
    with torch.no_grad():

        num = torch.numel(X)

        flag_min = 0
        flag_max = 0


        if min_X == None:
            min_X = torch.round(torch.min(X).mul(1.0/delta)).mul(delta)
            flag_min = 1

        if max_X == None:
            max_X = torch.round(torch.max(X).mul(1.0/delta)).mul(delta)
            flag_max = 1

        if flag_min == 0:
            min_X = torch.tensor([min_X], dtype=torch.float32)

        if flag_max == 0:
            max_X = torch.tensor([max_X], dtype=torch.float32)


        max_X = max_X + delta


        bins = ((max_X - min_X).mul(1/delta)).clone().detach().int()
        bins = bins.item()


        min_X = min_X.item()
        max_X = max_X.item()


        f_X = torch.histc(X, bins, min_X, max_X)/num
        c_X = torch.linspace(min_X, max_X - delta, bins)

        c_X = torch.round(c_X/delta)*delta


        return c_X, f_X


def getEmptyCF(left, right, delta):

    bins = round((right - left)/delta) + 1

    c_X = torch.linspace(left, right, bins)

    f_X = c_X.clone() - c_X


    return c_X, f_X

def getEmptyCF_plus(left, right, delta, num):

    bins = round((right - left)/delta) + 1

    c_X = torch.linspace(left, right, bins)

    f_X = c_X.clone() - c_X
    f_X = f_X.unsqueeze(0)

    for i in range(num - 1):
        f_T = c_X.clone() - c_X
        f_T = f_T.unsqueeze(0)

        f_X = torch.cat((f_X, f_T), dim = 0)

    return c_X, f_X



def getRandCF(left, right, delta):

    bins = round((right - left)/delta) + 1

    c_X = torch.linspace(left, right, bins)

    f_X = torch.abs(torch.randn(bins))
    f_X = f_X/torch.sum(f_X)


    return c_X, f_X


def getRandCF_plus(left, right, delta, num):

    bins = round((right - left)/delta) + 1

    c_X = torch.linspace(left, right, bins)

    f_X = torch.abs(torch.randn(bins))
    f_X = f_X/torch.sum(f_X)
    f_X = f_X.unsqueeze(0)

    for i in range(num - 1):
        f_T = torch.abs(torch.randn(bins))
        f_T = f_T/torch.torch.sum(f_T)
        f_T = f_T.unsqueeze(0)

        f_X = torch.cat((f_X, f_T), dim = 0)


    return c_X, f_X

def getRandCF_multi(left, right, delta, num, byte):

    c_X, f_X = getRandCF_plus(left, right, delta, num)
    f_X = f_X.unsqueeze(-1)

    for i in range(1, byte):
        c_T, f_T = getRandCF_plus(left, right, delta, num)
        f_T = f_T.unsqueeze(-1)

        f_X = torch.cat((f_X, f_T), dim = -1)

    return c_X, f_X



def corDisVal(c_X, f_X, c_Y, f_Y):

    with torch.no_grad():

        left  = torch.max(torch.min(c_X), torch.min(c_Y))
        right = torch.min(torch.max(c_X), torch.max(c_Y))

        border = c_X[1] - c_X[0]


        pos_l = (torch.abs(c_X - left)  < 0.5*border).nonzero()
        pos_r = (torch.abs(c_X - right) < 0.5*border).nonzero()

        f_X = f_X[pos_l:pos_r + 1]


        pos_l = (torch.abs(c_Y - left)  < 0.5*border).nonzero()
        pos_r = (torch.abs(c_Y - right) < 0.5*border).nonzero()

        f_Y = f_Y[pos_l:pos_r + 1]

        return torch.sum(f_X.mul(f_Y))/(f_X.mul(f_X).sum().sqrt() * f_Y.mul(f_Y).sum().sqrt() )


class ProductDis_dev(Function):

    @staticmethod
    def forward(ctx, vc_X, vf_X, vc_W, vf_W, c_Z, f_Z,
                border = 0.1,
                params = {'zero_swap': True, 'zero_approx': True, 'normal': True}):

        with torch.no_grad():

            px0 = 0
            pw0 = 0

            if params['zero_swap']:
                px0 = vf_X[vc_X == 0]
                pw0 = vf_W[vc_W == 0]

                if torch.numel(px0) == 0:
                    px0 = 0

                if torch.numel(pw0) == 0:
                    pw0 = 0

                if pw0 > px0:
                    c_X = vc_X
                    f_X = vf_X

                    c_W = vc_W
                    f_W = vf_W
                else:
                    c_X = vc_W
                    f_X = vf_W

                    c_W = vc_X
                    f_W = vf_X



            N_X = torch.numel(c_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_X = c_X.expand(N_Z, N_X).t()



            W = c_Z / cc_X
            pos = torch.round( W/border ) - torch.round( torch.min(c_W/border) )


            ff_W = torch.cat((f_W, f_W[-2:-1]), dim=0)
            ff_W[N_W] = 0.0


            pos[ ~( (pos > -1) & (pos < torch.numel(c_W) ) ) ] = N_W
            pos = pos.long()

            mf_W = ff_W[pos]


            p_X = f_X.clone()
            p_X[c_X == 0] = 0

            mp_X = p_X.expand(N_Z, N_X).t()


            cc_X = c_X.clone()
    #            cc_X[c_X == 0] = 1
            mc_X = cc_X.expand(N_Z, N_X).t()
            mc_X = torch.abs(1/mc_X)
            mc_X[mc_X == float('inf') ] = 0


            f_Z = torch.sum( mp_X.mul(mf_W).mul(mc_X), dim = 0)


            # when x equals to 0
            if params['zero_approx']:

                deltax = 2.0*border
                x_l = 0 - 1.0*border
                x_r = 0 + 1.0*border

                w_l = c_Z/x_l
                w_r = c_Z/x_r

                idx_l = torch.round(w_l/border) - torch.round( torch.min(c_W/border) )
                idx_r = torch.round(w_r/border) - torch.round( torch.min(c_W/border) )


                idx_l[ ~( (idx_l > -1) & (idx_l < torch.numel(c_W) ) ) ] = N_W
                idx_r[ ~( (idx_r > -1) & (idx_r < torch.numel(c_W) ) ) ] = N_W

                idx_l = idx_l.long()
                idx_r = idx_r.long()


                zp_W = (ff_W[idx_l] + ff_W[idx_r])/deltax
                zp_W = zp_W*1.13


                zf_X = f_X[torch.abs(c_X) < 0.5*border]
                if torch.numel(zf_X) != 1:
                    zf_X = 1

                value = zp_W*zf_X

                f_Z = f_Z + value




            if params['normal']:

                f_Z = f_Z/torch.sum(f_Z)

#        print("forward")
        ctx.save_for_backward(c_Z, vc_W, vc_X, vf_X, torch.tensor([border]))

        return c_Z, f_Z

    @staticmethod
    def backward(ctx, c_grad_output, f_grad_output):

        with torch.no_grad():


            # 0 的情况没有像forward那样好好处理

            c_Z, c_W, c_X, f_X, border = ctx.saved_tensors
            border = border.item()


            N_X = torch.numel(f_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_W = c_W.expand(N_Z, N_W).t()
            X = c_Z / cc_W

            pos = torch.round( X/border ) - torch.round( torch.min(c_X/border) )


            pos[ ~( (pos > -1) & (pos < torch.numel(c_X) ) ) ] = N_X
            pos = pos.long()


            ff_X = torch.cat((f_X, f_X[-2:-1]), dim=0)
            ff_X[N_X] = 0.0

            mf_X = ff_X[pos]


            cc_W = c_W.clone()
            mc_W = cc_W.expand(N_Z, N_W).t()
            mc_W = torch.abs(1/mc_W)
            mc_W[mc_W == float('inf')] = 0



            df_Z = f_grad_output.clone()

            dt_W = (df_Z*mf_X).mul(mc_W)

            df_W = torch.sum(dt_W, dim=1)


        return None, None, None, df_W, None, None, None, None

class ProductDis_fast(Function):


    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, c_Z, f_Z,
                border = 0.1,
                params = {'zero_swap': True, 'zero_approx': True, 'normal': True}):

        with torch.no_grad():
            N_X = torch.numel(c_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_X = c_X.expand(N_Z, N_X).t()



            W = c_Z / cc_X
            pos = torch.round( W/border ) - torch.round( torch.min(c_W/border) )


            ff_W = torch.cat((f_W, f_W[-2:-1]), dim=0)
            ff_W[N_W] = 0.0


            pos[ ~( (pos > -1) & (pos < torch.numel(c_W) ) ) ] = N_W
            pos = pos.long()

            mf_W = ff_W[pos]


            p_X = f_X.clone()
            p_X[c_X == 0] = 0

            mp_X = p_X.expand(N_Z, N_X).t()


            cc_X = c_X.clone()
    #            cc_X[c_X == 0] = 1
            mc_X = cc_X.expand(N_Z, N_X).t()
            mc_X = torch.abs(1/mc_X)
            mc_X[mc_X == float('inf') ] = 0


            f_Z = torch.sum( mp_X.mul(mf_W).mul(mc_X), dim = 0)


        ctx.save_for_backward(c_Z, c_W, c_X, f_X, torch.tensor([border]))

        return c_Z, f_Z

    @staticmethod
    def backward(ctx, c_grad_output, f_grad_output):

        with torch.no_grad():


            # 0 的情况没有像forward那样好好处理

            c_Z, c_W, c_X, f_X, border = ctx.saved_tensors
            border = border.item()


            N_X = torch.numel(f_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_W = c_W.expand(N_Z, N_W).t()
            X = c_Z / cc_W

            pos = torch.round( X/border ) - torch.round( torch.min(c_X/border) )


            pos[ ~( (pos > -1) & (pos < torch.numel(c_X) ) ) ] = N_X
            pos = pos.long()


            ff_X = torch.cat((f_X, f_X[-2:-1]), dim=0)
            ff_X[N_X] = 0.0

            mf_X = ff_X[pos]


            cc_W = c_W.clone()
            mc_W = cc_W.expand(N_Z, N_W).t()
            mc_W = torch.abs(1/mc_W)
            mc_W[mc_W == float('inf')] = 0



            df_Z = f_grad_output.clone()

            dt_W = (df_Z*mf_X).mul(mc_W)

            df_W = torch.sum(dt_W, dim=1)


        return None, None, None, df_W, None, None, None, None






class ProductDis(Function):


    @staticmethod
    def forward(ctx, vc_X, vf_X, vc_W, vf_W, c_Z, f_Z,
                border = 0.1,
                params = {'zero_swap': True, 'zero_approx': True, 'normal': True}):

        with torch.no_grad():

            px0 = 0
            pw0 = 0

            if params['zero_swap']:
                px0 = vf_X[vc_X == 0]
                pw0 = vf_W[vc_W == 0]

                if torch.numel(px0) == 0:
                    px0 = 0

                if torch.numel(pw0) == 0:
                    pw0 = 0

                if pw0 > px0:
                    c_X = vc_X
                    f_X = vf_X

                    c_W = vc_W
                    f_W = vf_W
                else:
                    c_X = vc_W
                    f_X = vf_W

                    c_W = vc_X
                    f_W = vf_X



            N_X = torch.numel(c_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_X = c_X.expand(N_Z, N_X).t()



            W = c_Z / cc_X
            pos = torch.round( W/border ) - torch.round( torch.min(c_W/border) )


            ff_W = torch.cat((f_W, f_W[-2:-1]), dim=0)
            ff_W[N_W] = 0.0


            pos[ ~( (pos > -1) & (pos < torch.numel(c_W) ) ) ] = N_W
            pos = pos.long()

            mf_W = ff_W[pos]


            p_X = f_X.clone()
            p_X[c_X == 0] = 0

            mp_X = p_X.expand(N_Z, N_X).t()


            cc_X = c_X.clone()
    #            cc_X[c_X == 0] = 1
            mc_X = cc_X.expand(N_Z, N_X).t()
            mc_X = torch.abs(1/mc_X)
            mc_X[mc_X == float('inf') ] = 0


            f_Z = torch.sum( mp_X.mul(mf_W).mul(mc_X), dim = 0)


            # when x equals to 0
            if params['zero_approx']:

                deltax = 2.0*border
                x_l = 0 - 1.0*border
                x_r = 0 + 1.0*border

                w_l = c_Z/x_l
                w_r = c_Z/x_r

                idx_l = torch.round(w_l/border) - torch.round( torch.min(c_W/border) )
                idx_r = torch.round(w_r/border) - torch.round( torch.min(c_W/border) )


                idx_l[ ~( (idx_l > -1) & (idx_l < torch.numel(c_W) ) ) ] = N_W
                idx_r[ ~( (idx_r > -1) & (idx_r < torch.numel(c_W) ) ) ] = N_W

                idx_l = idx_l.long()
                idx_r = idx_r.long()


                zp_W = (ff_W[idx_l] + ff_W[idx_r])/deltax
                zp_W = zp_W*1.13


                zf_X = f_X[torch.abs(c_X) < 0.5*border]
                if torch.numel(zf_X) != 1:
                    zf_X = 1

                value = zp_W*zf_X

                f_Z = f_Z + value




            if params['normal']:

                f_Z = f_Z/torch.sum(f_Z)

#        print("forward")
        ctx.save_for_backward(c_Z, vc_W, vc_X, vf_X, torch.tensor([border]))

        return c_Z, f_Z

    @staticmethod
    def backward(ctx, c_grad_output, f_grad_output):

        with torch.no_grad():


            # 0 的情况没有像forward那样好好处理

            c_Z, c_W, c_X, f_X, border = ctx.saved_tensors
            border = border.item()


            N_X = torch.numel(f_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_W = c_W.expand(N_Z, N_W).t()
            X = c_Z / cc_W

            pos = torch.round( X/border ) - torch.round( torch.min(c_X/border) )


            pos[ ~( (pos > -1) & (pos < torch.numel(c_X) ) ) ] = N_X
            pos = pos.long()


            ff_X = torch.cat((f_X, f_X[-2:-1]), dim=0)
            ff_X[N_X] = 0.0

            mf_X = ff_X[pos]


            cc_W = c_W.clone()
            mc_W = cc_W.expand(N_Z, N_W).t()
            mc_W = torch.abs(1/mc_W)
            mc_W[mc_W == float('inf')] = 0



            df_Z = f_grad_output.clone()

            dt_W = (df_Z*mf_X).mul(mc_W)

            df_W = torch.sum(dt_W, dim=1)


        return None, None, None, df_W, None, None, None, None



class ProductDis_plus(Function):


    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, c_Z,
                border = 0.1,
                params = {'zero_swap': True, 'zero_approx': True, 'normal': True}):

        with torch.no_grad():
            N_X = torch.numel(c_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_X = c_X.expand(N_Z, N_X).t()



            W = c_Z / cc_X
            pos = torch.round( W/border ) - torch.round( torch.min(c_W/border) )


            ff_W = torch.cat((f_W, f_W[-2:-1]), dim=0)
            ff_W[N_W] = 0.0


            pos[ ~( (pos > -1) & (pos < torch.numel(c_W) ) ) ] = N_W
            pos = pos.long()

            mf_W = ff_W[pos]


            p_X = f_X.clone()
            p_X[c_X == 0] = 0

            mp_X = p_X.expand(N_Z, N_X).t()


            cc_X = c_X.clone()
    #            cc_X[c_X == 0] = 1
            mc_X = cc_X.expand(N_Z, N_X).t()
            mc_X = torch.abs(1/mc_X)
            mc_X[mc_X == float('inf') ] = 0


            f_Z = torch.sum( mp_X.mul(mf_W).mul(mc_X), dim = 0)


#        print('forward')


        ctx.save_for_backward(c_Z, c_W, c_X, f_X, torch.tensor([border]))

        return c_Z, f_Z

    @staticmethod
    def backward(ctx, c_grad_output, f_grad_output):

        with torch.no_grad():


            # 0 的情况没有像forward那样好好处理

            c_Z, c_W, c_X, f_X, border = ctx.saved_tensors
            border = border.item()


            N_X = torch.numel(f_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_W = c_W.expand(N_Z, N_W).t()
            X = c_Z / cc_W

            pos = torch.round( X/border ) - torch.round( torch.min(c_X/border) )


            pos[ ~( (pos > -1) & (pos < torch.numel(c_X) ) ) ] = N_X
            pos = pos.long()


            ff_X = torch.cat((f_X, f_X[-2:-1]), dim=0)
            ff_X[N_X] = 0.0

            mf_X = ff_X[pos]


            cc_W = c_W.clone()
            mc_W = cc_W.expand(N_Z, N_W).t()
            mc_W = torch.abs(1/mc_W)
            mc_W[mc_W == float('inf')] = 0



            df_Z = f_grad_output.clone()

            dt_W = (df_Z*mf_X).mul(mc_W)

            df_W = torch.sum(dt_W, dim=1)


#        print("backward")


        return None, None, None, df_W, None, None, None, None




class DifferentiateDis_bias(Function):

    @staticmethod
    def forward(ctx, c_X, f_X, c_B, f_B, c_Z,
                border = 0.1,
                params = {'zero_swap': True, 'zero_approx': True, 'normal': True}):

#         print("in function")


        NUM_X, NUM_W, LEN_X = f_X.shape

        NUM_W, LEN_B = f_B.shape

        LEN_Z = torch.numel(c_Z)


#         print("f_X.shape:", f_X.shape)
#         print("f_B.shape:", f_B.shape)
#         print("c_Z.shape:", c_Z.shape)


        cc_B = c_B.expand(LEN_Z, LEN_B).t()

#         print("c_Z.shape:", c_Z.shape)
#         print("cc_B.shape:", cc_B.shape)
        X = c_Z - cc_B

        pos = torch.round( X/border ) - torch.round( torch.min(c_X/border) )


        pos[ ~( (pos > -1) & (pos < torch.numel(c_X) ) ) ] = LEN_X
        pos = pos.long()


        ff_X = torch.cat((f_X, f_X[:,:,  -2:-1]), dim=2)
        ff_X[:, :, LEN_X] = 0.0

        mf_X = ff_X[:, :, pos]

#         print("mf_X.shape:", mf_X.shape)
#
#
#         print("f_B.shape:", f_B.shape)


        mf_X = mf_X.permute(0, 3, 1, 2)

        f_Z = torch.sum( mf_X[:, :].mul(f_B), dim=3 ).permute(0, 2, 1)

#         print("mf_X.shape:", mf_X.shape)
#
#         print("f_Z.shape:", f_Z.shape)
#
#
#
#         print("below borderline --------------------------")


#        print(f_X.shape)

        ctx.save_for_backward(mf_X)




        return c_Z, f_Z

    @staticmethod
    def backward(ctx, c_grad_output, f_grad_output):

#         print("backward")
#
#         print("borderline ----------------------------")

        smf_X, = ctx.saved_tensors
        smf_X = smf_X.permute(3, 0, 2, 1)

        df_Z = f_grad_output


        df_B = smf_X[:].mul(df_Z)

        df_B = torch.sum(df_B, dim=3)
        df_B = df_B.permute(1, 2, 0)


#         print(df_B.shape)
#
#
#         print("smf_X.shape:", smf_X.shape)
#         print("df_Z.shape:", df_Z.shape)
#
#         print("borderline ----------------------------")

        return None, None, None, df_B, None, None, None, None



class DifferentiateDis_multi(Function):

    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, c_Z,
                border = 0.1,
                params = {'zero_swap': True, 'zero_approx': True, 'normal': True}):

        with torch.no_grad():

            N_X, L_X = f_X.shape
            N_W, L_W = f_W.shape


            NUM_X, LEN_X = f_X.shape
            NUM_W, LEN_W = f_W.shape

            LEN_Z = torch.numel(c_Z)


            cc_W = c_W.expand(LEN_Z, LEN_W).t()
            X = c_Z - cc_W


            pos = torch.round( X/border ) - torch.round( torch.min(c_X/border) )


            pos[ ~( (pos > -1) & (pos < torch.numel(c_X) ) ) ] = LEN_X
            pos = pos.long()


            ff_X = torch.cat((f_X, f_X[:, -2:-1]), dim=1)
            ff_X[:, LEN_X] = 0.0

            mf_X = ff_X[:, pos]


            mf_X = mf_X.expand(NUM_W, NUM_X, LEN_W, LEN_Z).permute(1, 0, 2, 3)


            mf_W = f_W.clone()
            mf_W[:, c_W == 0] = 0
            mf_W = mf_W.expand(LEN_Z, NUM_W, LEN_W).permute(1, 2, 0)


#             cc_W = c_W.clone()
#             cc_W = cc_W.expand(LEN_Z, LEN_W).t()
#             mc_W = torch.abs(1/cc_W)
#             mc_W[mc_W == float('inf')] = 0


 #           print("mc_W:", mc_W.shape)
#             print("mf_W:", mf_W.shape)
#             print("mf_X:", mf_X.shape)

            f_Z = mf_X[:].mul(mf_W)


#            f_Z = mf_X[:].mul(mf_W[:].mul(mc_W))



            f_Z = torch.sum(f_Z, dim = 2)

#        ctx.save_for_backward( mf_X, mc_W)
        ctx.save_for_backward(mf_X)

        return c_Z, f_Z


    @staticmethod
    def backward(ctx, c_grad_output, f_grad_output):

        with torch.no_grad():

#             mf_X, = ctx.saved_tensors
#
#             df_Z = f_grad_output.clone()


#             print("backward")
#             print("mf_X.shape:", mf_X.shape)
#
#             print("df_Z.shape:", df_Z.shape)


            smf_X, = ctx.saved_tensors
#            print("bacdword borderline -------------------------")

#            print("c_grad_output:", c_grad_output.shape)
#            print("f_grad_output:", f_grad_output.shape)

            df_Z = f_grad_output.clone()

            dt_W = smf_X
            #[:, :].mul(smc_W)
            dt_W = dt_W.permute(2, 0, 1, 3)

            df_W = dt_W[:].mul(df_Z)

            df_W = df_W.permute(1, 2, 0, 3)

            df_W = torch.sum(df_W, dim = 3)




        return None, None, None, df_W, None, None, None





class DifferentiateDis(Function):

    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, c_Z,
                border = 0.1):

        with torch.no_grad():


            N_X = torch.numel(c_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_X = c_X.expand(N_Z, N_X).t()



            W = c_Z - cc_X
            pos = torch.round( W/border ) - torch.round( torch.min(c_W/border) )


            ff_W = torch.cat((f_W, f_W[-2:-1]), dim=0)
            ff_W[N_W] = 0.0


            pos[ ~( (pos > -1) & (pos < torch.numel(c_W) ) ) ] = N_W
            pos = pos.long()

            mf_W = ff_W[pos]


            p_X = f_X.clone()
            p_X[c_X == 0] = 0

            mp_X = p_X.expand(N_Z, N_X).t()


            f_Z = torch.sum( mp_X.mul(mf_W), dim = 0)


            return c_Z, f_Z



    @staticmethod
    def backward(ctx, c_grad_output, f_grad_output):

        return None, None, None, None, None, None




class ProductDis_multi(Function):


    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, c_Z,
                border = 0.1,
                params = {'zero_swap': True, 'zero_approx': True, 'normal': True}):

        with torch.no_grad():

            N_X, L_X = f_X.shape
            N_W, L_W = f_W.shape


            NUM_X, LEN_X = f_X.shape
            NUM_W, LEN_W = f_W.shape

            LEN_Z = torch.numel(c_Z)



            cc_W = c_W.expand(LEN_Z, LEN_W).t()
            X = c_Z / cc_W


#            cc_W = c_W.expand(LEN_Z, LEN_W).t()
            mc_W = torch.abs(1/cc_W)
            mc_W[mc_W == float('inf')] = 0


            pos = torch.round( X/border ) - torch.round( torch.min(c_X/border) )


            pos[ ~( (pos > -1) & (pos < torch.numel(c_X) ) ) ] = LEN_X
            pos = pos.long()


            ff_X = torch.cat((f_X, f_X[:, -2:-1]), dim=1)
            ff_X[:, LEN_X] = 0.0

            mf_X = ff_X[:, pos]


            mf_X = mf_X.expand(NUM_W, NUM_X, LEN_W, LEN_Z).permute(1, 0, 2, 3)


            mf_W = f_W.clone()
            mf_W[:, c_W == 0] = 0
            mf_W = mf_W.expand(LEN_Z, NUM_W, LEN_W).permute(1, 2, 0)




            f_Z = mf_X[:].mul(mf_W[:].mul(mc_W))


            f_Z = torch.sum(f_Z, dim = 2)

        ctx.save_for_backward( mf_X, mc_W)


#        ctx.save_for_backward(c_Z, c_W, c_X, f_X, torch.tensor([border]))

        return c_Z, f_Z

    @staticmethod
    def backward(ctx, c_grad_output, f_grad_output):

#        print("start backword")

        with torch.no_grad():


            # 0 的情况没有像forward那样好好处理

            smf_X, smc_W = ctx.saved_tensors
#            print("bacdword borderline -------------------------")

#            print("c_grad_output:", c_grad_output.shape)
#            print("f_grad_output:", f_grad_output.shape)

            df_Z = f_grad_output.clone()

            dt_W = smf_X[:, :].mul(smc_W)
            dt_W = dt_W.permute(2, 0, 1, 3)

            df_W = dt_W[:].mul(df_Z)

            df_W = df_W.permute(1, 2, 0, 3)

            df_W = torch.sum(df_W, dim = 3)






#            print("df_W.shape:", df_W.shape)

#
#             print("dt_W.shape:", dt_W.shape)
#
#
#             print("smf_X.shape:", smf_X.shape)
#             print("smc_W.shape:", smc_W.shape)
#
#            print("bacdword borderline -------------------------")


#            c_Z, c_W, c_X, f_X, border = ctx.saved_tensors

#        print("finish backward")



        return None, None, None, df_W, None, None, None, None







class DifferentiateDis_multi_test(Function):

    m_time1 = 0.0

    m_flag = 0
    m_pos = 0

    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, c_Z,
                border = 0.1,
                params = {'zero_swap': True, 'zero_approx': True, 'normal': True}):

        with torch.no_grad():

            N_X, L_X = f_X.shape
            N_W, L_W = f_W.shape


            NUM_X, LEN_X = f_X.shape
            NUM_W, LEN_W = f_W.shape

            LEN_Z = torch.numel(c_Z)


            if DifferentiateDis_multi_test.m_flag == 0:

                cc_W = c_W.expand(LEN_Z, LEN_W).t()
                X = c_Z - cc_W


                pos = torch.round( X/border ) - torch.round( torch.min(c_X/border) )


                pos[ ~( (pos > -1) & (pos < torch.numel(c_X) ) ) ] = LEN_X - 1
                pos = pos.long()

                DifferentiateDis_multi_test.m_flag = 1
                DifferentiateDis_multi_test.m_pos = pos

            else:
                pos = DifferentiateDis_multi_test.m_pos


            ff_X = f_X
            ff_X[:, LEN_X - 1] = 0.0

            mf_X = ff_X[:, pos]

            mf_X = mf_X.expand(NUM_W, NUM_X, LEN_W, LEN_Z).permute(1, 0, 2, 3)

            mf_W = f_W.expand(LEN_Z, NUM_W, LEN_W).permute(1, 2, 0)



            f_Z = mf_X[:].mul(mf_W)

            f_Z = torch.sum(f_Z, dim = 2)

        ctx.save_for_backward(mf_X)

        return c_Z, f_Z


    @staticmethod
    def backward(ctx, c_grad_output, f_grad_output):

        with torch.no_grad():

#             mf_X, = ctx.saved_tensors
#
#             df_Z = f_grad_output.clone()


#             print("backward")
#             print("mf_X.shape:", mf_X.shape)
#
#             print("df_Z.shape:", df_Z.shape)


            smf_X, = ctx.saved_tensors
#            print("bacdword borderline -------------------------")

#            print("c_grad_output:", c_grad_output.shape)
#            print("f_grad_output:", f_grad_output.shape)

            df_Z = f_grad_output.clone()

            dt_W = smf_X
            #[:, :].mul(smc_W)
            dt_W = dt_W.permute(2, 0, 1, 3)

            df_W = dt_W[:].mul(df_Z)

            df_W = df_W.permute(1, 2, 0, 3)

            df_W = torch.sum(df_W, dim = 3)




        return None, None, None, df_W, None, None, None




class ProductDis_multi_test(Function):

    m_pos = 0
    m_mc_W = 0

    m_cc_W = 0


    m_flag = 0

    m_time1 = 0.0
    m_time2 = 0.0


    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, c_Z,
                border = 0.1,
                params = {'zero_swap': True, 'zero_approx': True, 'normal': True}):

        with torch.no_grad():

            N_X, L_X = f_X.shape
            N_W, L_W = f_W.shape


            NUM_X, LEN_X = f_X.shape
            NUM_W, LEN_W = f_W.shape

            LEN_Z = torch.numel(c_Z)



            if ProductDis_multi_test.m_flag == 0:

                cc_W = c_W.expand(LEN_Z, LEN_W).t()
                X = c_Z / cc_W


                pos = torch.round( X/border ) - torch.round( torch.min(c_X/border) )


                pos[ ~( (pos > -1) & (pos < torch.numel(c_X) ) ) ] = LEN_X - 1
                pos = pos.long()


                mc_W = torch.abs(1/cc_W)
                mc_W[mc_W == float('inf')] = 0


#                cc_W = c_W.expand(LEN_Z, LEN_W).t()

#                ProductDis_multi_test.m_cc_W = cc_W
                ProductDis_multi_test.m_flag = 1
                ProductDis_multi_test.m_pos = pos
                ProductDis_multi_test.m_mc_W = mc_W

            else:
                pos = ProductDis_multi_test.m_pos
                mc_W = ProductDis_multi_test.m_mc_W
#                cc_W = ProductDis_multi_test.m_cc_W


            ff_X = f_X
            ff_X[:, LEN_X - 1] = 0.0



            mf_X = ff_X[:, pos]


            mf_X = mf_X.expand(NUM_W, NUM_X, LEN_W, LEN_Z).permute(1, 0, 2, 3)

            mf_W = f_W.expand(LEN_Z, NUM_W, LEN_W).permute(1, 2, 0)


            f_Z = mf_X[:].mul(mf_W[:].mul(mc_W))


            f_Z = torch.sum(f_Z, dim = 2)

        ctx.save_for_backward( mf_X, mc_W)


#        ctx.save_for_backward(c_Z, c_W, c_X, f_X, torch.tensor([border]))

        return c_Z, f_Z

    @staticmethod
    def backward(ctx, c_grad_output, f_grad_output):

#        print("start backword")

        with torch.no_grad():


            # 0 的情况没有像forward那样好好处理

            smf_X, smc_W = ctx.saved_tensors
#            print("bacdword borderline -------------------------")

#            print("c_grad_output:", c_grad_output.shape)
#            print("f_grad_output:", f_grad_output.shape)

            df_Z = f_grad_output.clone()

            dt_W = smf_X[:, :].mul(smc_W)
            dt_W = dt_W.permute(2, 0, 1, 3)

            df_W = dt_W[:].mul(df_Z)

            df_W = df_W.permute(1, 2, 0, 3)

            df_W = torch.sum(df_W, dim = 3)






#            print("df_W.shape:", df_W.shape)

#
#             print("dt_W.shape:", dt_W.shape)
#
#
#             print("smf_X.shape:", smf_X.shape)
#             print("smc_W.shape:", smc_W.shape)
#
#            print("bacdword borderline -------------------------")


#            c_Z, c_W, c_X, f_X, border = ctx.saved_tensors

#        print("finish backward")



        return None, None, None, df_W, None, None, None, None







class ProductDis_multiW(Function):


    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, c_Z,
                border = 0.1,
                params = {'zero_swap': True, 'zero_approx': True, 'normal': True}):

        with torch.no_grad():
            N_X = torch.numel(c_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_X = c_X.expand(N_Z, N_X).t()

            W = c_Z / cc_X
            pos = torch.round( W/border ) - torch.round( torch.min(c_W/border) )


            ff_W = torch.cat((f_W, f_W[:, -2:-1]), dim = 1)
            ff_W[:, N_W] = 0.0


            pos[ ~( (pos > -1) & (pos < torch.numel(c_W) ) ) ] = N_W
            pos = pos.long()

            mf_W = ff_W[:, pos]


            p_X = f_X.clone()

            # when x equals to 0, the 1/x equals to infinity
            p_X[c_X == 0] = 0

            mp_X = p_X.expand(N_Z, N_X).t()


            cc_X = c_X.clone()
    #            cc_X[c_X == 0] = 1
            mc_X = cc_X.expand(N_Z, N_X).t()
            mc_X = torch.abs(1/mc_X)
            mc_X[mc_X == float('inf') ] = 0


            f_Z = mf_W[:].mul(mp_X)[:].mul(mc_X)

            f_Z = torch.sum(f_Z, dim = 1)

#
#             f_Z = torch.sum( mp_X.mul(mf_W).mul(mc_X), dim = 0)


        print('forward')


        ctx.save_for_backward(c_Z, c_W, c_X, f_X, torch.tensor([border]))

        return c_Z, f_Z

    @staticmethod
    def backward(ctx, c_grad_output, f_grad_output):

        with torch.no_grad():


            # 0 的情况没有像forward那样好好处理

            c_Z, c_W, c_X, f_X, border = ctx.saved_tensors
            border = border.item()


            N_X = torch.numel(f_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_W = c_W.expand(N_Z, N_W).t()
            X = c_Z / cc_W

            pos = torch.round( X/border ) - torch.round( torch.min(c_X/border) )


            pos[ ~( (pos > -1) & (pos < torch.numel(c_X) ) ) ] = N_X
            pos = pos.long()


            ff_X = torch.cat((f_X, f_X[-2:-1]), dim=0)
            ff_X[N_X] = 0.0

            mf_X = ff_X[pos]


            cc_W = c_W.clone()
            mc_W = cc_W.expand(N_Z, N_W).t()
            mc_W = torch.abs(1/mc_W)
            mc_W[mc_W == float('inf')] = 0



            df_Z = f_grad_output.clone()

            dt_W = (df_Z*mf_X).mul(mc_W)

            df_W = torch.sum(dt_W, dim=1)


        print("backward")


        return None, None, None, df_W, None, None, None, None






class ProductDis_fast(Function):


    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, c_Z,
                border = 0.1,
                params = {'zero_swap': True, 'zero_approx': True, 'normal': True}):

        with torch.no_grad():
            N_X = torch.numel(c_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_W = c_W.expand(N_Z, N_W).t()
            X = c_Z / cc_W

            pos = torch.round( X/border ) - torch.round( torch.min(c_X/border) )


            pos[ ~( (pos > -1) & (pos < torch.numel(c_X) ) ) ] = N_X
            pos = pos.long()


            ff_X = torch.cat((f_X, f_X[-2:-1]), dim=0)
            ff_X[N_X] = 0.0

            mf_X = ff_X[pos]



            mf_W = f_W.clone()
            mf_W[c_W == 0] = 0

            mf_W = mf_W.expand(N_Z, N_W).t()



            cc_W = c_W.clone()
            mc_W = cc_W.expand(N_Z, N_W).t()
            mc_W = torch.abs(1/mc_W)
            mc_W[mc_W == float('inf')] = 0



            f_Z = torch.sum( mf_W.mul(mf_X).mul(mc_W), dim = 0 )



#        print('forward')


        ctx.save_for_backward( mf_X, mc_W)

        return c_Z, f_Z

    @staticmethod
    def backward(ctx, c_grad_output, f_grad_output):

        with torch.no_grad():


            # 0 的情况没有像forward那样好好处理
            smf_X, smc_W = ctx.saved_tensors



            df_Z = f_grad_output.clone()

            dt_W = (df_Z * smf_X).mul(smc_W)

            df_W = torch.sum(dt_W, dim=1)


#        print("backward")


        return None, None, None, df_W, None, None, None, None



class ProductDis_test(Function):


    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, c_Z,
                border = 0.1,
                params = {'zero_swap': True, 'zero_approx': True, 'normal': True}):

        with torch.no_grad():
            N_X = torch.numel(c_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_W = c_W.expand(N_Z, N_W).t()
            X = c_Z / cc_W

            pos = torch.round( X/border ) - torch.round( torch.min(c_X/border) )


            pos[ ~( (pos > -1) & (pos < torch.numel(c_X) ) ) ] = N_X
            pos = pos.long()


            ff_X = torch.cat((f_X, f_X[-2:-1]), dim=0)
            ff_X[N_X] = 0.0

            mf_X = ff_X[pos]



            mf_W = f_W.clone()
            mf_W[c_W == 0] = 0

            mf_W = mf_W.expand(N_Z, N_W).t()



            cc_W = c_W.clone()
            mc_W = cc_W.expand(N_Z, N_W).t()
            mc_W = torch.abs(1/mc_W)
            mc_W[mc_W == float('inf')] = 0



            f_Z = torch.sum( mf_W.mul(mf_X).mul(mc_W), dim = 0 )





        ctx.save_for_backward( mf_X, mc_W)

        return c_Z, f_Z

    @staticmethod
    def backward(ctx, c_grad_output, f_grad_output):

        with torch.no_grad():


            # 0 的情况没有像forward那样好好处理
            smf_X, smc_W = ctx.saved_tensors



            df_Z = f_grad_output.clone()

            dt_W = (df_Z * smf_X).mul(smc_W)

            df_W = torch.sum(dt_W, dim=1)


#        print("backward")


        return None, None, None, df_W, None, None, None, None













class ProDisFunction:

    def productDis_plus(c_X, f_X, c_W, f_W, c_Z, border, params, product_func):


        row, column = f_W.shape

        c_Z, f_Z = product_func(c_X, f_X, c_W, f_W[0], c_Z)
        f_Z = f_Z.unsqueeze(0)

        for i in range(row - 1):
            c_T, f_T = product_func(c_X, f_X, c_W, f_W[i + 1], c_Z)
            f_T = f_T.unsqueeze(0)

            f_Z = torch.cat((f_Z, f_T), dim = 0)


        return c_Z, f_Z



    def productDis_test(c_X, f_X, c_W, f_W, c_Z, border, params, product_func):

        row, column = f_W.shape

        c_T, f_Z = product_func(c_X, f_X, c_W, f_W[0], c_Z)
        f_Z = f_Z.unsqueeze(0)


        for i in range(row - 1):
            c_T, f_T = product_func(c_X, f_X, c_W, f_W[i + 1], c_Z)
            f_T = f_T.unsqueeze(0)


        return c_Z, f_Z.clone()


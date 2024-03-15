python -u exp2.py -pa_im /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/input/ \
                  -pa_gt /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/groundtruth/  \
                  -pa_fg /home/cqzhao/projects/matrix/data_test/fgimgs/dynamicBackground/fountain01_v6/ \
                  -pa_net /home/cqzhao/projects/matrix/data_test/refinenet/network/baseline/highway \
                  -pa_out /home/cqzhao/projects/matrix/data_test/refinenet/outputmask/baseline/highway  \
                  -gpuid 0 \
                  -version v10   \
                  -imgs_idx 1 4 5 6 3 2 3 
#>> temp.output


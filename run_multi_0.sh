echo "start training and testing"
python  -u train_test_SBR.py \
  --datatypes baseline baseline baseline baseline dynamicBackground dynamicBackground dynamicBackground dynamicBackground dynamicBackground dynamicBackground \
  --videos office highway pedestrians PETS2006 canoe fountain01 fountain02 overpass fall boats \
  --index_list 1879 831 484 945 972 1149 743 2360 2647 2013  \
  --len_index_list 1 1 1 1 1 1 1 1 1 1  \
  --test_dataset CDNet2014 CDNet2014 CDNet2014 CDNet2014 CDNet2014 CDNet2014 CDNet2014 CDNet2014 CDNet2014 CDNet2014  \
  --cuda 1   \
  --is_train False     \
  --net_type u_net \
  --test_layer 32 \
  --network_input /home/guanfang/Projects/one_model_test_multi_LA/network_reproduce_one_model/network_dis_0100_dark.pt \
  --visual_output /home/guanfang/Projects/results/all_dataset_all_seen_without_coarse/cdnet/results_visual/ \
  --txt_output /home/guanfang/Projects/results/all_dataset_all_seen_without_coarse/cdnet/results/ \
  --report_output /home/guanfang/Projects/results/all_dataset_all_seen_without_coarse/cdnet/report/
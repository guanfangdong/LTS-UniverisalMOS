echo "start baseline/office 0"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/baseline/office/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/baseline/office/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/baseline/office/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_baseline_office.output \


echo "start baseline/highway 1"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/baseline/highway/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/baseline/highway/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/baseline/highway/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_baseline_highway.output \


echo "start baseline/pedestrians 2"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/baseline/pedestrians/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/baseline/pedestrians/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/baseline/pedestrians/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_baseline_pedestrians.output \


echo "start baseline/PETS2006 3"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/baseline/PETS2006/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/baseline/PETS2006/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/baseline/PETS2006/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_baseline_PETS2006.output \


echo "start dynamicBackground/canoe 4"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/dynamicBackground/canoe/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/dynamicBackground/canoe/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/dynamicBackground/canoe/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_dynamicBackground_canoe.output \


echo "start dynamicBackground/fountain01 5"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/dynamicBackground/fountain01/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/dynamicBackground/fountain01/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/dynamicBackground/fountain01/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_dynamicBackground_fountain01.output \


echo "start dynamicBackground/fountain02 6"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/dynamicBackground/fountain02/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/dynamicBackground/fountain02/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/dynamicBackground/fountain02/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_dynamicBackground_fountain02.output \


echo "start dynamicBackground/overpass 7"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/dynamicBackground/overpass/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/dynamicBackground/overpass/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/dynamicBackground/overpass/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_dynamicBackground_overpass.output \


echo "start dynamicBackground/fall 8"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/dynamicBackground/fall/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/dynamicBackground/fall/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/dynamicBackground/fall/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_dynamicBackground_fall.output \


echo "start dynamicBackground/boats 9"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/dynamicBackground/boats/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/dynamicBackground/boats/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/dynamicBackground/boats/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_dynamicBackground_boats.output \


echo "start badWeather/blizzard 10"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/badWeather/blizzard/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/badWeather/blizzard/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/badWeather/blizzard/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_badWeather_blizzard.output \


echo "start badWeather/skating 11"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/badWeather/skating/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/badWeather/skating/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/badWeather/skating/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_badWeather_skating.output \


echo "start badWeather/snowFall 12"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/badWeather/snowFall/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/badWeather/snowFall/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/badWeather/snowFall/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_badWeather_snowFall.output \


echo "start badWeather/wetSnow 13"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/badWeather/wetSnow/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/badWeather/wetSnow/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/badWeather/wetSnow/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_badWeather_wetSnow.output \


echo "start cameraJitter/badminton 14"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/cameraJitter/badminton/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/cameraJitter/badminton/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/cameraJitter/badminton/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_cameraJitter_badminton.output \


echo "start cameraJitter/boulevard 15"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/cameraJitter/boulevard/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/cameraJitter/boulevard/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/cameraJitter/boulevard/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_cameraJitter_boulevard.output \


echo "start cameraJitter/sidewalk 16"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/cameraJitter/sidewalk/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/cameraJitter/sidewalk/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/cameraJitter/sidewalk/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_cameraJitter_sidewalk.output \


echo "start cameraJitter/traffic 17"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/cameraJitter/traffic/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/cameraJitter/traffic/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/cameraJitter/traffic/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_cameraJitter_traffic.output \


echo "start shadow/backdoor 18"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/shadow/backdoor/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/shadow/backdoor/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/shadow/backdoor/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_shadow_backdoor.output \


echo "start shadow/bungalows 19"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/shadow/bungalows/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/shadow/bungalows/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/shadow/bungalows/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_shadow_bungalows.output \


echo "start shadow/busStation 20"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/shadow/busStation/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/shadow/busStation/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/shadow/busStation/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_shadow_busStation.output \


echo "start shadow/copyMachine 21"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/shadow/copyMachine/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/shadow/copyMachine/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/shadow/copyMachine/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_shadow_copyMachine.output \


echo "start shadow/cubicle 22"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/shadow/cubicle/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/shadow/cubicle/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/shadow/cubicle/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_shadow_cubicle.output \


echo "start shadow/peopleInShade 23"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/shadow/peopleInShade/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/shadow/peopleInShade/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/shadow/peopleInShade/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_shadow_peopleInShade.output \


echo "start intermittentObjectMotion/abandonedBox 24"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/intermittentObjectMotion/abandonedBox/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/intermittentObjectMotion/abandonedBox/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/intermittentObjectMotion/abandonedBox/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_intermittentObjectMotion_abandonedBox.output \


echo "start intermittentObjectMotion/parking 25"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/intermittentObjectMotion/parking/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/intermittentObjectMotion/parking/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/intermittentObjectMotion/parking/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_intermittentObjectMotion_parking.output \


echo "start intermittentObjectMotion/sofa 26"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/intermittentObjectMotion/sofa/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/intermittentObjectMotion/sofa/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/intermittentObjectMotion/sofa/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_intermittentObjectMotion_sofa.output \


echo "start intermittentObjectMotion/streetLight 27"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/intermittentObjectMotion/streetLight/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/intermittentObjectMotion/streetLight/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/intermittentObjectMotion/streetLight/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_intermittentObjectMotion_streetLight.output \


echo "start intermittentObjectMotion/tramstop 28"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/intermittentObjectMotion/tramstop/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/intermittentObjectMotion/tramstop/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/intermittentObjectMotion/tramstop/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_intermittentObjectMotion_tramstop.output \


echo "start intermittentObjectMotion/winterDriveway 29"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/intermittentObjectMotion/winterDriveway/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/intermittentObjectMotion/winterDriveway/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/intermittentObjectMotion/winterDriveway/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_intermittentObjectMotion_winterDriveway.output \


echo "start turbulence/turbulence0 30"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/turbulence/turbulence0/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/turbulence/turbulence0/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/turbulence/turbulence0/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_turbulence_turbulence0.output \


echo "start turbulence/turbulence1 31"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/turbulence/turbulence1/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/turbulence/turbulence1/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/turbulence/turbulence1/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_turbulence_turbulence1.output \


echo "start turbulence/turbulence2 32"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/turbulence/turbulence2/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/turbulence/turbulence2/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/turbulence/turbulence2/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_turbulence_turbulence2.output \


echo "start turbulence/turbulence3 33"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/turbulence/turbulence3/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/turbulence/turbulence3/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/turbulence/turbulence3/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_turbulence_turbulence3.output \


echo "start nightVideos/bridgeEntry 34"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/nightVideos/bridgeEntry/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/nightVideos/bridgeEntry/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/nightVideos/bridgeEntry/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_nightVideos_bridgeEntry.output \


echo "start nightVideos/busyBoulvard 35"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/nightVideos/busyBoulvard/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/nightVideos/busyBoulvard/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/nightVideos/busyBoulvard/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_nightVideos_busyBoulvard.output \


echo "start nightVideos/fluidHighway 36"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/nightVideos/fluidHighway/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/nightVideos/fluidHighway/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/nightVideos/fluidHighway/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_nightVideos_fluidHighway.output \


echo "start nightVideos/streetCornerAtNight 37"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/nightVideos/streetCornerAtNight/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/nightVideos/streetCornerAtNight/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/nightVideos/streetCornerAtNight/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_nightVideos_streetCornerAtNight.output \


echo "start nightVideos/tramStation 38"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/nightVideos/tramStation/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/nightVideos/tramStation/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/nightVideos/tramStation/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_nightVideos_tramStation.output \


echo "start nightVideos/winterStreet 39"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/nightVideos/winterStreet/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/nightVideos/winterStreet/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/nightVideos/winterStreet/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_nightVideos_winterStreet.output \


echo "start PTZ/continuousPan 40"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/PTZ/continuousPan/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/PTZ/continuousPan/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/PTZ/continuousPan/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_PTZ_continuousPan.output \


echo "start PTZ/intermittentPan 41"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/PTZ/intermittentPan/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/PTZ/intermittentPan/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/PTZ/intermittentPan/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_PTZ_intermittentPan.output \


echo "start PTZ/twoPositionPTZCam 42"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/PTZ/twoPositionPTZCam/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/PTZ/twoPositionPTZCam/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/PTZ/twoPositionPTZCam/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_PTZ_twoPositionPTZCam.output \


echo "start PTZ/zoomInZoomOut 43"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/PTZ/zoomInZoomOut/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/PTZ/zoomInZoomOut/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/PTZ/zoomInZoomOut/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_PTZ_zoomInZoomOut.output \


echo "start thermal/corridor 44"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/thermal/corridor/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/thermal/corridor/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/thermal/corridor/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_thermal_corridor.output \


echo "start thermal/diningRoom 45"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/thermal/diningRoom/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/thermal/diningRoom/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/thermal/diningRoom/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_thermal_diningRoom.output \


echo "start thermal/lakeSide 46"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/thermal/lakeSide/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/thermal/lakeSide/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/thermal/lakeSide/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_thermal_lakeSide.output \


echo "start thermal/library 47"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/thermal/library/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/thermal/library/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/thermal/library/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_thermal_library.output \


echo "start thermal/park 48"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/thermal/park/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/thermal/park/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/thermal/park/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_thermal_park.output \


echo "start lowFramerate/port_0_17fps 49"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/lowFramerate/port_0_17fps/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/lowFramerate/port_0_17fps/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/lowFramerate/port_0_17fps/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_lowFramerate_port_0_17fps.output \


echo "start lowFramerate/tramCrossroad_1fps 50"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/lowFramerate/tramCrossroad_1fps/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/lowFramerate/tramCrossroad_1fps/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/lowFramerate/tramCrossroad_1fps/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_lowFramerate_tramCrossroad_1fps.output \


echo "start lowFramerate/tunnelExit_0_35fps 51"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/lowFramerate/tunnelExit_0_35fps/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/lowFramerate/tunnelExit_0_35fps/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/lowFramerate/tunnelExit_0_35fps/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_lowFramerate_tunnelExit_0_35fps.output \


echo "start lowFramerate/turnpike_0_5fps 52"
python -u DIDL_detect_sp.py \
-pa_net  /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/ \
-gpuid 0 \
-idx_net 69 \
-pa_im /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/lowFramerate/turnpike_0_5fps/input/ \
-pa_gt /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/dataset/CDNet2014_full/lowFramerate/turnpike_0_5fps/groundtruth/ \
-ft_im jpg \
-ft_gt png \
-pa_out /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/fgimgs/lowFramerate/turnpike_0_5fps/ \
>> /media/gf/6b8a0261-518c-4082-81e5-1bd7d7492cd7/drive/results/one_model_cdnet_ep69_round0/log/results_lowFramerate_turnpike_0_5fps.output \



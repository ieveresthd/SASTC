## ===== vgg11-pyramidnet20 =====
#python3 main_distill_distribute.py --snn_arch vgg11 --ann_arch pyramidnet20 --ta_path result/ann/pyramidnet20/ann_model_best.pth.tar --gpu_id 1,2,3 --log --save
## ===== vgg11-wrn_28_4 =====
#python3 main_distill_distribute.py --snn_arch vgg11 --ann_arch wrn_28_4 --ta_path result/ann/wrn_28_4/ann_model_best.pth.tar --gpu_id 1,2,3 --log --save
## ===== resnet18-wrn_28_4 =====
#python3 main_distill_distribute.py --snn_arch resnet18 --ann_arch wrn_28_4 --ta_path result/ann/wrn_28_4/ann_model_best.pth.tar --gpu_id 0,1,2,3 --log --save
## ===== resnet18-pyramidnet20 =====
#python3 main_distill_distribute.py --snn_arch resnet18 --ann_arch pyramidnet20 --ta_path result/ann/pyramidnet20/ann_model_best.pth.tar --gpu_id 0,1,2,3 --log --save

## check wrn_16_2 ann edition
#python3 train_teacher.py --arch wrn_16_2 -lr 0.1 --weight-decay 5e-4 --log --save --gpu_id 1,2,3
## ------------------------------------ SASTC T=3 --------------------------------------
## ===== wrn_16_2 -pyramidnet20 =====
#python3 main_distill_distribute.py --T 3 --snn_arch wrn_16_2 --ann_arch pyramidnet20 --ta_path result/ann/pyramidnet20/ann_model_best.pth.tar --gpu_id 1,2,3 --log --save
## ===== wrn_16_2 -wrn_28_4 =====
#python3 main_distill_distribute.py --T 3 --snn_arch wrn_16_2 --ann_arch wrn_28_4 --ta_path result/ann/wrn_28_4/ann_model_best.pth.tar --gpu_id 1,2,3 --log --save
## ===== wrn_16_2 -resnet19 =====
#python3 main_distill_distribute.py --T 3 --snn_arch wrn_16_2 --ann_arch resnet19 --ta_path result/ann/resnet19/ann_model_best.pth.tar --gpu_id 1,2,3 --log --save

# --------------------------- KD T=7 ---------------------------------
#python3 main_distill_distribute.py --T 7 --batch-size 64 --distill_type KD --snn_arch wrn_16_2 --ann_arch wrn_28_4 --ta_path result/ann/wrn_28_4/ann_model_best.pth.tar --gpu_id 0,1,2,3 --log --save
#python3 main_distill_distribute.py --T 7 --batch-size 64 --distill_type KD --snn_arch wrn_16_2 --ann_arch pyramidnet20 --ta_path result/ann/pyramidnet20/ann_model_best.pth.tar --gpu_id 0,1,2,3 --log --save
#python3 main_distill_distribute.py --T 7 --batch-size 64 --distill_type KD --snn_arch wrn_16_2 --ann_arch resnet19 --ta_path result/ann/resnet19/ann_model_best.pth.tar --gpu_id 0,1,2,3 --log --save
# --------------------------- KD T=3 ---------------------------------
#python3 main_distill_distribute.py --T 3 --distill_type KD --snn_arch wrn_16_2 --ann_arch wrn_28_4 --ta_path result/ann/wrn_28_4/ann_model_best.pth.tar --gpu_id 0,1,2,3 --log --save
#python3 main_distill_distribute.py --T 3 --distill_type KD --snn_arch wrn_16_2 --ann_arch pyramidnet20 --ta_path result/ann/pyramidnet20/ann_model_best.pth.tar --gpu_id 0,1,2,3 --log --save
#python3 main_distill_distribute.py --T 3 --distill_type KD --snn_arch wrn_16_2 --ann_arch resnet19 --ta_path result/ann/resnet19/ann_model_best.pth.tar --gpu_id 0,1,2,3 --log --save

# --------------------------- KD T=2 ---------------------------------
#python3 main_distill_distribute.py --T 2 --distill_type KD --snn_arch wrn_16_2 --ann_arch wrn_28_4 --ta_path result/ann/wrn_28_4/ann_model_best.pth.tar --gpu_id 0,1,2,3 --log --save
#python3 main_distill_distribute.py --T 2 --distill_type KD --snn_arch wrn_16_2 --ann_arch pyramidnet20 --ta_path result/ann/pyramidnet20/ann_model_best.pth.tar --gpu_id 0,1,2,3 --log --save
#python3 main_distill_distribute.py --T 2 --distill_type KD --snn_arch wrn_16_2 --ann_arch resnet19 --ta_path result/ann/resnet19/ann_model_best.pth.tar --gpu_id 0,1,2,3 --log --save
#python3 main_distill_distribute.py --T 7 --batch-size 64 --distill_type KD --snn_arch wrn_16_2 --ann_arch resnet19 --ta_path result/ann/resnet19/ann_model_best.pth.tar --gpu_id 0,1,2,3 --log --save

# ===== SASTC wrn_16_2 =====
python3 main_distill_distribute.py --batch-size 16 --T 2 --beta 800 --distill_type SASTC --snn_arch wrn_16_2 --ann_arch pyramidnet20 --ta_path result/ann/pyramidnet20/ann_model_best.pth.tar --gpu_id 0,1,2,3 --address tcp://127.0.0.1:2371 --log --save
python3 main_distill_distribute.py --batch-size 16 --T 2 --beta 800 --distill_type SASTC --snn_arch wrn_16_2 --ann_arch wrn_28_4 --ta_path result/ann/wrn_28_4/ann_model_best.pth.tar --gpu_id 0,1,2,3 --address tcp://127.0.0.1:2372 --log --save
python3 main_distill_distribute.py --batch-size 16 --T 2 --beta 800 --distill_type SASTC --snn_arch wrn_16_2 --ann_arch resnet19 --ta_path result/ann/resnet19/ann_model_best.pth.tar --gpu_id 0,1,2,3 --address tcp://127.0.0.1:2373 --log --save

python3 main_distill_distribute.py --batch-size 16 --T 3 --beta 800 --distill_type SASTC --snn_arch wrn_16_2 --ann_arch pyramidnet20 --ta_path result/ann/pyramidnet20/ann_model_best.pth.tar --gpu_id 0,1,2,3 --address tcp://127.0.0.1:2374 --log --save
python3 main_distill_distribute.py --batch-size 16 --T 3 --beta 800 --distill_type SASTC --snn_arch wrn_16_2 --ann_arch wrn_28_4 --ta_path result/ann/wrn_28_4/ann_model_best.pth.tar --gpu_id 0,1,2,3 --address tcp://127.0.0.1:2375 --log --save
python3 main_distill_distribute.py --batch-size 16 --T 3 --beta 800 --distill_type SASTC --snn_arch wrn_16_2 --ann_arch resnet19 --ta_path result/ann/resnet19/ann_model_best.pth.tar --gpu_id 0,1,2,3 --address tcp://127.0.0.1:2376 --log --save

python3 main_distill_distribute.py --batch-size 16 --T 7 --beta 800 --distill_type SASTC --snn_arch wrn_16_2 --ann_arch pyramidnet20 --ta_path result/ann/pyramidnet20/ann_model_best.pth.tar --gpu_id 0,1,2,3 --address tcp://127.0.0.1:2377 --log --save
python3 main_distill_distribute.py --batch-size 16 --T 7 --beta 800 --distill_type SASTC --snn_arch wrn_16_2 --ann_arch wrn_28_4 --ta_path result/ann/wrn_28_4/ann_model_best.pth.tar --gpu_id 0,1,2,3 --address tcp://127.0.0.1:2378 --log --save
python3 main_distill_distribute.py --batch-size 16 --T 7 --beta 800 --distill_type SASTC --snn_arch wrn_16_2 --ann_arch resnet19 --ta_path result/ann/resnet19/ann_model_best.pth.tar --gpu_id 0,1,2,3 --address tcp://127.0.0.1:2379 --log --save

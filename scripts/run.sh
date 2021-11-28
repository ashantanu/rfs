# ======================
# exampler commands on miniImageNet
# ======================

# supervised pre-training
python train_supervised.py --trial pretrain --model_path ./model_checkpoint/ --tb_path ./tboard/ --data_root ./data/

# distillation
# setting '-a 1.0' should give simimlar performance
python train_distillation.py -r 0.5 -a 0.5 --path_t /path/to/teacher.pth --trial born1 --model_path /path/to/save --tb_path /path/to/tensorboard --data_root /path/to/data_root

# evaluation
python eval_fewshot.py --model_path ./model_checkpoint/mini_simple.pth --data_root ./data/miniImageNet/

# ======================
# exampler commands for moco on miniImageNet
# ======================
# evaluation
python eval_fewshot_moco.py --model_path ../CMC/84_backup/84_miniimagenet_models/84_MoCo0.999_softmax_16384_resnet12_lr_0.03_decay_0.0001_bsz_128_crop_0.2_aug_CJ/current.pth --data_root ./data/miniImageNet/

python eval_fewshot_moco.py --model_path ../CMC/84_backup/84_miniimagenet_models/84_MoCo0.999_softmax_16384_resnet12_lr_0.03_decay_0.0001_bsz_128_crop_0.2_aug_CJ/ckpt_epoch_10.pth --data_root ./data/miniImageNet/


# ======================
# exampler commands for moco on miniImageNet - model with RFS resnet12
# ======================
# evaluation
python eval_fewshot_moco_2.py --model_path ../CMC/12_84_miniimagenet_models/84_MoCo0.999_softmax_16384_resnet12_lr_0.03_decay_0.0001_bsz_128_crop_0.2_aug_CJ/checkpoint_model.pth --data_root ./data/miniImageNet/
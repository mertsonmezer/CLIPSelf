NPROC_PER_NODE=4

torchrun --nproc_per_node $NPROC_PER_NODE -m training.main \
  --name clipself_coco_6_save6_test1_eva_vitl14_24layers \
  --model EVA02-CLIP-L-14-336 \
  --pretrained eva \
  --embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy \
  --batch-size 4 \
  --epochs 6 \
  --lr 1e-5 \
  --wd 0.1 \
  --alpha 0.95 \
  --train-ratio 1.0 \
  --downsample-factor 14 \
  --det-image-size 896 \
  --workers 4 \
  --warmup 1000 \
  --lock-image \
  --lock-image-unlocked-groups 24 \
  --extract-type v2 \
  --dataset-type grid_distill \
  --test-type coco_panoptic \
  --train-data data/coco/annotations/instances_train2017.json \
  --val-data data/coco/annotations/panoptic_val2017.json \
  --train-image-root data/coco/train2017 \
  --val-image-root data/coco/val2017 \
  --cache-dir checkpoints/EVA02_CLIP_L_336_psz14_s6B.pt \
  --zeroshot-frequency 1 \
  --log-every-n-steps 50 \
  --save-frequency 6
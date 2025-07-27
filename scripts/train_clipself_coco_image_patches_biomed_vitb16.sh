NPROC_PER_NODE=4

torchrun --nproc_per_node $NPROC_PER_NODE -m training.main \
  --name clipself_coco_6_save6_test1_biomed_vitb16_12layers_2 \
  --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
  --embed-path metadata/coco_panoptic_clip_hand_craft_biomed.npy \
  --batch-size 2 \
  --epochs 6 \
  --lr 1e-5 \
  --wd 0.1 \
  --alpha 0.7 \
  --train-ratio 0.05 \
  --downsample-factor 16 \
  --det-image-size 224 \
  --workers 4 \
  --warmup 1000 \
  --lock-image \
  --lock-image-unlocked-groups 12 \
  --extract-type v2 \
  --dataset-type grid_distill \
  --test-type coco_panoptic \
  --train-data data/coco/annotations/instances_train2017.json \
  --val-data data/coco/annotations/panoptic_val2017.json \
  --train-image-root data/coco/train2017 \
  --val-image-root data/coco/val2017 \
  --zeroshot-frequency 1 \
  --log-every-n-steps 50 \
  --save-frequency 6
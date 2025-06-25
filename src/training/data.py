import os
import random
from dataclasses import dataclass

from PIL import Image
from pycocotools.coco import COCO
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from open_clip.transform import get_scale


class ProposalDistillDataset(Dataset):
  """Placeholder for future proposal distillation dataset."""

  def __init__(self, *args, **kwargs):
    raise NotImplementedError


class GridDistillDataset(Dataset):
  """COCO grid patch dataset used for CLIPSelf training."""

  def __init__(
    self,
    annotation_file: str,
    image_root: str,
    transforms,
    max_split: int = 4,
    crop_size: int | tuple[int, int] = 224,
    max_boxes: int = 20,
  ) -> None:
    self.coco = COCO(annotation_file)
    self.image_root = image_root
    self.transforms = transforms
    self.max_split = max_split
    self.max_boxes = max_boxes
    if not isinstance(crop_size, (list, tuple)):
      crop_size = (crop_size, crop_size)
    self.crop_size = crop_size
    self.image_ids = list(self.coco.imgs.keys())
    self._init_grid_templates()

  def _init_grid_templates(self) -> None:
    self.box_templates = {}
    for m in range(1, self.max_split + 1):
      for n in range(1, self.max_split + 1):
        grid_x, grid_y = torch.meshgrid(
          torch.linspace(0, 1, n + 1),
          torch.linspace(0, 1, m + 1),
          indexing="xy",
        )
        x0y0 = torch.stack([grid_x[:-1, :-1], grid_y[:-1, :-1]], dim=-1)
        x1y1 = torch.stack([grid_x[1:, 1:], grid_y[1:, 1:]], dim=-1)
        boxes = torch.cat([x0y0, x1y1], dim=-1).view(-1, 4)
        self.box_templates[(m, n)] = boxes

  def __len__(self) -> int:  # noqa: D401 - simple override
    return len(self.image_ids)

  def _read_image(self, file_name: str) -> Image.Image:
    image_path = os.path.join(self.image_root, file_name)
    return Image.open(image_path).convert("RGB")

  def _sample_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
    indices = list(range(len(boxes)))
    random.shuffle(indices)
    indices = indices[: self.max_boxes]
    return boxes[indices]

  def __getitem__(self, idx: int):
    image_id = self.image_ids[idx]
    info = self.coco.imgs[image_id]
    image = self._read_image(info["file_name"])
    new_image = self.transforms[0](image)
    scale = get_scale(image, new_image)

    choice = random.choice(list(self.box_templates.keys()))
    boxes = self._sample_boxes(self.box_templates[choice])

    crops = []
    for box in boxes:
      x0, y0, x1, y1 = (box * torch.tensor([image.width, image.height, image.width, image.height])).tolist()
      crop = image.crop((x0, y0, x1, y1))
      crop = self.transforms[1](crop)
      crops.append(crop)
    if not crops:
      crop = self.transforms[1](image.crop((0, 0, image.width // 4, image.height // 4)))
      crops = [crop]
      boxes = boxes[:1]

    norm_boxes = torch.zeros(self.max_boxes, 5, dtype=torch.float32)
    crop_tensor = torch.zeros(self.max_boxes, 3, *self.crop_size)

    _, h, w = new_image.shape
    boxes = boxes.clone()
    boxes[:, :4] *= scale
    boxes[:, [0, 2]] /= w
    boxes[:, [1, 3]] /= h

    num = len(crops)
    norm_boxes[:num, :4] = boxes
    norm_boxes[:num, 4] = 1.0
    crop_tensor[:num] = torch.stack(crops)

    return new_image, norm_boxes, crop_tensor


@dataclass
class DataInfo:
  dataloader: DataLoader
  sampler: DistributedSampler | None = None

  def set_epoch(self, epoch: int) -> None:
    if self.sampler is not None:
      self.sampler.set_epoch(epoch)


def build_dataloader(dataset: Dataset, is_train: bool, args) -> DataInfo:
  sampler = DistributedSampler(dataset) if args.distributed else None
  shuffle = is_train and sampler is None
  dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=shuffle,
    num_workers=args.workers,
    sampler=sampler,
    pin_memory=True,
    drop_last=is_train,
  )
  dataloader.num_samples = len(dataset)
  dataloader.num_batches = len(dataloader)
  return DataInfo(dataloader, sampler)


def get_dataset_class(dataset_type: str):
  if dataset_type == "grid_distill":
    return GridDistillDataset
  if dataset_type == "proposals_distill":
    return ProposalDistillDataset
  raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch: int = 0):
  preprocess_train, preprocess_val = preprocess_fns
  data = {}

  if args.train_data:
    dataset_cls = get_dataset_class(args.dataset_type)
    dataset = dataset_cls(
      args.train_data,
      args.train_image_root,
      preprocess_train,
      max_split=args.max_split,
      crop_size=args.input_size,
      max_boxes=args.max_boxes,
    )
    data["train"] = build_dataloader(dataset, True, args)

  if args.val_data:
    dataset_cls = get_dataset_class(args.dataset_type)
    dataset = dataset_cls(
      args.val_data,
      args.val_image_root,
      preprocess_val,
      max_split=args.max_split,
      crop_size=args.input_size,
      max_boxes=args.max_boxes,
    )
    data["val"] = build_dataloader(dataset, False, args)

  return data


def demo_grid_distill(annotation_file: str, image_root: str, num_samples: int = 2) -> None:
  """Demonstrate :class:`GridDistillDataset` loading.

  Parameters
  ----------
  annotation_file: str
      Path to the COCO annotation file.
  image_root: str
      Root directory containing the images.
  num_samples: int, optional
      Number of samples to iterate through (default: 2).
  """

  from torchvision.transforms import Compose, Resize, ToTensor

  image_tf = Compose([Resize(1024), ToTensor()])
  crop_tf = Compose([Resize(224), ToTensor()])

  dataset = GridDistillDataset(annotation_file, image_root, [image_tf, crop_tf])
  loader = DataLoader(dataset, batch_size=1)

  for i, batch in enumerate(loader):
    img, boxes, crops = batch
    print("Image shape:", list(img.shape))
    print("Boxes shape:", list(boxes.shape))
    print("Crops shape:", list(crops.shape))
    print()
    if i >= num_samples - 1:
      break


if __name__ == "__main__":  # pragma: no cover - manual demo
  import argparse

  parser = argparse.ArgumentParser(description="GridDistillDataset demo")
  parser.add_argument("annotation_file", help="Path to COCO annotation JSON")
  parser.add_argument("image_root", help="Directory containing images")
  parser.add_argument("--num-samples", type=int, default=2, dest="num_samples")
  args = parser.parse_args()

  demo_grid_distill(args.annotation_file, args.image_root, args.num_samples)

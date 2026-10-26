#!/usr/bin/env python3
"""Build pretrain_full_checkpoint.pth (student/teacher wrapper) from the DINO
ImageNet ViT-S/8 backbone weights. Student is DDP-wrapped in the trainer
(keys 'module.backbone.*'); the ViT teacher is not (keys 'backbone.*').
DINO heads are left to random init."""
import sys, torch
from pathlib import Path
repo = Path(sys.argv[1])
sd = torch.load(repo / "dino_deitsmall8_pretrain.pth", map_location="cpu")
student = {f"module.backbone.{k}": v for k, v in sd.items()}
teacher = {f"backbone.{k}": v for k, v in sd.items()}
torch.save({"student": student, "teacher": teacher}, repo / "pretrain_full_checkpoint.pth")
print("wrote", repo / "pretrain_full_checkpoint.pth", len(sd), "tensors")

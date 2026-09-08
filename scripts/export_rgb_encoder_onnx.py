"""Export the RGB multi-task model to ONNX.

Example:
  pip install -e .[model]
  python scripts/export_rgb_encoder_onnx.py --out models/rgb_multitask.onnx

This exports random/untrained weights unless you supply --checkpoint. It is an
engineering/export example, not a trained garment model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from garment_grader.models.rgb_multitask import RGBMultiTaskNet


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: RGBMultiTaskNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        return (
            y["category_logits"],
            y["brand_embedding"],
            y["color_logits"],
            y["style_logits"],
            y["sleeve_logits"],
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("models/rgb_multitask.onnx"))
    ap.add_argument("--checkpoint", type=Path)
    args = ap.parse_args()

    model = RGBMultiTaskNet(
        num_categories=64,
        num_colors=16,
        num_styles=24,
        num_sleeves=4,
        pretrained=False,
    )
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()

    wrapper = ExportWrapper(model)
    dummy = torch.randn(1, 3, 224, 224)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        input_names=["image"],
        output_names=["category_logits", "brand_embedding", "color_logits", "style_logits", "sleeve_logits"],
        dynamic_axes={"image": {0: "batch"}},
        opset_version=17,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

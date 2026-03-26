#!/usr/bin/env python3
"""Create a clean moodyMix model directory by stripping the model.diffusion_model. prefix.
Data offsets are preserved exactly - only header tensor names are changed."""
import json, struct, os, shutil

def create_clean_moodymix():
    src = "moodyMix_zitV10DPO.safetensors"
    out_dir = "zimage-turbo-moodymix-clean"
    os.makedirs(f"{out_dir}/transformer", exist_ok=True)

    with open(src, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)
        tensor_data = f.read()

    meta = json.loads(header_bytes)

    # Strip prefix from all tensor names, keep data_offsets unchanged
    new_meta = {"__metadata__": meta.get("__metadata__", {})}
    for name, info in meta.items():
        if name == "__metadata__":
            continue
        clean = name.replace("model.diffusion_model.", "")
        new_meta[clean] = info

    print(f"Total tensors: {len(new_meta) - 1}")

    # Spot-check key tensors
    for check in ["x_embedder.weight", "x_embedder.bias",
                  "final_layer.linear.weight", "final_layer.adaLN_modulation.1.weight",
                  "cap_embedder.1.weight", "x_pad_token", "t_embedder.mlp.0.weight"]:
        if check in new_meta:
            print(f"  ✅ {check}: shape={new_meta[check]['shape']}")
        else:
            print(f"  ❌ MISSING: {check}")

    # Write new safetensors file (header only changes, data is identical)
    new_header = json.dumps(new_meta, separators=(",", ":")).encode("utf-8")
    out_file = f"{out_dir}/transformer/diffusion_pytorch_model.safetensors"
    with open(out_file, "wb") as f:
        f.write(struct.pack("<Q", len(new_header)))
        f.write(new_header)
        f.write(tensor_data)

    # Write index.json
    weight_map = {k: "diffusion_pytorch_model.safetensors"
                  for k in new_meta if k != "__metadata__"}
    index = {"metadata": {"total_size": len(tensor_data)}, "weight_map": weight_map}
    with open(f"{out_dir}/transformer/diffusion_pytorch_model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Copy config.json from original ZIT (architecture params are the same)
    shutil.copy2("zimage-turbo/transformer/config.json",
                 f"{out_dir}/transformer/config.json")

    # Copy shared components from original ZIT
    for item in ["model_index.json", "text_encoder", "tokenizer", "vae"]:
        src_path = f"zimage-turbo/{item}"
        dst_path = f"{out_dir}/{item}"
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        elif os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

    print(f"\n✅ Created: {out_dir}")
    print(f"   File size: {os.path.getsize(out_file) / 1e9:.1f} GB")

if __name__ == "__main__":
    create_clean_moodymix()

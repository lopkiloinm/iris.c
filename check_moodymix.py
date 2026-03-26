#!/usr/bin/env python3
import json

print("=== Analyzing moodyMix structure ===")

# Load moodyMix
with open('/Users/martinliu/Downloads/iris.c-main/moodyMix_zitV10DPO.safetensors', 'rb') as f:
    import struct
    header_len = struct.unpack('<Q', f.read(8))[0]
    header_data = f.read(header_len).decode('utf-8')
    moodymix_metadata = json.loads(header_data)

# Load original
with open('/Users/martinliu/Downloads/iris.c-main/zimage-turbo/transformer/diffusion_pytorch_model.safetensors.index.json') as f:
    original_index = json.load(f)

print("=== moodyMix Key Components ===")
moodymix_keys = set()
for name in moodymix_metadata.keys():
    if name != "__metadata__":
        clean_name = name.replace("model.diffusion_model.", "")
        moodymix_keys.add(clean_name)

# Check for essential components
essential = [
    "all_x_embedder.2-1.weight",
    "all_x_embedder.2-1.bias", 
    "all_final_layer.2-1.adaLN_modulation.1.weight",
    "all_final_layer.2-1.adaLN_modulation.1.bias",
    "all_final_layer.2-1.linear.weight",
    "all_final_layer.2-1.linear.bias"
]

print("Essential components check:")
for comp in essential:
    if comp in moodymix_keys:
        print(f"  ✅ {comp}")
    else:
        print(f"  ❌ {comp}")

print(f"\nmoodyMix has {len(moodymix_keys)} unique tensor names")
print(f"Original expects {len(original_index['weight_map'])} tensor names")

# Show what moodyMix has for layers
layer_count = 0
for name in moodymix_keys:
    if name.startswith("layers.") and ".attention.qkv.weight" in name:
        layer_count += 1

print(f"moodyMix has {layer_count} layers with fused attention")
print(f"Original has 30 layers")

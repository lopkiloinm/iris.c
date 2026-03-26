#!/usr/bin/env python3
import json
import struct
import os
import shutil

def create_fused_compatible_model():
    print("=== Creating fused attention compatible moodyMix model ===")
    
    # Load moodyMix
    moodymix_file = "/Users/martinliu/Downloads/iris.c-main/moodyMix_zitV10DPO.safetensors"
    with open(moodymix_file, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_data = f.read(header_len).decode('utf-8')
        moodymix_metadata = json.loads(header_data)
        moodymix_tensor_data = f.read()
    
    # Load original for missing components
    original_index_file = '/Users/martinliu/Downloads/iris.c-main/zimage-turbo/transformer/diffusion_pytorch_model.safetensors.index.json'
    with open(original_index_file) as f:
        original_index = json.load(f)
    
    # Create hybrid metadata with moodyMix tensors + missing original components
    hybrid_metadata = {"__metadata__": moodymix_metadata.get("__metadata__", {})}
    weight_map = {}
    
    # Add all moodyMix tensors (with name conversion)
    for name, info in moodymix_metadata.items():
        if name == "__metadata__":
            continue
        
        # Remove prefix
        clean_name = name.replace("model.diffusion_model.", "")
        hybrid_metadata[clean_name] = info
        weight_map[clean_name] = "diffusion_pytorch_model.safetensors"
    
    # Add missing essential components from original
    missing_essential = [
        "all_x_embedder.2-1.weight",
        "all_x_embedder.2-1.bias",
        "all_final_layer.2-1.adaLN_modulation.1.weight", 
        "all_final_layer.2-1.adaLN_modulation.1.bias",
        "all_final_layer.2-1.linear.weight",
        "all_final_layer.2-1.linear.bias"
    ]
    
    print(f"Adding {len(missing_essential)} missing essential components...")
    
    # Read original tensor data
    base_dir = "/Users/martinliu/Downloads/iris.c-main/zimage-turbo/transformer"
    original_tensor_data = {}
    
    # Load all original files
    for file_name in ["diffusion_pytorch_model-00001-of-00003.safetensors", 
                     "diffusion_pytorch_model-00002-of-00003.safetensors",
                     "diffusion_pytorch_model-00003-of-00003.safetensors"]:
        file_path = f"{base_dir}/{file_name}"
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                orig_header_len = struct.unpack('<Q', f.read(8))[0]
                orig_header_data = f.read(orig_header_len).decode('utf-8')
                orig_metadata = json.loads(orig_header_data)
                orig_tensor_data_section = f.read()
                
                for orig_name, orig_info in orig_metadata.items():
                    if orig_name != "__metadata__":
                        original_tensor_data[orig_name] = (orig_info, orig_tensor_data_section)
    
    # Add missing components
    for missing in missing_essential:
        if missing in original_tensor_data:
            info, data = original_tensor_data[missing]
            hybrid_metadata[missing] = info
            weight_map[missing] = "diffusion_pytorch_model.safetensors"
            print(f"  ✅ Added {missing}")
        else:
            print(f"  ❌ Could not find {missing}")
    
    print(f"Hybrid model has {len(hybrid_metadata)} tensors")
    
    # Create output directory
    output_dir = "/Users/martinliu/Downloads/iris.c-main/zimage-turbo-moodymix-fused"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/transformer", exist_ok=True)
    
    # Write index.json
    index_data = {
        "metadata": {
            "total_size": len(moodymix_tensor_data)
        },
        "weight_map": weight_map
    }
    
    with open(f"{output_dir}/transformer/diffusion_pytorch_model.safetensors.index.json", 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # Write safetensors file
    output_file = f"{output_dir}/transformer/diffusion_pytorch_model.safetensors"
    new_header_json = json.dumps(hybrid_metadata, separators=(',', ':'))
    new_header_len = len(new_header_json.encode('utf-8'))
    
    with open(output_file, 'wb') as f:
        f.write(struct.pack('<Q', new_header_len))
        f.write(new_header_json.encode('utf-8'))
        f.write(moodymix_tensor_data)
    
    # Copy other files
    for item in ["model_index.json", "text_encoder", "tokenizer", "vae"]:
        src = f"/Users/martinliu/Downloads/iris.c-main/zimage-turbo/{item}"
        dst = f"{output_dir}/{item}"
        if os.path.exists(src):
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    
    print(f"✅ Created fused-compatible model: {output_dir}")
    print(f"Test with: ./iris -d {output_dir} -p 'test prompt' -o test.png")

if __name__ == "__main__":
    create_fused_compatible_model()

#!/usr/bin/env python3
import json
import struct
import os
import shutil

def create_proper_fused_model():
    print("=== Creating proper fused attention model ===")
    
    # Load moodyMix
    moodymix_file = "/Users/martinliu/Downloads/iris.c-main/moodyMix_zitV10DPO.safetensors"
    with open(moodymix_file, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_data = f.read(header_len).decode('utf-8')
        moodymix_metadata = json.loads(header_data)
        moodymix_tensor_data = f.read()
    
    # Create output directory
    output_dir = "/Users/martinliu/Downloads/iris.c-main/zimage-turbo-moodymix-proper"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/transformer", exist_ok=True)
    
    # Start with moodyMix metadata (with prefix removed)
    final_metadata = {"__metadata__": moodymix_metadata.get("__metadata__", {})}
    
    for name, info in moodymix_metadata.items():
        if name == "__metadata__":
            continue
        # Remove prefix
        clean_name = name.replace("model.diffusion_model.", "")
        final_metadata[clean_name] = info
    
    # Load original files to extract missing tensor data
    base_dir = "/Users/martinliu/Downloads/iris.c-main/zimage-turbo/transformer"
    
    # Read original files and extract missing tensors
    original_tensors = {}
    for file_name in ["diffusion_pytorch_model-00001-of-00003.safetensors", 
                     "diffusion_pytorch_model-00002-of-00003.safetensors",
                     "diffusion_pytorch_model-00003-of-00003.safetensors"]:
        file_path = f"{base_dir}/{file_name}"
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                orig_header_len = struct.unpack('<Q', f.read(8))[0]
                orig_header_data = f.read(orig_header_len).decode('utf-8')
                orig_metadata = json.loads(orig_header_data)
                
                # Find tensor offsets
                offset = 8 + orig_header_len
                for orig_name, orig_info in orig_metadata.items():
                    if orig_name != "__metadata__":
                        # Calculate tensor data offset
                        shape = orig_info["shape"]
                        if len(shape) >= 2:
                            tensor_size = shape[0] * shape[1] * 2  # BF16
                        elif len(shape) == 1:
                            tensor_size = shape[0] * 2  # BF16
                        else:
                            tensor_size = 2  # BF16 scalar
                        original_tensors[orig_name] = (orig_info, offset, tensor_size)
                        offset += tensor_size
    
    # Extract missing tensor data and add to final metadata
    missing_essential = [
        "all_x_embedder.2-1.weight",
        "all_x_embedder.2-1.bias",
        "all_final_layer.2-1.adaLN_modulation.1.weight", 
        "all_final_layer.2-1.adaLN_modulation.1.bias",
        "all_final_layer.2-1.linear.weight",
        "all_final_layer.2-1.linear.bias"
    ]
    
    # Read original file data to extract tensors
    original_file_data = {}
    for file_name in ["diffusion_pytorch_model-00001-of-00003.safetensors", 
                     "diffusion_pytorch_model-00002-of-00003.safetensors",
                     "diffusion_pytorch_model-00003-of-00003.safetensors"]:
        file_path = f"{base_dir}/{file_name}"
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                header_len = struct.unpack('<Q', f.read(8))[0]
                header_data = f.read(header_len)
                original_file_data[file_name] = f.read()
    
    # Extract and add missing tensors
    additional_tensor_data = b""
    for missing in missing_essential:
        if missing in original_tensors:
            info, offset, size = original_tensors[missing]
            final_metadata[missing] = info
            
            # Find which file contains this tensor
            for file_name, file_data in original_file_data.items():
                if offset < len(file_data):
                    # Extract tensor data
                    tensor_data = file_data[offset:offset+size]
                    additional_tensor_data += tensor_data
                    print(f"  ✅ Added {missing} ({size} bytes)")
                    break
        else:
            print(f"  ❌ Could not find {missing}")
    
    # Combine moodyMix data with additional tensors
    final_tensor_data = moodymix_tensor_data + additional_tensor_data
    
    # Create weight map
    weight_map = {name: "diffusion_pytorch_model.safetensors" for name in final_metadata.keys() if name != "__metadata__"}
    
    # Write index.json
    index_data = {
        "metadata": {
            "total_size": len(final_tensor_data)
        },
        "weight_map": weight_map
    }
    
    with open(f"{output_dir}/transformer/diffusion_pytorch_model.safetensors.index.json", 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # Write final safetensors file
    output_file = f"{output_dir}/transformer/diffusion_pytorch_model.safetensors"
    new_header_json = json.dumps(final_metadata, separators=(',', ':'))
    new_header_len = len(new_header_json.encode('utf-8'))
    
    with open(output_file, 'wb') as f:
        f.write(struct.pack('<Q', new_header_len))
        f.write(new_header_json.encode('utf-8'))
        f.write(final_tensor_data)
    
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
    
    # Copy transformer config.json
    config_src = f"/Users/martinliu/Downloads/iris.c-main/zimage-turbo/transformer/config.json"
    config_dst = f"{output_dir}/transformer/config.json"
    if os.path.exists(config_src):
        shutil.copy2(config_src, config_dst)
        print(f"✅ Copied transformer config.json")
    
    print(f"✅ Created proper fused model: {output_dir}")
    print(f"Final model has {len(final_metadata)} tensors")
    print(f"Test with: ./iris -d {output_dir} -p 'test prompt' -o test.png")

if __name__ == "__main__":
    create_proper_fused_model()

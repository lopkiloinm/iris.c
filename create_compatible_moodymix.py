#!/usr/bin/env python3
import json
import struct
import os

def create_compatible_model():
    print("=== Creating compatible moodyMix model ===")
    
    # Load moodyMix tensors
    input_file = "/Users/martinliu/Downloads/iris.c-main/moodyMix_zitV10DPO.safetensors"
    with open(input_file, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_data = f.read(header_len).decode('utf-8')
        moodymix_metadata = json.loads(header_data)
        tensor_data = f.read()
    
    # Load original index for reference
    with open('/Users/martinliu/Downloads/iris.c-main/zimage-turbo/transformer/diffusion_pytorch_model.safetensors.index.json') as f:
        original_index = json.load(f)
    
    print(f"Original has {len(original_index['weight_map'])} tensors")
    print(f"MoodyMix has {len(moodymix_metadata)} tensors")
    
    # Create mapping from moodyMix to original format
    new_metadata = {"__metadata__": moodymix_metadata.get("__metadata__", {})}
    weight_map = {}
    
    # Map tensors with naming conversions
    for original_name in original_index['weight_map'].keys():
        # Try to find equivalent in moodyMix
        moodymix_name = f"model.diffusion_model.{original_name}"
        
        if moodymix_name in moodymix_metadata:
            # Direct mapping
            new_metadata[original_name] = moodymix_metadata[moodymix_name]
            weight_map[original_name] = "diffusion_pytorch_model.safetensors"
        else:
            # Try fused attention mapping
            if "attention.to_q.weight" in original_name:
                layer_name = original_name.replace("attention.to_q.weight", "attention.qkv.weight")
                moodymix_fused = f"model.diffusion_model.{layer_name}"
                if moodymix_fused in moodymix_metadata:
                    print(f"  Mapping {moodymix_fused} -> {original_name} (Q part)")
                    # We'll need to split the fused tensor later
                    new_metadata[original_name] = moodymix_metadata[moodymix_fused]
                    weight_map[original_name] = "diffusion_pytorch_model.safetensors"
            
            elif "attention.to_k.weight" in original_name:
                layer_name = original_name.replace("attention.to_k.weight", "attention.qkv.weight")
                moodymix_fused = f"model.diffusion_model.{layer_name}"
                if moodymix_fused in moodymix_metadata:
                    print(f"  Mapping {moodymix_fused} -> {original_name} (K part)")
                    new_metadata[original_name] = moodymix_metadata[moodymix_fused]
                    weight_map[original_name] = "diffusion_pytorch_model.safetensors"
            
            elif "attention.to_v.weight" in original_name:
                layer_name = original_name.replace("attention.to_v.weight", "attention.qkv.weight")
                moodymix_fused = f"model.diffusion_model.{layer_name}"
                if moodymix_fused in moodymix_metadata:
                    print(f"  Mapping {moodymix_fused} -> {original_name} (V part)")
                    new_metadata[original_name] = moodymix_metadata[moodymix_fused]
                    weight_map[original_name] = "diffusion_pytorch_model.safetensors"
            
            else:
                print(f"  WARNING: No mapping found for {original_name}")
    
    print(f"Mapped {len(weight_map)} tensors")
    
    # Create the new model directory
    output_dir = "/Users/martinliu/Downloads/iris.c-main/zimage-turbo-moodymix-v2"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/transformer", exist_ok=True)
    
    # Write index.json
    index_data = {
        "metadata": {
            "total_size": len(tensor_data)
        },
        "weight_map": weight_map
    }
    
    with open(f"{output_dir}/transformer/diffusion_pytorch_model.safetensors.index.json", 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # Write safetensors with new metadata
    output_file = f"{output_dir}/transformer/diffusion_pytorch_model.safetensors"
    new_header_json = json.dumps(new_metadata, separators=(',', ':'))
    new_header_len = len(new_header_json.encode('utf-8'))
    
    with open(output_file, 'wb') as f:
        f.write(struct.pack('<Q', new_header_len))
        f.write(new_header_json.encode('utf-8'))
        f.write(tensor_data)
    
    # Copy other files
    import shutil
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
    
    print(f"✅ Created compatible model: {output_dir}")
    print(f"Test with: ./iris -d {output_dir} -p 'test prompt' -o test.png")

if __name__ == "__main__":
    create_compatible_model()

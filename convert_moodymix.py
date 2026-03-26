#!/usr/bin/env python3
import json
import struct
import os

def convert_moodymix_to_iris():
    input_file = "/Users/martinliu/Downloads/iris.c-main/moodyMix_zitV10DPO.safetensors"
    output_dir = "/Users/martinliu/Downloads/iris.c-main/zimage-turbo-moodymix"
    
    print("=== Converting moodyMix to Iris format ===")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/transformer", exist_ok=True)
    
    # Read the original safetensors
    with open(input_file, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_data = f.read(header_len).decode('utf-8')
        metadata = json.loads(header_data)
        
        # Read all tensor data
        tensor_data = f.read()
    
    print(f"Loaded {len(metadata)} tensors")
    
    # Create new metadata for Iris format
    new_metadata = {"__metadata__": {}}
    weight_map = {}
    
    # Process tensors - remove "model.diffusion_model." prefix
    for name, info in metadata.items():
        if name == "__metadata__":
            new_metadata[name] = info
            continue
            
        # Remove the prefix
        new_name = name
        if name.startswith("model.diffusion_model."):
            new_name = name.replace("model.diffusion_model.", "", 1)
        
        # Store tensor info (same shape, dtype, etc.)
        new_metadata[new_name] = info
        weight_map[new_name] = "diffusion_pytorch_model.safetensors"
    
    # Create index.json like the original
    index_data = {
        "metadata": {
            "total_size": len(tensor_data)  # Use actual tensor data length
        },
        "weight_map": weight_map
    }
    
    # Write index.json
    with open(f"{output_dir}/transformer/diffusion_pytorch_model.safetensors.index.json", 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # Write the converted safetensors file
    output_file = f"{output_dir}/transformer/diffusion_pytorch_model.safetensors"
    
    # Prepare new header
    new_header_json = json.dumps(new_metadata, separators=(',', ':'))
    new_header_len = len(new_header_json.encode('utf-8'))
    
    with open(output_file, 'wb') as f:
        # Write header length and header
        f.write(struct.pack('<Q', new_header_len))
        f.write(new_header_json.encode('utf-8'))
        
        # Write tensor data (same as original)
        f.write(tensor_data)
    
    print(f"✅ Converted {len(new_metadata)-1} tensors")
    print(f"✅ Output: {output_file}")
    print(f"✅ Index: {output_dir}/transformer/diffusion_pytorch_model.safetensors.index.json")
    
    # Copy other necessary files from original zimage-turbo
    import shutil
    
    files_to_copy = [
        "model_index.json",
        "text_encoder",
        "tokenizer", 
        "vae"
    ]
    
    for file_or_dir in files_to_copy:
        src = f"/Users/martinliu/Downloads/iris.c-main/zimage-turbo/{file_or_dir}"
        dst = f"{output_dir}/{file_or_dir}"
        
        if os.path.exists(src):
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            print(f"✅ Copied {file_or_dir}")
    
    print()
    print("🎉 Conversion complete!")
    print(f"Test with: ./iris -d {output_dir} -p 'a test prompt' -o test.png")

if __name__ == "__main__":
    convert_moodymix_to_iris()

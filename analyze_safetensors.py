#!/usr/bin/env python3
import json
import struct

def analyze_safetensors(filepath):
    try:
        with open(filepath, 'rb') as f:
            # Read header length
            header_len = struct.unpack('<Q', f.read(8))[0]
            header_data = f.read(header_len).decode('utf-8')
            metadata = json.loads(header_data)
        
        print(f"=== {filepath.split('/')[-1]} ===")
        print(f"Total tensors: {len(metadata)}")
        
        # Analyze tensor names
        layer_types = {}
        sample_tensors = []
        double_block_count = 0
        single_block_count = 0
        
        for name, info in metadata.items():
            if name == "__metadata__":
                continue
                
            # Count layer types
            if "double_blocks" in name:
                layer_types["double_blocks"] = layer_types.get("double_blocks", 0) + 1
                double_block_count += 1
            elif "single_blocks" in name:
                layer_types["single_blocks"] = layer_types.get("single_blocks", 0) + 1
                single_block_count += 1
            elif "context_refiner" in name:
                layer_types["context_refiner"] = layer_types.get("context_refiner", 0) + 1
            elif "all_final_layer" in name:
                layer_types["final_layer"] = layer_types.get("final_layer", 0) + 1
            elif "embedder" in name:
                layer_types["embedder"] = layer_types.get("embedder", 0) + 1
            elif "input_blocks" in name:
                layer_types["input_blocks"] = layer_types.get("input_blocks", 0) + 1
            elif "output_blocks" in name:
                layer_types["output_blocks"] = layer_types.get("output_blocks", 0) + 1
            elif "middle_blocks" in name:
                layer_types["middle_blocks"] = layer_types.get("middle_blocks", 0) + 1
            else:
                layer_types["other"] = layer_types.get("other", 0) + 1
            
            # Collect sample names
            if len(sample_tensors) < 20:
                sample_tensors.append(f"{name}: {info.get('shape', 'N/A')}")
        
        print("\n=== Layer Distribution ===")
        for layer_type, count in sorted(layer_types.items()):
            print(f"{layer_type}: {count} tensors")
        
        print(f"\n=== Block Analysis ===")
        print(f"Double block tensors: {double_block_count}")
        print(f"Single block tensors: {single_block_count}")
        
        print("\n=== Sample Tensors ===")
        for tensor in sample_tensors:
            print(tensor)
            
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    analyze_safetensors("/Users/martinliu/Downloads/iris.c-main/moodyMix_zitV10DPO.safetensors")

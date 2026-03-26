#!/usr/bin/env python3
import json
import struct

def analyze_other_tensors(filepath):
    try:
        with open(filepath, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header_data = f.read(header_len).decode('utf-8')
            metadata = json.loads(header_data)
        
        print("=== Analyzing 'other' tensors ===")
        other_patterns = {}
        other_samples = []
        
        for name, info in metadata.items():
            if name == "__metadata__":
                continue
                
            # Check for double/single blocks (should be 0)
            if "double_blocks" in name or "single_blocks" in name:
                continue
            elif "context_refiner" in name or "embedder" in name:
                continue
            else:
                # This is an "other" tensor
                pattern = name.split('.')[2] if '.' in name else name
                other_patterns[pattern] = other_patterns.get(pattern, 0) + 1
                
                if len(other_samples) < 20:
                    other_samples.append(f"{name}: {info.get('shape', 'N/A')}")
        
        print("=== Other Tensor Patterns ===")
        for pattern, count in sorted(other_patterns.items()):
            print(f"{pattern}: {count} tensors")
            
        print("\n=== Sample Other Tensors ===")
        for tensor in other_samples:
            print(tensor)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_other_tensors("/Users/martinliu/Downloads/iris.c-main/moodyMix_zitV10DPO.safetensors")

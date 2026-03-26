#!/usr/bin/env python3
import json

# Check what patch_size the model should have
with open('/Users/martinliu/Downloads/iris.c-main/zimage-turbo-moodymix-proper/model_index.json') as f:
    model_index = json.load(f)

print('=== Model index ===')
print(json.dumps(model_index, indent=2))

# Check if there's any config that might specify patch_size
print('\\n=== Looking for patch_size info ===')
import os
for root, dirs, files in os.walk('/Users/martinliu/Downloads/iris.c-main/zimage-turbo-moodymix-proper'):
    for file in files:
        if file.endswith('.json') and 'config' in file.lower():
            print(f'Found config: {os.path.join(root, file)}')
            try:
                with open(os.path.join(root, file)) as f:
                    config = json.load(f)
                    if 'patch_size' in str(config):
                        print(f'  Contains patch_size: {config}')
            except:
                pass

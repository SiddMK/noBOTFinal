import sys
sys.path.append("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT")

import unifyllm
import inspect

# Check where UnifiedRetriever is defined
print("=" * 60)
print("UnifiedRetriever is defined in:")
print(inspect.getfile(unifyllm.UnifiedRetriever))
print("=" * 60)

# Check the source file
print("\nFile location:")
print(unifyllm.__file__)
print("=" * 60)

# List all files in the module
import os
module_dir = os.path.dirname(unifyllm.__file__)
print("\nAll Python files in directory:")
for file in os.listdir(module_dir):
    if file.endswith('.py'):
        print(f"  - {file}")
        

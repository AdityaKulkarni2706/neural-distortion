import json
import os

# 1. Load the weights
with open("Weights/model_weights.json", "r") as f:
    data = json.load(f)

# 2. Format them as C++ arrays
def to_cpp_array(name, values):
    # Convert list of floats to a comma-separated string
    content = ", ".join(f"{v:.8f}f" for v in values)
    return f"static const float {name}[{len(values)}] = {{ {content} }};"

# 3. Generate the header file content
header_content = f"""#pragma once

// Auto-generated weights for Neural Distortion Plugin
// Generated from export_to_cpp.py

namespace NeuralWeights {{
    {to_cpp_array("w1", data["layer1_weights"])}
    {to_cpp_array("b1", data["layer1_bias"])}
    
    {to_cpp_array("w2", data["layer2_weights"])}
    {to_cpp_array("b2", data["layer2_bias"])}

    static const int input_size = 1;
    static const int hidden_size = {len(data["layer1_bias"])};
    static const int output_size = 1;
}}
"""

# 4. Save to Weights.h
with open("Weights/Weights.h", "w") as f:
    f.write(header_content)

print("Success! Created 'Weights.h'. You can now include this in your C++ project.")
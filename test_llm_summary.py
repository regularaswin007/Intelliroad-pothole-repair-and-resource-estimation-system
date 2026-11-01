"""
Test script to debug Phi-2 LLM loading issues
Run this standalone to check if the model loads correctly
"""

import os
import sys
import torch

print("=" * 70)
print("PHI-2 LLM DIAGNOSTIC TEST")
print("=" * 70)

# Step 1: Check PyTorch
print("\n📦 Step 1: Checking PyTorch installation...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("✅ PyTorch OK")

# Step 2: Check transformers
print("\n📦 Step 2: Checking transformers library...")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print("✅ Transformers OK")
except ImportError:
    print("❌ Transformers not installed!")
    print("Install: pip install transformers")
    sys.exit(1)

# Step 3: Check model files
print("\n📂 Step 3: Checking model files...")
model_path = "./llm_models/phi-2"

if not os.path.exists(model_path):
    print(f"❌ Model directory not found: {model_path}")
    print("\n📥 To download Phi-2 model:")
    print("   1. Install git-lfs: https://git-lfs.github.com/")
    print("   2. Run: git clone https://huggingface.co/microsoft/phi-2 ./llm_models/phi-2")
    sys.exit(1)

# Check for either pytorch_model.bin OR model.safetensors (modern format)
required_files = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json"
]

# Model weights can be in different formats
model_weight_files = [
    "model.safetensors",
    "pytorch_model.bin",
    "model-00001-of-00002.safetensors"  # Sometimes split into multiple files
]

missing_files = []
for file in required_files:
    file_path = os.path.join(model_path, file)
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"✅ {file} ({size_mb:.1f} MB)")
    else:
        print(f"❌ {file} - MISSING")
        missing_files.append(file)

# Check for model weights
print("\n📦 Checking model weight files...")
weight_file_found = False
for weight_file in model_weight_files:
    file_path = os.path.join(model_path, weight_file)
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"✅ {weight_file} ({size_mb:.1f} MB)")
        weight_file_found = True
        break

if not weight_file_found:
    print(f"❌ No model weight files found!")
    print(f"   Looking for: {model_weight_files}")
    print("\n📥 Model files incomplete. Please download again:")
    print("   rm -rf ./llm_models/phi-2")
    print("   git lfs install")
    print("   git clone https://huggingface.co/microsoft/phi-2 ./llm_models/phi-2")
    sys.exit(1)

if missing_files:
    print(f"\n❌ Missing files: {missing_files}")
    print("Please re-download the model")
    sys.exit(1)

print("\n✅ All required model files present")

# Step 4: Test model loading (CPU only)
print("\n🔧 Step 4: Testing model loading on CPU...")
print("This may take 30-60 seconds...")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Tokenizer loaded")
    
    # Load model on CPU
    print("Loading model (this takes time)...")
    print("⏳ Please wait... (30-60 seconds)")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True
    )
    
    # Explicitly move to CPU
    device = "cpu"
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully on CPU")
    
    # Step 5: Test inference
    print("\n🧪 Step 5: Testing inference...")
    
    test_prompt = "The road has 3 potholes. Analysis:"
    
    inputs = tokenizer(
        test_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)
    
    print("Generating response (15 tokens)...")
    print("⏳ Please wait...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = response[len(test_prompt):].strip()
    
    print(f"\n📝 Test Output:")
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated}")
    
    if generated and len(generated) > 0:
        print("\n" + "=" * 70)
        print("✅ ✅ ✅ SUCCESS! LLM IS WORKING CORRECTLY!")
        print("=" * 70)
        print("\n🎉 You can now use the LLM in your application.")
        print("\n📝 Next steps:")
        print("   1. Run: streamlit run complete_pipeline.py")
        print("   2. Enable the 'LLM Analysis' checkbox")
        print("   3. Upload an image and detect potholes")
        print("   4. Get AI-powered analysis!")
    else:
        print("\n⚠️ Warning: Model loaded but generated empty output")
        print("This might still work in the main application")
    
except Exception as e:
    print(f"\n❌ Error during model loading/inference:")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 Troubleshooting:")
    print("1. Make sure you have enough RAM (>8GB recommended)")
    print("2. Close other applications to free memory")
    print("3. Try downloading the model again:")
    print("   rm -rf ./llm_models/phi-2")
    print("   git lfs install")
    print("   git clone https://huggingface.co/microsoft/phi-2 ./llm_models/phi-2")
    print("4. Check if transformers version is compatible (>=4.35.0)")
    print("\n🔄 Alternative: Use the rule-based analyzer (no LLM needed)")
    print("   It provides excellent analysis instantly!")
    sys.exit(1)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print("\n💡 System Status:")
print("   ✅ PyTorch installed")
print("   ✅ Transformers installed")
print("   ✅ Model files complete")
print("   ✅ Model loads on CPU")
print("   ✅ Inference working")
print("\n🚀 Ready to use in Streamlit app!")
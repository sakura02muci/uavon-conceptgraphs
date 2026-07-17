"""
Test script to verify UAV-ON evaluation setup.

Checks:
1. Dataset files exist
2. ConceptGraphs modules load correctly
3. AirSim connection works
4. Single episode can run
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

def check_dataset():
    """Check if UAV-ON dataset exists."""
    print("\n1. Checking UAV-ON dataset...")
    
    dataset_dir = Path(__file__).resolve().parents[2] / "UAV-ON-dataset" / "valset"
    
    if not dataset_dir.exists():
        print(f"   ❌ Dataset directory not found: {dataset_dir}")
        return False
    
    json_files = list(dataset_dir.glob("*.json"))
    print(f"   ✓ Found {len(json_files)} scene files")
    
    if len(json_files) < 14:
        print(f"   ⚠️  Expected 14 scenes, found {len(json_files)}")
    
    # Check Barnyard as example
    barnyard = dataset_dir / "Barnyard.json"
    if barnyard.exists():
        import json
        with open(barnyard, 'r') as f:
            episodes = json.load(f)
        print(f"   ✓ Barnyard.json: {len(episodes)} episodes")
    else:
        print("   ❌ Barnyard.json not found")
        return False
    
    return True


def check_modules():
    """Check if required modules can be imported."""
    print("\n2. Checking Python modules...")
    
    required = [
        ('airsim', 'AirSim'),
        ('torch', 'PyTorch'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('clip', 'OpenAI CLIP'),
    ]
    
    all_ok = True
    for module_name, display_name in required:
        try:
            __import__(module_name)
            print(f"   ✓ {display_name}")
        except ImportError:
            print(f"   ❌ {display_name} not found")
            all_ok = False
    
    # Check our modules
    try:
        from conceptgraphs_uav import ConceptGraphBuilder
        print("   ✓ ConceptGraphs UAV")
    except ImportError as e:
        print(f"   ❌ ConceptGraphs UAV: {e}")
        all_ok = False
    
    try:
        from conceptgraphs_uav.clip_detector import CLIPDetector
        print("   ✓ CLIP Detector")
    except ImportError as e:
        print(f"   ❌ CLIP Detector: {e}")
        all_ok = False
    
    return all_ok


def check_airsim():
    """Check AirSim connection."""
    print("\n3. Checking AirSim connection...")
    
    try:
        import airsim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(False)
        print("   ✓ Connected to AirSim")
        print(f"   ✓ API version: {client.getServerVersion()}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to connect: {e}")
        print("   ⚠️  Make sure UE4 environment is running!")
        return False


def check_clip_detector():
    """Check CLIP detector initialization."""
    print("\n4. Checking CLIP detector...")
    
    try:
        from conceptgraphs_uav.clip_detector import CLIPDetector
        import numpy as np
        
        print("   Loading CLIP model...")
        detector = CLIPDetector()
        
        # Test detection on dummy image
        dummy_image = np.random.randint(0, 255, (144, 256, 3), dtype=np.uint8)
        result = detector.classify_image(dummy_image, top_k=1)
        
        print(f"   ✓ CLIP detector loaded")
        print(f"   ✓ Test detection: '{result[0][0]}' (conf={result[0][1]:.3f})")
        print(f"   ✓ Categories: {len(detector.categories)}")
        
        return True
    except Exception as e:
        print(f"   ❌ CLIP detector failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_mini_test():
    """Run a minimal end-to-end test."""
    print("\n5. Running mini end-to-end test...")
    
    try:
        import airsim
        import numpy as np
        from conceptgraphs_uav.clip_detector import CLIPDetector
        
        # Connect
        client = airsim.MultirotorClient()
        client.confirmConnection()
        
        # Get one image
        print("   Getting observation from AirSim...")
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        
        if len(responses) == 0:
            print("   ❌ No images received")
            return False
        
        # Parse image
        response = responses[0]
        img_array = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_array = img_array.reshape(response.height, response.width, 3)
        
        print(f"   ✓ Got image: {img_array.shape}")
        
        # Run CLIP detection
        detector = CLIPDetector()
        result = detector.classify_image(img_array, top_k=1)
        
        print(f"   ✓ Detection: '{result[0][0]}' (conf={result[0][1]:.3f})")
        print("   ✓ Mini test passed!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Mini test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("UAV-ON Evaluation Setup Verification")
    print("="*70)
    
    results = []
    
    # Run checks
    results.append(("Dataset", check_dataset()))
    results.append(("Modules", check_modules()))
    results.append(("AirSim", check_airsim()))
    results.append(("CLIP Detector", check_clip_detector()))
    
    # Only run mini test if everything else passed
    if all(r[1] for r in results):
        results.append(("End-to-End Test", run_mini_test()))
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    for name, status in results:
        symbol = "✓" if status else "❌"
        print(f"{symbol} {name}")
    
    if all(r[1] for r in results):
        print("\n✅ All checks passed! Ready to run evaluation.")
        print("\nNext steps:")
        print("  python scripts/eval_simple_uavon.py \\")
        print("      --dataset ../UAV-ON-dataset/valset/Barnyard.json \\")
        print("      --num_episodes 3 \\")
        print("      --strategy clip")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Dataset: Download UAV-ON dataset to ../UAV-ON-dataset/")
        print("  - Modules: pip install -r requirements.txt")
        print("  - AirSim: Start UE4 environment before running this script")
    
    print("="*70)


if __name__ == "__main__":
    main()

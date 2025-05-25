"""
Main entry point for Building Damage Analysis LLM System
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment and check dependencies"""
    try:
        import torch
        import transformers
        import langchain
        import streamlit
        import pandas
        import cv2
        import PIL

        logger.info("✅ All required dependencies are installed")

        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            device = "cuda"
        else:
            logger.info("⚠️ CUDA not available, using CPU")
            device = "cpu"

        return device

    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)


def test_data_loading():
    """Test data loading functionality"""
    logger.info("Testing data loading...")

    try:
        from data_loader import test_data_loader

        test_data_loader()
        logger.info("✅ Data loading test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    logger.info("Testing model creation...")

    try:
        from models import create_model

        device = "cpu"  # Use CPU for testing
        model = create_model(device)
        logger.info("✅ Model creation test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Model creation test failed: {e}")
        return False


def test_langchain_integration():
    """Test LangChain integration"""
    logger.info("Testing LangChain integration...")

    try:
        from langchain_integration import create_damage_analysis_pipeline

        pipeline = create_damage_analysis_pipeline(device="cpu")
        logger.info("✅ LangChain integration test passed")
        return True
    except Exception as e:
        logger.error(f"❌ LangChain integration test failed: {e}")
        return False


def run_training(device: str = "cpu", epochs: int = None):
    """Run model training"""
    logger.info("Starting model training...")

    try:
        from trainer import train_model
        from config import TRAINING_CONFIG

        # Update config if epochs specified
        config = TRAINING_CONFIG.copy()
        if epochs:
            config["num_epochs"] = epochs

        trainer = train_model(device=device, config=config)
        logger.info("✅ Training completed successfully")
        return trainer

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return None


def run_analysis(image_path: str, query: str = None, device: str = "cpu"):
    """Run single image analysis"""
    logger.info(f"Analyzing image: {image_path}")

    try:
        from langchain_integration import analyze_building_damage

        if query is None:
            query = "건물의 피해 상황을 분석해주세요."

        result = analyze_building_damage(
            image_path=image_path, query=query, device=device, generate_report=True
        )

        if result["success"]:
            logger.info("✅ Analysis completed successfully")

            # Print results
            print("\n" + "=" * 50)
            print("건물 피해 분석 결과")
            print("=" * 50)

            if result["report"]:
                print(result["report"])
            else:
                analysis = result["analysis_result"]
                print(f"분석 ID: {analysis.damage_id}")
                print(f"신뢰도: {analysis.confidence_score:.2%}")
                print(f"상세 분석: {analysis.damage_analysis}")

        else:
            logger.error("❌ Analysis failed")
            print("분석에 실패했습니다.")

        return result

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return None


def run_streamlit_app(port: int = 8501):
    """Run Streamlit web application"""
    logger.info(f"Starting Streamlit app on port {port}")

    try:
        import subprocess
        import sys

        # Run streamlit app
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.port",
            str(port),
            "--server.address",
            "0.0.0.0",
        ]

        subprocess.run(cmd)

    except Exception as e:
        logger.error(f"❌ Streamlit app failed to start: {e}")


def run_tests():
    """Run all system tests"""
    logger.info("Running system tests...")

    tests = [
        ("Environment Setup", setup_environment),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("LangChain Integration", test_langchain_integration),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        try:
            if test_name == "Environment Setup":
                device = test_func()
                results[test_name] = True
            else:
                results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            results[test_name] = False

    # Print test summary
    print("\n" + "=" * 50)
    print("테스트 결과 요약")
    print("=" * 50)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print(f"\n전체 테스트: {'✅ 모두 통과' if all_passed else '❌ 일부 실패'}")

    return all_passed


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Building Damage Analysis LLM System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run system tests
  python main.py test
  
  # Train the model
  python main.py train --device cuda --epochs 5
  
  # Analyze a single image
  python main.py analyze --image path/to/image.jpg --query "분석 요청"
  
  # Start Streamlit web app
  python main.py app --port 8501
  
  # Start Streamlit app on custom port
  python main.py app --port 8080
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run system tests")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training",
    )
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single image")
    analyze_parser.add_argument("--image", required=True, help="Path to image file")
    analyze_parser.add_argument("--query", help="Analysis query")
    analyze_parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for analysis",
    )

    # Streamlit app command
    app_parser = subparsers.add_parser("app", help="Start Streamlit web application")
    app_parser.add_argument("--port", type=int, default=8501, help="Port number")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup environment
    device = setup_environment()

    # Execute command
    if args.command == "test":
        success = run_tests()
        sys.exit(0 if success else 1)

    elif args.command == "train":
        device = args.device if args.device == "cuda" and device == "cuda" else "cpu"
        trainer = run_training(device=device, epochs=args.epochs)
        sys.exit(0 if trainer else 1)

    elif args.command == "analyze":
        if not Path(args.image).exists():
            logger.error(f"Image file not found: {args.image}")
            sys.exit(1)

        device = args.device if args.device == "cuda" and device == "cuda" else "cpu"
        result = run_analysis(args.image, args.query, device=device)
        sys.exit(0 if result and result["success"] else 1)

    elif args.command == "app":
        run_streamlit_app(port=args.port)


if __name__ == "__main__":
    main()

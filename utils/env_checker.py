#!/usr/bin/env python3
"""
환경변수 설정 상태 확인 스크립트
배포 전에 실행하여 환경변수가 올바르게 설정되었는지 확인
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()


def check_environment_variables() -> Dict[str, Any]:
    """환경변수 상태 확인"""

    print("🔧 환경변수 설정 상태 확인")
    print("=" * 50)

    # 필수 환경변수
    required_vars = {
        "DEVICE": {
            "value": os.getenv("DEVICE", "cpu"),
            "default": "cpu",
            "description": "PyTorch 연산 디바이스",
        },
        "CACHE_DIR": {
            "value": os.getenv("CACHE_DIR", "./cache"),
            "default": "./cache",
            "description": "캐시 디렉토리",
        },
        "LOGS_DIR": {
            "value": os.getenv("LOGS_DIR", "./logs"),
            "default": "./logs",
            "description": "로그 디렉토리",
        },
    }

    # 선택적 환경변수
    optional_vars = {
        "OPENAI_API_KEY": {
            "value": os.getenv("OPENAI_API_KEY"),
            "description": "OpenAI API 키",
        },
        "HUGGINGFACE_TOKEN": {
            "value": os.getenv("HUGGINGFACE_TOKEN"),
            "description": "HuggingFace 토큰",
        },
        "MODEL_PATH": {
            "value": os.getenv("MODEL_PATH"),
            "description": "모델 파일 경로",
        },
    }

    results = {"required": {}, "optional": {}, "status": "success"}

    print("\n📋 필수 환경변수:")
    for var_name, var_info in required_vars.items():
        value = var_info["value"]
        is_set = value != var_info["default"]

        status = "✅ 설정됨" if is_set else f"⚠️  기본값 사용 ({var_info['default']})"
        print(f"  {var_name}: {value} - {status}")
        print(f"    설명: {var_info['description']}")

        results["required"][var_name] = {
            "value": value,
            "is_set": is_set,
            "status": "ok",
        }

    print("\n📋 선택적 환경변수:")
    for var_name, var_info in optional_vars.items():
        value = var_info["value"]
        is_set = value is not None

        if is_set:
            # API 키는 일부만 표시
            if "API" in var_name or "TOKEN" in var_name:
                display_value = f"{value[:8]}..." if len(value) > 8 else value
            else:
                display_value = value
            status = "✅ 설정됨"
        else:
            display_value = "설정되지 않음"
            status = "ℹ️  선택사항"

        print(f"  {var_name}: {display_value} - {status}")
        print(f"    설명: {var_info['description']}")

        results["optional"][var_name] = {
            "value": value,
            "is_set": is_set,
            "status": "ok",
        }

    return results


def check_directories():
    """디렉토리 생성 가능 여부 확인"""
    print("\n📁 디렉토리 확인:")

    directories = [
        Path(os.getenv("CACHE_DIR", "./cache")),
        Path(os.getenv("LOGS_DIR", "./logs")),
    ]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if directory.exists():
                print(f"  ✅ {directory}: 생성 성공")
            else:
                print(f"  ❌ {directory}: 생성 실패")
        except Exception as e:
            print(f"  ❌ {directory}: 오류 - {e}")


def check_imports():
    """주요 모듈 import 가능 여부 확인"""
    print("\n📦 모듈 Import 확인:")

    modules = [
        ("streamlit", "Streamlit 웹 프레임워크"),
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("langchain", "LangChain"),
        ("chromadb", "ChromaDB"),
        ("plotly", "Plotly"),
        ("pandas", "Pandas"),
        ("PIL", "Pillow"),
    ]

    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}: {description}")
        except ImportError as e:
            print(f"  ❌ {module_name}: {description} - {e}")


def check_files():
    """필수 파일 존재 여부 확인"""
    print("\n📄 필수 파일 확인:")

    files = [
        ("streamlit_app.py", "메인 애플리케이션"),
        ("requirements.txt", "패키지 의존성"),
        ("vector_store_faiss.py", "FAISS 벡터스토어 모듈"),
        ("config.py", "설정 파일"),
        (".streamlit/config.toml", "Streamlit 설정"),
        ("standard/", "표준 데이터 (디렉토리)"),
    ]

    for file_path, description in files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}: {description}")
        else:
            print(f"  ⚠️  {file_path}: {description} - 파일 없음")


def main():
    """메인 실행 함수"""
    print("🚀 배포 환경 체크 시작\n")

    # 환경변수 확인
    env_results = check_environment_variables()

    # 디렉토리 확인
    check_directories()

    # 모듈 import 확인
    check_imports()

    # 파일 확인
    check_files()

    print("\n" + "=" * 50)
    print("✅ 환경 체크 완료!")
    print("\n💡 배포 팁:")
    print("  1. Streamlit Cloud에서 Secrets 설정을 잊지 마세요")
    print("  2. 대용량 모델 파일은 .gitignore에 포함되어 있습니다")
    print("  3. 환경변수가 없어도 기본 기능은 동작합니다")

    return env_results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)

#!/usr/bin/env python3
"""
배포 환경용 커스텀 모델 업로드 스크립트
GitHub Releases에 모델 파일을 자동으로 업로드합니다.
"""

import os
import sys
import requests
import json
from pathlib import Path
from typing import Optional


def check_github_token() -> Optional[str]:
    """GitHub Personal Access Token 확인"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("❌ GITHUB_TOKEN 환경변수가 설정되지 않았습니다.")
        print(
            "📝 GitHub → Settings → Developer settings → Personal access tokens에서 토큰 생성"
        )
        print("🔑 필요한 권한: repo (Full control of private repositories)")
        return None
    return token


def get_repo_info() -> tuple:
    """GitHub 저장소 정보 확인"""
    try:
        # git remote 정보에서 저장소 정보 추출
        import subprocess

        result = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True, text=True
        )

        if result.returncode != 0:
            print("❌ Git 저장소가 아니거나 origin remote가 없습니다.")
            return None, None

        remote_url = result.stdout.strip()

        # GitHub URL에서 owner/repo 추출
        if "github.com" in remote_url:
            if remote_url.startswith("https://"):
                # https://github.com/owner/repo.git
                parts = remote_url.replace("https://github.com/", "").replace(
                    ".git", ""
                )
            elif remote_url.startswith("git@"):
                # git@github.com:owner/repo.git
                parts = remote_url.replace("git@github.com:", "").replace(".git", "")
            else:
                print("❌ GitHub URL 형식을 인식할 수 없습니다.")
                return None, None

            owner, repo = parts.split("/")
            return owner, repo
        else:
            print("❌ GitHub 저장소가 아닙니다.")
            return None, None

    except Exception as e:
        print(f"❌ 저장소 정보 확인 실패: {e}")
        return None, None


def create_release(
    token: str, owner: str, repo: str, tag: str, name: str
) -> Optional[str]:
    """GitHub Release 생성"""
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    data = {
        "tag_name": tag,
        "name": name,
        "body": f"커스텀 모델 배포를 위한 릴리즈 v{tag}",
        "draft": False,
        "prerelease": False,
    }

    try:
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 201:
            release_data = response.json()
            print(f"✅ Release 생성 성공: {release_data['html_url']}")
            return release_data["upload_url"].replace("{?name,label}", "")
        elif response.status_code == 422:
            # Release가 이미 존재하는 경우
            print(f"⚠️ Release {tag}가 이미 존재합니다. 기존 Release 사용")

            # 기존 Release 정보 가져오기
            get_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
            get_response = requests.get(get_url, headers=headers)

            if get_response.status_code == 200:
                release_data = get_response.json()
                return release_data["upload_url"].replace("{?name,label}", "")
            else:
                print(f"❌ 기존 Release 정보 가져오기 실패: {get_response.status_code}")
                return None
        else:
            print(f"❌ Release 생성 실패: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Release 생성 중 오류: {e}")
        return None


def upload_asset(upload_url: str, token: str, file_path: Path) -> bool:
    """Release에 파일 업로드"""

    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/octet-stream",
    }

    upload_url_with_name = f"{upload_url}?name={file_path.name}"

    try:
        print(
            f"📤 파일 업로드 시작: {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f}MB)"
        )

        with open(file_path, "rb") as f:
            response = requests.post(upload_url_with_name, headers=headers, data=f)

        if response.status_code == 201:
            asset_data = response.json()
            print(f"✅ 파일 업로드 성공: {asset_data['browser_download_url']}")
            return True
        else:
            print(f"❌ 파일 업로드 실패: {response.status_code}")
            print(response.text)
            return False

    except Exception as e:
        print(f"❌ 파일 업로드 중 오류: {e}")
        return False


def main():
    """메인 실행 함수"""
    print("🚀 커스텀 모델 배포 스크립트 시작")

    # 1. GitHub 토큰 확인
    token = check_github_token()
    if not token:
        sys.exit(1)

    # 2. 저장소 정보 확인
    owner, repo = get_repo_info()
    if not owner or not repo:
        sys.exit(1)

    print(f"📁 저장소: {owner}/{repo}")

    # 3. 모델 파일 확인
    model_files = []

    yolo_path = Path("train/models/custom_yolo_damage.pt")
    if yolo_path.exists():
        model_files.append(yolo_path)
        print(
            f"✅ YOLO 모델 발견: {yolo_path} ({yolo_path.stat().st_size / 1024 / 1024:.1f}MB)"
        )
    else:
        print(f"⚠️ YOLO 모델 없음: {yolo_path}")

    clip_path = Path("train/models/clip_finetuned.pt")
    if clip_path.exists():
        model_files.append(clip_path)
        print(
            f"✅ CLIP 모델 발견: {clip_path} ({clip_path.stat().st_size / 1024 / 1024:.1f}MB)"
        )
    else:
        print(f"⚠️ CLIP 모델 없음: {clip_path}")

    if not model_files:
        print("❌ 업로드할 모델 파일이 없습니다.")
        sys.exit(1)

    # 4. Release 태그 입력
    tag = input("📝 Release 태그를 입력하세요 (예: v1.0.0): ").strip()
    if not tag:
        tag = "v1.0.0"

    name = f"Custom Models {tag}"

    # 5. Release 생성
    upload_url = create_release(token, owner, repo, tag, name)
    if not upload_url:
        sys.exit(1)

    # 6. 파일 업로드
    success_urls = []
    for model_file in model_files:
        if upload_asset(upload_url, token, model_file):
            download_url = f"https://github.com/{owner}/{repo}/releases/download/{tag}/{model_file.name}"
            success_urls.append((model_file.name, download_url))

    # 7. 결과 출력
    if success_urls:
        print("\n🎉 배포 완료! 다음 환경변수를 설정하세요:")
        print("=" * 60)

        for file_name, url in success_urls:
            if "yolo" in file_name.lower():
                print(f"CUSTOM_YOLO_URL = {url}")
            elif "clip" in file_name.lower():
                print(f"CUSTOM_CLIP_URL = {url}")

        print("=" * 60)
        print("\n📋 Streamlit Cloud 설정:")
        print("1. Streamlit Cloud → Manage app → Settings → Secrets")
        print("2. 위의 환경변수들을 추가")
        print("3. 앱을 재시작하여 커스텀 모델 사용")

        print(
            f"\n🔗 Release 페이지: https://github.com/{owner}/{repo}/releases/tag/{tag}"
        )
    else:
        print("❌ 모든 파일 업로드에 실패했습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()

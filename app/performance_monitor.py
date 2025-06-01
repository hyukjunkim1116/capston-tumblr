"""
성능 모니터링 유틸리티 - 로컬 vs 배포 환경 성능 비교
"""

import time
import logging
import psutil
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """성능 모니터링 및 비교 클래스"""

    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.memory_start = None

    def start_monitoring(self, operation_name: str):
        """성능 모니터링 시작"""
        self.operation_name = operation_name
        self.start_time = time.time()
        self.memory_start = self._get_memory_usage()

        logger.info(f"📊 성능 모니터링 시작: {operation_name}")

    def end_monitoring(self) -> Dict[str, Any]:
        """성능 모니터링 종료 및 결과 반환"""
        if not self.start_time:
            return {}

        end_time = time.time()
        memory_end = self._get_memory_usage()

        metrics = {
            "operation": self.operation_name,
            "duration": end_time - self.start_time,
            "memory_start_mb": self.memory_start,
            "memory_end_mb": memory_end,
            "memory_delta_mb": memory_end - self.memory_start,
            "timestamp": datetime.now().isoformat(),
        }

        # 환경 정보 추가
        from app.config import get_app_config

        config = get_app_config()
        metrics.update(
            {
                "environment": config["environment"],
                "is_deployment": config["is_deployment"],
                "device": config["device"],
                "max_image_size": config["max_image_size"],
            }
        )

        logger.info(
            f"✅ 성능 모니터링 완료: {self.operation_name} ({metrics['duration']:.2f}초)"
        )

        # 메트릭 저장
        self._save_metrics(metrics)

        return metrics

    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 (MB)"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def _save_metrics(self, metrics: Dict[str, Any]):
        """메트릭을 파일에 저장"""
        try:
            metrics_file = Path("cache/performance_metrics.jsonl")
            metrics_file.parent.mkdir(exist_ok=True)

            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")

        except Exception as e:
            logger.warning(f"⚠️ 메트릭 저장 실패: {e}")


def performance_wrapper(operation_name: str):
    """성능 모니터링 데코레이터"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            monitor.start_monitoring(operation_name)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                metrics = monitor.end_monitoring()

                # Streamlit 세션에 메트릭 저장
                if hasattr(st.session_state, "performance_metrics"):
                    st.session_state.performance_metrics.append(metrics)
                else:
                    st.session_state.performance_metrics = [metrics]

        return wrapper

    return decorator


@st.cache_data
def load_performance_history() -> List[Dict[str, Any]]:
    """성능 히스토리 로드"""
    try:
        metrics_file = Path("cache/performance_metrics.jsonl")
        if not metrics_file.exists():
            return []

        metrics = []
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    metrics.append(json.loads(line.strip()))
                except:
                    continue

        return metrics[-100:]  # 최근 100개만

    except Exception as e:
        logger.warning(f"⚠️ 성능 히스토리 로드 실패: {e}")
        return []


def display_performance_comparison():
    """성능 비교 대시보드 표시"""
    st.subheader("📊 성능 비교 대시보드")

    history = load_performance_history()
    current_session = getattr(st.session_state, "performance_metrics", [])

    if not history and not current_session:
        st.info("아직 성능 데이터가 없습니다.")
        return

    # 환경별 데이터 분리
    local_metrics = [m for m in history if not m.get("is_deployment", False)]
    deployment_metrics = [m for m in history if m.get("is_deployment", False)]

    # 비교 표 생성
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💻 로컬 환경")
        if local_metrics:
            avg_duration = sum(m["duration"] for m in local_metrics) / len(
                local_metrics
            )
            avg_memory = sum(m["memory_delta_mb"] for m in local_metrics) / len(
                local_metrics
            )

            st.metric("평균 처리 시간", f"{avg_duration:.2f}초")
            st.metric("평균 메모리 사용", f"{avg_memory:.1f}MB")
            st.metric("총 실행 횟수", len(local_metrics))
        else:
            st.info("로컬 환경 데이터 없음")

    with col2:
        st.markdown("### 🌐 배포 환경")
        if deployment_metrics:
            avg_duration = sum(m["duration"] for m in deployment_metrics) / len(
                deployment_metrics
            )
            avg_memory = sum(m["memory_delta_mb"] for m in deployment_metrics) / len(
                deployment_metrics
            )

            st.metric("평균 처리 시간", f"{avg_duration:.2f}초")
            st.metric("평균 메모리 사용", f"{avg_memory:.1f}MB")
            st.metric("총 실행 횟수", len(deployment_metrics))
        else:
            st.info("배포 환경 데이터 없음")

    # 현재 세션 성능
    if current_session:
        st.markdown("### 🔄 현재 세션 성능")
        latest = current_session[-1]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("마지막 처리 시간", f"{latest['duration']:.2f}초")
        with col2:
            st.metric("메모리 사용량", f"{latest['memory_delta_mb']:.1f}MB")
        with col3:
            env_emoji = "🌐" if latest["is_deployment"] else "💻"
            st.metric("실행 환경", f"{env_emoji} {latest['environment']} (고정확도)")


def get_performance_recommendations() -> List[str]:
    """성능 권장사항 (모든 환경에서 고정확도 설정)"""
    from app.config import get_app_config

    config = get_app_config()

    recommendations = [
        "🎯 모든 환경에서 고정확도 모드로 실행됩니다",
        "⚡ TTA(Test Time Augmentation)가 활성화되어 정확도가 향상됩니다",
        "📊 배치 크기 4로 설정되어 있습니다",
        "🔥 최고 품질 GPT-4o 모델을 사용합니다",
        "📏 이미지 최대 크기: 2048px로 설정되어 있습니다",
    ]

    if config["is_deployment"]:
        recommendations.append(
            "🌐 배포 환경에서도 로컬과 동일한 고정확도 설정을 사용합니다"
        )
    else:
        recommendations.append("💻 로컬 환경에서 최적의 성능을 제공합니다")

    return recommendations


# 성능 모니터링 적용을 위한 유틸리티 함수들
def optimize_image_processing():
    """이미지 처리 최적화 (모든 환경에서 고성능 설정)"""
    from app.config import get_app_config

    config = get_app_config()

    # 모든 환경에서 고성능 설정
    import torch

    if torch.cuda.is_available():
        torch.set_num_threads(8)  # 더 많은 CPU 스레드 사용
        logger.info("🚀 CUDA 가속 및 멀티스레딩 최적화 적용")
    else:
        torch.set_num_threads(8)  # CPU 환경에서도 멀티스레딩
        logger.info("💻 CPU 멀티스레딩 최적화 적용")

    if config["is_deployment"]:
        # 배포 환경도 이제 고성능 설정 사용 (메모리 정리만 유지)
        import gc

        gc.collect()  # 가비지 컬렉션
        logger.info("🧹 메모리 정리 완료")
    else:
        logger.info("💻 로컬 환경 설정 유지")

    logger.info(f"🎯 고정확도 모드 활성화 - 환경: {config['environment']}")


@st.cache_data
def get_system_info() -> Dict[str, Any]:
    """시스템 정보 조회"""
    try:
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "memory_available_gb": psutil.virtual_memory().available / 1024**3,
            "disk_free_gb": psutil.disk_usage(".").free / 1024**3,
        }
    except:
        return {"error": "시스템 정보 조회 실패"}


def display_system_info():
    """시스템 정보 표시"""
    st.subheader("🖥️ 시스템 정보")

    info = get_system_info()

    if "error" not in info:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("CPU 코어", f"{info['cpu_count']}개")
        with col2:
            st.metric("총 메모리", f"{info['memory_total_gb']:.1f}GB")
        with col3:
            st.metric("사용 가능 메모리", f"{info['memory_available_gb']:.1f}GB")
        with col4:
            st.metric("디스크 여유공간", f"{info['disk_free_gb']:.1f}GB")
    else:
        st.error("시스템 정보를 조회할 수 없습니다.")

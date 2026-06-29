"""프로젝트 전역 설정 (경로, 상수, 시각화 테마).

원본 데이터(중앙회 제공)는 공모전 규정상 외부 공개가 금지되므로
`data/raw/` 아래에 두고 .gitignore 로 제외한다.
원본이 없을 경우 `data/sample/` 의 합성 샘플로 자동 대체된다.
"""
from __future__ import annotations

from pathlib import Path

# ── 경로 ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLE = ROOT / "data" / "sample"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_TBL = ROOT / "outputs" / "tables"

# 중앙회 제공 원본 파일명 (data/raw/ 에 복사해 두면 우선 사용)
ACCIDENT_FILE = "★2021-2025 학교안전사고 데이터.xlsx"
COMPENSATION_FILE = "★2021-2025 학교안전사고 보상 데이터.xlsx"

# 합성 샘플 파일명 (공개·재현용)
SAMPLE_ACCIDENT_FILE = "sample_accident.xlsx"
SAMPLE_COMPENSATION_FILE = "sample_compensation.xlsx"

YEARS = ["2021", "2022", "2023", "2024", "2025"]

# ── 도메인 상수 ───────────────────────────────────────────────────────
SCHOOL_LEVELS = ["유치원", "초등학교", "중학교", "고등학교", "특수학교", "기타학교"]

# 학교급별 학년 순서 (정렬·범주화용)
GRADE_ORDER = {
    "유치원": ["유아"],
    "초등학교": ["1학년", "2학년", "3학년", "4학년", "5학년", "6학년"],
    "중학교": ["1학년", "2학년", "3학년"],
    "고등학교": ["1학년", "2학년", "3학년"],
}

# 보상금(심각도) 구성 항목 → 합계가 '총보상금'
MONEY_COLS = ["요양급여", "장해급여", "간병급여", "유족급여", "장례비", "위로금", "보전비용"]

# 분석에 사용할 핵심 범주 컬럼
GROUP_KEYS = ["학교급", "사고자학년", "사고형태", "사고장소"]


def ensure_dirs() -> None:
    """출력 디렉터리를 생성한다."""
    for d in (OUT_FIG, OUT_TBL):
        d.mkdir(parents=True, exist_ok=True)

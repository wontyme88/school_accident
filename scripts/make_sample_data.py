"""공개·재현용 합성(가짜) 샘플 데이터 생성기.

⚠️ 중앙회 제공 원본 데이터는 공모전 규정상 외부 공개가 불가능하다.
   GitHub 등 공개 환경에서 코드를 '재현'할 수 있도록, 원본과 '동일한 스키마'를 갖되
   값은 난수로 생성한 가짜 데이터를 만든다. (실제 통계적 의미는 없음)

원본과 컬럼 구조·시트구성(연도별)을 동일하게 맞춰, 동일 파이프라인이 그대로 돈다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402

RNG = np.random.default_rng(20260629)

LEVELS = ["유치원", "초등학교", "중학교", "고등학교", "특수학교", "기타학교"]
GRADES_BY_LEVEL = {
    "유치원": ["유아"],
    "초등학교": [f"{i}학년" for i in range(1, 7)],
    "중학교": [f"{i}학년" for i in range(1, 4)],
    "고등학교": [f"{i}학년" for i in range(1, 4)],
    "특수학교": [f"{i}학년" for i in range(1, 4)],
    "기타학교": [f"{i}학년" for i in range(1, 4)],
}
REGIONS = ["서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종",
           "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
ACCIDENT_TYPES = ["넘어짐", "고정된 물체와의 부딪힘", "사람과의 부딪힘", "긁힘, 찔림",
                  "움직이는 물체와의 부딪힘", "스포츠 활동 중 충격을 가함", "베임, 절단",
                  "1미터 미만의 높이에서 떨어짐", "1미터 이상의 높이에서 떨어짐",
                  "물체 사이에 끼임·눌림", "교통사고", "그밖의 손상 사고"]
PLACES = ["운동장", "복도", "강당(체육관)", "일반(교과)교실", "계단",
          "특별교실(과학실 외)", "교문/주변", "화장실", "급식실/식당", "기타 교외"]
PARTS = ["치아", "눈", "무릎", "발목", "손목", "머리", "어깨", "팔", "다리", "손가락"]
ACTIVITIES = ["체육", "쉬는시간", "점심시간", "식사시간(간식 포함)", "휴식",
              "그 밖의 교육활동 시간", "등교", "하교", "기타"]
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
ACT_TIME = ["체육", "쉬는시간", "식사시간(간식 포함)", "점심시간", "등교", "하교", "휴식", "기타"]

N_PER_YEAR = 4000  # 샘플 규모(연도당)


def _weighted_grade(level):
    g = GRADES_BY_LEVEL[level]
    return RNG.choice(g)


def _make_accident_year(year: int) -> pd.DataFrame:
    n = N_PER_YEAR
    levels = RNG.choice(LEVELS, size=n, p=[.12, .40, .22, .20, .03, .03])
    grades = [_weighted_grade(l) for l in levels]
    months = RNG.integers(1, 13, size=n)
    return pd.DataFrame({
        "구분": [f"A{year}{i:06d}" for i in range(n)],
        "지역": RNG.choice(REGIONS, size=n),
        "학교급": levels,
        "사고자구분": "일반학생",
        "사고자학년": grades,
        "사고자성별": RNG.choice(["남", "여"], size=n, p=[.58, .42]),
        "사고연월": [f"{year}-{m:02d}" for m in months],
        "사고발생시각": [f"{RNG.integers(8,17):02d}:{RNG.integers(0,60):02d}" for _ in range(n)],
        "사고요일": RNG.choice(WEEKDAYS, size=n, p=[.19, .19, .19, .18, .17, .04, .04]),
        "사고시간": RNG.choice(ACT_TIME, size=n),
        "사고장소": RNG.choice(PLACES, size=n),
        "사고부위": RNG.choice(PARTS, size=n),
        "사고형태": RNG.choice(ACCIDENT_TYPES, size=n,
                            p=_norm([.30, .18, .10, .07, .07, .07, .04,
                                     .05, .03, .03, .02, .04])),
        "사고당시활동": RNG.choice(ACTIVITIES, size=n),
    })


def _make_comp_year(year: int) -> pd.DataFrame:
    n = int(N_PER_YEAR * 0.6)
    levels = RNG.choice(LEVELS, size=n, p=[.12, .40, .22, .20, .03, .03])
    grades = [_weighted_grade(l) for l in levels]
    # 보상금: 대부분 소액, 일부 고액(장해/유족) → 로그정규 + 희박한 큰 값
    요양 = (RNG.lognormal(mean=12.0, sigma=1.1, size=n)).astype("int64")
    장해 = np.where(RNG.random(n) < 0.02,
                   RNG.lognormal(16, 1.0, size=n), 0).astype("int64")
    간병 = np.where(RNG.random(n) < 0.03,
                   RNG.lognormal(14, 0.8, size=n), 0).astype("int64")
    유족 = np.where(RNG.random(n) < 0.001,
                   RNG.lognormal(18, 0.5, size=n), 0).astype("int64")
    return pd.DataFrame({
        "구분": [f"F{year}{i:06d}" for i in range(n)],
        "지역": RNG.choice(REGIONS, size=n),
        "학교급": levels,
        "사고자구분": "일반학생",
        "사고자학년": grades,
        "사고자성별": RNG.choice(["남", "여"], size=n, p=[.58, .42]),
        "사고시간": RNG.choice(ACT_TIME, size=n),
        "사고장소": RNG.choice(PLACES, size=n),
        "사고부위": RNG.choice(PARTS, size=n),
        "사고형태": RNG.choice(ACCIDENT_TYPES, size=n,
                            p=_norm([.30, .18, .10, .07, .07, .07, .04,
                                     .05, .03, .03, .02, .04])),
        "사고당시활동": RNG.choice(ACTIVITIES, size=n),
        "요양급여": 요양, "장해급여": 장해, "간병급여": 간병, "유족급여": 유족,
        "장례비": np.where(유족 > 0, RNG.integers(3_000_000, 6_000_000, n), 0),
        "위로금": RNG.integers(0, 200_000, n),
        "보전비용": RNG.integers(0, 50_000, n),
    })


def _norm(p):
    p = np.array(p, dtype=float)
    return p / p.sum()


def main():
    C.DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
    acc_path = C.DATA_SAMPLE / C.SAMPLE_ACCIDENT_FILE
    cmp_path = C.DATA_SAMPLE / C.SAMPLE_COMPENSATION_FILE

    with pd.ExcelWriter(acc_path) as w:
        for y in C.YEARS:
            _make_accident_year(int(y)).to_excel(w, sheet_name=y, index=False)
    with pd.ExcelWriter(cmp_path) as w:
        for y in C.YEARS:
            _make_comp_year(int(y)).to_excel(w, sheet_name=y, index=False)

    print(f"[완료] 합성 사고 데이터  → {acc_path.relative_to(C.ROOT)}")
    print(f"[완료] 합성 보상 데이터  → {cmp_path.relative_to(C.ROOT)}")
    print("※ 값은 난수입니다. 실제 통계적 의미는 없으며 코드 재현용입니다.")


if __name__ == "__main__":
    main()

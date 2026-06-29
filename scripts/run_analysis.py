"""학교안전사고 분석 전체 파이프라인 실행.

원본(data/raw/)이 있으면 원본으로, 없으면 합성 샘플로 자동 실행한다.
산출물:
  - outputs/figures/*.png   : 보고서용 그림
  - outputs/tables/*.csv     : 집계 표 + 최종 가이드라인 데이터

사용:  python scripts/run_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C          # noqa: E402
from src import data_loader as dl    # noqa: E402
from src import analysis as an       # noqa: E402
from src import viz                  # noqa: E402


def save_table(df: pd.DataFrame, name: str) -> None:
    C.ensure_dirs()
    path = C.OUT_TBL / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [표]   {path.relative_to(C.ROOT)}  ({len(df):,} rows)")


def main() -> None:
    viz.setup_korean_font()
    real = dl.is_using_real_data()
    print("=" * 64)
    print(f"데이터 소스: {'중앙회 원본' if real else '합성 샘플(공개·재현용)'}")
    print("=" * 64)

    # ── 적재 ────────────────────────────────────────────────────────
    acc = dl.load_accident()
    cmp = dl.load_compensation()
    print(f"사고 데이터 : {acc.shape[0]:,} 행 / {acc.shape[1]} 열")
    print(f"보상 데이터 : {cmp.shape[0]:,} 행 / {cmp.shape[1]} 열")
    print(f"총 보상지급액: {cmp['총보상금'].sum():,} 원\n")

    # ── 1. 기초 분포 ────────────────────────────────────────────────
    level_counts = an.count_by(acc, ["학교급"], "건수")
    save_table(level_counts, "t01_학교급별_건수")
    viz.bar_count_by_level(level_counts)

    trend = an.yearly_trend(acc)
    save_table(trend, "t02_연도별_추이")
    viz.line_yearly(trend)

    monthly = an.monthly_pattern(acc)
    viz.bar_monthly(monthly)

    weekday = an.weekday_pattern(acc)
    viz.bar_weekday(weekday)

    # ── 2. 장소·부위 분포 ───────────────────────────────────────────
    place = an.count_by(acc, ["사고장소"], "사고건수")
    save_table(place, "t03_장소별_건수")
    viz.bar_topn(place, "사고장소", "사고건수",
                 "사고장소별 발생 건수 Top12", "fig05_장소별", top=12, color="#C44E52")

    part = an.count_by(acc, ["사고부위"], "사고건수")
    save_table(part, "t04_부위별_건수")
    viz.bar_topn(part, "사고부위", "사고건수",
                 "사고부위별 발생 건수 Top12", "fig06_부위별", top=12, color="#937860")

    activity = an.count_by(acc, ["사고시간"], "사고건수")
    viz.bar_topn(activity, "사고시간", "사고건수",
                 "활동(시간)대별 발생 건수 Top10", "fig07_활동별", top=10, color="#4C72B0")

    # ── 3. 초등학교 학년 × 사고형태 히트맵 ─────────────────────────
    elem = dl.apply_grade_order(acc[acc["학교급"] == "초등학교"], "초등학교")
    viz.heatmap_grade_type(elem, "초등학교", "fig08_초등_학년x사고형태")

    # ── 4. 심각도(보상금) 분석 ──────────────────────────────────────
    sev_type = an.severity_by_type(cmp)
    save_table(sev_type, "t05_사고형태별_심각도")

    comp = an.money_composition(cmp)
    viz.bar_money_composition(comp)

    # ── 5. 위험지수: 빈도 × 심각도 ──────────────────────────────────
    # 사고형태 단위(사분면용): 빈도 vs 평균보상금
    freq_type = an.count_by(acc, ["사고형태"], "사고건수")
    risk_type = pd.merge(freq_type, sev_type, on="사고형태", how="inner")
    risk_type["종합위험지수"] = risk_type["사고건수"] * risk_type["평균보상금"]
    save_table(risk_type.sort_values("종합위험지수", ascending=False),
               "t06_사고형태별_위험지수")
    viz.scatter_risk_quadrant(risk_type)

    # 학교급·학년·사고형태·장소 단위 종합위험지수
    risk = an.risk_index(acc, cmp)
    risk = an.risk_quadrant(risk)
    save_table(risk, "t07_종합위험지수_전체")
    viz.bar_top_risk(risk)

    # ── 6. 최종 산출물: 학년별 안전 가이드라인 데이터 ──────────────
    guide = an.guideline_table(risk, top_n=5)
    save_table(guide, "t08_학년별_안전가이드라인_TOP5")

    print("\n── 예시: 추천 결과 (초등학교 1학년) ──")
    print(an.recommend(risk, "초등학교", "1학년").to_string(index=False))

    print("\n[완료] 모든 그림·표가 outputs/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()

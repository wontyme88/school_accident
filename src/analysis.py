"""핵심 분석 로직.

위험(Risk)을 '빈도(frequency) × 심각도(severity)' = 기대손실(expected loss)로 정의한다.

    종합위험지수 = 사고건수 × 평균보상금
                = 발생빈도(얼마나 자주) × 1건당 평균 피해규모(얼마나 크게)

사고 데이터(발생 빈도)와 보상 데이터(피해 심각도)는 별개 데이터셋이므로,
공통 범주 키(학교급·학년·사고형태·사고장소)로 결합해 위험지수를 산출한다.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


# ── 기초 집계 ─────────────────────────────────────────────────────────
def count_by(df: pd.DataFrame, cols: list[str], name: str = "건수") -> pd.DataFrame:
    """범주 컬럼 기준 건수 집계."""
    return (
        df.groupby(cols, observed=True)
        .size()
        .reset_index(name=name)
        .sort_values(name, ascending=False)
    )


def yearly_trend(acc: pd.DataFrame) -> pd.DataFrame:
    """집계연도별 사고 건수 추이."""
    return (
        acc.groupby("집계연도", observed=True)
        .size()
        .reset_index(name="사고건수")
        .sort_values("집계연도")
    )


def monthly_pattern(acc: pd.DataFrame) -> pd.DataFrame:
    """월별 사고 건수(계절성). 사고월 결측은 제외."""
    sub = acc.dropna(subset=["사고월"])
    return (
        sub.groupby(sub["사고월"].astype(int), observed=True)
        .size()
        .reset_index(name="사고건수")
        .rename(columns={"사고월": "월"})
        .sort_values("월")
    )


def weekday_pattern(acc: pd.DataFrame) -> pd.DataFrame:
    """요일별 사고 건수."""
    order = ["월", "화", "수", "목", "금", "토", "일"]
    out = count_by(acc, ["사고요일"], "사고건수")
    out["사고요일"] = pd.Categorical(out["사고요일"], categories=order, ordered=True)
    return out.dropna(subset=["사고요일"]).sort_values("사고요일")


# ── 심각도(보상금) 분석 ───────────────────────────────────────────────
def severity_by_type(cmp: pd.DataFrame) -> pd.DataFrame:
    """사고형태별 보상 통계(건수·총액·평균). 평균보상금 = 심각도."""
    return (
        cmp.groupby("사고형태", observed=True)
        .agg(보상건수=("총보상금", "count"),
             총보상금=("총보상금", "sum"),
             평균보상금=("총보상금", "mean"))
        .reset_index()
        .sort_values("평균보상금", ascending=False)
    )


def money_composition(cmp: pd.DataFrame) -> pd.DataFrame:
    """보상금 구성 항목별 총액(요양/장해/간병/유족/장례비/위로금/보전비용)."""
    totals = cmp[C.MONEY_COLS].sum().sort_values(ascending=False)
    return totals.reset_index().rename(columns={"index": "급여항목", 0: "총액"})


# ── 위험지수(빈도 × 심각도) ───────────────────────────────────────────
def risk_index(
    acc: pd.DataFrame,
    cmp: pd.DataFrame,
    keys: list[str] | None = None,
    levels: list[str] | None = None,
) -> pd.DataFrame:
    """범주 키 단위 종합위험지수.

    Parameters
    ----------
    keys : 결합·집계 기준 컬럼 (기본: 학교급·학년·사고형태·사고장소)
    levels : 대상 학교급 (기본: 초·중·고). '유아' 학년은 자동 제외.
    """
    keys = keys or C.GROUP_KEYS
    levels = levels or ["초등학교", "중학교", "고등학교"]

    acc_f = acc[acc["학교급"].isin(levels) & (acc["사고자학년"] != "유아")]
    cmp_f = cmp[cmp["학교급"].isin(levels) & (cmp["사고자학년"] != "유아")]

    freq = (
        acc_f.groupby(keys, observed=True)
        .size()
        .reset_index(name="사고건수")
    )
    sev = (
        cmp_f.groupby(keys, observed=True)
        .agg(보상건수=("총보상금", "count"), 총보상금=("총보상금", "sum"))
        .reset_index()
    )

    risk = pd.merge(freq, sev, on=keys, how="inner")
    risk["평균보상금"] = risk["총보상금"] / risk["보상건수"]
    risk["종합위험지수"] = risk["사고건수"] * risk["평균보상금"]
    return risk.sort_values("종합위험지수", ascending=False).reset_index(drop=True)


def risk_quadrant(risk: pd.DataFrame) -> pd.DataFrame:
    """빈도 vs 심각도 중앙값을 기준으로 4분면 위험유형을 분류한다.

    - 고빈도·고심각 : 최우선 관리 (빈번하고 피해도 큼)
    - 저빈도·고심각 : 중대사고 대비 (드물지만 치명적)
    - 고빈도·저심각 : 일상 안전수칙 (잦지만 경미)
    - 저빈도·저심각 : 관찰 영역
    """
    out = risk.copy()
    f_med = out["사고건수"].median()
    s_med = out["평균보상금"].median()

    def label(r):
        hi_f = r["사고건수"] >= f_med
        hi_s = r["평균보상금"] >= s_med
        if hi_f and hi_s:
            return "고빈도·고심각(최우선)"
        if not hi_f and hi_s:
            return "저빈도·고심각(중대사고)"
        if hi_f and not hi_s:
            return "고빈도·저심각(일상수칙)"
        return "저빈도·저심각(관찰)"

    out["위험유형"] = out.apply(label, axis=1)
    return out


def guideline_table(risk: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """학교급·학년별 '위험 사고유형·장소 Top N' 가이드라인 데이터를 생성한다.

    이것이 본 프로젝트의 최종 산출물: 학년별 안전 가이드라인에 바로 쓰는 데이터.
    """
    cols = ["학교급", "사고자학년", "사고형태", "사고장소",
            "사고건수", "평균보상금", "종합위험지수"]
    out = (
        risk.sort_values(["학교급", "사고자학년", "종합위험지수"],
                         ascending=[True, True, False])
        .groupby(["학교급", "사고자학년"], observed=True)
        .head(top_n)
        .loc[:, cols]
        .reset_index(drop=True)
    )
    out["위험순위"] = (
        out.groupby(["학교급", "사고자학년"], observed=True).cumcount() + 1
    )
    return out


def recommend(risk: pd.DataFrame, 학교급: str, 사고자학년: str, top_n: int = 5) -> pd.DataFrame:
    """학교급·학년 입력 → 가장 위험한 사고유형·장소 Top N 추천."""
    sub = risk[(risk["학교급"] == 학교급) & (risk["사고자학년"] == 사고자학년)]
    return (
        sub.sort_values("종합위험지수", ascending=False)
        .loc[:, ["사고형태", "사고장소", "사고건수", "평균보상금", "종합위험지수"]]
        .head(top_n)
        .reset_index(drop=True)
    )

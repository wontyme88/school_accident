"""시각화 유틸리티. 한글 폰트 설정 + 보고서용 그림 생성 함수."""
from __future__ import annotations

import platform

import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일 저장
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as C


def setup_korean_font() -> None:
    """OS별 한글 폰트를 설정한다."""
    system = platform.system()
    if system == "Darwin":
        font = "AppleGothic"
    elif system == "Windows":
        font = "Malgun Gothic"
    else:
        font = "NanumGothic"
    plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"
    sns.set_theme(style="whitegrid", font=font, rc={"axes.unicode_minus": False})


def _save(fig, name: str) -> None:
    C.ensure_dirs()
    path = C.OUT_FIG / f"{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [그림] {path.relative_to(C.ROOT)}")


def _won(x: float) -> str:
    """원 단위를 보기 좋은 만원/억 단위 문자열로."""
    if x >= 1e8:
        return f"{x/1e8:.1f}억"
    if x >= 1e4:
        return f"{x/1e4:.0f}만"
    return f"{x:.0f}"


def bar_count_by_level(level_counts, name="fig01_학교급별_사고건수"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=level_counts, x="학교급", y="건수", hue="학교급",
                legend=False, palette="Blues_d", ax=ax)
    ax.set_title("학교급별 학교안전사고 발생 건수 (2021–2025)")
    ax.set_xlabel(""); ax.set_ylabel("사고건수")
    for c in ax.containers:
        ax.bar_label(c, fmt="{:,.0f}", fontsize=8)
    _save(fig, name)


def line_yearly(trend, name="fig02_연도별_추이"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=trend, x="집계연도", y="사고건수", marker="o", ax=ax)
    ax.set_title("연도별 학교안전사고 추이")
    ax.set_xlabel("집계연도"); ax.set_ylabel("사고건수")
    ax.set_xticks(trend["집계연도"])
    for x, y in zip(trend["집계연도"], trend["사고건수"]):
        ax.annotate(f"{y:,}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    _save(fig, name)


def bar_monthly(monthly, name="fig03_월별_계절성"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=monthly, x="월", y="사고건수", color="#4C72B0", ax=ax)
    ax.set_title("월별 학교안전사고 발생 (계절성)")
    ax.set_xlabel("월"); ax.set_ylabel("사고건수")
    _save(fig, name)


def bar_weekday(weekday, name="fig04_요일별"):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.barplot(data=weekday, x="사고요일", y="사고건수", color="#55A868", ax=ax)
    ax.set_title("요일별 학교안전사고 발생")
    ax.set_xlabel(""); ax.set_ylabel("사고건수")
    _save(fig, name)


def bar_topn(df, cat_col, val_col, title, name, top=12, color="#C44E52"):
    sub = df.nlargest(top, val_col)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=sub, y=cat_col, x=val_col, color=color, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(val_col); ax.set_ylabel("")
    for c in ax.containers:
        ax.bar_label(c, fmt="{:,.0f}", fontsize=8, padding=2)
    _save(fig, name)


def heatmap_grade_type(acc_level, level_name, name, top_types=10):
    """학년 × 사고형태 히트맵 (특정 학교급)."""
    top = acc_level["사고형태"].value_counts().nlargest(top_types).index
    sub = acc_level[acc_level["사고형태"].isin(top)]
    pivot = (
        sub.groupby(["사고형태", "사고자학년"], observed=True)
        .size().reset_index(name="건수")
        .pivot(index="사고형태", columns="사고자학년", values="건수")
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                cbar_kws={"label": "사고건수"}, ax=ax)
    ax.set_title(f"{level_name} 학년 × 사고형태 발생 분포")
    ax.set_xlabel("사고자학년"); ax.set_ylabel("사고형태")
    _save(fig, name)


def scatter_risk_quadrant(severity_freq, name="fig09_위험_사분면"):
    """빈도 vs 평균보상금 산점도 + 4분면 경계선.

    심각도가 익사·익수(1.2억) 등으로 수십~수백 배 차이 나므로 로그-로그 축을 사용한다.
    점 크기는 종합위험지수에 비례시켜 '실질 위험'을 한눈에 보이게 한다.
    """
    import numpy as np
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    d = severity_freq.copy()

    sizes = 60 + 900 * (d["종합위험지수"] / d["종합위험지수"].max())
    ax.scatter(d["사고건수"], d["평균보상금"], s=sizes, alpha=0.5,
               color="#4C72B0", edgecolor="white", linewidth=0.6)

    f_med, s_med = d["사고건수"].median(), d["평균보상금"].median()
    ax.axvline(f_med, color="gray", ls="--", lw=1)
    ax.axhline(s_med, color="gray", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")

    # 사분면 라벨
    xmin, xmax = d["사고건수"].min() * 0.6, d["사고건수"].max() * 1.6
    ymin, ymax = d["평균보상금"].min() * 0.6, d["평균보상금"].max() * 1.6
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    q = dict(fontsize=9, alpha=0.35, fontweight="bold", color="#333")
    ax.text(xmax*0.5, ymax*0.7, "고빈도·고심각\n(최우선)", ha="right", va="top", **q)
    ax.text(xmin*1.6, ymax*0.7, "저빈도·고심각\n(중대사고)", ha="left", va="top", **q)
    ax.text(xmax*0.5, ymin*1.6, "고빈도·저심각\n(일상수칙)", ha="right", va="bottom", **q)
    ax.text(xmin*1.6, ymin*1.6, "저빈도·저심각\n(관찰)", ha="left", va="bottom", **q)

    # 주요 사고형태 라벨 (상위 위험 + 고심각)
    show = set(d.nlargest(6, "종합위험지수").index) | set(d.nlargest(4, "평균보상금").index)
    for i in show:
        r = d.loc[i]
        ax.annotate(f" {r['사고형태']}", (r["사고건수"], r["평균보상금"]),
                    fontsize=8, alpha=0.95, va="center")

    ax.set_xlabel("사고건수 (빈도, 로그축)")
    ax.set_ylabel("평균보상금 (심각도, 원, 로그축)")
    ax.set_title("사고형태별 위험 사분면 (빈도 × 심각도) · 점 크기 = 종합위험지수")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _won(v)))
    _save(fig, name)


def bar_top_risk(risk, name="fig10_위험지수_Top", top=12):
    sub = risk.nlargest(top, "종합위험지수").copy()
    sub["라벨"] = (sub["학교급"] + " " + sub["사고자학년"].astype(str)
                  + " · " + sub["사고형태"] + " @" + sub["사고장소"])
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=sub, y="라벨", x="종합위험지수", color="#8172B3", ax=ax)
    ax.set_title("종합위험지수 Top (학교급·학년·사고형태·장소)")
    ax.set_xlabel("종합위험지수 = 사고건수 × 평균보상금"); ax.set_ylabel("")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _won(v)))
    _save(fig, name)


def bar_money_composition(comp, name="fig11_보상금_구성"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=comp, x="급여항목", y="총액", color="#CCB974", ax=ax)
    ax.set_title("보상금 구성 항목별 총 지급액")
    ax.set_xlabel(""); ax.set_ylabel("총액(원)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _won(v)))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, name)

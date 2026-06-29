"""데이터 적재 및 전처리.

- 중앙회 원본(data/raw/)이 있으면 사용하고, 없으면 합성 샘플(data/sample/)을 사용한다.
- 5개 연도 시트를 하나로 합치며, 시트명(연도)을 '집계연도' 컬럼으로 보존한다.
- 보상 데이터는 금액 컬럼을 정수화하고 '총보상금'(심각도 대용치)을 생성한다.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def _resolve_paths() -> tuple[str, str, bool]:
    """(사고파일, 보상파일, is_real) 경로를 반환. 원본 우선, 없으면 샘플."""
    # 원본 위치 후보: data/raw/  또는  중앙회 제공 폴더(둘 다 .gitignore 처리)
    raw_dirs = [C.DATA_RAW, C.ROOT / "★중앙회 제공데이터"]
    for d in raw_dirs:
        real_acc = d / C.ACCIDENT_FILE
        real_cmp = d / C.COMPENSATION_FILE
        if real_acc.exists() and real_cmp.exists():
            return str(real_acc), str(real_cmp), True

    sample_acc = C.DATA_SAMPLE / C.SAMPLE_ACCIDENT_FILE
    sample_cmp = C.DATA_SAMPLE / C.SAMPLE_COMPENSATION_FILE
    if sample_acc.exists() and sample_cmp.exists():
        return str(sample_acc), str(sample_cmp), False

    raise FileNotFoundError(
        "원본 데이터(data/raw/)도, 합성 샘플(data/sample/)도 찾을 수 없습니다.\n"
        "  • 원본: data/raw/ 에 중앙회 제공 xlsx 2개를 복사하세요.\n"
        "  • 샘플: `python scripts/make_sample_data.py` 로 합성 샘플을 생성하세요."
    )


def _concat_year_sheets(path: str) -> pd.DataFrame:
    """연도별 시트를 읽어 '집계연도' 컬럼과 함께 결합한다."""
    sheets = pd.read_excel(path, sheet_name=C.YEARS)
    frames = []
    for year, df in sheets.items():
        df = df.copy()
        df["집계연도"] = int(year)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_accident() -> pd.DataFrame:
    """학교안전사고(발생) 데이터를 적재·전처리한다."""
    acc_path, _, _ = _resolve_paths()
    df = _concat_year_sheets(acc_path)

    # 사고연월(YYYY-MM 문자열) → 연/월 파생
    if "사고연월" in df.columns:
        ym = pd.to_datetime(df["사고연월"], format="%Y-%m", errors="coerce")
        df["사고연도"] = ym.dt.year
        df["사고월"] = ym.dt.month
    return df


def load_compensation() -> pd.DataFrame:
    """학교안전사고 보상 데이터를 적재하고 '총보상금'을 생성한다."""
    _, cmp_path, _ = _resolve_paths()
    df = _concat_year_sheets(cmp_path)

    for col in C.MONEY_COLS:
        if col not in df.columns:
            df[col] = 0
        s = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype("int64")

    df["총보상금"] = df[C.MONEY_COLS].sum(axis=1)
    return df


def is_using_real_data() -> bool:
    """원본 데이터를 쓰는지 여부."""
    return _resolve_paths()[2]


def apply_grade_order(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """특정 학교급 데이터의 '사고자학년'을 순서형 범주로 변환한다."""
    order = C.GRADE_ORDER.get(level)
    if not order:
        return df
    out = df.copy()
    out = out[out["사고자학년"].isin(order)]
    out["사고자학년"] = pd.Categorical(out["사고자학년"], categories=order, ordered=True)
    return out

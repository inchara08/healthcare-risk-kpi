"""
Stage 1 (cont.): Parse CMS SynPUF raw CSVs from ZIP archives into typed DataFrames.

Column naming follows CMS SynPUF data dictionary v1.0.
Output is a dict of DataFrames ready for pandera validation.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)

# ─── Column type maps ────────────────────────────────────────────────────────

BENE_DATE_COLS = ["BENE_BIRTH_DT", "BENE_DEATH_DT"]
BENE_INT_COLS = [
    "BENE_SEX_IDENT_CD",
    "BENE_RACE_CD",
    "SP_ALZHDMTA",
    "SP_CHF",
    "SP_CHRNKIDN",
    "SP_CNCR",
    "SP_COPD",
    "SP_DEPRESSN",
    "SP_DIABETES",
    "SP_ISCHMCHT",
    "SP_OSTEOPRS",
    "SP_RA_OA",
    "SP_STRKETIA",
    "BENE_ESRD_IND",
    "BENE_HI_CVRAGE_TOT_MONS",
    "BENE_SMI_CVRAGE_TOT_MONS",
    "BENE_HMO_CVRAGE_TOT_MONS",
    "PLAN_CVRG_MOS_NUM",
]
BENE_FLOAT_COLS = [
    "MEDREIMB_IP",
    "BENRES_IP",
    "PPPYMT_IP",
    "MEDREIMB_OP",
    "BENRES_OP",
    "PPPYMT_OP",
    "MEDREIMB_CAR",
    "BENRES_CAR",
    "PPPYMT_CAR",
]

INPATIENT_DATE_COLS = ["CLM_ADMSN_DT", "NCH_BENE_DSCHRG_DT"]
INPATIENT_INT_COLS = [
    "CLM_DRG_CD",
    "NCH_BENE_BLOOD_PNTS_FRNSHD_QTY",
    "CLM_ADMSN_SOURCE_CD",
    "PTNT_DSCHRG_STUS_CD",
    "CLM_IP_ADMSN_TYPE_CD",
    "NCH_PTNT_STATUS_IND_CD",
]
INPATIENT_FLOAT_COLS = [
    "CLM_PMT_AMT",
    "NCH_PRMRY_PYR_CLM_PD_AMT",
    "CLM_PASS_THRU_AMT",
    "NCH_BENE_IP_DDCTBL_AMT",
    "NCH_BENE_PTA_COINSRNC_LBLTY_AM",
    "NCH_BENE_BLOOD_PNTS_FRNSHD_QTY",
    "CLM_UTLZTN_DAY_CNT",
    "NCH_BENE_DSCHRG_DT",
]

# Wide ICD9 and procedure columns
ICD9_DIAG_COLS = [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)]
ICD9_PROC_COLS = [f"ICD9_PRCDR_CD_{i}" for i in range(1, 7)]
HCFASPC_COLS = [f"HCFASPC_PRNCPAL_DGNS_CD_{i}" for i in range(1, 11)]


class SynPUFDataset(NamedTuple):
    beneficiaries: pd.DataFrame
    inpatient: pd.DataFrame


def _parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
    return df


def _coerce_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")
    return df


def _coerce_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    return df


def _read_csv_from_zip(zip_path: Path, encoding: str = "latin-1") -> pd.DataFrame:
    """Extract the first .csv found in a ZIP archive and return as DataFrame."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found in {zip_path}")
        with zf.open(csv_names[0]) as f:
            raw = io.TextIOWrapper(f, encoding=encoding)
            df = pd.read_csv(raw, dtype=str, low_memory=False)
    logger.info("Loaded %s: %d rows, %d cols", zip_path.name, len(df), len(df.columns))
    return df


def load_beneficiaries(zip_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate beneficiary summary files."""
    frames = []
    for path in zip_paths:
        if "beneficiary" not in path.name.lower() and "bene" not in path.name.lower():
            continue
        df = _read_csv_from_zip(path)
        df = _parse_dates(df, BENE_DATE_COLS)
        df = _coerce_int(df, BENE_INT_COLS)
        df = _coerce_float(df, BENE_FLOAT_COLS)
        frames.append(df)

    if not frames:
        raise ValueError("No beneficiary ZIP files found in provided paths")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["DESYNPUF_ID"])
    logger.info("Beneficiaries loaded: %d unique records", len(combined))
    return combined


def load_inpatient_claims(zip_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate inpatient claims files."""
    frames = []
    for path in zip_paths:
        if "inpatient" not in path.name.lower():
            continue
        df = _read_csv_from_zip(path)
        df = _parse_dates(df, INPATIENT_DATE_COLS)
        df = _coerce_int(df, INPATIENT_INT_COLS)
        df = _coerce_float(df, INPATIENT_FLOAT_COLS)
        frames.append(df)

    if not frames:
        raise ValueError("No inpatient ZIP files found in provided paths")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["CLM_ID"])
    logger.info("Inpatient claims loaded: %d unique records", len(combined))
    return combined


def _read_csv_direct(csv_path: Path) -> pd.DataFrame:
    """Read a CSV file directly (for synthetic generated data)."""
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    logger.info("Loaded %s: %d rows, %d cols", csv_path.name, len(df), len(df.columns))
    return df


def load_raw_data(raw_dir: Path) -> SynPUFDataset:
    """Load SynPUF data from raw_dir (supports both ZIP and CSV formats)."""
    zip_paths = sorted(raw_dir.glob("*.zip"))
    csv_paths = sorted(raw_dir.glob("*.csv"))

    if not zip_paths and not csv_paths:
        raise FileNotFoundError(
            f"No data files found in {raw_dir}. Run 'make download' first."
        )

    if csv_paths:
        # Synthetic generated CSVs
        bene_files = [p for p in csv_paths if "beneficiary" in p.name.lower() or "bene" in p.name.lower()]
        inpatient_files = [p for p in csv_paths if "inpatient" in p.name.lower()]

        bene_frames = []
        for p in bene_files:
            df = _read_csv_direct(p)
            df = _parse_dates(df, BENE_DATE_COLS)
            df = _coerce_int(df, BENE_INT_COLS)
            df = _coerce_float(df, BENE_FLOAT_COLS)
            bene_frames.append(df)

        inpatient_frames = []
        for p in inpatient_files:
            df = _read_csv_direct(p)
            df = _parse_dates(df, INPATIENT_DATE_COLS)
            df = _coerce_int(df, INPATIENT_INT_COLS)
            df = _coerce_float(df, INPATIENT_FLOAT_COLS)
            inpatient_frames.append(df)

        if not bene_frames:
            raise FileNotFoundError(f"No beneficiary CSV found in {raw_dir}")
        if not inpatient_frames:
            raise FileNotFoundError(f"No inpatient CSV found in {raw_dir}")

        bene = pd.concat(bene_frames, ignore_index=True).drop_duplicates(subset=["DESYNPUF_ID"])
        inpatient = pd.concat(inpatient_frames, ignore_index=True).drop_duplicates(subset=["CLM_ID"])
        logger.info("Beneficiaries: %d unique records", len(bene))
        logger.info("Inpatient claims: %d unique records", len(inpatient))
        return SynPUFDataset(beneficiaries=bene, inpatient=inpatient)

    # ZIP fallback (original CMS downloads if available)
    bene = load_beneficiaries(zip_paths)
    inpatient = load_inpatient_claims(zip_paths)
    return SynPUFDataset(beneficiaries=bene, inpatient=inpatient)

"""Shared pytest fixtures using the 500-row sample dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SAMPLE_DIR = Path("data/samples")


@pytest.fixture(scope="session")
def sample_beneficiaries() -> pd.DataFrame:
    """500-row synthetic beneficiary DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 500
    bene_ids = [f"BENE{i:06d}" for i in range(n)]
    birth_years = rng.integers(1920, 1960, size=n)
    birth_dates = pd.to_datetime(
        [f"{y}-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}" for y in birth_years]
    )
    return pd.DataFrame(
        {
            "DESYNPUF_ID": bene_ids,
            "BENE_BIRTH_DT": birth_dates,
            "BENE_DEATH_DT": pd.NaT,
            "BENE_SEX_IDENT_CD": pd.array(rng.choice([1, 2], size=n), dtype="Int16"),
            "BENE_RACE_CD": pd.array(rng.choice([1, 2, 3, 4, 5, 6], size=n), dtype="Int16"),
            "SP_ALZHDMTA": pd.array(rng.choice([0, 1, 2], size=n), dtype="Int16"),
            "SP_CHF": pd.array(rng.choice([0, 1, 2], size=n), dtype="Int16"),
            "SP_CHRNKIDN": pd.array(rng.choice([0, 1, 2], size=n), dtype="Int16"),
            "SP_CNCR": pd.array(rng.choice([0, 1, 2], size=n), dtype="Int16"),
            "SP_COPD": pd.array(rng.choice([0, 1, 2], size=n), dtype="Int16"),
            "SP_DEPRESSN": pd.array(rng.choice([0, 1, 2], size=n), dtype="Int16"),
            "SP_DIABETES": pd.array(rng.choice([0, 1, 2], size=n), dtype="Int16"),
            "SP_ISCHMCHT": pd.array(rng.choice([0, 1, 2], size=n), dtype="Int16"),
            "SP_OSTEOPRS": pd.array(rng.choice([0, 1, 2], size=n), dtype="Int16"),
            "SP_RA_OA": pd.array(rng.choice([0, 1, 2], size=n), dtype="Int16"),
            "SP_STRKETIA": pd.array(rng.choice([0, 1, 2], size=n), dtype="Int16"),
            "BENE_HI_CVRAGE_TOT_MONS": pd.array(rng.integers(0, 13, size=n), dtype="Int16"),
            "BENE_SMI_CVRAGE_TOT_MONS": pd.array(rng.integers(0, 13, size=n), dtype="Int16"),
            "BENE_HMO_CVRAGE_TOT_MONS": pd.array(rng.integers(0, 13, size=n), dtype="Int16"),
            "PLAN_CVRG_MOS_NUM": pd.array(rng.integers(0, 13, size=n), dtype="Int16"),
            "MEDREIMB_IP": rng.uniform(0, 50000, size=n).astype("float32"),
            "MEDREIMB_OP": rng.uniform(0, 10000, size=n).astype("float32"),
            "MEDREIMB_CAR": rng.uniform(0, 5000, size=n).astype("float32"),
        }
    )


@pytest.fixture(scope="session")
def sample_inpatient(sample_beneficiaries) -> pd.DataFrame:
    """~1500-row synthetic inpatient claims DataFrame for testing."""
    rng = np.random.default_rng(42)
    bene_ids = sample_beneficiaries["DESYNPUF_ID"].values
    n_claims = 1500
    claim_bene = rng.choice(bene_ids, size=n_claims)

    admit_dates = pd.to_datetime(pd.date_range("2008-01-01", "2010-12-31", periods=n_claims))
    los = rng.integers(1, 20, size=n_claims)
    discharge_dates = admit_dates + pd.to_timedelta(los, unit="D")

    return pd.DataFrame(
        {
            "CLM_ID": [f"CLM{i:08d}" for i in range(n_claims)],
            "DESYNPUF_ID": claim_bene,
            "CLM_ADMSN_DT": admit_dates,
            "NCH_BENE_DSCHRG_DT": discharge_dates,
            "PRVDR_NUM": [f"PROV{rng.integers(1, 20):03d}" for _ in range(n_claims)],
            "CLM_DRG_CD": pd.array(rng.integers(1, 999, size=n_claims), dtype="Int16"),
            "CLM_PMT_AMT": rng.uniform(500, 100000, size=n_claims).astype("float32"),
            "CLM_PASS_THRU_AMT": rng.uniform(0, 5000, size=n_claims).astype("float32"),
            "CLM_ADMSN_SOURCE_CD": pd.array(rng.integers(1, 9, size=n_claims), dtype="Int16"),
            "CLM_IP_ADMSN_TYPE_CD": pd.array(rng.integers(1, 4, size=n_claims), dtype="Int16"),
            "PTNT_DSCHRG_STUS_CD": pd.array(rng.integers(1, 30, size=n_claims), dtype="Int16"),
            "CLM_UTLZTN_DAY_CNT": pd.array(los, dtype="Int16"),
            "ICD9_DGNS_CD_1": [f"{rng.integers(100,999)}" for _ in range(n_claims)],
            "ICD9_DGNS_CD_2": [
                f"{rng.integers(100,999)}" if rng.random() > 0.3 else None for _ in range(n_claims)
            ],
        }
    )


@pytest.fixture(scope="session")
def sample_patient_features(sample_inpatient, sample_beneficiaries) -> pd.DataFrame:
    """Synthetic analytics.patient_features for feature pipeline tests."""
    rng = np.random.default_rng(42)
    n = len(sample_inpatient)
    bene_birth = dict(
        zip(
            sample_beneficiaries["DESYNPUF_ID"],
            sample_beneficiaries["BENE_BIRTH_DT"],
        )
    )

    admit = sample_inpatient["CLM_ADMSN_DT"]
    discharge = sample_inpatient["NCH_BENE_DSCHRG_DT"]
    bene_ids = sample_inpatient["DESYNPUF_ID"].values

    age = [
        int((pd.Timestamp(a) - pd.Timestamp(bene_birth.get(b, "1940-01-01"))).days / 365)
        for a, b in zip(admit, bene_ids)
    ]

    return pd.DataFrame(
        {
            "claim_id": sample_inpatient["CLM_ID"].values,
            "bene_id": bene_ids,
            "admit_dt": admit.values,
            "discharge_dt": discharge.values,
            "provider_id": sample_inpatient["PRVDR_NUM"].values,
            "drg_cd": sample_inpatient["CLM_DRG_CD"].values,
            "admit_source_cd": sample_inpatient["CLM_ADMSN_SOURCE_CD"].values,
            "admit_type_cd": sample_inpatient["CLM_IP_ADMSN_TYPE_CD"].values,
            "discharge_status_cd": sample_inpatient["PTNT_DSCHRG_STUS_CD"].values,
            "claim_pmt_amt": sample_inpatient["CLM_PMT_AMT"].values,
            "pass_thru_amt": sample_inpatient["CLM_PASS_THRU_AMT"].values,
            "utilization_day_cnt": sample_inpatient["CLM_UTLZTN_DAY_CNT"].values,
            "admitting_icd9": None,
            "age_at_admit": age,
            "los_days": (discharge - admit).dt.days.values,
            "died_within_30d": rng.integers(0, 2, size=n),
            "elixhauser_count": rng.integers(0, 10, size=n),
            "prior_admits_90d": rng.integers(0, 5, size=n),
            "prior_admits_365d": rng.integers(0, 10, size=n),
            "days_since_last_admit": rng.integers(1, 365, size=n),
            "sex_cd": rng.choice([1, 2], size=n),
            "race_cd": rng.choice([1, 2, 3, 4, 5, 6], size=n),
            "hmo_coverage_months": rng.integers(0, 13, size=n),
            "hi_coverage_months": rng.integers(0, 13, size=n),
            "medreimb_ip": rng.uniform(0, 50000, size=n).astype(float),
            "death_dt": pd.NaT,
        }
    )


@pytest.fixture(scope="session")
def sample_conditions(sample_beneficiaries) -> pd.DataFrame:
    """Long-format chronic conditions for the sample beneficiaries."""
    condition_map = {
        "SP_ALZHDMTA": "alzheimers",
        "SP_CHF": "chf",
        "SP_CHRNKIDN": "chronic_kidney",
        "SP_CNCR": "cancer",
        "SP_COPD": "copd",
        "SP_DEPRESSN": "depression",
        "SP_DIABETES": "diabetes",
        "SP_ISCHMCHT": "ischemic_heart",
        "SP_OSTEOPRS": "osteoporosis",
        "SP_RA_OA": "ra_oa",
        "SP_STRKETIA": "stroke_tia",
    }
    rng = np.random.default_rng(42)
    rows = []
    for bene_id in sample_beneficiaries["DESYNPUF_ID"].values:
        for cond in condition_map.values():
            if rng.random() > 0.7:
                rows.append((bene_id, cond, rng.choice([1, 2])))
    return pd.DataFrame(rows, columns=["bene_id", "condition_code", "indicator"])

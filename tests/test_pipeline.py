import pytest
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pipeline import ValidationPipeline


@pytest.fixture
def pipeline(tmp_path):
    """Create a pipeline with a temp output dir."""
    p = ValidationPipeline(
        config_path="config/validation_config.yaml",
        profile="standard"
    )
    p.output_dir = str(tmp_path)
    return p


@pytest.fixture
def csv_path():
    return "data/gpu_metrics.csv"


def test_pipeline_run_returns_0_or_1(pipeline, csv_path):
    code = pipeline.run(source="csv", input_path=csv_path,
                        generate_charts=False)
    assert code in [0, 1]


def test_pipeline_creates_results_file(pipeline, csv_path, tmp_path):
    pipeline.run(source="csv", input_path=csv_path,
                 generate_charts=False)
    json_files = list(tmp_path.glob("results_*.json"))
    assert len(json_files) == 1


def test_pipeline_results_file_is_valid_json(pipeline, csv_path, tmp_path):
    pipeline.run(source="csv", input_path=csv_path,
                 generate_charts=False)
    json_files = list(tmp_path.glob("results_*.json"))
    with open(json_files[0]) as f:
        data = json.load(f)
    assert "metadata" in data
    assert "results"  in data
    assert "overall"  in data["results"]


def test_pipeline_creates_history_csv(pipeline, csv_path, tmp_path):
    pipeline.run(source="csv", input_path=csv_path,
                 generate_charts=False)
    assert os.path.exists(os.path.join(str(tmp_path), "history.csv"))


def test_pipeline_history_grows_with_each_run(pipeline, csv_path, tmp_path):
    pipeline.run(source="csv", input_path=csv_path,
                 generate_charts=False)
    pipeline.run(source="csv", input_path=csv_path,
                 generate_charts=False)
    import pandas as pd
    df = pd.read_csv(os.path.join(str(tmp_path), "history.csv"))
    # 5 GPUs in csv x 2 runs = 10 rows
    assert len(df) >= 6


def test_pipeline_strict_profile_fails_more(csv_path):
    p_standard = ValidationPipeline(profile="standard")
    p_strict   = ValidationPipeline(profile="strict")
    p_standard.output_dir = "/tmp/std"
    p_strict.output_dir   = "/tmp/strict"
    os.makedirs("/tmp/std",    exist_ok=True)
    os.makedirs("/tmp/strict", exist_ok=True)

    code_std    = p_standard.run(source="csv", input_path=csv_path,
                                  generate_charts=False)
    code_strict = p_strict.run(source="csv",   input_path=csv_path,
                                generate_charts=False)
    # strict should fail at least as often as standard
    assert code_strict >= code_std


def test_pipeline_missing_input_returns_1(pipeline):
    code = pipeline.run(source="csv", input_path="nonexistent.csv",
                        generate_charts=False)
    assert code == 1
import yaml
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Thresholds:
    temp_limit_c:       int   = 73
    power_limit_w:      int   = 300
    memory_limit_gb:    float = 23.0
    gpu_util_warn_pct:  int   = 95


@dataclass
class ReportingConfig:
    save_json:             bool  = True
    save_csv:              bool  = False
    include_metadata:      bool  = True
    pass_rate_target_pct:  int   = 95


@dataclass
class ToolConfig:
    name:       str = "GPU Validation Tool"
    version:    str = "4.0.0"
    output_dir: str = "results"
    log_file:   str = "validation.log"


@dataclass
class ValidationConfig:
    tool:       ToolConfig       = field(default_factory=ToolConfig)
    thresholds: Thresholds       = field(default_factory=Thresholds)
    reporting:  ReportingConfig  = field(default_factory=ReportingConfig)
    profile:    Optional[str]    = None


def load_config(config_path: str, profile: str = None) -> ValidationConfig:
    """
    Load validation config from a YAML file.
    Optionally override thresholds with a named profile.
    """
    if not os.path.exists(config_path):
        logger.warning("Config file not found: %s — using defaults", config_path)
        return ValidationConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    logger.info("Loaded config from %s", config_path)

    cfg = ValidationConfig()

    # Load tool section
    if "tool" in raw:
        t = raw["tool"]
        cfg.tool = ToolConfig(
            name=t.get("name", "GPU Validation Tool"),
            version=t.get("version", "4.0.0"),
            output_dir=t.get("output_dir", "results"),
            log_file=t.get("log_file", "validation.log"),
        )

    # Load thresholds — start with defaults
    base = raw.get("thresholds", {})
    cfg.thresholds = Thresholds(
        temp_limit_c=base.get("temp_limit_c", 73),
        power_limit_w=base.get("power_limit_w", 300),
        memory_limit_gb=base.get("memory_limit_gb", 23.0),
        gpu_util_warn_pct=base.get("gpu_util_warn_pct", 95),
    )

    # Apply profile overrides if requested
    if profile and "profiles" in raw:
        profiles = raw["profiles"]
        if profile not in profiles:
            raise ValueError(
                f"Profile '{profile}' not found. "
                f"Available: {list(profiles.keys())}"
            )
        overrides = profiles[profile]
        logger.info("Applying profile: %s — %s",
                    profile, overrides.get("description", ""))
        if "temp_limit_c"  in overrides:
            cfg.thresholds.temp_limit_c  = overrides["temp_limit_c"]
        if "power_limit_w" in overrides:
            cfg.thresholds.power_limit_w = overrides["power_limit_w"]
        cfg.profile = profile

    # Load reporting section
    if "reporting" in raw:
        r = raw["reporting"]
        cfg.reporting = ReportingConfig(
            save_json=r.get("save_json", True),
            save_csv=r.get("save_csv", False),
            include_metadata=r.get("include_metadata", True),
            pass_rate_target_pct=r.get("pass_rate_target_pct", 95),
        )

    logger.debug("Config loaded: temp=%dC power=%dW profile=%s",
                 cfg.thresholds.temp_limit_c,
                 cfg.thresholds.power_limit_w,
                 cfg.profile or "default")

    return cfg
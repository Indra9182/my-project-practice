"""Quick demo — shows all three patterns working together."""
import sys, os, logging
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

from config import load_config
from patterns import (
    get_strategy, create_reader,
    ValidationRunner, LoggingObserver, FileObserver
)

# Load config
cfg = load_config("config/validation_config.yaml", profile="strict")
print(f"\nProfile: strict | Temp limit: {cfg.thresholds.temp_limit_c}C")

# Factory pattern — create reader automatically
reader = create_reader("csv")
gpus   = reader.read(filepath="data/gpu_metrics.csv")
print(f"Loaded {len(gpus)} GPUs via {reader.source_name}")

# Strategy pattern — swap validation strategy easily
for strategy_name in ["thermal", "power", "full"]:
    strategy = get_strategy(strategy_name)
    runner   = ValidationRunner(strategy)

    # Observer pattern — add listeners
    runner.add_observer(LoggingObserver())
    runner.add_observer(FileObserver("results/failures.log"))

    result = runner.run(gpus, cfg.thresholds)
    print(f"\nStrategy '{strategy_name}': {result['overall']} "
          f"({result['summary']['pass_rate']} pass rate)")
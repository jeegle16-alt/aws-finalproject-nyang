# train_entrypoint.py
import json
import subprocess
from pathlib import Path

HP_JSON = Path("/opt/ml/input/config/hyperparameters.json")


def load_hps():
    if HP_JSON.exists():
        return json.loads(HP_JSON.read_text())
    return {}


def main():
    hp = load_hps()

    s3_public = hp.get("s3-public-uri")
    s3_etl = hp.get("s3-etl-uri")
    lookback = hp.get("lookback-days", "30")
    s3_model_prefix = hp.get("s3-model-prefix")

    cmd = ["python", "-u", "src/runtime/train.py"]

    if s3_public:
        cmd += ["--s3-public-feature-uri", str(s3_public)]
    if s3_etl:
        cmd += ["--s3-etl-uri", str(s3_etl)]
    cmd += ["--lookback-days", str(lookback)]

    if s3_model_prefix:
        cmd += ["--s3-model-prefix", str(s3_model_prefix)]

    print("[ENTRYPOINT] hyperparameters =", hp)
    print("[ENTRYPOINT] cmd =", " ".join(cmd))

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
import json

from smilesbrain.utils import stable_seed, write_run_manifest


def test_stable_seed_is_process_independent():
    assert stable_seed("ketamine") == stable_seed("ketamine")
    assert stable_seed("ketamine") != stable_seed("LSD")


def test_manifest_records_seed_and_input_hash(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("seed: 7\n")
    output = tmp_path / "result.csv"
    manifest = write_run_manifest(output, config={"name": "test"}, seed=7,
                                  input_paths=[config])
    payload = json.loads(manifest.read_text())
    assert payload["seed"] == 7
    assert str(config) in payload["input_sha256"]
    assert payload["config"]["name"] == "test"

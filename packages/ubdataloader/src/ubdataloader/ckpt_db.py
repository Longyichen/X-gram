import json
import duckdb
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    remote_path: List[str]
    proportion: List[float]

    seq_len_each_sample: int
    global_batch_size: int
    text_chunk_size: int

    pack_method: str


@dataclass
class StateDict:
    consumed_samples: int
    token_it: int
    epoch: int
    sample_in_epoch: int
    streaming_dict: Dict[str, Any]


class CkptDB:
    def __init__(
        self,
        is_master: bool,
        ckpt_path: str,
    ):
        self.is_master = is_master
        self.ckpt_path = ckpt_path
        self.ckpt_db = None

    def _create_database(self, dataset_config: DatasetConfig):
        self.ckpt_db.execute(
            "CREATE TABLE IF NOT EXISTS state_dict ("
            "consumed_samples INTEGER PRIMARY KEY, "
            "token_it INTEGER, "
            "epoch INTEGER, "
            "sample_in_epoch INTEGER, "
            "streaming_dict JSON"
            ");"
            "CREATE TABLE IF NOT EXISTS checksum (consumed_samples INTEGER PRIMARY KEY, tokens_checksum INTEGER, labels_checksum INTEGER);"
            "CREATE TABLE IF NOT EXISTS dataset_config (id INTEGER PRIMARY KEY, config JSON);"
            "INSERT OR IGNORE INTO dataset_config (id, config) VALUES (0, ?);",
            [
                json.dumps(
                    {
                        "remote_path": dataset_config.remote_path,
                        "proportion": dataset_config.proportion,
                        "seq_len_each_sample": dataset_config.seq_len_each_sample,
                        "global_batch_size": dataset_config.global_batch_size,
                        "text_chunk_size": dataset_config.text_chunk_size,
                        "pack_method": dataset_config.pack_method,
                    }
                )
            ],
        )

    def save(
        self,
        state_dict: StateDict,
        dataset_config: DatasetConfig,
    ):
        assert self.is_master, "only the master process can save the state dict"

        if self.ckpt_db is None:
            self.ckpt_db = duckdb.connect(self.ckpt_path)
            self._create_database(dataset_config)

        self.ckpt_db.execute(
            (
                "INSERT OR REPLACE INTO state_dict "
                "(consumed_samples, token_it, epoch, sample_in_epoch, streaming_dict) "
                "VALUES (?, ?, ?, ?, ?)"
            ),
            [
                state_dict.consumed_samples,
                state_dict.token_it,
                state_dict.epoch,
                state_dict.sample_in_epoch,
                json.dumps(state_dict.streaming_dict),
            ],
        )
        self.ckpt_db.commit()

    def load(self, consumed_samples: int) -> Tuple[DatasetConfig, StateDict]:
        if self.ckpt_db is None:
            self.ckpt_db = duckdb.connect(self.ckpt_path, read_only=True)

        dataset_config = DatasetConfig(
            **json.loads(
                self.ckpt_db.execute(
                    "SELECT config FROM dataset_config WHERE id = 0"
                ).fetchone()[0]
            )
        )

        (
            consumed_samples,
            token_it,
            epoch,
            sample_in_epoch,
            streaming_dict,
        ) = self.ckpt_db.execute(
            "SELECT consumed_samples, token_it, epoch, sample_in_epoch, streaming_dict FROM state_dict WHERE consumed_samples = ?",
            [consumed_samples],
        ).fetchone()

        state_dict = StateDict(
            consumed_samples=consumed_samples,
            token_it=token_it,
            epoch=epoch,
            sample_in_epoch=sample_in_epoch,
            streaming_dict=json.loads(streaming_dict),
        )

        self.ckpt_db.close()
        self.ckpt_db = None

        return dataset_config, state_dict

    def save_checksum(
        self, consumed_samples: int, tokens_checksum: int, labels_checksum: int
    ):
        if self.ckpt_db is None:
            self.ckpt_db = duckdb.connect(self.ckpt_path)

        self.ckpt_db.execute(
            "INSERT OR REPLACE INTO checksum (consumed_samples, tokens_checksum, labels_checksum) VALUES (?, ?, ?)",
            [consumed_samples, tokens_checksum, labels_checksum],
        )
        self.ckpt_db.commit()

    def load_checksum(self, consumed_samples: int):
        if self.ckpt_db is None:
            self.ckpt_db = duckdb.connect(self.ckpt_path, read_only=True)
        return self.ckpt_db.execute(
            "SELECT tokens_checksum, labels_checksum FROM checksum WHERE consumed_samples = ?",
            [consumed_samples],
        ).fetchone()

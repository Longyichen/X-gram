import numpy
from dataclasses import dataclass
from streaming import StreamingDataset, Stream
from typing import Optional, List, Dict, Any


@dataclass
class TextChunkState:
    epoch: int
    sample_in_epoch: int
    streaming_dict: Dict[str, Any]


@dataclass
class TextTokenItem:
    # payload
    text: str
    token: List[int]
    # metadata
    ck_tot: int
    ck_idx: int


@dataclass
class TextTokenChunk:
    # payload
    text_token_payloads: List[TextTokenItem]
    # state of the text chunk
    text_chunk_state: TextChunkState


class TextDataset:
    def __init__(
        self,
        local_path: List[str],
        remote_path: List[Optional[str]],
        proportion: List[float],
        world_size: int,
        chunk_size: int,
        epoch: int = 0,
        sample_in_epoch: int = 0,
        streaming_dict: Dict[str, Any] = None,
        use_token_column: Optional[str] = None,
    ):
        self.world_size = world_size
        self.chunk_size = chunk_size
        assert self.chunk_size % self.world_size == 0, "chunk_size % world_size == 0"
        print("world_size: %s chunk_size: %s " % (self.world_size, self.chunk_size))

        # resume from the checkpoint
        self.epoch = epoch
        self.sample_in_epoch = sample_in_epoch

        self.use_token_column = use_token_column

        streams = self.create_streams(local_path, remote_path, proportion)

        # NOTE : single process mode to get same sample in all worker
        # so the replication and num_canonical_nodes is 1
        # but there are multiple processes in each worker
        # so we use file lock to ensure each dataset be initialized one by one,
        # due the streaming dataset need to create the shared memory file, there name will be the same
        self.stream_dataset = StreamingDataset(
            streams=streams,
            shuffle=False,
            num_canonical_nodes=1,
            replication=1,
            batch_size=self.chunk_size,
            keep_zip=True,
            predownload=self.chunk_size * 8,
            cache_limit="8gb",
        )

        print(
            "Init the streaming dataset done, dataset length: %s"
            % len(self.stream_dataset)
        )

        if streaming_dict is not None:
            self.stream_dataset.load_state_dict(streaming_dict)
            print("Load the streaming dataset state dict done")

    def create_streams(
        self,
        local_path: List[str],
        remote_path: List[Optional[str]],
        proportion: List[float],
    ) -> List[Stream]:
        streams = []
        for idx, (local, remote, prop) in enumerate(
            zip(local_path, remote_path, proportion)
        ):
            local = None if local == "None" else local
            remote = None if remote == "None" else remote

            assert not (local is None and remote is None), (
                "local and remote cannot both be None"
            )

            if local is None:
                local = f"/tmp/ubdataloader_cache_{idx}"
                print(f"WARNING: local is None, use the temporary directory: {local}")

            print(
                "Create stream dataset [%d]: local_path: %s, remote_path: %s, proportion: %s"
                % (idx, local, remote, prop)
            )

            streams.append(
                Stream(
                    remote=remote,
                    local=local,
                    repeat=prop,
                )
            )
        return streams

    def text_chunk_state(self) -> TextChunkState:
        return TextChunkState(
            epoch=self.epoch,
            sample_in_epoch=self.sample_in_epoch,
            streaming_dict=self.stream_dataset.state_dict(
                self.sample_in_epoch, from_beginning=True
            ),
        )

    def __iter__(self):
        # the stream dataset will be re-created in the next epoch
        # so it's infinite loop

        state = None
        all_items = []

        while True:
            for data in self.stream_dataset:
                self.sample_in_epoch += 1

                if state is None:
                    state = self.text_chunk_state()

                if self.use_token_column is None:
                    text = data["text"]
                    token = None
                else:
                    text = None
                    token = numpy.frombuffer(
                        data[self.use_token_column], dtype=numpy.int32
                    ).tolist()

                item = TextTokenItem(
                    text=text,
                    token=token,
                    ck_tot=int(data["ck_tot"]),
                    ck_idx=int(data["ck_idx"]),
                )

                all_items.append(item)

                if len(all_items) == self.chunk_size:
                    yield TextTokenChunk(
                        text_token_payloads=all_items,
                        text_chunk_state=state,
                    )
                    all_items = []
                    state = None

            print(
                f"!!!WARNING: Streaming dataset epoch done, epoch: {self.epoch}, use the next epoch: {self.epoch + 1}"
            )
            # the stream dataset will be re-created in the next epoch
            self.sample_in_epoch = 0
            self.epoch += 1

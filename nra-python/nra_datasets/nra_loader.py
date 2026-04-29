"""
HuggingFace Datasets loader script for NRA archives.
"""

import datasets
import nra

class NRADatasetBuilder(datasets.GeneratorBasedBuilder):
    """HuggingFace Datasets builder for .nra archives."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", description="Load NRA archive"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Dataset loaded from Neural Ready Archive (NRA)",
            features=datasets.Features({
                "file_id": datasets.Value("string"),
                "bytes": datasets.Value("binary"),
            }),
        )

    def _split_generators(self, dl_manager):
        # Determine the archive path from data_files
        data_files = self.config.data_files
        if data_files is None:
            raise ValueError("data_files must be specified (path to .nra archive)")
        
        if isinstance(data_files, dict):
            archive_path = data_files["train"][0]
        elif isinstance(data_files, (list, tuple)):
            archive_path = data_files[0]
        else:
            archive_path = str(data_files)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"archive_path": archive_path},
            ),
        ]

    def _generate_examples(self, archive_path):
        # If it's an HTTP URL, use CloudArchive for zero-download streaming
        if str(archive_path).startswith(("http://", "https://")):
            archive = nra.CloudArchive(str(archive_path))
        else:
            archive = nra.BetaArchive(str(archive_path))
            
        for idx, file_id in enumerate(archive.file_ids()):
            yield idx, {
                "file_id": file_id,
                "bytes": bytes(bytearray(archive.read_file(file_id))),
            }

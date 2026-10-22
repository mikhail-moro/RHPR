"""Contracts for public source decoding and held-out target isolation."""
from pathlib import Path
import gzip
import struct
import tempfile
import unittest

from scripts.prepare_gr_beauty_data import decode_example, merge_splits, read_examples


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def message(tag, content):
    return varint((tag << 3) | 2) + varint(len(content)) + content


class BeautyPreparationTests(unittest.TestCase):
    def test_example_preserves_utf8_text_and_zero_id(self):
        # tf.train.Example -> Features map -> Feature.{bytes_list,int64_list}.
        text = message(1, message(1, "Title: crème".encode()))
        identity = message(3, message(1, varint(0)))
        entries = [message(1, message(1, b"text") + message(2, text)),
                   message(1, message(1, b"id") + message(2, identity))]
        self.assertEqual(decode_example(message(1, b"".join(entries))),
                         {"text": ["Title: crème".encode()], "id": [0]})

    def test_packed_and_unpacked_int64_lists(self):
        packed = message(3, message(1, varint(1) + varint(300)))
        unpacked = message(3, b"\x08" + varint(1) + b"\x08" + varint(300))
        for feature in (packed, unpacked):
            example = message(1, message(1, message(1, b"sequence_data") + message(2, feature)))
            self.assertEqual(decode_example(example), {"sequence_data": [1, 300]})

    def test_truncated_tfrecord_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.tfrecord.gz"
            with gzip.open(path, "wb") as stream:
                stream.write(struct.pack("<Q", 10) + b"\0" * 4 + b"short")
            with self.assertRaisesRegex(ValueError, "truncated TFRecord payload"):
                list(read_examples(path))

    def test_full_train_history_retained_while_source_eval_is_clipped(self):
        train = list(range(35))
        records = merge_splits({0: train}, {0: train[-20:] + [35]},
                               {0: (train + [35])[-20:] + [36]})
        self.assertEqual(records, [{"user_id": 0, "train": train, "validation": 35, "test": 36}])

    def test_wrong_target_shift_or_user_set_fails(self):
        for validation, testing in (({0: [1, 2, 3]}, {0: [1, 2, 4]}),
                                    ({1: [1, 2, 3]}, {0: [1, 2, 3, 4]})):
            with self.assertRaises(ValueError):
                merge_splits({0: [1, 2]}, validation, testing)


if __name__ == "__main__":
    unittest.main()

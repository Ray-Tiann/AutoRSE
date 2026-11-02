import base64
import copy
import unittest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
from autorse_leaderboard.receipts import sign_receipt, verify_receipt


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        key = Ed25519PrivateKey.generate()
        self.private = base64.b64encode(key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).decode()
        self.keys = {"test": {"algorithm": "Ed25519", "public_key": base64.b64encode(
            key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()}}
        self.payload = {"evaluated_at": "2026-10-30T00:00:00+00:00", "participant": "test-user", "score": 0.123}
        self.receipt = sign_receipt(self.payload, self.private, "test")

    def test_round_trip(self):
        self.assertEqual(verify_receipt(self.receipt, self.keys), self.payload)

    def test_tamper_and_unknown_key(self):
        for field, value in (("participant", "attacker"), ("score", 1.0)):
            changed = copy.deepcopy(self.receipt)
            changed[field] = value
            with self.assertRaises(ValueError):
                verify_receipt(changed, self.keys)
        with self.assertRaises(ValueError):
            verify_receipt(self.receipt, {})

    def test_revoked_and_time_bound(self):
        self.keys["test"]["not_before"] = "2026-10-31T00:00:00+00:00"
        with self.assertRaises(ValueError):
            verify_receipt(self.receipt, self.keys)
        self.keys["test"].pop("not_before")
        self.keys["test"]["revoked"] = True
        with self.assertRaises(ValueError):
            verify_receipt(self.receipt, self.keys)

    def test_unsigned_nan(self):
        with self.assertRaises(ValueError):
            verify_receipt(self.payload, self.keys)
        with self.assertRaises(ValueError):
            sign_receipt({"score": float("nan")}, self.private, "test")


if __name__ == "__main__":
    unittest.main()

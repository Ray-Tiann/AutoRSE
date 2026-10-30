"""Ed25519 receipts: public verification never needs the evaluator's private key."""
import base64
import json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

DOMAIN = b"AutoRSE evaluation receipt v1\x00"


def canonical_payload(payload):
    if not isinstance(payload, dict) or "signature" in payload:
        raise ValueError("invalid_receipt_payload")
    return DOMAIN + json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")


def sign_receipt(payload, private_key_b64, key_id):
    key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64, validate=True))
    signature = key.sign(canonical_payload(payload))
    return {**payload, "signature": {"algorithm": "Ed25519", "key_id": key_id,
                                    "value": base64.b64encode(signature).decode("ascii")}}


def verify_receipt(receipt, trusted_keys):
    try:
        signature = receipt["signature"]
        if set(signature) != {"algorithm", "key_id", "value"} or signature["algorithm"] != "Ed25519":
            raise ValueError()
        keys = trusted_keys.get("keys", trusted_keys)
        entry = keys[signature["key_id"]]
        if entry["algorithm"] != "Ed25519" or entry.get("revoked", False):
            raise ValueError()
        payload = {key: value for key, value in receipt.items() if key != "signature"}
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(entry["public_key"], validate=True))
        key.verify(base64.b64decode(signature["value"], validate=True), canonical_payload(payload))
        # Key validity can be bounded when rotating or revoking a deployment key.
        stamp = payload.get("evaluated_at", "")
        from datetime import datetime
        actual = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if actual.utcoffset() is None:
            raise ValueError()
        for name, lower in (("not_before", True), ("not_after", False)):
            if entry.get(name):
                bound = datetime.fromisoformat(entry[name].replace("Z", "+00:00"))
                if (actual < bound if lower else actual > bound):
                    raise ValueError()
        return payload
    except Exception:
        raise ValueError("untrusted_or_modified_evaluation_receipt") from None

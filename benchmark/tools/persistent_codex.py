from harbor.agents.installed.codex import Codex
from tools.persistent_installed import PersistentInstalled


class PersistentCodex(PersistentInstalled, Codex):
    harness = 'codex'

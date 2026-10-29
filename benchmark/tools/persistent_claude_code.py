from harbor.agents.installed.claude_code import ClaudeCode
from tools.persistent_installed import PersistentInstalled


class PersistentClaudeCode(PersistentInstalled, ClaudeCode):
    harness = 'claude-code'

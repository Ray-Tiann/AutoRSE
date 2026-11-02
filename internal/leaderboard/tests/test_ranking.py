import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autorse_leaderboard.ranking import build_leaderboard, DATA_REVISION, TASK_ORDER


def summary(identifier, scores=None):
    return {"id": identifier, "participant": "researcher", "kind": "benchmark", "publishable": True,
            "data_revision": DATA_REVISION,
            "configuration": {"name": identifier, "agent": "agent", "harness": "Codex",
                              "agent_version": "1", "model": identifier, "model_version": "1",
                              "framework_commit": "a" * 40, "code_url": "https://github.com/example/agent"},
            "scores": list(scores or [.5] * 6), "sd": [.01] * 6, "trial_counts": [3] * 6,
            "source_url": "https://github.com/example/results/commit/" + "a" * 40}


class RankingTests(unittest.TestCase):
    def test_empty_board_is_honest_and_versioned(self):
        data = build_leaderboard([])
        self.assertEqual(data["rows"], [])
        self.assertEqual(data["task_order"], list(TASK_ORDER))
        self.assertEqual(data["metadata"]["data_revision"], DATA_REVISION)
        self.assertEqual(data["metadata"]["overall_cohort_count"], 0)
        self.assertTrue(all("baseline_score" not in task for task in data["tasks"]))

    def test_single_reviewed_configuration(self):
        row = build_leaderboard([summary("one")])["rows"][0]
        self.assertEqual(row["task_ranks"], [1] * 6)
        self.assertEqual((row["mean_task_rank"], row["rank_sum"], row["overall_rank"]), (1, 6, 1))
        self.assertNotIn("relative_gain_percent", row)
        self.assertEqual(row["provenance"], "Reviewed community")

    def test_midrank_at_four_decimal_precision(self):
        rows = build_leaderboard([summary("a", [.10004] * 6), summary("b", [.10003] * 6),
                                  summary("c", [.10005] * 6)])["rows"]
        indexed = {row["id"]: row for row in rows}
        self.assertEqual(indexed["a"]["task_ranks"], [2.5] * 6)
        self.assertEqual(indexed["b"]["task_ranks"], [2.5] * 6)
        self.assertEqual(indexed["c"]["task_ranks"], [1] * 6)
        self.assertEqual(indexed["a"]["scores"], [.10004] * 6)

    def test_overall_competition_ties_have_no_gain_tiebreaker(self):
        data = build_leaderboard([summary("a", [.9, .8, .9, .8, .9, .8]),
                                  summary("b", [.8, .9, .8, .9, .8, .9]), summary("c", [.7] * 6)])
        self.assertEqual([r["overall_rank"] for r in data["rows"]], [1, 1, 3])
        self.assertEqual([r["mean_task_rank"] for r in data["rows"]], [1.5, 1.5, 3])

    def test_input_order_does_not_change_published_rows(self):
        source = [summary("b", [.7] * 6), summary("a", [.7] * 6), summary("c", [.6] * 6)]
        self.assertEqual(build_leaderboard(source), build_leaderboard(list(reversed(source))))

    def test_input_summaries_are_not_mutated(self):
        source = [summary("one"), summary("two", [.6] * 6)]
        before = copy.deepcopy(source)
        build_leaderboard(source)
        self.assertEqual(source, before)

    def test_duplicate_ids_and_mixed_revisions_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_leaderboard([summary("a"), summary("a")])
        other = summary("b"); other["data_revision"] = "autorse-public-v3-20270101"
        with self.assertRaisesRegex(ValueError, "Mixed"):
            build_leaderboard([summary("a"), other])

    def test_demo_incomplete_and_extra_trials_are_ineligible(self):
        mutations = [("kind", "demo"), ("trial_counts", [3, 3, 3, 2, 3, 3]),
                     ("trial_counts", [4] * 6), ("scores", [.5] * 5), ("sd", [.1] * 5)]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                row = summary("a"); row[key] = value
                with self.assertRaises(ValueError):build_leaderboard([row])

    def test_nonfinite_boolean_or_negative_sd_fail(self):
        for field, value in (("scores", float("nan")), ("scores", float("inf")), ("scores", True),
                             ("sd", -.1), ("sd", False), ("trial_counts", True)):
            with self.subTest(field=field, value=value):
                row = summary("a"); row[field][0] = value
                with self.assertRaises(ValueError):build_leaderboard([row])

    def test_required_display_fields_and_safe_provenance_url(self):
        for field in ("id", "participant", "data_revision"):
            row = summary("a"); row[field] = ""
            with self.assertRaises(ValueError):build_leaderboard([row])
        for field in ("model", "harness"):
            row = summary("a"); row["configuration"][field] = ""
            with self.assertRaises(ValueError):build_leaderboard([row])
        row = summary("a"); row["source_url"] = "javascript:alert(1)"
        with self.assertRaises(ValueError):build_leaderboard([row])
        row["source_url"] = None
        self.assertEqual(build_leaderboard([row])["rows"][0]["source_url"], "")

    def test_unpublishable_summary_is_rejected(self):
        for flag in (False, None):
            row = summary("a"); row["publishable"] = flag
            with self.assertRaises(ValueError):build_leaderboard([row])
        row = summary("a"); row.pop("publishable")
        with self.assertRaises(ValueError):build_leaderboard([row])


class FrontendCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[2] / "hf-leaderboard/index.html"
        cls.html = cls.path.read_text()
        cls.paper = json.loads(re.search(r'<script type="application/json" id="seed-data">(.*?)</script>', cls.html, re.S).group(1))
        cls.empty = json.loads(re.search(r'<script type="application/json" id="community-seed-data">(.*?)</script>', cls.html, re.S).group(1))

    def test_historical_results_and_empty_community_remain_separate(self):
        self.assertEqual(self.empty["rows"], [])
        self.assertEqual(len(self.paper["rows"]), 7)
        self.assertEqual(self.paper["metadata"]["provenance"], "Paper reference")
        self.assertEqual(self.empty["metadata"]["data_revision"], DATA_REVISION)
        self.assertNotEqual(self.paper["metadata"]["provenance"], self.empty["metadata"]["provenance"])

    @unittest.skipUnless(shutil.which("node"), "Node is required for frontend core parity")
    def test_frontend_preserves_both_rankings_without_mixing_or_recomputing_filters(self):
        core = re.search(r'<script>(\(function\(root,factory\).*?)</script>', self.html, re.S).group(1)
        sources = [summary("a", [.10004] * 6), summary("b", [.10003] * 6), summary("c", [.10005] * 6)]
        sources[1]["configuration"]["harness"] = "Other harness"
        community = build_leaderboard(sources)
        payload = {"paper": self.paper, "empty": self.empty, "community": community}
        code = "const module={exports:{}};\n" + core + "\nconst api=module.exports,data=" + json.dumps(payload) + ";\n" + r'''
if(!api.validData(data.paper)||!api.validData(data.empty)||!api.validData(data.community))throw Error('valid cohort refused');
if(api.fixedFour(.14415)!=='0.1442'||api.fixedFour(.10004999999999999)!=='0.1000'||api.fixedFour(1e-8)!=='0.0000')throw Error('decimal display policy differs');
const historical=api.ranking(data.paper);
if(historical.length!==7||historical[0].row.id!=='claude-code-claude-opus-4-8'||historical[0].rankSum!==14)throw Error('historical results changed');
const ranks=api.ranking(data.community);
if(ranks.map(x=>x.row.id).join(',')!=='c,a,b'||ranks.map(x=>x.rank).join(',')!=='1,2,2')throw Error('community rank mismatch');
const task=api.ranking(data.community,'HSTU');
if(task.map(x=>x.rank).join(',')!=='1,2.5,2.5'||task[1].value!==.10004)throw Error('raw task scores or midranks changed');
const before=api.ranking(data.community).map(x=>[x.row.id,x.rank,x.meanRank]);
const filtered=api.ranking(data.community,'overall','Codex').map(x=>[x.row.id,x.rank,x.meanRank]);
if(JSON.stringify(before.filter(x=>x[0]!=='b'))!==JSON.stringify(filtered))throw Error('filter changed cohort');
const mixed={...data.community,rows:[...data.community.rows,...data.paper.rows]};
if(api.validData(mixed))throw Error('historical and community cohort accepted together');
if(api.ranking(data.empty).length!==0)throw Error('empty community gained fabricated rows');
console.log('frontend cohort parity passed');
'''
        result = subprocess.run([shutil.which("node"), "-e", code], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node is required for script syntax checks")
    def test_every_executable_script_parses(self):
        scripts = re.findall(r'<script>(.*?)</script>', self.html, re.S)
        for script in scripts:
            result = subprocess.run([shutil.which("node"), "-e", "new Function(" + json.dumps(script) + ");"], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":unittest.main()

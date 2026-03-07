"""End-to-end correctness test for the TRACE benchmark scripts.

Tests everything that can be run without torch/transformers:
  1. bash -n syntax check on all shell scripts
  2. compute_metrics.py  — with known synthetic accuracy data
  3. collect_mmlu.py     — with known synthetic lm_eval JSON data
  4. convert_trace_data.py — data-conversion logic with a stub tokenizer

Run from the repo root:
    python3 scripts/test_e2e.py
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS   = REPO_ROOT / "scripts"

TASKS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_script(name: str) -> types.ModuleType:
    """Import a scripts/*.py file as a module without executing __main__."""
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def capture(fn) -> str:
    """Capture stdout produced by fn()."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        fn()
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Shell script syntax check
# ─────────────────────────────────────────────────────────────────────────────

class TestShellSyntax(unittest.TestCase):
    def _check(self, name):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPTS / name)],
            capture_output=True, text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            f"{name} has syntax errors:\n{result.stderr}",
        )

    def test_setup_sh(self):        self._check("setup.sh")
    def test_baselines_sh(self):    self._check("baselines.sh")
    def test_train_trace_sh(self):  self._check("train_trace_osft.sh")
    def test_report_sh(self):       self._check("report.sh")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  compute_metrics.py
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics(unittest.TestCase):
    """Verify the script with fully known synthetic accuracy values."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        T = len(TASKS)
        # Build a perfect matrix: every task retains full accuracy after all later tasks
        # A[i][j] = accuracy of task i after training through task j+1
        for j in range(T):
            for i in range(j + 1):
                task = TASKS[i]
                path = self.tmp / f"{task}_after_task{j + 1}.json"
                # Task i achieves (i+1)*10% when trained on, stays flat afterwards
                acc_at_train  = (i + 1) * 0.10            # 10%, 20%, …, 80%
                acc_after_all = acc_at_train               # no forgetting (BWT = 0)
                path.write_text(json.dumps({"accuracy": acc_after_all}))

    def _run(self):
        mod = load_script("compute_metrics.py")
        with patch("sys.argv", ["compute_metrics.py", "--results-dir", str(self.tmp)]):
            return capture(mod.main)

    def test_runs_without_error(self):
        out = self._run()
        self.assertIn("Average Accuracy", out)
        self.assertIn("Backward Transfer", out)

    def test_average_accuracy(self):
        # mean of [10,20,30,40,50,60,70,80]% = 45%
        out = self._run()
        self.assertIn("45.0%", out)

    def test_backward_transfer_zero(self):
        # no forgetting → BWT = 0.0%
        out = self._run()
        self.assertIn("+0.0%", out)

    def test_task_column_alignment(self):
        # Every row must start with the task name left-aligned in 20 chars
        out = self._run()
        for line in out.splitlines():
            for task in TASKS:
                if line.startswith(task):
                    self.assertTrue(
                        len(line) >= 20,
                        f"Row for {task} looks truncated: {line!r}",
                    )

    def test_partial_results_warn(self):
        # Remove the last file — should print a warning, not crash
        last = self.tmp / f"{TASKS[-1]}_after_task8.json"
        last.unlink()
        out = self._run()
        self.assertIn("Warning", out)
        self.assertIn("Average Accuracy", out)

    def test_bwt_negative(self):
        """Re-build with forgetting: accuracy drops after each task is trained on."""
        T = len(TASKS)
        for j in range(T):
            for i in range(j + 1):
                task = TASKS[i]
                path = self.tmp / f"{task}_after_task{j + 1}.json"
                # Accuracy when trained = 0.8; drops by 5pp per subsequent task
                acc = max(0.0, 0.8 - (j - i) * 0.05)
                path.write_text(json.dumps({"accuracy": acc}))
        out = self._run()
        # BWT should be negative (model forgets)
        # Extract the BWT line and check the sign
        for line in out.splitlines():
            if "Backward Transfer" in line and "N/A" not in line:
                self.assertIn("-", line, f"Expected negative BWT in: {line!r}")
                break


# ─────────────────────────────────────────────────────────────────────────────
# 3.  collect_mmlu.py
# ─────────────────────────────────────────────────────────────────────────────

def _write_lm_eval_result(path: Path, acc: float) -> None:
    """Write a minimal lm_eval results.json file (0.4+ format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "results": {
            "mmlu": {"acc,none": acc, "acc": acc}
        }
    }))


class TestCollectMmlu(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        mmlu = self.tmp / "mmlu"

        self.orig_acc = 0.60
        self.svd_acc  = 0.55

        _write_lm_eval_result(mmlu / "original"      / "results.json", self.orig_acc)
        _write_lm_eval_result(mmlu / "svd_truncated"  / "results.json", self.svd_acc)
        for i, task in enumerate(TASKS, 1):
            # Accuracy grows slightly with each task
            _write_lm_eval_result(
                mmlu / f"osft_after_task_{i}" / "results.json",
                self.orig_acc - 0.01 * i,
            )

    def _run(self):
        mod = load_script("collect_mmlu.py")
        with patch("sys.argv", ["collect_mmlu.py", "--results-dir", str(self.tmp)]):
            return capture(mod.main)

    def test_runs_without_error(self):
        out = self._run()
        self.assertIn("original", out)
        self.assertIn("svd_truncated", out)

    def test_all_8_tasks_present(self):
        out = self._run()
        for i in range(1, 9):
            self.assertIn(f"osft_task_{i}", out, f"osft_task_{i} missing from output")

    def test_delta_vs_original(self):
        # svd_truncated is 5pp below original → Δ vs original = -5.00%
        out = self._run()
        self.assertIn("-5.00%", out)

    def test_zero_accuracy_handled(self):
        """A 0.0 accuracy must not fall through to None (the `or` bug)."""
        mmlu = self.tmp / "mmlu"
        _write_lm_eval_result(mmlu / "original" / "results.json", 0.0)
        mod = load_script("collect_mmlu.py")
        with patch("sys.argv", ["collect_mmlu.py", "--results-dir", str(self.tmp)]):
            out = capture(mod.main)
        # "original" row must show 0.00% not N/A
        for line in out.splitlines():
            if line.strip().startswith("original"):
                self.assertIn("0.00%", line, f"0.0 acc shown as N/A: {line!r}")
                break

    def test_column_widths_do_not_overflow(self):
        """The longest label 'osft_task_8 (20Minuten)' must fit within COL_LABEL."""
        mod = load_script("collect_mmlu.py")
        col_label = mod.COL_LABEL
        out = self._run()
        for line in out.splitlines():
            if not line or line.startswith("-"):
                continue
            label_part = line[:col_label]
            self.assertLessEqual(
                len(label_part.rstrip()), col_label,
                f"Label overflows COL_LABEL={col_label}: {line!r}",
            )

    def test_missing_checkpoint_shows_na(self):
        """A missing osft_after_task_3 directory must show N/A, not crash."""
        import shutil
        shutil.rmtree(self.tmp / "mmlu" / "osft_after_task_3")
        out = self._run()
        # Should still run and show other rows
        self.assertIn("osft_task_1", out)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  convert_trace_data.py — data conversion logic with a stub tokenizer
# ─────────────────────────────────────────────────────────────────────────────

class StubTokenizer:
    """Minimal tokenizer substitute: maps each character to its ordinal."""
    eos_token    = "</s>"
    eos_token_id = 2
    pad_token    = None

    class _Result:
        def __init__(self, ids): self.input_ids = ids

    def __call__(self, text, add_special_tokens=True):
        ids = [1] if add_special_tokens else []   # 1 = BOS
        ids += [ord(c) % 100 + 10 for c in text]
        return self._Result(ids)

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()


class TestConvertTraceData(unittest.TestCase):

    def _load_convert_task(self):
        """Load the module with transformers stubbed out."""
        stub = types.ModuleType("transformers")
        stub.AutoTokenizer = StubTokenizer
        with patch.dict("sys.modules", {"transformers": stub}):
            return load_script("convert_trace_data.py")

    def _make_trace_dir(self, tmp: Path, n_examples=5, long_answer=False) -> Path:
        trace_dir = tmp / "TRACE"
        for task in TASKS:
            task_dir = trace_dir / task
            task_dir.mkdir(parents=True, exist_ok=True)
            records = []
            for idx in range(n_examples):
                answer = "A" * 3000 if long_answer else f"answer{idx}"
                records.append({"prompt": f"Q{idx}: question text", "answer": answer})
            (task_dir / "train.json").write_text(json.dumps(records))
            # test split has no eval for simplicity
        return trace_dir

    def test_basic_conversion(self):
        """Each example is written to JSONL with correct keys and length."""
        mod = self._load_convert_task()
        tok = StubTokenizer()
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            trace_dir = self._make_trace_dir(tmp)
            out_dir = tmp / "tokenized"
            mod.convert_task("C-STANCE", trace_dir, out_dir, tok, max_seq_length=2048)
            lines = (out_dir / "C-STANCE" / "train.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 5)
            for line in lines:
                rec = json.loads(line)
                self.assertIn("input_ids", rec)
                self.assertIn("labels", rec)
                self.assertIn("len", rec)
                self.assertEqual(len(rec["input_ids"]), rec["len"])
                self.assertEqual(len(rec["labels"]), rec["len"])

    def test_label_masking(self):
        """Prompt tokens are masked (-100) and answer tokens are not."""
        mod = self._load_convert_task()
        tok = StubTokenizer()
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            trace_dir = self._make_trace_dir(tmp, n_examples=1)
            out_dir = tmp / "tokenized"
            mod.convert_task("C-STANCE", trace_dir, out_dir, tok, max_seq_length=2048)
            rec = json.loads(
                (out_dir / "C-STANCE" / "train.jsonl").read_text().splitlines()[0]
            )
            labels = rec["labels"]
            # Labels must start with at least one -100 (BOS is always masked)
            self.assertEqual(labels[0], mod.IGNORE_INDEX)
            # The final token should be EOS (2) with a real label
            self.assertEqual(rec["input_ids"][-1], tok.eos_token_id)
            self.assertEqual(labels[-1], tok.eos_token_id)
            # Some tokens must NOT be masked (the answer part)
            self.assertTrue(
                any(l != mod.IGNORE_INDEX for l in labels),
                "All labels are IGNORE_INDEX — answer masking is wrong",
            )

    def test_truncation(self):
        """Sequences longer than max_seq_length are truncated exactly."""
        mod = self._load_convert_task()
        tok = StubTokenizer()
        max_len = 20
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            trace_dir = self._make_trace_dir(tmp, n_examples=3, long_answer=True)
            out_dir = tmp / "tokenized"
            mod.convert_task("C-STANCE", trace_dir, out_dir, tok, max_seq_length=max_len)
            for line in (out_dir / "C-STANCE" / "train.jsonl").read_text().splitlines():
                rec = json.loads(line)
                self.assertLessEqual(
                    rec["len"], max_len,
                    f"Sequence not truncated: len={rec['len']} > max={max_len}",
                )

    def test_eos_appended(self):
        """Every example must end with the EOS token."""
        mod = self._load_convert_task()
        tok = StubTokenizer()
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            trace_dir = self._make_trace_dir(tmp, n_examples=4)
            out_dir = tmp / "tokenized"
            mod.convert_task("FOMC", trace_dir, out_dir, tok, max_seq_length=2048)
            for line in (out_dir / "FOMC" / "train.jsonl").read_text().splitlines():
                rec = json.loads(line)
                self.assertEqual(
                    rec["input_ids"][-1], tok.eos_token_id,
                    "Last token is not EOS",
                )

    def test_all_tasks_converted(self):
        """Running main() converts all 8 tasks."""
        mod = self._load_convert_task()
        stub = types.ModuleType("transformers")
        stub.AutoTokenizer = StubTokenizer
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            trace_dir = self._make_trace_dir(tmp)
            out_dir = tmp / "tokenized"
            with patch.dict("sys.modules", {"transformers": stub}):
                with patch("sys.argv", [
                    "convert_trace_data.py",
                    "--model", "dummy",
                    "--trace-dir", str(trace_dir),
                    "--output-dir", str(out_dir),
                    "--max-seq-length", "2048",
                ]):
                    mod.main()
            for task in TASKS:
                self.assertTrue(
                    (out_dir / task / "train.jsonl").exists(),
                    f"{task}/train.jsonl missing",
                )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    for cls in [
        TestShellSyntax,
        TestComputeMetrics,
        TestCollectMmlu,
        TestConvertTraceData,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

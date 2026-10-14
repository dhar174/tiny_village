"""Model cache setup uses isolated, owned directories and no model imports."""

import importlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch


CACHE_SETTINGS = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME")


class TestModelCache(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="tiny-village-cache-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for name in CACHE_SETTINGS:
            os.environ.pop(name, None)
        self.home = self.root / "profile"
        self.home.mkdir()
        self.home_patch = patch("pathlib.Path.home", return_value=self.home)
        self.home_patch.start()
        self.addCleanup(self.home_patch.stop)

    def configure(self):
        self.assertIsNotNone(importlib.util.find_spec("tiny_cache"), "Owned-cache helper is missing")
        return importlib.import_module("tiny_cache").configure_model_cache()

    def test_default_uses_user_huggingface_cache(self):
        expected = self.home / ".cache" / "huggingface"
        self.assertEqual(Path(self.configure()), expected / "hub")
        self.assertEqual(os.environ["HF_HOME"], str(expected))
        self.assertEqual(os.environ["HF_HUB_CACHE"], str(expected / "hub"))
        self.assertNotIn("TRANSFORMERS_CACHE", os.environ)

    def test_xdg_cache_home_is_supported(self):
        os.environ["XDG_CACHE_HOME"] = str(self.root / "xdg")
        self.assertEqual(Path(self.configure()), self.root / "xdg" / "huggingface" / "hub")

    def test_explicit_home_takes_precedence_over_xdg(self):
        os.environ["XDG_CACHE_HOME"] = str(self.root / "unused")
        os.environ["HF_HOME"] = str(self.root / "configured")
        self.assertEqual(Path(self.configure()), self.root / "configured" / "hub")
        self.assertFalse((self.root / "unused").exists())

    def test_distinct_hub_and_legacy_overrides_are_preserved(self):
        configured = {"HF_HOME": str(self.root / "hf"), "HF_HUB_CACHE": str(self.root / "hub"),
                      "TRANSFORMERS_CACHE": str(self.root / "legacy")}
        os.environ.update(configured)
        self.assertEqual(Path(self.configure()), self.root / "legacy")
        self.assertEqual({name: os.environ[name] for name in configured}, configured)
        self.assertTrue(all(Path(path).is_dir() for path in configured.values()))

    def test_hub_override_is_used_without_legacy_override(self):
        os.environ["HF_HUB_CACHE"] = str(self.root / "custom-hub")
        self.assertEqual(Path(self.configure()), self.root / "custom-hub")

    def test_repeated_setup_preserves_cache_contents(self):
        cache = Path(self.configure())
        sentinel = cache / "existing-model.bin"
        sentinel.write_bytes(b"existing model data")
        self.assertEqual(Path(self.configure()), cache)
        self.assertEqual(sentinel.read_bytes(), b"existing model data")

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_private_permissions_do_not_depend_on_umask(self):
        previous = os.umask(0)
        try:
            cache = Path(self.configure())
        finally:
            os.umask(previous)
        for path in (cache, cache.parent, cache.parent.parent):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700, str(path))
            self.assertEqual(path.stat().st_uid, os.geteuid())

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_existing_owned_directories_are_restricted_without_touching_files(self):
        os.environ["HF_HOME"] = str(self.root / "existing")
        home = Path(os.environ["HF_HOME"])
        home.mkdir(mode=0o777)
        home.chmod(0o777)
        sentinel = home / "keep"
        sentinel.write_bytes(b"keep")
        sentinel.chmod(0o640)
        self.configure()
        self.assertEqual(stat.S_IMODE(home.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(sentinel.stat().st_mode), 0o640)
        self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_non_directory_cache_is_rejected_without_environment_changes(self):
        path = self.root / "not-a-directory"
        path.write_text("keep")
        os.environ["HF_HUB_CACHE"] = str(path)
        before = dict(os.environ)
        with self.assertRaisesRegex(RuntimeError, "owned.*cache|cache.*owned"):
            self.configure()
        self.assertEqual(path.read_text(), "keep")
        self.assertEqual(dict(os.environ), before)

    @unittest.skipUnless(os.name == "posix", "POSIX symlink semantics")
    def test_symlink_cache_is_rejected_without_changing_target(self):
        target = self.root / "other-cache"
        target.mkdir()
        target.chmod(0o750)
        link = self.root / "linked-cache"
        link.symlink_to(target, target_is_directory=True)
        os.environ["HF_HOME"] = str(link)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.configure()
        self.assertTrue(link.is_symlink())
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o750)

    @unittest.skipUnless(os.name == "posix", "POSIX symlink semantics")
    def test_symlink_ancestor_is_rejected(self):
        target = self.root / "other-parent"
        target.mkdir()
        link = self.root / "linked-parent"
        link.symlink_to(target, target_is_directory=True)
        os.environ["HF_HOME"] = str(link / "cache")
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.configure()
        self.assertFalse((target / "cache").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX ownership")
    def test_foreign_ownership_is_rejected_before_chmod(self):
        self.assertIsNotNone(importlib.util.find_spec("tiny_cache"), "Owned-cache helper is missing")
        import tiny_cache
        home = self.root / "foreign"
        home.mkdir()
        os.environ["HF_HOME"] = str(home)
        real_geteuid = os.geteuid()
        with patch.object(tiny_cache.os, "geteuid", return_value=real_geteuid + 1):
            with patch.object(tiny_cache.os, "fchmod", side_effect=AssertionError("Must not chmod foreign cache")):
                with self.assertRaisesRegex(RuntimeError, "owned"):
                    self.configure()

    def test_filesystem_error_is_actionable_and_does_not_change_environment(self):
        before = dict(os.environ)
        with patch("os.mkdir", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(RuntimeError, "HF_HOME.*owned|owned.*HF_HOME"):
                self.configure()
        self.assertEqual(dict(os.environ), before)

    def test_security_configuration_is_preserved(self):
        settings = {"REQUESTS_CA_BUNDLE": str(self.root / "ca.pem"), "HF_HUB_DISABLE_SSL_VERIFICATION": "0"}
        os.environ.update(settings)
        self.configure()
        self.assertEqual({name: os.environ[name] for name in settings}, settings)

    def test_cache_configuration_does_not_import_models(self):
        os.environ["HF_HOME"] = str(self.root / "stdlib-only")
        result = subprocess.run(
            [sys.executable, "-c", """
import importlib.abc, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'transformers', 'torch', 'huggingface_hub', 'tiny_memories', 'tiny_brain_io'}:
            raise AssertionError('Cache setup imported a model library: ' + fullname)
sys.meta_path.insert(0, NoModels())
from tiny_cache import configure_model_cache
from pathlib import Path
assert Path(configure_model_cache()).is_dir()
"""], capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class TestModelCacheIntegration(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="tiny-village-cache-integration-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.environment = os.environ.copy()
        for name in CACHE_SETTINGS:
            self.environment.pop(name, None)
        self.environment.update(
            HF_HOME=str(self.root / "hf"), HF_HUB_CACHE=str(self.root / "hub"),
            TRANSFORMERS_CACHE=str(self.root / "transformers"), TMPDIR=str(self.root),
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MPLCONFIGDIR=str(self.root / "matplotlib"),
        )

    def run_python(self, source):
        result = subprocess.run(
            [sys.executable, "-c", "import faulthandler; faulthandler.dump_traceback_later(25, exit=True)\n" + textwrap.dedent(source)],
            cwd=Path(__file__).resolve().parents[1], env=self.environment,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-6000:])

    def test_import_orders_and_repeat_startup_preserve_owned_and_shared_data(self):
        shared = self.root / "hf_test_cache"
        shared.mkdir(mode=0o700)
        (shared / "unrelated").write_bytes(b"never modify shared caches")
        orders = [("tiny_memories", "tiny_brain_io"), ("tiny_brain_io", "tiny_memories")]
        for order in orders:
            with self.subTest(order=order):
                self.run_python(f"""
                    import importlib, importlib.abc, os, sys
                    from pathlib import Path
                    configured = {{name: os.environ[name] for name in ('HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE')}}
                    class OptionalModelsUnavailable(importlib.abc.MetaPathFinder):
                        def find_spec(self, fullname, path=None, target=None):
                            if fullname == 'transformers':
                                assert all(Path(value).is_dir() for value in configured.values()), 'Models imported before owned cache setup'
                            if fullname.split('.')[0] in {{'transformers', 'nltk', 'spacy'}}:
                                raise ImportError('Controlled unavailable optional dependency: ' + fullname)
                    sys.meta_path.insert(0, OptionalModelsUnavailable())
                    for name in {order!r}:
                        module = importlib.import_module(name)
                        assert module.cache_dir == configured['TRANSFORMERS_CACHE']
                    assert {{name: os.environ[name] for name in configured}} == configured
                    sentinel = Path(configured['TRANSFORMERS_CACHE']) / 'existing-model.bin'
                    if sentinel.exists():
                        assert sentinel.read_bytes() == b'reusable model data'
                    else:
                        sentinel.write_bytes(b'reusable model data')
                """)
        self.assertEqual((shared / "unrelated").read_bytes(), b"never modify shared caches")
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(shared.stat().st_mode), 0o700)
            for name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
                path = Path(self.environment[name])
                self.assertTrue(path.is_dir(), str(path))
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
                self.assertEqual(path.stat().st_uid, os.geteuid())

    def test_real_model_loaders_receive_selected_cache_and_preserve_verification(self):
        self.run_python("""
            import importlib.abc, os, sys
            from pathlib import Path
            expected = os.environ['TRANSFORMERS_CACHE']
            original_security = {name: os.environ.get(name) for name in
                ('REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'HF_HUB_DISABLE_SSL_VERIFICATION')}
            class OptionalModelsUnavailable(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'transformers', 'nltk', 'spacy'}:
                        raise ImportError('Controlled unavailable optional dependency: ' + fullname)
            sys.meta_path.insert(0, OptionalModelsUnavailable())
            import tiny_memories as memory
            import tiny_brain_io as brain
            calls = []
            class DownloadBoundary:
                @classmethod
                def from_pretrained(cls, name, **kwargs):
                    assert kwargs['cache_dir'] == expected, kwargs
                    assert kwargs.get('trust_remote_code') is True
                    assert set(kwargs) <= {'cache_dir', 'trust_remote_code', 'revision'}, kwargs
                    directory = Path(kwargs['cache_dir'])
                    assert directory.is_dir()
                    (directory / 'download-fixture.bin').write_bytes(b'isolated downloader fixture')
                    calls.append((name, kwargs))
                    return cls()
                def to(self, device): return self
                def eval(self): return self
                def zero_grad(self): return self
            memory.AutoModel = memory.AutoTokenizer = DownloadBoundary
            memory.TORCH_AVAILABLE = False
            embedding = memory.EmbeddingModel()
            assert embedding.is_initialized
            brain.AutoModelForCausalLM = brain.AutoTokenizer = DownloadBoundary
            brain.TRANSFORMERS_AVAILABLE = True
            brain.TORCH_AVAILABLE = False
            model = brain.TinyBrainIO('offline-fixture', {'offline-fixture': {'revision': 'fixed-test-rev'}})
            assert model.model is not None and model.tokenizer is not None and model.load_error is None
            assert len(calls) == 4
            assert calls[2][1]['revision'] == 'fixed-test-rev'
            assert (Path(expected) / 'download-fixture.bin').read_bytes() == b'isolated downloader fixture'
            assert {name: os.environ.get(name) for name in original_security} == original_security
        """)


if __name__ == "__main__":
    unittest.main()

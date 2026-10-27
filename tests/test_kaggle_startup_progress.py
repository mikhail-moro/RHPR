"""A Python-start marker must be emitted before filesystem/dependency setup."""
import ast
import contextlib
import io
import json
import unittest
from scripts.build_kaggle_bundle import _render_runner


class StartupProgressTests(unittest.TestCase):
    def test_python_start_is_visible_before_writes_or_heavy_imports(self):
        source=_render_runner(json.dumps({'name':'startup-fixture'}),'source-fixture')
        tree=ast.parse(source)
        end=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Expr)
                 and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                 and n.value.func.attr=='mkdir')
        prefix=ast.Module(body=tree.body[:end],type_ignores=[])
        output=io.StringIO()
        with contextlib.redirect_stdout(output):exec(compile(prefix,'startup-prefix','exec'),{})
        self.assertEqual(json.loads(output.getvalue()),dict(phase='kernel_python_started',
            git_sha='source-fixture',experiment='startup-fixture'))


if __name__=='__main__':unittest.main()

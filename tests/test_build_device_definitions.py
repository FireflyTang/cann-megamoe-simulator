import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/build_device.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location('build_device', SCRIPT)
build_device = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_device)


class BuildDeviceDefinitions(unittest.TestCase):
    def test_parse_and_bind_kernel_definitions(self):
        definitions = build_device.kernel_definitions(['MEGAMOE_RESEARCH_DISPATCH_WAVES=12',
                                                       'MEGAMOE_RESEARCH_COMBINE_AIV0_RELAY=1'])
        self.assertEqual(definitions, {'MEGAMOE_RESEARCH_DISPATCH_WAVES': 12,
                                      'MEGAMOE_RESEARCH_COMBINE_AIV0_RELAY': 1})
        with tempfile.TemporaryDirectory() as name:
            recipe = Path(name) / 'MegaMoe.py'
            recipe.write_text("custom_compile_options = {'__ALLSOC__': ['-UENABLE_PROFILING']},\n")
            additions = build_device.bind_kernel_definitions(recipe, definitions)
        self.assertEqual(additions, ['-DMEGAMOE_RESEARCH_DISPATCH_WAVES=12',
                                     '-DMEGAMOE_RESEARCH_COMBINE_AIV0_RELAY=1'])

    def test_reject_invalid_and_tool_owned_definitions(self):
        for values in (['lower_case=1'], ['MEGAMOE_X=12j'], ['MEGAMOE_X=1', 'MEGAMOE_X=2'],
                       ['ENABLE_PROFILING=1'], ['TILINGKEY_COMM_MODE=0']):
            with self.assertRaises(ValueError):
                build_device.kernel_definitions(values)


if __name__ == '__main__':
    unittest.main()

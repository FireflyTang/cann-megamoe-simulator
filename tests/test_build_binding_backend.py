import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location("build_binding", SCRIPTS / "build_binding.py")
build_binding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_binding)


PROFILER_CONFIG = """#pragma once
#define PROF_N_CORES 84
#define PROF_AIV_CORES 56
#define PROF_SLOTS_PER_CORE 16384
#define PROF_RECORD_BYTES 16
#define PROF_RINGIDX_ALIGN 64
"""

EVENTS = """#pragma once
enum ProfEventId : uint32_t {
    KERNEL_BEGIN = 0x0001,
    KERNEL_END = 0x00ff,
};
"""

WRITER = """#pragma once
static_assert(PROF_N_CORES == MEGAMOE_COMPILED_PROF_N_CORES, "abi");
static_assert(PROF_AIV_CORES == MEGAMOE_COMPILED_PROF_AIV_CORES, "abi");
static_assert(PROF_SLOTS_PER_CORE == MEGAMOE_COMPILED_PROF_SLOTS_PER_CORE, "abi");
static_assert(PROF_RECORD_BYTES == MEGAMOE_COMPILED_PROF_RECORD_BYTES, "abi");
static_assert(PROF_RINGIDX_ALIGN == MEGAMOE_COMPILED_PROF_RINGIDX_ALIGN, "abi");
"""


class BuildBindingBackend(unittest.TestCase):
    def _run(self, mode):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            generated = root / "generated"
            generated.mkdir()
            (generated / "profiler_config.h").write_text(PROFILER_CONFIG)
            (generated / "profile_events_generated.h").write_text(EVENTS)
            (generated / "pair_map.json").write_text("{}\n")
            attempt = root / "attempt"
            attempt.mkdir()
            writer = root / "writer.h"
            writer.write_text(WRITER)
            recipe = root / "recipe.py"
            recipe.write_text(
                "custom_compile_options = {'__ALLSOC__': ["
                f"'-DENABLE_PROFILING=1', '-DMEGAMOE_PROFILE_DETAIL=1', "
                f"'-I{generated}', '-I{writer.parent}']}},\n"
            )
            token = build_binding.begin(recipe, generated, attempt, mode, "w8a8", writer)
            return token

    def test_sim_build_selects_sim_backend(self):
        token = self._run("sim")
        self.assertIn("-DMEGAMOE_PROFILE_BACKEND=1", token["compile_options"])
        self.assertEqual(token["profile_backend"], 1)

    def test_board_build_selects_board_backend(self):
        token = self._run("full")
        self.assertIn("-DMEGAMOE_PROFILE_BACKEND=0", token["compile_options"])
        self.assertEqual(token["profile_backend"], 0)


if __name__ == "__main__":
    unittest.main()

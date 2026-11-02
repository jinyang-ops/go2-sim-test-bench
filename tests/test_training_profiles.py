import copy
import subprocess
import sys
import unittest
from types import SimpleNamespace

from robot_test.common import configure_training_randomization


def config():
    return SimpleNamespace(noise=SimpleNamespace(), domain_rand=SimpleNamespace(),
                           control=SimpleNamespace(stiffness=50, damping=2),
                           reward=SimpleNamespace(torque_scale=-0.0002))


class TrainingProfileTests(unittest.TestCase):
    def test_mass_only_changes_mass_fields_relative_to_nominal(self):
        nominal, mass = config(), config()
        configure_training_randomization(nominal, "nominal")
        configure_training_randomization(mass, "mass_only")
        fields = vars(nominal.domain_rand)
        changed = {key for key in fields if getattr(nominal.domain_rand, key) != getattr(mass.domain_rand, key)}
        self.assertEqual(changed, {"randomize_base_mass", "added_mass_range"})
        self.assertEqual(vars(nominal.noise), vars(mass.noise))
        self.assertEqual(vars(nominal.control), vars(mass.control))
        self.assertEqual(vars(nominal.reward), vars(mass.reward))
        self.assertEqual(mass.domain_rand.added_mass_range, [0.0, 2.0])

    def test_switching_from_full_to_mass_clears_unrelated_perturbations(self):
        reused = config()
        configure_training_randomization(reused, "full")
        configure_training_randomization(reused, "mass_only")
        fresh = config()
        configure_training_randomization(fresh, "mass_only")
        self.assertEqual(vars(reused.domain_rand), vars(fresh.domain_rand))
        self.assertEqual(vars(reused.noise), vars(fresh.noise))

    def test_invalid_profile_is_rejected_without_mutation(self):
        cfg = config()
        before = copy.deepcopy(cfg)
        with self.assertRaises(ValueError):
            configure_training_randomization(cfg, "invalid")
        self.assertEqual(cfg, before)

    def test_cli_rejects_combining_full_and_mass_profiles_before_gpu_import(self):
        result = subprocess.run([sys.executable, "-m", "robot_test.train", "--qa-mass-rand", "--qa-domain-rand"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("not allowed with argument", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()

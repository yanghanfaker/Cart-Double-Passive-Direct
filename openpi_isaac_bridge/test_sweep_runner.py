import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_gain_sweep


class SweepRoutingTests(unittest.TestCase):
    def test_long_approach_routes_explicit_flag_and_unique_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);commands=[]
            class Process:
                def __init__(self,command,**kwargs):
                    commands.append(command);output=Path(command[command.index("--output")+1])
                    report={"joint_telemetry_summary":{},"actuator_audit":{},"physics_joint_summary":{},
                            "openpi_result":{"probe_completed":False,"termination_reason":"near_object_boundary"},"openpi_setup":{}}
                    (output/"motion_openpi-step.json").write_text(json.dumps(report));self.stdout=io.StringIO("")
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def wait(self):return 0
            args=["runner","--isaaclab-root",str(root),"--profiles","kp2p25-kd0p25","--modes","openpi-step",
                  "--openpi-approach","--openpi-approach-long","--openpi-replan-steps","5","--openpi-steps","400"]
            with patch.object(sys,"argv",args),patch.object(run_gain_sweep.subprocess,"Popen",Process):
                self.assertEqual(run_gain_sweep.main(),0)
            self.assertIn("--openpi-approach-long",commands[0])
            self.assertEqual(commands[0][commands[0].index("--openpi-steps")+1],"400")
            manifest=json.loads((root/"logs/openpi_gain_sweep/latest_gain_sweep.json").read_text())
            self.assertTrue(manifest["profiles"][0].endswith("-approach-long"))
            self.assertTrue(manifest["openpi_approach_long"])

    def test_segmented_approach_routes_60_steps_and_chunk5(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);commands=[]
            class Process:
                def __init__(self,command,**kwargs):
                    commands.append(command);output=Path(command[command.index("--output")+1])
                    report={"joint_telemetry_summary":{},"actuator_audit":{},"physics_joint_summary":{},
                            "openpi_result":{"probe_completed":False,"completed_steps":20,"termination_reason":"segment_progress_less_than_2mm"},"openpi_setup":{}}
                    (output/"motion_openpi-step.json").write_text(json.dumps(report));self.stdout=io.StringIO("")
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def wait(self):return 0
            args=["runner","--isaaclab-root",str(root),"--profiles","kp2p25-kd0p25","--modes","openpi-step",
                  "--openpi-approach","--openpi-replan-steps","5","--openpi-steps","60"]
            with patch.object(sys,"argv",args),patch.object(run_gain_sweep.subprocess,"Popen",Process):
                self.assertEqual(run_gain_sweep.main(),0)
            self.assertEqual(len(commands),1)
            self.assertIn("--openpi-approach",commands[0])
            self.assertEqual(commands[0][commands[0].index("--openpi-steps")+1],"60")
            manifest=json.loads((root/"logs/openpi_gain_sweep/latest_gain_sweep.json").read_text())
            self.assertEqual(manifest["runs"][0]["openpi_result"]["termination_reason"],"segment_progress_less_than_2mm")

    def test_chunk_comparison_routes_two_runs_and_preserves_server_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=root/"server.log";log.write_text("loaded checkpoint fixture")
            commands=[]
            class Process:
                def __init__(self,command,**kwargs):
                    commands.append(command)
                    output=Path(command[command.index("--output")+1])
                    report={"joint_telemetry_summary":{},"actuator_audit":{},"physics_joint_summary":{},
                            "openpi_result":{"probe_completed":True,"preview_images":[]},"openpi_setup":{}}
                    (output/"motion_openpi-step.json").write_text(json.dumps(report))
                    self.stdout=io.StringIO("")
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def wait(self):return 0
            argv=["runner","--isaaclab-root",str(root),"--profiles","kp2p25-kd0p25",
                  "--modes","openpi-step","--openpi-replan-steps","1","5","--policy-server-log",str(log)]
            with patch.object(sys,"argv",argv),patch.object(run_gain_sweep.subprocess,"Popen",Process):
                self.assertEqual(run_gain_sweep.main(),0)
            self.assertEqual([c[c.index("--openpi-replan-steps")+1] for c in commands],["1","5"])
            manifest=json.loads((root/"logs/openpi_gain_sweep/latest_gain_sweep.json").read_text())
            self.assertEqual(len(manifest["profiles"]),2)
            for run in manifest["runs"]:
                self.assertEqual((Path(run["output"])/"policy_server_startup.log").read_bytes(),log.read_bytes())
                self.assertFalse(run["policy_server_log"]["truncated"])

    def test_server_log_capture_has_explicit_size_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"server.log";source.write_bytes(b"x"*(128*1024+20))
            result=run_gain_sweep.capture_policy_server_log(source,root)
            self.assertTrue(result["truncated"])
            self.assertEqual(result["captured_bytes"],128*1024)
            self.assertEqual((root/"policy_server_startup.log").stat().st_size,128*1024)

    def test_chunk_case_expansion_rejects_duplicate_lengths(self):
        case={"name":"candidate","profile":"kp2p25-kd0p25","joint_overrides":{}}
        with self.assertRaises(ValueError):run_gain_sweep.expand_openpi_replan_cases([case],[1,1])
        expanded=run_gain_sweep.expand_openpi_replan_cases([case],[1,5])
        self.assertEqual([c["name"] for c in expanded],["candidate-chunk1","candidate-chunk5"])
        self.assertEqual(case["name"],"candidate")

    def test_openpi_mode_loads_client_extra_and_preserves_failed_probe_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            commands=[]
            class FakeProcess:
                def __init__(self,command,**kwargs):
                    commands.append(command)
                    output=Path(command[command.index("--output")+1])
                    report={"joint_telemetry_summary":{},"actuator_audit":{},"physics_joint_summary":{},
                            "openpi_result":{"probe_completed":False,"completed_steps":0,"failure_reason":"server unavailable","preview_images":[]},
                            "openpi_setup":{"state_frame":"world"}}
                    (output/"motion_openpi-step.json").write_text(json.dumps(report))
                    self.stdout=io.StringIO("")
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def wait(self):return 0
            argv=["run_gain_sweep.py","--isaaclab-root",str(root),"--profiles","kp2p25-kd0p25",
                  "--modes","openpi-step","--openpi-steps","2","--server-port","8001"]
            with patch.object(sys,"argv",argv),patch.object(run_gain_sweep.subprocess,"Popen",FakeProcess):
                self.assertEqual(run_gain_sweep.main(),0)
            self.assertEqual(commands[0][:6],["uv","run","--extra","viser","--extra","openpi"])
            self.assertIn("--record-physics",commands[0])
            self.assertEqual(commands[0][commands[0].index("--openpi-steps")+1],"2")
            self.assertEqual(commands[0][commands[0].index("--server-port")+1],"8001")
            manifest=json.loads((root/"logs/openpi_gain_sweep/latest_gain_sweep.json").read_text())
            self.assertEqual(manifest["status"],"completed")
            self.assertFalse(manifest["runs"][0]["openpi_result"]["probe_completed"])

    def test_failed_grasp_metric_is_exportable_completed_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenarios = [{"name":"home-fast","object_offset_xy_m":[0,0],"object_yaw_deg":0,"seed":0,"orient_step_deg":.5},
                         {"name":"left-slow","object_offset_xy_m":[-.02,.02],"object_yaw_deg":20,"seed":1,"orient_step_deg":.25}]
            scenario_file = root / "grasp-scenarios.json"
            scenario_file.write_text(json.dumps(scenarios))
            routed = []
            class FakeProcess:
                def __init__(self, command, **kwargs):
                    output = Path(command[command.index("--output")+1])
                    index=command.index("--grasp-object-offset")
                    routed.append((list(map(float,command[index+1:index+3])),
                        float(command[command.index("--grasp-object-yaw")+1]),
                        float(command[command.index("--grasp-orient-step-deg")+1]),
                        int(command[command.index("--grasp-seed")+1])))
                    report = {"joint_telemetry_summary": {}, "actuator_audit": {}, "physics_joint_summary": {},
                              "grasp_result": {"success": False, "failure_reason": "empty close", "trace": [{"phase":"close"}]},
                              "grasp_setup": {"ee_sensor_offset_m": [0,0,0.1034]}}
                    (output / "motion_grasp.json").write_text(json.dumps(report))
                    if "--record-physics" not in command:
                        raise AssertionError("grasp must force-enable physics tracing")
                    self.stdout = io.StringIO("")
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def wait(self): return 0
            argv = ["run_gain_sweep.py", "--isaaclab-root", str(root), "--profiles", "kp2p25-kd0p25", "--modes", "grasp",
                    "--grasp-scenarios-json", str(scenario_file)]
            with patch.object(sys, "argv", argv), patch.object(run_gain_sweep.subprocess, "Popen", FakeProcess):
                self.assertEqual(run_gain_sweep.main(), 0)
            manifest = json.loads((root / "logs/openpi_gain_sweep/latest_gain_sweep.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(len(manifest["runs"]),2)
            self.assertEqual(routed,[([0,0],0,.5,0), ([-.02,.02],20,.25,1)])
            self.assertEqual(manifest["runs"][1]["grasp_scenario"],scenarios[1])
            self.assertFalse(manifest["runs"][0]["grasp_result"]["success"])
            self.assertEqual(manifest["runs"][0]["grasp_result"]["trace"], [{"phase":"close"}])

    def test_cartesian_run_retains_trace_and_checks_all_six_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario_file = root / "scenarios.json"
            scenario_file.write_text(json.dumps([
                {"name": "home", "start_offset_m": [0,0,0], "amplitude_scale": 1},
                {"name": "offset", "start_offset_m": [0,0.015,0.01], "amplitude_scale": 2}]))
            routed = []
            axes = [{"axis": a, "passed": True, "outbound_gain": 0.95,
                     "trace": [{"position_error_m": 0.00001, "ee_pos_root_m": [0.4, 0, 0.3]}]}
                    for a in ("x", "y", "z", "rx", "ry", "rz")]
            class FakeProcess:
                def __init__(self, command, **kwargs):
                    output = Path(command[command.index("--output")+1])
                    index = command.index("--stream-start-offset")
                    offset = list(map(float, command[index+1:index+4]))
                    scale = float(command[command.index("--stream-amplitude-scale")+1])
                    routed.append((offset, scale))
                    report = {"joint_telemetry_summary": {}, "actuator_audit": {},
                              "all_stream_tests_passed": True, "axes": axes,
                              "thresholds": {"gain_range": [0.7, 1.3]},
                              "target_semantics": "current measured pose relative",
                              "stream_amplitude_scale": scale,
                              "stream_start": {"offset_root_m": offset, "settled_last_5_steps": True}}
                    (output / "motion_libero-stream.json").write_text(json.dumps(report))
                    self.stdout = io.StringIO("")
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def wait(self): return 0
            argv = ["run_gain_sweep.py", "--isaaclab-root", str(root), "--profiles", "baseline",
                    "--modes", "libero-stream", "--stream-scenarios-json", str(scenario_file)]
            with patch.object(sys, "argv", argv), patch.object(run_gain_sweep.subprocess, "Popen", FakeProcess):
                self.assertEqual(run_gain_sweep.main(), 0)
            manifest = json.loads((root / "logs/openpi_gain_sweep/latest_gain_sweep.json").read_text())
            run = manifest["runs"][0]
            self.assertEqual(len(manifest["runs"]), 2)
            self.assertEqual(routed, [([0,0,0],1), ([0,0.015,0.01],2)])
            self.assertEqual(manifest["runs"][1]["stream_condition"]["stream_amplitude_scale"], 2)
            self.assertTrue(manifest["runs"][1]["stream_condition"]["stream_start"]["settled_last_5_steps"])
            self.assertEqual(run["axes"], axes)
            self.assertEqual(run["stream_thresholds"]["gain_range"], [0.7, 1.3])
            self.assertEqual(run["cartesian_stream_screen"]["outbound_gain_range"], [0.9, 1.1])
            self.assertTrue(run["cartesian_stream_screen"]["all_passed"])

    def test_named_cases_route_distinct_overrides_and_export_without_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "cases.json"
            cases = [{"name": f"case-{kd}", "profile": "kp2p25-kd0p25",
                      "joint_overrides": {"panda_joint2": {"damping": kd}}} for kd in [30, 40]]
            config.write_text(json.dumps(cases))
            launched = []
            class FakeProcess:
                def __init__(self, command, **kwargs):
                    output = Path(command[command.index("--output")+1])
                    overrides = json.loads(Path(command[command.index("--joint-gains-json")+1]).read_text())
                    launched.append((command, overrides))
                    report = {"joint_telemetry_summary": {}, "actuator_audit": overrides,
                              "physics_joint_summary": {}, "joint_steps": []}
                    (output / "motion_joint-step.json").write_text(json.dumps(report))
                    self.stdout = io.StringIO("")
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def wait(self): return 0
            argv = ["run_gain_sweep.py", "--isaaclab-root", str(root), "--gain-cases-json", str(config),
                    "--modes", "joint-step"]
            with patch.object(sys, "argv", argv), patch.object(run_gain_sweep.subprocess, "Popen", FakeProcess):
                self.assertEqual(run_gain_sweep.main(), 0)
            manifest = json.loads((root / "logs/openpi_gain_sweep/latest_gain_sweep.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["profiles"], ["case-30", "case-40"])
            self.assertEqual([o["panda_joint2"]["damping"] for _, o in launched], [30, 40])
            self.assertEqual(len({r["output"] for r in manifest["runs"]}), 2)
            for command, overrides in launched:
                self.assertEqual(command[command.index("--gain-profile")+1], "kp2p25-kd0p25")
                self.assertIn("--record-physics", command)
            self.assertEqual([r["actuator_audit"] for r in manifest["runs"]], [o for _, o in launched])


if __name__ == "__main__":
    unittest.main()

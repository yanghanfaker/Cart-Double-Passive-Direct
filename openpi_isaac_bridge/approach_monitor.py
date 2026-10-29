"""Diagnostic-only progress checks; never produce or alter policy actions."""
import math


def position(value):
    if len(value)!=3 or not all(math.isfinite(v) for v in value):
        raise ValueError("expected finite XYZ position")
    return list(value)


class ApproachMonitor:
    def __init__(self, tcp_world, object_world):
        self.start_tcp=position(tcp_world)
        self.start_object=position(object_world)
        self.initial_distance=math.dist(self.start_tcp,self.start_object)
        self.last_distance=self.segment_start_distance=self.initial_distance
        self.count=0
        self.segments=[]
        self.stop_reason="near_object_boundary" if self.initial_distance<=.08 else None

    def update(self, row):
        if row["step"]!=self.count+1:
            raise ValueError("approach samples must be consecutive executed steps")
        tcp=position(row["state_after_world"][:3])
        obj=position(row["object_pos_world_m_for_diagnostic_only"])
        self.count+=1
        self.last_distance=math.dist(tcp,obj)
        if self.last_distance<=.08:
            self.stop_reason="near_object_boundary"
        elif self.last_distance-self.initial_distance>.005:
            self.stop_reason="distance_increased_more_than_5mm"
        checkpoint=None
        if self.count%20==0:
            progress=self.segment_start_distance-self.last_distance
            checkpoint={"end_step":self.count,"start_distance_m":self.segment_start_distance,
                        "end_distance_m":self.last_distance,"distance_reduction_m":progress,
                        "tcp_world_m":tcp,"net_tcp_displacement_m":[b-a for a,b in zip(self.start_tcp,tcp)]}
            self.segments.append(checkpoint)
            self.segment_start_distance=self.last_distance
            if progress<.002 and self.stop_reason is None:
                self.stop_reason="segment_progress_less_than_2mm"
        return {"distance_to_object_m":self.last_distance,"checkpoint":checkpoint,"stop_reason":self.stop_reason}

    def summary(self):
        return {"initial_distance_m":self.initial_distance,"last_distance_m":self.last_distance,
                "executed_steps":self.count,"segments":self.segments,"stop_reason":self.stop_reason,
                "thresholds":{"segment_steps":20,"min_segment_progress_m":.002,
                              "max_distance_increase_from_initial_m":.005,"near_object_distance_m":.08},
                "scope":"simulator truth for diagnostic stop checks only; no target generation; no grasp success claim"}

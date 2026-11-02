"""Capture a physics sample before upstream reset_idx overwrites root_states."""


class SnapshotMixin:
    def check_termination(self):
        super().check_termination()
        forces = self.contact_forces[:, self.termination_contact_indices, :]
        self.qa_snapshot = {
            "state": self.root_states.detach().clone(),
            "velocity": self.base_lin_vel.detach().clone(),
            "command": self.commands[:, :3].detach().clone(),
            "torques": self.torques.detach().clone(),
            "contact": (forces.norm(dim=-1) > 1.0).any(dim=1).detach().clone(),
            "done": self.reset_buf.detach().clone(),
            "timeout": self.time_out_buf.detach().clone(),
        }

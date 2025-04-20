import os
import time
import scipy
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation


def transform_matrix_to_7d(transform):
    # transform is a numpy 4x4 homogenous transformation
    # Extract translation (last column of the matrix)
    translation = transform[:3, 3]
    # Extract rotation matrix (top-left 3x3 submatrix)
    rotation_matrix = transform[:3, :3]
    # Convert rotation matrix to quaternion (x, y, z, w order)
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    # Convert quaternion to (w, x, y, z order)
    quaternion = quaternion[[3, 0, 1, 2]]
    # Combine quaternion and translation into a 7-dimensional array
    result = np.concatenate((translation, quaternion))
    return result


class FrankaKinematics:
    def __init__(self):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_model_path = os.path.join(current_file_dir, "urdf/franka_no_ee_pin.urdf")
        self.pin_model, _, _ = pin.buildModelsFromUrdf(urdf_model_path)
        self.pin_data = self.pin_model.createData()
        self.ee_link_id = self.pin_model.getFrameId("franka_ee")
        self.ik_joint_bounds = scipy.optimize.Bounds(self.pin_model.lowerPositionLimit, self.pin_model.upperPositionLimit)
    
    def forward_kinematics(self, joint_position):
        # joint_position is a numpy array of shape (num_joints,)
        pin.framesForwardKinematics(self.pin_model, self.pin_data, joint_position)       
        # returns a numpy array of shape (7,) representing a pose (wxyz order) 
        return transform_matrix_to_7d(self.pin_data.oMf[self.ee_link_id].np)
    
    def inverse_kinematics(self, ee_pose, init_joint_pos, loss_tol=1e-3, max_iter=3, verbose=True):
        # Convert desired pose into proper format
        ee_pose = np.concatenate([ee_pose[:3], ee_pose[4:], ee_pose[3:4]]) # xyzw
        x_ee_des = pin.XYZQUATToSE3(ee_pose)

        def ik_loss_function(q) -> float:
            # update frame information
            pin.framesForwardKinematics(self.pin_model, self.pin_data, np.asarray(q))
            # computes the relative transformation between desired and current ee pose
            dMf = x_ee_des.actInv(self.pin_data.oMf[self.ee_link_id])
            # 6D twist representing the relative transformation
            err = pin.log(dMf).vector
            # scalar loss
            return np.linalg.norm(err)**2

        iteration = 0
        loss, best_loss = np.inf, np.inf
        best_joint_pos, optim_success = None, False
        ik_optim_options = {'ftol': 1e-9, 'disp': False, 'eps': 1e-8, 'maxiter': 10000}

        while iteration < max_iter and loss > loss_tol:
            if iteration == 1:
                # if the first iteration fails, try starting optimization with Franka's home configuration
                init_joint_pos = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
            elif iteration > 1:
                # if the second iteration fails, try starting optimization with random joint configuration
                init_joint_pos = np.random.uniform(self.ik_joint_bounds.lb, self.ik_joint_bounds.ub)

            # start optimization
            optim_result = scipy.optimize.minimize(
                ik_loss_function, init_joint_pos, method="SLSQP", tol=1e-6,
                options=ik_optim_options,
                bounds=self.ik_joint_bounds,
            )
            loss = optim_result.fun
            if loss < best_loss:
                best_joint_pos = optim_result.x
                best_loss = loss
                optim_success = optim_result.success
            if verbose:
                print(f"Target pose: {ee_pose} | Iteration: {iteration} | Loss: {loss}")
            iteration += 1

        return best_joint_pos, (best_loss < loss_tol) and optim_success, best_loss



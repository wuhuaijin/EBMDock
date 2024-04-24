import os
import functools
import torch


def G_fn(protein_coords, x, sigma):
    # protein_coords: (n,3) ,  x: (m,3), output: (m,)
    e = torch.exp(- torch.sum((protein_coords.view(1, -1, 3) - x.view(-1,1,3)) ** 2, dim=2) / float(sigma) )  # (m, n)
    return - sigma * torch.log(1e-3 +  e.sum(dim=1) )


def compute_body_intersection_loss(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma = 25., surface_ct=10.):
    assert model_ligand_coors_deform.shape[1] == 3
    loss = torch.mean( torch.clamp(surface_ct - G_fn(bound_receptor_repres_nodes_loc_array, model_ligand_coors_deform, sigma), min=0) ) + \
           torch.mean( torch.clamp(surface_ct - G_fn(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma), min=0) )
    return loss


class Transformation(torch.nn.Module):
    '''
    euler_angles_to_matrix: default by 'ZYX' extrinsic rotation
    '''
    def __init__(self, args):
        self.args = args
        super(Transformation, self).__init__()

    def _axis_angle_rotation(self, axis: str, angle):
        """
        Return the rotation matrices for one of the rotations about an axis
        of which Euler angles describe, for each value of the angle given.

        Args:
            axis: Axis label "X" or "Y or "Z".
            angle: any shape tensor of Euler angles in radians

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)

        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        if axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        if axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        
        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

    def euler_angles_to_matrix(self, euler_angles, convention: str):
        """
        Convert rotations given as Euler angles in radians to rotation matrices.

        Args:
            euler_angles: Euler angles in radians as tensor of shape (..., 3).
            convention: Convention string of three uppercase letters from
                {"X", "Y", and "Z"}.

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
            raise ValueError("Invalid input euler angles.")
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        matrices = map(self._axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
        return functools.reduce(torch.matmul, matrices)

    def _angle_from_tan(
        self, axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
    ) -> torch.Tensor:
        """
        Extract the first or third Euler angle from the two members of
        the matrix which are positive constant times its sine and cosine.

        Args:
            axis: Axis label "X" or "Y or "Z" for the angle we are finding.
            other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
                convention.
            data: Rotation matrices as tensor of shape (..., 3, 3).
            horizontal: Whether we are looking for the angle for the third axis,
                which means the relevant entries are in the same row of the
                rotation matrix. If not, they are in the same column.
            tait_bryan: Whether the first and third axes in the convention differ.

        Returns:
            Euler Angles in radians for each matrix in data as a tensor
            of shape (...).
        """

        i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
        if horizontal:
            i2, i1 = i1, i2
        even = (axis + other_axis) in ["XY", "YZ", "ZX"]
        if horizontal == even:
            return torch.atan2(data[..., i1], data[..., i2])
        if tait_bryan:
            return torch.atan2(-data[..., i2], data[..., i1])
        return torch.atan2(data[..., i2], -data[..., i1])


    def _index_from_letter(self, letter: str) -> int:
        if letter == "X":
            return 0
        if letter == "Y":
            return 1
        if letter == "Z":
            return 2
        raise ValueError("letter must be either X, Y or Z.")


    def matrix_to_euler_angles(self, matrix: torch.Tensor, convention: str) -> torch.Tensor:
        """
        Convert rotations given as rotation matrices to Euler angles in radians.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).
            convention: Convention string of three uppercase letters.

        Returns:
            Euler angles in radians as tensor of shape (..., 3).
        """
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        i0 = self._index_from_letter(convention[0])
        i2 = self._index_from_letter(convention[2])
        tait_bryan = i0 != i2
        if tait_bryan:
            central_angle = torch.asin(
                matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
            )
        else:
            central_angle = torch.acos(matrix[..., i0, i0])

        o = (
            self._angle_from_tan(
                convention[0], convention[1], matrix[..., i2], False, tait_bryan
            ),
            central_angle,
            self._angle_from_tan(
                convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
            ),
        )
        return torch.stack(o, -1)

    def forward(self, pos, R, t):

        pos_new = []        
        for i in range(len(pos)):
            pos[i] = pos[i].to(self.args['device']) 
            T_align = self.euler_angles_to_matrix(R[i], 'ZYX')
            pos_new.append(( T_align @ pos[i].t() ).t() + t[i])
        
        return pos_new

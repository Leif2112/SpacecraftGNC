import numpy as np
from gnc.attitude.rigid_body import (
    axis_angle_to_quaternion,
    quaternion_to_dcm,
    dcm_to_euler_angles,
    angular_momentum,
    rotational_kinetic_energy,
    attitude_rhs,
)

def test_axis_angle_to_quaternion():
    axis = np.array([1, 0, 0])
    angle = np.pi / 2  # 90 degrees
    q = axis_angle_to_quaternion(axis, angle)
    norm = np.linalg.norm(q)
    print("\n[Quaternion Normalisation Test]")
    print(f"  Norm = {norm:.10f} (should be 1.0)", "✅" if abs(norm - 1.0) < 1e-10 else "❌")
    assert np.allclose(norm, 1.0, atol=1e-10)

def test_quaternion_to_dcm_and_back():
    axis = np.array([0, 1, 0])
    angle = np.pi / 3
    q = axis_angle_to_quaternion(axis, angle)
    dcm = quaternion_to_dcm(q)
    euler = dcm_to_euler_angles(dcm)
    finite = np.isfinite(euler).all()
    print("\n[Quaternion → DCM → Euler Conversion Test]")
    print(f"  DCM shape: {dcm.shape}")
    print(f"  Euler angles (deg): {np.rad2deg(euler)}", "✅" if finite else "❌")
    assert dcm.shape == (3, 3)
    assert finite

def test_angular_momentum_and_energy():
    J = np.diag([2500, 5000, 6500])
    omega = np.array([0.001, 0.002, 0.003])
    H = angular_momentum(J, omega)
    T = rotational_kinetic_energy(J, omega)
    valid_energy = T > 0 and np.isfinite(T)
    print("\n[Angular Momentum & Kinetic Energy Test]")
    print(f"  H = {H}")
    print(f"  T = {T:.6f} km^2/s^2", "✅" if valid_energy else "❌")
    assert H.shape == (3,)
    assert valid_energy

def test_attitude_rhs_free_dynamics():
    J = np.diag([2500, 5000, 6500])
    omega0 = np.array([0.001, 0.002, 0.003])
    q0 = axis_angle_to_quaternion(np.array([0, 0, 1]), 0)
    state = np.concatenate((q0, omega0))
    rhs = attitude_rhs(0, state, J)
    finite = np.isfinite(rhs).all()
    correct_shape = rhs.shape == (7,)
    print("\n[Attitude Dynamics RHS Test]")
    print(f"  Output shape: {rhs.shape}", "✅" if correct_shape else "❌")
    print(f"  Values: {np.round(rhs, 8)}", "✅" if finite else "❌")
    assert correct_shape
    assert finite

"""
Python port of CCNY Robotics Lab's imu_tools complementary_filter.cpp.

Original project:
https://github.com/CCNYRoboticsLab/imu_tools

Quaternion convention:
    [w, x, y, z]

The internal state represents the global frame relative to the body frame.
get_orientation() returns its inverse, matching the C++ implementation.

BSD 3-Clause License

Copyright (c) 2015, City University of New York
CCNY Robotics Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _as_vector3(values: Iterable[float], name: str) -> FloatArray:
    vector = np.asarray(tuple(values), dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains a non-finite value")
    return vector


def _as_quaternion(values: Iterable[float], name: str = "quaternion") -> FloatArray:
    quaternion = np.asarray(tuple(values), dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"{name} must contain exactly four values")
    if not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{name} contains a non-finite value")
    return quaternion


def normalize_vector(vector: Iterable[float]) -> FloatArray:
    v = _as_vector3(vector, "vector")
    norm = float(np.linalg.norm(v))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Cannot normalize a zero-length vector")
    return v / norm


def normalize_quaternion(quaternion: Iterable[float]) -> FloatArray:
    q = _as_quaternion(quaternion)
    norm = float(np.linalg.norm(q))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Cannot normalize a zero-length quaternion")
    return q / norm


def invert_quaternion(quaternion: Iterable[float]) -> FloatArray:
    """Quaternion conjugate. Assumes the quaternion is normalized."""
    q = _as_quaternion(quaternion)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_multiply(
    first: Iterable[float],
    second: Iterable[float],
) -> FloatArray:
    """Hamilton product, r = first * second."""
    p0, p1, p2, p3 = _as_quaternion(first, "first quaternion")
    q0, q1, q2, q3 = _as_quaternion(second, "second quaternion")

    return np.array(
        [
            p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
            p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
            p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
            p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0,
        ],
        dtype=np.float64,
    )


def rotate_vector_by_quaternion(
    vector: Iterable[float],
    quaternion: Iterable[float],
) -> FloatArray:
    x, y, z = _as_vector3(vector, "vector")
    q0, q1, q2, q3 = _as_quaternion(quaternion)

    return np.array(
        [
            (q0*q0 + q1*q1 - q2*q2 - q3*q3) * x
            + 2.0 * (q1*q2 - q0*q3) * y
            + 2.0 * (q1*q3 + q0*q2) * z,

            2.0 * (q1*q2 + q0*q3) * x
            + (q0*q0 - q1*q1 + q2*q2 - q3*q3) * y
            + 2.0 * (q2*q3 - q0*q1) * z,

            2.0 * (q1*q3 - q0*q2) * x
            + 2.0 * (q2*q3 + q0*q1) * y
            + (q0*q0 - q1*q1 - q2*q2 + q3*q3) * z,
        ],
        dtype=np.float64,
    )


def scale_quaternion(gain: float, delta_q: Iterable[float]) -> FloatArray:
    """
    Scale a correction quaternion toward identity.

    This follows the original implementation:
    - SLERP branch when scalar part is negative
    - LERP branch otherwise
    """
    if not 0.0 <= gain <= 1.0:
        raise ValueError("gain must be in [0, 1]")

    dq = normalize_quaternion(delta_q)
    dq0, dq1, dq2, dq3 = dq

    if dq0 < 0.0:
        angle = math.acos(float(np.clip(dq0, -1.0, 1.0)))
        sin_angle = math.sin(angle)

        if abs(sin_angle) <= np.finfo(np.float64).eps:
            result = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            a = math.sin(angle * (1.0 - gain)) / sin_angle
            b = math.sin(angle * gain) / sin_angle
            result = np.array(
                [a + b*dq0, b*dq1, b*dq2, b*dq3],
                dtype=np.float64,
            )
    else:
        result = np.array(
            [
                (1.0 - gain) + gain*dq0,
                gain*dq1,
                gain*dq2,
                gain*dq3,
            ],
            dtype=np.float64,
        )

    return normalize_quaternion(result)


class ComplementaryFilter:
    """Quaternion complementary filter ported from imu_tools."""

    GRAVITY = 9.81
    ANGULAR_VELOCITY_THRESHOLD = 0.2
    ACCELERATION_THRESHOLD = 0.1
    DELTA_ANGULAR_VELOCITY_THRESHOLD = 0.01

    def __init__(
        self,
        gain_acc: float = 0.01,
        gain_mag: float = 0.01,
        bias_alpha: float = 0.01,
        do_bias_estimation: bool = True,
        do_adaptive_gain: bool = False,
    ) -> None:
        self.gain_acc = self._validated_gain(gain_acc, "gain_acc")
        self.gain_mag = self._validated_gain(gain_mag, "gain_mag")
        self.bias_alpha = self._validated_gain(bias_alpha, "bias_alpha")

        self.do_bias_estimation = bool(do_bias_estimation)
        self.do_adaptive_gain = bool(do_adaptive_gain)

        self.initialized = False
        self.steady_state = False

        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._omega_previous = np.zeros(3, dtype=np.float64)
        self._omega_bias = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _validated_gain(value: float, name: str) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        return value

    def set_orientation(self, quaternion: Iterable[float]) -> None:
        """
        Set the externally visible orientation.

        The internal state stores the inverse, matching the C++ implementation.
        """
        self._q = invert_quaternion(normalize_quaternion(quaternion))
        self.initialized = True

    def get_orientation(self) -> FloatArray:
        """Return orientation as normalized [w, x, y, z]."""
        return invert_quaternion(self._q)

    def get_angular_velocity_bias(self) -> FloatArray:
        return self._omega_bias.copy()

    def check_state(
        self,
        acceleration: Iterable[float],
        angular_velocity: Iterable[float],
    ) -> bool:
        acc = _as_vector3(acceleration, "acceleration")
        omega = _as_vector3(angular_velocity, "angular_velocity")

        acceleration_magnitude = float(np.linalg.norm(acc))
        if abs(acceleration_magnitude - self.GRAVITY) > self.ACCELERATION_THRESHOLD:
            return False

        if np.any(
            np.abs(omega - self._omega_previous)
            > self.DELTA_ANGULAR_VELOCITY_THRESHOLD
        ):
            return False

        if np.any(
            np.abs(omega - self._omega_bias)
            > self.ANGULAR_VELOCITY_THRESHOLD
        ):
            return False

        return True

    def update_biases(
        self,
        acceleration: Iterable[float],
        angular_velocity: Iterable[float],
    ) -> None:
        acc = _as_vector3(acceleration, "acceleration")
        omega = _as_vector3(angular_velocity, "angular_velocity")

        self.steady_state = self.check_state(acc, omega)

        if self.steady_state:
            self._omega_bias += self.bias_alpha * (
                omega - self._omega_bias
            )

        self._omega_previous = omega.copy()

    def get_prediction(
        self,
        angular_velocity: Iterable[float],
        dt: float,
    ) -> FloatArray:
        omega = _as_vector3(angular_velocity, "angular_velocity")
        dt = float(dt)

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite value")

        wx, wy, wz = omega - self._omega_bias
        q0, q1, q2, q3 = self._q

        predicted = np.array(
            [
                q0 + 0.5*dt*( wx*q1 + wy*q2 + wz*q3),
                q1 + 0.5*dt*(-wx*q0 - wy*q3 + wz*q2),
                q2 + 0.5*dt*( wx*q3 - wy*q0 - wz*q1),
                q3 + 0.5*dt*(-wx*q2 + wy*q1 - wz*q0),
            ],
            dtype=np.float64,
        )
        return normalize_quaternion(predicted)

    @staticmethod
    def get_measurement_acc(
        acceleration: Iterable[float],
    ) -> FloatArray:
        ax, ay, az = normalize_vector(acceleration)

        if az >= 0.0:
            q0 = math.sqrt(max((az + 1.0) * 0.5, 0.0))
            if q0 <= np.finfo(np.float64).eps:
                raise ValueError("Accelerometer orientation is singular")
            q = np.array(
                [q0, -ay/(2.0*q0), ax/(2.0*q0), 0.0],
                dtype=np.float64,
            )
        else:
            x_term = math.sqrt(max((1.0 - az) * 0.5, 0.0))
            if x_term <= np.finfo(np.float64).eps:
                raise ValueError("Accelerometer orientation is singular")
            q = np.array(
                [-ay/(2.0*x_term), x_term, 0.0, ax/(2.0*x_term)],
                dtype=np.float64,
            )

        return normalize_quaternion(q)

    @staticmethod
    def get_measurement_acc_mag(
        acceleration: Iterable[float],
        magnetic_field: Iterable[float],
    ) -> FloatArray:
        q_acc = ComplementaryFilter.get_measurement_acc(acceleration)
        mx, my, mz = _as_vector3(magnetic_field, "magnetic_field")
        q0, q1, q2, _ = q_acc

        lx = (
            (q0*q0 + q1*q1 - q2*q2) * mx
            + 2.0*(q1*q2) * my
            - 2.0*(q0*q2) * mz
        )
        ly = (
            2.0*(q1*q2) * mx
            + (q0*q0 - q1*q1 + q2*q2) * my
            + 2.0*(q0*q1) * mz
        )

        gamma = lx*lx + ly*ly
        if gamma <= np.finfo(np.float64).eps:
            raise ValueError("Horizontal magnetic-field magnitude is too small")

        beta = math.sqrt(max(gamma + lx*math.sqrt(gamma), 0.0))
        if beta <= np.finfo(np.float64).eps:
            raise ValueError("Magnetometer correction is singular")

        q_mag = np.array(
            [
                beta / math.sqrt(2.0*gamma),
                0.0,
                0.0,
                ly / (math.sqrt(2.0)*beta),
            ],
            dtype=np.float64,
        )

        return normalize_quaternion(quaternion_multiply(q_acc, q_mag))

    @staticmethod
    def get_acc_correction(
        acceleration: Iterable[float],
        predicted_q: Iterable[float],
    ) -> FloatArray:
        acc = normalize_vector(acceleration)
        p = normalize_quaternion(predicted_q)

        predicted_gravity = rotate_vector_by_quaternion(
            acc,
            invert_quaternion(p),
        )
        gx, gy, gz = predicted_gravity

        dq0 = math.sqrt(max((gz + 1.0) * 0.5, 0.0))
        if dq0 <= np.finfo(np.float64).eps:
            # This singular case corresponds to a 180-degree correction.
            # Pick a valid horizontal rotation axis.
            horizontal = math.hypot(gx, gy)
            if horizontal <= np.finfo(np.float64).eps:
                return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
            return normalize_quaternion(
                [0.0, -gy/horizontal, gx/horizontal, 0.0]
            )

        return normalize_quaternion(
            [dq0, -gy/(2.0*dq0), gx/(2.0*dq0), 0.0]
        )

    @staticmethod
    def get_mag_correction(
        magnetic_field: Iterable[float],
        predicted_q: Iterable[float],
    ) -> FloatArray:
        magnetic = _as_vector3(magnetic_field, "magnetic_field")
        p = normalize_quaternion(predicted_q)

        lx, ly, _ = rotate_vector_by_quaternion(
            magnetic,
            invert_quaternion(p),
        )

        gamma = lx*lx + ly*ly
        if gamma <= np.finfo(np.float64).eps:
            raise ValueError("Horizontal magnetic-field magnitude is too small")

        beta = math.sqrt(max(gamma + lx*math.sqrt(gamma), 0.0))
        if beta <= np.finfo(np.float64).eps:
            raise ValueError("Magnetometer correction is singular")

        return normalize_quaternion(
            [
                beta / math.sqrt(2.0*gamma),
                0.0,
                0.0,
                ly / (math.sqrt(2.0)*beta),
            ]
        )

    @classmethod
    def get_adaptive_gain(
        cls,
        alpha: float,
        acceleration: Iterable[float],
    ) -> float:
        alpha = cls._validated_gain(alpha, "alpha")
        acceleration_magnitude = float(
            np.linalg.norm(_as_vector3(acceleration, "acceleration"))
        )
        error = abs(acceleration_magnitude - cls.GRAVITY) / cls.GRAVITY

        error1 = 0.1
        error2 = 0.2

        if error < error1:
            factor = 1.0
        elif error < error2:
            factor = (error2 - error) / (error2 - error1)
        else:
            factor = 0.0

        return factor * alpha

    def update(
        self,
        acceleration: Iterable[float],
        angular_velocity: Iterable[float],
        dt: float,
        magnetic_field: Optional[Iterable[float]] = None,
    ) -> FloatArray:
        """
        Update the filter and return orientation [w, x, y, z].

        Parameters
        ----------
        acceleration:
            Accelerometer measurement [ax, ay, az], normally in m/s^2.
        angular_velocity:
            Gyroscope measurement [wx, wy, wz], in rad/s.
        dt:
            Time step in seconds.
        magnetic_field:
            Optional magnetometer measurement [mx, my, mz].
        """
        acc = _as_vector3(acceleration, "acceleration")
        omega = _as_vector3(angular_velocity, "angular_velocity")

        if not self.initialized:
            if magnetic_field is None:
                self._q = self.get_measurement_acc(acc)
            else:
                self._q = self.get_measurement_acc_mag(
                    acc,
                    magnetic_field,
                )
            self.initialized = True
            return self.get_orientation()

        if self.do_bias_estimation:
            self.update_biases(acc, omega)

        predicted_q = self.get_prediction(omega, dt)

        delta_acc = self.get_acc_correction(acc, predicted_q)
        gain_acc = (
            self.get_adaptive_gain(self.gain_acc, acc)
            if self.do_adaptive_gain
            else self.gain_acc
        )
        delta_acc = scale_quaternion(gain_acc, delta_acc)
        corrected_q = quaternion_multiply(predicted_q, delta_acc)

        if magnetic_field is not None:
            delta_mag = self.get_mag_correction(
                magnetic_field,
                corrected_q,
            )
            delta_mag = scale_quaternion(self.gain_mag, delta_mag)
            corrected_q = quaternion_multiply(corrected_q, delta_mag)

        self._q = normalize_quaternion(corrected_q)
        return self.get_orientation()

    def reset(self) -> None:
        self.initialized = False
        self.steady_state = False
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._omega_bias = np.zeros(3, dtype=np.float64)
        self._omega_previous = np.zeros(3, dtype=np.float64)


if __name__ == "__main__":
    # Small stationary example. A real application should use measured dt.
    filter_ = ComplementaryFilter(
        gain_acc=0.01,
        do_bias_estimation=True,
        do_adaptive_gain=False,
    )

    dt = 0.02
    acceleration = np.array([0.0, 0.0, 9.81])
    angular_velocity = np.array([0.0, 0.0, 0.0])

    for _ in range(100):
        orientation = filter_.update(
            acceleration=acceleration,
            angular_velocity=angular_velocity,
            dt=dt,
        )

    print("Quaternion [w, x, y, z]:", orientation)

#!/usr/bin/env python
"""Supporting functions."""
import numpy as np

def trace_field_line(coords, interpolator, tracer):
    # Trace in the direction of the field.
    tracer.follow_field_direction = True
    tracer.compute(coords, interpolator)
    coords_forward = tracer.points

    # Trace in the opposite direction of the field.
    tracer.follow_field_direction = False
    tracer.compute(coords, interpolator)
    coords_backward = tracer.points

    x = np.concatenate((np.flip(np.array(coords_backward)[:, 0])[0:-1], np.array(coords_forward)[:, 0]))
    y = np.concatenate((np.flip(np.array(coords_backward)[:, 1])[0:-1], np.array(coords_forward)[:, 1]))
    z = np.concatenate((np.flip(np.array(coords_backward)[:, 2])[0:-1], np.array(coords_forward)[:, 2]))

    return x, y, z

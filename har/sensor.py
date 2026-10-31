import os, streamlit.components.v1 as components
_f = components.declare_component("har_sensor_stream", path=os.path.join(os.path.dirname(__file__), "sensor_component"))
def sensor_stream(key="sensor"):
    """Returns {session, i0, rate, samples[[ax,ay,az,gx,gy,gz]...]} about once per second, or None."""
    return _f(key=key, default=None)

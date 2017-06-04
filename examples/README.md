# Examples
## Parabint Trajectories

There are three types of (one-dimensional) trajectories:
 - `Ramp`: a segment with constant acceleration
 - `ParabolicCurve`: a concatenation of `Ramp`s
 - `ParabolicCurvesND`: an n-dimensional trajectory consisting of n `ParabolicCurvesND`s of equal duration
```python
from parabint.trajectory import Ramp, ParabolicCurve, ParabolicCurvesND

ramp1 = Ramp(0, 1, 0.5)
ramp2 = Ramp(ramp1.v1, -0.7, 0.8)
curve1 = ParabolicCurve(ramps=[ramp1, ramp2])

curve1.PlotVel() # visualize the velocity profile

ramp3 = Ramp(0.2, -0.66, 1.0)
ramp4 = Ramp(ramp3.v1, 0, 0.3)
curve2 = ParabolicCurve(ramps=[ramp3, ramp4])
curvesnd = ParabolicCurvesND(curves=[curve1, curve2])

curvesnd.PlotVel() # visualize the velocity profile
```

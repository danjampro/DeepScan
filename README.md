## DeepScan v0.61 

![A picture](https://raw.githubusercontent.com/danjampro/DeepScan/master/Examples/example_deblending.png)

DeepScan is a Python3 package designed to detect low surface brightness features in astronomical data without fragmenting extended structure. The software was originally introduced in [Prole et al. (2018)](https://doi.org/10.1093/mnras/sty1021).

DeepScan has gone through significant changes since its first release. The most significant development has been the introduction of a de-blending algorithm, designed to overcome limitations encountered by other software.

Please see the examples to see how to use DeepScan. 

The software is under continual development and any questions or feedback would be welcomed.

**Basic Usage**

See ~/Examples/example_basic.py. Basic usage Looks like this:

```python
from deepscan.deepscan import DeepScan

#"data" is your data as a 2D np.array.
result = DeepScan(data) 

```

**Installation:**

	$pip install deepscan

**Contact:**

Dan Prole (<proled@cardiff.ac.uk>)


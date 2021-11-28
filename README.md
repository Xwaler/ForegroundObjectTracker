# ForegroundObjectTracker

## Install

- Download the latest .whl release
- ```pip install -U ForegroundObjectTracker-X.X.X-py3-none-any.whl```


## Usage

```python
from ForegroundObjectTracker import ForegroundObjectTracker

tracker = ForegroundObjectTracker()

"""From a file"""
for detections in tracker.run_from_file(footage_path='footage/footage3.mp4', display=True, save_footage=False):
    print(detections)
    
"""From individual frames"""
frames = [...]
for frame in frames:
    detections = tracker.forward(frame, display=False, save_footage=False)
tracker.cv_cleanup()
```
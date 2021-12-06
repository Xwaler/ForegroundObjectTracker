# ForegroundObjectTracker

---

**ForegroundObjectTracker** will identify and extract **in real time** any moving objects from a series of images.

## Installation

- Download the latest **.whl** release
- ```pip install -U ForegroundObjectTracker-X.X.X-py3-none-any.whl```


## Usage

Here is a basic implementation of this package :
```python
from ForegroundObjectTracker import ForegroundObjectTracker

tracker = ForegroundObjectTracker()

"""From a file"""
for detections in tracker.run_from_file(footage_path='footage/footage3.mp4'):
    print(detections)

"""From individual frames"""
frames = [...]
for frame in frames:
    detections = tracker.forward(frame)
tracker.cv_cleanup()
```

For a more complex and customizable usage, please refer to the [repository wiki](../../wiki).

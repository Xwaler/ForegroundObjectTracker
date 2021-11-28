from ForegroundObjectTracker import ForegroundObjectTracker


if __name__ == '__main__':
    tracker = ForegroundObjectTracker()

    """From a file"""
    for detections in tracker.run_from_file(footage_path='footage/footage3.mp4', display=True, save_footage=False):
        print(detections)

    """From individual frames"""
    # for frame in frames:
    #     detections = tracker.forward(frame, display=False, save_footage=False)
    # tracker.cv_cleanup()

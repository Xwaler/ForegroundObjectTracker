from ForegroundObjectsTracker import ForegroundObjectsTracker


def main():
    writer = None
    previous_frame = None
    sparse_points_to_track = None
    filters = None
    known_detections = [[], [], [], []]

    for n, frame in enumerate(frame_generator(FOOTAGE_PATH)):
        if previous_frame is None:
            previous_frame = frame
            h, w = frame.shape[:2]
            sparse_points_to_track = compute_sparse_points_to_track(w, h)
            filters = np.ceil([
                CONTOUR_MIN_WIDTH_RATIO * w,
                CONTOUR_MAX_WIDTH_RATIO * w,
                CONTOUR_MIN_HEIGHT_RATIO * h,
                CONTOUR_MAX_HEIGHT_RATIO * h,
            ])
            continue

        """
        ABS DIFF, GAUSSIAN THRESHOLD, MORPHOLOGY CLOSING, FULLY CORRECTED WITH KALMAN FILTER
        """
        s = time()
        diff = cv2.absdiff(previous_frame, frame)
        gray_frame = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        threshold = apply_gaussian_threshold(gray_frame)
        morphology = apply_morphology_closing(threshold)
        contours = find_contours(morphology)
        bounding_rects = get_bounding_rects(contours)
        assign_rect_to_detections(bounding_rects, known_detections[0], filters)
        result_abs = display_detections(frame, known_detections[0])
        time_abs = (time() - s) * 1000.

        """
        DENSE OPTICAL FLOW, GAUSSIAN THRESHOLD, MORPHOLOGY CLOSING, FULLY CORRECTED WITH KALMAN FILTER
        """
        s = time()
        flow = apply_dense_optical_flow(previous_frame, frame)
        threshold = apply_gaussian_threshold(flow)
        filtered = apply_blur(threshold)
        morphology = apply_morphology_closing(filtered)
        contours = find_contours(morphology)
        bounding_rects = get_bounding_rects(contours)
        assign_rect_to_detections(bounding_rects, known_detections[1], filters)
        result_dense = display_detections(frame, known_detections[1])
        time_dense = (time() - s) * 1000.

        """
        SPARSE OPTICAL FLOW, GAUSSIAN THRESHOLD, MORPHOLOGY CLOSING, FULLY CORRECTED WITH KALMAN FILTER
        """
        s = time()
        feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7, useHarrisDetector=True)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_points_to_track = cv2.goodFeaturesToTrack(gray_frame, **feature_params)
        corrected_frame, homography_matrix = apply_sparse_optical_flow(
            previous_frame, frame, best_points_to_track
        )
        threshold = apply_gaussian_threshold(corrected_frame)
        filtered = apply_blur(threshold)
        morphology = apply_morphology_closing(filtered)
        contours = find_contours(morphology)
        bounding_rects = get_bounding_rects(contours)
        assign_rect_to_detections(bounding_rects, known_detections[2], filters)
        result_sparse = display_detections(frame, known_detections[2])
        time_sparse = (time() - s) * 1000.

        """
        CUSTOM SPARSE OPTICAL FLOW, GAUSSIAN THRESHOLD, MORPHOLOGY CLOSING, FULLY CORRECTED WITH KALMAN FILTER
        """
        s = time()
        best_points_to_track = filter_best_points_to_track(sparse_points_to_track, known_detections[3])
        corrected_frame, homography_matrix = apply_sparse_optical_flow(
            previous_frame, frame, best_points_to_track
        )
        threshold = apply_gaussian_threshold(corrected_frame)
        filtered = apply_blur(threshold)
        morphology = apply_morphology_closing(filtered)
        contours = find_contours(morphology)
        bounding_rects = get_bounding_rects(contours)
        assign_rect_to_detections(bounding_rects, known_detections[3], filters)
        result_custom_sparse = display_detections(frame, known_detections[3])
        time_custom_sparse = (time() - s) * 1000.

        previous_frame = frame
        print(f'\nFrame {n}')
        pprint({k: f'{np.round(np.mean(values), 3)} ms' for k, values in TIMEIT_HISTORY.items()})

        labeled_images = add_labels([
            result_abs,
            result_dense,
            result_sparse,
            result_custom_sparse,
        ], labels=[
            "Absolute difference", "Dense optical flow", "Sparse optical flow", "Custom sparse optical flow (SELECTED)"
        ], times=np.round([
            time_abs, time_dense, time_sparse, time_custom_sparse
        ], 3))
        processed_frame = concat_views(labeled_images)
        resized = resize_with_aspect_ratio(processed_frame, width=DISPLAY_WINDOW_WIDTH)
        cv2.imshow('Display', resized)

        if RENDER_SAVE_TO_FILE:
            if writer is None:
                writer = cv2.VideoWriter(
                    'benchmark.mp4', cv2.VideoWriter_fourcc(*'mp4v'), RENDER_FPS, processed_frame.shape[:2][::-1]
                )
            writer.write(processed_frame)

        if cv2.waitKey(1) in [27, ord('q'), ord('Q')]:
            break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

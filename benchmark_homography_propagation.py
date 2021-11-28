from main import *


def main():
    writer = None
    previous_frame = None
    sparse_points_to_track = None
    filters = None
    known_detections = [[], []]

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
        WITHOUT HOMOGRAPHY PROPAGATION TO KALMAN FILTER
        """
        best_points_to_track = filter_best_points_to_track(sparse_points_to_track, known_detections[0])
        corrected_frame, homography_matrix = apply_sparse_optical_flow(
            previous_frame, frame, best_points_to_track
        )
        threshold = apply_gaussian_threshold(corrected_frame)
        filtered = apply_blur(threshold)
        morphology = apply_morphology_closing(filtered)
        contours = find_contours(morphology)
        bounding_rects = get_bounding_rects(contours)
        assign_rect_to_detections(bounding_rects, known_detections[0], filters)
        result = display_detections(frame, known_detections[0], rect_color=(0, 0, 255), trajectory_color=(0, 0, 255))

        """
        WITH HOMOGRAPHY PROPAGATION TO KALMAN FILTER
        """
        s = time()
        best_points_to_track = filter_best_points_to_track(sparse_points_to_track, known_detections[1])
        corrected_frame, homography_matrix = apply_sparse_optical_flow(
            previous_frame, frame, best_points_to_track
        )
        project_detections_on_new_plane(known_detections[1], homography_matrix)
        threshold = apply_gaussian_threshold(corrected_frame)
        filtered = apply_blur(threshold)
        morphology = apply_morphology_closing(filtered)
        contours = find_contours(morphology)
        bounding_rects = get_bounding_rects(contours)
        assign_rect_to_detections(bounding_rects, known_detections[1], filters)
        result = display_detections(result, known_detections[1], rect_color=(0, 255, 0), trajectory_color=(0, 255, 0))
        time_with = (time() - s) * 1000.

        previous_frame = frame
        print(f'\nFrame {n}')
        pprint({k: f'{np.round(np.mean(values), 3)} ms' for k, values in TIMEIT_HISTORY.items()})

        labeled_images = add_labels([
            result,
        ], labels=[
            "Homography propagation (Red: OFF | Green: ON)"
        ], times=np.round([
            time_with
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

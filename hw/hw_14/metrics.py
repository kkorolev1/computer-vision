import numpy as np


def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here
    lu_x = max(bbox1[0], bbox2[0])
    lu_y = max(bbox1[1], bbox2[1])
    rd_x = min(bbox1[2], bbox2[2])
    rd_y = min(bbox1[3], bbox2[3])
    if rd_x <= lu_x or rd_y <= lu_y:
        return 0
    else:
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        inter_area = (rd_x - lu_x) * (rd_y - lu_y)
        return inter_area / (bbox1_area + bbox2_area - inter_area)


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 5: Update matches with current matched IDs
        frame_obj_dict = {detection[0]: detection for detection in frame_obj}
        frame_hyp_dict = {detection[0]: detection for detection in frame_hyp}

        frame_busy_ids = set()
        hyp_busy_ids = set()

        for gt_id, hyp_id in matches.items():
            if gt_id in frame_obj_dict and hyp_id in frame_hyp_dict:
                iou = iou_score(frame_obj_dict[gt_id][1:], frame_hyp_dict[hyp_id][1:])
                if iou > threshold:  # it's a match
                    dist_sum += iou
                    match_count += 1
                    frame_busy_ids.add(gt_id)
                    hyp_busy_ids.add(hyp_id)

        for gt_detection in frame_obj:
            ious = [float("-inf")] * len(frame_hyp)
            for i, hyp_detection in enumerate(frame_hyp):
                if gt_detection[0] not in frame_busy_ids and hyp_detection[0] not in hyp_busy_ids:
                    iou = iou_score(gt_detection[1:], hyp_detection[1:])
                    if iou > threshold:
                        ious[i] = iou

            if np.max(ious) != float("-inf"):
                max_index = np.argmax(ious)
                hyp_id = frame_hyp[max_index][0]
                matches[gt_detection[0]] = hyp_id
                dist_sum += ious[max_index]
                match_count += 1
                frame_busy_ids.add(gt_detection[0])
                hyp_busy_ids.add(hyp_id)



    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error

        # Step 6: Update matches with current matched IDs

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        frame_obj_dict = {detection[0]: detection[1:] for detection in frame_obj}
        frame_hyp_dict = {detection[0]: detection[1:] for detection in frame_hyp}

        for gt_id, hyp_id in matches.items():
            if gt_id in frame_obj_dict and hyp_id in frame_hyp_dict:
                iou = iou_score(frame_obj_dict[gt_id], frame_hyp_dict[hyp_id])
                if iou > threshold:  # it's a match
                    dist_sum += iou
                    match_count += 1
                    del frame_obj_dict[gt_id]
                    del frame_hyp_dict[hyp_id]

        detections_pairs = []
        for gt_id, gt_bbox in frame_obj_dict.items():
            for hyp_id, hyp_bbox in frame_hyp_dict.items():
                iou = iou_score(gt_bbox, hyp_bbox)
                if iou > threshold:
                    detections_pairs += [(iou, gt_id, hyp_id)]

        detections_pairs.sort(reverse=True)
        new_matches = {}

        for iou, gt_id, hyp_id in detections_pairs:
            if gt_id in frame_obj_dict and hyp_id in frame_hyp_dict:
                match_count += 1
                dist_sum += iou
                new_matches[gt_id] = hyp_id
                if gt_id in matches and matches[gt_id] != hyp_id:
                    mismatch_error += 1
                del frame_obj_dict[gt_id]
                del frame_hyp_dict[hyp_id]

        for gt_id, hyp_id in new_matches.items():
            matches[gt_id] = hyp_id

        missed_count = len(frame_obj_dict)
        false_positive = len(frame_hyp_dict)

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (missed_count + false_positive + mismatch_error) / max(sum(map(len, obj)), 1)

    return MOTP, MOTA


def run_motp_mota():
    gt_tracks = [
        [[0, 0, 0, 150, 100], [1, 400, 0, 500, 150]],
        [[0, 50, 50, 200, 150], [1, 400, 100, 500, 250]],
        [[0, 150, 100, 300, 200], [1, 350, 150, 450, 300]]
    ]
    result_tracks = [
        [[0, 15, 15, 145, 115], [1, 380, 20, 510, 150]],
        [[0, 40, 40, 190, 160], [1, 400, 140, 520, 270]],
        [[0, 150, 150, 280, 240], [2, 330, 170, 450, 290]]
    ]

    gt_metrics = [0.665, 0.5]
    result_metrics = motp_mota(gt_tracks, result_tracks)

run_motp_mota()
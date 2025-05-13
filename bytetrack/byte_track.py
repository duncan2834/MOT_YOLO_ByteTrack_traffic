import numpy as np

from supervision.detection.core import Detections
from supervision.detection.utils import box_iou_batch
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.basetrack import BaseTrack, TrackState
from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter
from supervision.utils.internal import deprecated_parameter


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    
    def __init__(self, tlwh, score, class_ids):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)  # Bounding box dạng (top left x, top left y, width, height)
        self.kalman_filter = None  # Sẽ gán sau, dùng để dự đoán vị trí tiếp theo
        self.mean, self.covariance = None, None  # Trạng thái Kalman (mean vector và ma trận hiệp phương sai)
        self.is_activated = False  # Chưa active (chưa được xác nhận bởi một detection nào)
        
        self.score = score  # Confidence score của detection ban đầu
        self.class_ids = class_ids  # Nhãn lớp
        self.tracklet_len = 0  # Độ dài track
    
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0 # Nếu không phải trạng thái "Tracked", bỏ qua vận tốc (vh = 0)
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )
        
    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = []
            multi_covariance = []
            for i, st in enumerate(stracks):
                multi_mean.append(st.mean.copy())
                multi_covariance.append(st.covariance)
                # Nếu track không ở trạng thái Tracked, đặt mean[7] = 0 → vô hiệu hóa tốc độ thay đổi chiều cao (vh) để tránh dự đoán sai.
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                np.asarray(multi_mean), np.asarray(multi_covariance)
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
    def activate(self, kalman_filter, frame_id): # frame_id: ID của frame hiện tại, dùng để đánh dấu thời gian track bắt đầu
        # Khởi tạo một track mới
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id() # gan id moi cho object
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh) # chuyen tu tlwh[topleftx, toplefty, w, h] -> xyah
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # Nếu là frame đầu tiên của video, track này được đánh dấu là đã active
        if frame_id == 1:
            self.is_activated = True # Dùng để xác định track đã thực sự được sử dụng
        # Đánh dấu thời điểm bắt đầu theo dõi
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    def re_activate(self, new_track, frame_id, new_id=False): # reactivate 1 track cu sau khi lost nhung da khop voi object moi
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0 # reset lai tracklet
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
    
    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
    
    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy() # x, y, a, h
        ret[2] *= ret[3] # x, y, a * h = w, h
        ret[:2] -= ret[2:] / 2 # x - w/2, y - h/2, w, h
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2] # x1, y1, x1 + w, y1 + h
        return ret
    
    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)
    
    @staticmethod
    def tlbr_to_tlwh(tlbr): # chuyen tu topleft, bottomright -> tlwh
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_tlbr(tlwh): # chuyen tu tlwh -> topleft, bottomright
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret
    
    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)
    
def detections2boxes(detections):
    """
    Chuyển Supervision Detections sang numpy tensors.
    Args:
        detections (Detections): Detections/Targets in the format of sv.Detections.
    Returns:
        numpy tensors (x_min, y_min, x_max, y_max, confidence, class_id).
    """
    return np.hstack(
        (
            detections.xyxy,
            detections.confidence[:, np.newaxis],
            detections.class_id[:, np.newaxis],
        )
    )
    
class ByteTrack:
    """
    Initialize the ByteTrack object.

    <video controls>
        <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/annotate-video-with-traces.mp4" type="video/mp4">
    </video>

    Parameters:
        track_activation_threshold (float, optional): Detection confidence threshold
            for track activation. Increasing track_activation_threshold improves accuracy
            and stability but might miss true detections. Decreasing it increases
            completeness but risks introducing noise and instability.
        lost_track_buffer (int, optional): Number of frames to buffer when a track is lost.
            Increasing lost_track_buffer enhances occlusion handling, significantly
            reducing the likelihood of track fragmentation or disappearance caused
            by brief detection gaps.
        minimum_matching_threshold (float, optional): Threshold for matching tracks with detections.
            Increasing minimum_matching_threshold improves accuracy but risks fragmentation.
            Decreasing it improves completeness but risks false positives and drift.
        frame_rate (int, optional): The frame rate of the video.
    """
    # cảnh báo rằng một tham số cũ đã bị thay thế bằng tham số mới
    @deprecated_parameter(
        old_parameter="track_buffer",
        new_parameter="lost_track_buffer",
        map_function=lambda x: x,
        warning_message="`{old_parameter}` in `{function_name}` is deprecated and will "
        "be remove in `supervision-0.23.0`. Use '{new_parameter}' "
        "instead.",
    )
    @deprecated_parameter(
        old_parameter="track_thresh",
        new_parameter="track_activation_threshold",
        map_function=lambda x: x,
        warning_message="`{old_parameter}` in `{function_name}` is deprecated and will "
        "be remove in `supervision-0.23.0`. Use '{new_parameter}' "
        "instead.",
    )
    @deprecated_parameter(
        old_parameter="match_thresh",
        new_parameter="minimum_matching_threshold",
        map_function=lambda x: x,
        warning_message="`{old_parameter}` in `{function_name}` is deprecated and will "
        "be remove in `supervision-0.23.0`. Use '{new_parameter}' "
        "instead.",
    )
    
    def __init__(
        self,
        track_activation_threshold: 0.25, # Ngưỡng confidence để kích hoạt một track mới
        lost_track_buffer: 30, # sau 30 frame mà không xuất hiện lại thì xóa
        minimum_matching_threshold: 0.8, # ngưỡng matching kiểu iou
        frame_rate: 30,
    ):
        self.track_activation_threshold = track_activation_threshold
        self.minimum_matching_threshold = minimum_matching_threshold

        self.frame_id = 0
        self.det_thresh = self.track_activation_threshold + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)
        self.kalman_filter = KalmanFilter()

        self.tracked_tracks= [] # danh sách các track đang được theo dõi bình thường.
        self.lost_tracks = [] # các track tạm thời mất detection (đang trong buffer).
        self.removed_tracks = [] # các track đã bị loại bỏ vĩnh viễn (quá thời gian buffer hoặc không match nữa).
        
    def update_with_detections(self, detections):
        """
        Updates the tracker with the provided detections and returns the updated
        detection results.

        Args:
            detections (Detections): The detections to pass through the tracker.

        Example:
            ```python
            import supervision as sv
            from ultralytics import YOLO

            model = YOLO(<MODEL_PATH>)
            tracker = sv.ByteTrack()

            bounding_box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            def callback(frame: np.ndarray, index: int) -> np.ndarray:
                results = model(frame)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = tracker.update_with_detections(detections)

                labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

                annotated_frame = bounding_box_annotator.annotate(
                    scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels)
                return annotated_frame

            sv.process_video(
                source_path=<SOURCE_VIDEO_PATH>,
                target_path=<TARGET_VIDEO_PATH>,
                callback=callback
            )
            ```
        """

        tensors = detections2boxes(detections=detections) # chuyển -> dạng [x1, y1, x2, y2, score, class_id]
        tracks = self.update_with_tensors(tensors=tensors)

        if len(tracks) > 0:
            detection_bounding_boxes = np.asarray([det[:4] for det in tensors]) # lấy x1, y1, x2, y2
            track_bounding_boxes = np.asarray([track.tlbr for track in tracks]) # lấy x1, y1, x2, y2 của track

            ious = box_iou_batch(detection_bounding_boxes, track_bounding_boxes) # tính iou

            iou_costs = 1 - ious # iou distance
            # Ánh xạ optimal detection với track tương ứng (IoU ≥ 0.5).
            # Trả về danh sách matches: các cặp chỉ số [i_detection, i_track]
            matches, _, _ = matching.linear_assignment(iou_costs, 0.5)
            
            # Gán tracker_id cho detections
            detections.tracker_id = np.full(len(detections), -1, dtype=int)
            for i_detection, i_track in matches:
                detections.tracker_id[i_detection] = int(tracks[i_track].track_id)
                
            # Chỉ giữ lại các detection được gán tracker_id
            return detections[detections.tracker_id != -1]

        else:
            detections.tracker_id = np.array([], dtype=int)

            return detections
        
    def reset(self):
        """
        Resets the internal state of the ByteTrack tracker.

        This method clears the tracking data, including tracked, lost,
        and removed tracks, as well as resetting the frame counter. It's
        particularly useful when processing multiple videos sequentially,
        ensuring the tracker starts with a clean state for each new video.
        """
        # đơn giản là reset lại thôi, cho về từ đầu hết.
        self.frame_id = 0
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        BaseTrack.reset_counter()
        
    def update_with_tensors(self, tensors):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        class_ids = tensors[:, 5] # [x1, y1, x2, y2, confidencescore, classid], lay clsid
        scores = tensors[:, 4] # lay cscore
        bboxes = tensors[:, :4] # x1, y1, x2, y2
        
        remain_inds = scores > self.track_activation_threshold # > 0.25 thi lấy index, detection mạnh
        inds_low = scores > 0.1
        inds_high = scores < self.track_activation_threshold
        
        inds_second = np.logical_and(inds_low, inds_high) # những detection có score trung bình nằm giữa 0.1 và track_activation_threshold
        dets_second = bboxes[inds_second] # box trung bình
        dets = bboxes[remain_inds] # box mạnh
        scores_keep = scores[remain_inds] # score mạnh
        scores_second = scores[inds_second] # score trung bình
        
        class_ids_keep = class_ids[remain_inds] # cls id mạnh
        class_ids_second = class_ids[inds_second] # cls id trung bình
        
        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, c) # tlbr là bbox, s là score, c là clsid của các detection mạnh
                for (tlbr, s, c) in zip(dets, scores_keep, class_ids_keep)
            ]
        else:
            detections = []
            
        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]

        for track in self.tracked_tracks:
            if not track.is_activated: # track chưa activate thì cho vô unconfirmed
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
                
        """ Step 2: First association, with high score detection boxes"""
        # Gán các detection có confidence cao với các track đang theo dõi (tracked_stracks) hoặc các (lost_tracks)
        strack_pool = joint_tracks(tracked_stracks, self.lost_tracks)
        # Predict the current location with Kalman filter
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections) # tinh iou dis giua cac track da predict va bbox moi

        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.minimum_matching_threshold
        )
        # matches: danh sách các cặp (track_idx, detection_idx) được gán thành công
        # u_track: cac track ko duoc gan
        # u_detection: cac detection ko dc gan
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else: # neu track la lost, ma tim dc match moi thi reactivate
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                
        """ Step 3: Second association, with low score detection boxes"""
        # Tăng khả năng "giữ" track khi vật thể bị occlusion hoặc detection model không đủ tự tin
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, c)
                for (tlbr, s, c) in zip(dets_second, scores_second, class_ids_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track # cac track chua match
            if strack_pool[i].state == TrackState.Tracked # giu lai track
        ]
        
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked: # dang track thi cap nhat 
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else: # lost thi reactivate 
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                
        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        # Ghép các detection chưa match với track chưa xác nhận (unconfirmed)
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)

        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        # Những track chưa xác nhận nếu match được thì sẽ được "kích hoạt" và theo dõi tiếp
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
            
        # Track nào không match được → xóa luôn
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
            
        """ Step 4: Init new stracks"""
        # u_detection: là chỉ số (index) của các detection chưa được gán sau 2 vòng matching
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        
        """ Step 5: Update state"""
        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track) # xoa track

        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.Tracked
        ]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_starcks) # them cac track voi dc activate
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refind_stracks) # them cac track duoc tim thay lai
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks.extend(removed_stracks)
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        ) # Tránh trùng lặp track giữa tracked và lost, giữ lại track nào có thời gian tồn tại dài hơn
        output_stracks = [track for track in self.tracked_tracks if track.is_activated]

        return output_stracks # Trả ra danh sách các track đang theo dõi và đã được kích hoạt
            
        
def joint_tracks(track_list_a, track_list_b):
    # join 2 tracklist mà không bị duplicate track nào
    seen_track_ids = set()
    result = []

    for track in track_list_a + track_list_b:
        if track.track_id not in seen_track_ids:
            seen_track_ids.add(track.track_id)
            result.append(track)

    return result

def sub_tracks(track_list_a, track_list_b):
    # Trả về các track trong track_list_a nhưng loại bỏ những track có track_id trùng với các track trong track_list_b.
    tracks = {track.track_id: track for track in track_list_a}
    track_ids_b = {track.track_id for track in track_list_b}

    for track_id in track_ids_b:
        tracks.pop(track_id, None)

    return list(tracks.values())

def remove_duplicate_tracks(tracks_a, tracks_b):
    pairwise_distance = matching.iou_distance(tracks_a, tracks_b) # tinh iou distance
    matching_pairs = np.where(pairwise_distance < 0.15) # distance < 0.15 la match

    duplicates_a, duplicates_b = set(), set()
    for track_index_a, track_index_b in zip(*matching_pairs):
        time_a = tracks_a[track_index_a].frame_id - tracks_a[track_index_a].start_frame # thoi gian track song
        time_b = tracks_b[track_index_b].frame_id - tracks_b[track_index_b].start_frame
        if time_a > time_b: # track nao thoi gian song be hon thi coi la dup
            duplicates_b.add(track_index_b)
        else:
            duplicates_a.add(track_index_a)

    result_a = [
        track for index, track in enumerate(tracks_a) if index not in duplicates_a
    ] # giu lai track khong bi dup
    result_b = [
        track for index, track in enumerate(tracks_b) if index not in duplicates_b
    ]

    return result_a, result_b
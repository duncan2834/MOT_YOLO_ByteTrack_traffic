import numpy as np
import scipy.linalg

class KalmanFilter:
    # Kalman filter dành cho tracking bbox trong image space
    # 8D state space (x, y, a, h, vx, vy, va, vh)
    # bao gồm x, y là center bbox, a là aspect ratio, h là height, còn lại là các vận tốc tương ứng 
    
    # LƯU Ý: state space là bao gồm 8D, nma observation space(kgian quan sát) là (x, y, a, h) thôi, là vị trí bbox ấy
    def __init__(self):
        # khởi tạo các thành phần trong 8d state space, v ko đổi
        # thiết lập ma trận chuyển động, ma trận quan sát, và các hệ số nhiễu
        
        ndim, dt = 4, 1.0 # 4 là số chiều qsat được, dt là khoảng thời gian giữa 2 frame, giả định là 1
        
        # ma trận chuyển động F
        self._motion_mat = np.eye(2 * ndim, 2 * ndim) # taoj ma tran don vi 8x8
        for i in range(ndim):
            self._motion_mat[i][ndim + i] = dt
            
        """
        [[1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]]
        """
        # ma tran quan sat
        self._update_mat = np.eye(ndim, 2 * ndim) 
        # std
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        
    def initiate(self, measurement):
        # measurement : bbox (x, y, a, h) 
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos) # [0, 0, 0, 0]
        mean = np.r_[mean_pos, mean_vel]
        # độ lệch chuẩn của mỗi thành phần
        std = [
            2 * self._std_weight_position * measurement[3], # nhan voi h, do luong theo h
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        # ma trận hiệp phương sai 8x8, ma trận chéo, biểu thị độ không chắc chắn của trạng thái
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        # chay Kalman filter prediction
        # mean [x, y, a, h, vx, vy, va, vh]
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) # Q: ma trận nhiễu 
        mean = np.dot(mean, self._motion_mat.T) # [x_t+1] = x_t + vx_t
        # cập nhật ma trận hiệp phương sai theo công thức chuẩn kalman: P_t+1 = F·P·Fᵀ + Q
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance
    
    def project(self, mean, covariance):
        # chiếu về kgian quan sát(measurement space) để update
        # chuyen state space 8d ve observation space 4d
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std)) # sai so
        
        mean = np.dot(self._update_mat, mean) # 4x8 * 8x8 -> 4x8, chiếu mean 8 chiều về mean 4 chiều
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        ) # Chiếu ma trận hiệp phương sai P từ 8×8 về 4×4
        # P_meas = H · P_state · Hᵀ
        # H: measurement space
        # P_state: ma tran hiep phuong sai trang thai
        # P_meas: ma tran hiep phuong sai quan sat
        return mean, covariance + innovation_cov
    
    def multi_predict(self, mean, covariance):
        # predict cho nhieu doi tuong
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T # tu 8, N ve N, 8
        motion_cov = []
        for i in range(len(mean)): # N
            motion_cov.append(np.diag(sqr[i])) 
        motion_cov = np.asarray(motion_cov) # N, 8, 8
        
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2)) # F*P -> transpose về N, 8, 8
        covariance = np.dot(left, self._motion_mat.T) + motion_cov # F*P*F.T + Q

        return mean, covariance
    
    def update(self, mean, covariance, measurement):
        # kalman filter correction
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # Kalman Gain là trọng số cho biết nên tin vào đo thực (measurement) bao nhiêu so với dự đoán
        # Tinh kalman gain qua cholesky
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T # K = PH.T(HPH.T + R)-1
        innovation = measurement - projected_mean # y: sai so
        
        new_mean = mean + np.dot(innovation, kalman_gain.T) # x + Ky
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        ) # (I - KH)P
        return new_mean, new_covariance
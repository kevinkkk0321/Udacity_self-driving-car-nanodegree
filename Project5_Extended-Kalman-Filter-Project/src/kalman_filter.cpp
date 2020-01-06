#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * TODO: predict the state
   */
    x_ = F_ * x_; // + u;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
    VectorXd z_pred = H_ * x_;
    MatrixXd H_trans = H_.transpose();
    
    VectorXd y = z - z_pred;
    MatrixXd S = H_ * P_ * H_trans + R_;
    MatrixXd K = P_ * H_trans * S.inverse();
    
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
    float ro = sqrt(x_[0] * x_[0] + x_[1] * x_[1]);
    float phi = atan2(x_[1], x_[0]);
    float ro_dot;
    if (fabs(ro) < 0.0001) {
      ro_dot = 0;
    } else {
        ro_dot = (x_[0] * x_[2] + x_[1] * x_[3])/ro;
    }
    VectorXd z_pred(3);
    z_pred << ro, phi, ro_dot;
     
    //VectorXd z_pred = H_ * x_;
    MatrixXd H_trans = H_.transpose();
    
    VectorXd y = z - z_pred;
    //Normalize residual phi to be between -pi to pi
    y(1) = atan2(sin(y(1)), cos(y(1)));
    
    MatrixXd S = H_ * P_ * H_trans + R_;
    MatrixXd K = P_ * H_trans * S.inverse();
    
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

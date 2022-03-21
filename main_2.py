# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:32:47 2022

@author: Hariharan Natesh
"""

import numpy as np
from pr3_utils import *

from scipy.linalg import expm

class Visual_SLAM():
    
    def __init__(self,filename):
        
        self.t,self.features,self.linear_velocity,self.angular_velocity,self.K,self.b,self.imu_T_cam = load_data(filename)
        
        self.num = self.t.shape[1]
        self.T = np.zeros((4,4,self.num))
        self.T[:,:,0] = np.eye(4)
        
        feature = self.features[:,:,0]
        self.indices = np.arange(0,feature.shape[1]-1,10)
        self.M = len(self.indices)
        
        self.fsu = self.K[0,0]
        
        self.Ks = np.zeros((4,4))
        
        self.Ks[0:2,0:3] = self.K[0:2,0:3]
        self.Ks[2,0:3] = self.K[0,0:3]
        self.Ks[2,3] = -self.K[0,0]*self.b
        self.Ks[3,0:3] = self.K[1,0:3]
        
        self.P = np.zeros((3,4))
        
        self.P[0:3,0:3] = np.eye(3)
        
        self.cam_T_imu = np.linalg.inv(self.imu_T_cam)
    
    def homogenize(self,Y):
        
        ones = np.ones(Y.shape[1])
        homo = np.vstack((Y,ones))
        return homo

    def dehomogenize(self,Y):
        
        Y = Y/Y[-1,:]
        
        return Y[:-1,:]
    
    def skew_matrix(self,vector):
        
        skew_mat = np.zeros((3,3))
        
        skew_mat[0,1] = -vector[2,0]
        skew_mat[0,2] = vector[1,0]
        skew_mat[1,0] = vector[2,0]
        skew_mat[1,2] = -vector[0,0]
        skew_mat[2,0] = -vector[1,0]
        skew_mat[2,1] = vector[0,0]
        
        return skew_mat
        
    def twist_matrix(self,vector):
        
        omega = vector[3:6,0].reshape(-1,1)
        omega_hat = self.skew_matrix(omega)
        
        v = vector[0:3,0]
        u_hat = np.zeros((4,4))
        
        u_hat[0:3,0:3] =omega_hat
        u_hat[0:3,3] = v
        
        return u_hat
        
    
    
    def dead_reckoning(self):
        
        
        # print(t.shape)

        for i in range(1,self.num):
            
            tau = self.t[0,i] - self.t[0,i-1]
            
            general_vel = np.vstack((self.linear_velocity[:,i-1].reshape(-1,1),self.angular_velocity[:,i-1].reshape(-1,1)))
            twist_mat = self.twist_matrix(general_vel)
            self.T[:,:,i] = self.T[:,:,i-1] @ expm(tau*twist_mat)
        return self.T
        # visualize_trajectory_2d(T,'Trajectory 10 Dead reckoning')
        
    
    def prior(self):
        
        feature = self.features[:,:,0]
        
              

        feature = feature[:,self.indices]
        valid_indices = np.argwhere(feature[0,:]!=-1)
        
        # print(valid_indices)
    
        left_features = feature[0:2,:]
        right_features = feature[2:4,:]
        left_feat_valid = left_features[left_features!=-1].reshape(2,-1)
        right_feat_valid = right_features[right_features!=-1].reshape(2,-1)
        
        N_t = left_feat_valid.shape[1]
        
        V = np.eye(left_feat_valid.shape[0]*N_t)*0.01
        
        v_t = (np.random.multivariate_normal([0]*2*N_t,V)).reshape(2,N_t)
        
        z_t = left_feat_valid - v_t
        
        z_t_homo = self.homogenize(z_t)
        
        pi_coord = np.linalg.inv(self.K) @ z_t_homo
        
        depth = self.fsu*self.b/(left_feat_valid[0,:]-right_feat_valid[0,:])
        
        depth = depth.reshape(-1,N_t)
        
        cam_coord = pi_coord * depth 
        
        cam_coord_homo = self.homogenize(cam_coord)
        
        world_coord_homo = self.T[:,:,0] @ self.imu_T_cam @ cam_coord_homo
        
        world_coord = self.dehomogenize(world_coord_homo)
        
        
        
        mu_t = np.zeros((3,self.M))
        mu_t[:,valid_indices[:,0]] = world_coord
        
        sigma_t = np.eye(3*self.M)*0.01
        
        
        return mu_t,sigma_t,N_t,valid_indices 
    
            
    def mapping(self,i):
        
        
        if i==1:
            self.mu_t,self.sigma_t,N_t,self.valid_prior = self.prior()
            self.seen_set = set(self.valid_prior[:,0])
            # print("seen set at i=1",self.seen_set)
   
        
        
        T_inv = np.linalg.inv(self.T[:,:,i])
        
        feature = self.features[:,:,i]
        
        
        self.feature = feature[:,self.indices]
        valid_indices = np.argwhere(self.feature[0,:]!=-1)
        
        new_valid_set = set(valid_indices[:,0])
        # print("Valid indices",valid_indices)
        difference = new_valid_set.difference(self.seen_set)
        
        self.seen_set = set.union(self.seen_set, new_valid_set)
    
    
        left_features = self.feature[0:2,:]
        right_features = self.feature[2:4,:]
        left_feat_valid = left_features[left_features!=-1].reshape(2,-1)
        right_feat_valid = right_features[right_features!=-1].reshape(2,-1)
        
        N_t_next = left_feat_valid.shape[1]
        
        print(i,N_t_next)
        if N_t_next==0:
            return None,None
        V = np.eye(left_feat_valid.shape[0]*N_t_next)*0.1
        
        V_I = np.eye(4*N_t_next)*0.1
        
        v_t = (np.random.multivariate_normal([0]*2*N_t_next,V)).reshape(2,N_t_next)
        
        z_t = left_feat_valid - v_t
        
        z_t_homo = self.homogenize(z_t)
        
        pi_coord = np.linalg.inv(self.K) @ z_t_homo
        
        depth = self.fsu*self.b/(left_feat_valid[0,:]-right_feat_valid[0,:])
        
        depth = depth.reshape(-1,N_t_next)
        
        cam_coord = pi_coord * depth 
        
        cam_coord_homo = self.homogenize(cam_coord)
        
        world_coord_homo = self.T[:,:,i] @ self.imu_T_cam @ cam_coord_homo
        
        world_coord = self.dehomogenize(world_coord_homo)
        # print(world_coord)
        
        H_t_next = np.zeros((4*N_t_next,3*self.M))
        
        mj_homo = self.homogenize(self.mu_t)
            
        ans = self.cam_T_imu @ T_inv @ mj_homo
        for ind in range(N_t_next):
            
            v_ind = valid_indices[ind,0]
            
    
            dpi_dq = np.eye(4)
            dpi_dq[:,2] = -ans[0,v_ind]/ans[2,v_ind],-ans[1,v_ind]/ans[2,v_ind],0,-ans[3,v_ind]/ans[2,v_ind]
            dpi_dq[2,2] = 0
            
            dpi_dq*=(1/ans[2,v_ind])
            # print(dpi_dq)
            H_t_next[4*ind:4*ind+4,3*ind:3*ind+3] = self.Ks @ dpi_dq @ self.cam_T_imu @ T_inv @ self.P.T 
            
        # print(H_t_next)
        if len(difference)!=0:
            # print(i,difference)
            diff_indices = sorted(list(difference))
            print("Diff",diff_indices)
            self.mu_t[:,diff_indices] = world_coord[:,-len(difference):]
            return valid_indices,N_t_next
        
        
        
        K_t_next = self.sigma_t @ H_t_next.T @ np.linalg.inv(H_t_next @ self.sigma_t @ H_t_next.T + V_I)
        # print("K_t_next",K_t_next)
        ans_z_bar = self.cam_T_imu @ T_inv @ self.homogenize(self.mu_t[:,valid_indices[:,0]])
        ans_z_bar = (1/ans_z_bar[2,:]) * ans_z_bar
        ans_z_bar = self.Ks @ ans_z_bar
        mu_t_next = self.mu_t
        
        # print("difference",(feature[:,valid_indices[:,0]] - ans_z_bar ))
        innov_K = K_t_next @ ((self.feature[:,valid_indices[:,0]] - ans_z_bar )).reshape(-1,1,order='F')
        mu_t_next[:,valid_indices[:,0]] = self.mu_t[:,valid_indices[:,0]] + innov_K.reshape(3,-1,order='F')[:,:N_t_next]
        
        # print("mu_t_next",mu_t_next)
        sigma_t_next = (np.eye(3*self.M) - K_t_next @ H_t_next) @ self.sigma_t 
        
        self.mu_t = mu_t_next
        self.sigma_t = sigma_t_next
        
        return valid_indices,N_t_next
    
        # print((np.where(mu_t[0]!=0)[0]))
        # np.save('mu_t_10_mapping.npy',mu_t)
        
    
    def mapping_only(self): 
        
        for i in range(1,self.num):
            
            
            tau = self.t[0,i] - self.t[0,i-1]
            
            general_vel = np.vstack((self.linear_velocity[:,i-1].reshape(-1,1),self.angular_velocity[:,i-1].reshape(-1,1)))
            twist_mat = self.twist_matrix(general_vel)
            self.T[:,:,i] = self.T[:,:,i-1] @ expm(tau*twist_mat)
            
            T_inv = np.linalg.inv(self.T[:,:,i])
            
            omega_hat = self.skew_matrix(self.angular_velocity[:,i-1].reshape(-1,1))
            vel_hat = self.skew_matrix(self.linear_velocity[:,i-1].reshape(-1,1)) 
            
            
            valid_indices,N_t_next = self.mapping(i)
            
            if N_t_next == None:
                continue
        
    
    def prediction(self):
        

        
        sigma_pred_t = np.eye(6)*0.1
        
        W = np.diag([0.3,0.3,0.3,0.01,0.01,0.01])
            
        for i in range(1,self.num):
            
            
            tau = self.t[0,i] - self.t[0,i-1]
            
            general_vel = np.vstack((self.linear_velocity[:,i-1].reshape(-1,1),self.angular_velocity[:,i-1].reshape(-1,1)))
            twist_mat = self.twist_matrix(general_vel)
            self.T[:,:,i] = self.T[:,:,i-1] @ expm(tau*twist_mat)
            
            T_inv = np.linalg.inv(self.T[:,:,i])
            
            omega_hat = self.skew_matrix(self.angular_velocity[:,i-1].reshape(-1,1))
            vel_hat = self.skew_matrix(self.linear_velocity[:,i-1].reshape(-1,1)) 
            
            u_cap = np.zeros((6,6))
            u_cap[0:3,0:3] = omega_hat
            u_cap[0:3,3:6] = vel_hat 
            u_cap[3:6,3:6] = omega_hat 
            
            sigma_pred_next = expm(-tau*u_cap) @ sigma_pred_t @ expm(-tau*u_cap).T + W
            
            valid_indices,N_t_next = self.mapping(i)
            
            if N_t_next == None:
                continue
        
            # N_t_next,valid_indices = self.feature_extract(i)
            # if N_t_next==0:
            #     continue
            mj_homo = self.homogenize(self.mu_t)
            
            inside_pi = self.cam_T_imu @ T_inv @ mj_homo 
            
            pi_val = inside_pi/inside_pi[2,:]
            
            zbar_t_next = self.Ks @ pi_val 
            
            H_t_next = np.zeros((4*N_t_next,6))
            
            ans =  self.cam_T_imu @ T_inv @ mj_homo
            
            # print(ans.shape)
            # print(ans[2,1])
            for ind in range(N_t_next):
                
                v_ind = valid_indices[ind,0]
                # print("v_ind",v_ind)
                # print(ans[2,1])
                mu_inv_mj = T_inv @ mj_homo[:,v_ind]
                
                dpi_dq = np.eye(4)
                
                dpi_dq[:,2] = -ans[0,v_ind]/ans[2,v_ind],-ans[1,v_ind]/ans[2,v_ind],0,-ans[3,v_ind]/ans[2,v_ind]
                dpi_dq[2,2] = 0
                
                dpi_dq*=(1/ans[2,v_ind])

                
                circle_dot = np.zeros((4,6))
                
                circle_dot[0:3,0:3] = np.eye(3)
                circle_dot[0:3,3:6] = -self.skew_matrix((mu_inv_mj).reshape(-1,1))
                
                H_t_next[4*ind:4*ind+4,:] = -self.Ks @ dpi_dq @ self.cam_T_imu @ circle_dot
                
            V_I = np.eye(4*N_t_next)*0.01
            K_t_next = sigma_pred_next @ H_t_next.T @ np.linalg.pinv(H_t_next @ sigma_pred_next @ H_t_next.T + V_I)
            
            # print(K_t_next.shape)
            # print(self.feature[:,valid_indices[:,0]].shape)
            # print(zbar_t_next[:,valid_indices[:,0]].shape)
            inside_expm = K_t_next @ (self.feature[:,valid_indices[:,0]] - zbar_t_next[:,valid_indices[:,0]]).reshape(-1,1,order='F')
            self.T[:,:,i] = self.T[:,:,i] @ expm(self.twist_matrix(inside_expm))
            
            sigma_pred_t = (np.eye(6) - K_t_next @ H_t_next) @ sigma_pred_next
            
            
            
            
                
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        

if __name__ == '__main__':

    # Load the measurements
    filename = "./data/03.npz"
    # t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
    
    vslam = Visual_SLAM(filename)

    # (a) IMU Localization via EKF Prediction

    # (b) Landmark Mapping via EKF Update

    # (c) Visual-Inertial SLAM

    # You can use the function below to visualize the robot pose over time
    # visualize_trajectory_2d(world_T_imu, show_ori = True)
    
    # vslam.dead_reckoning()
    vslam.prediction()
    # vslam.mapping_only()
    visualize_trajectory_2d(vslam.T,vslam.mu_t,'VSLAM for 03')
    
    
    

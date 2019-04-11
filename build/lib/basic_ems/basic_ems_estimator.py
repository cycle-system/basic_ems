"""
Module that condense all the components of the EMS tested above
"""

# General Imports

import numpy as np
import numpy.matlib as matlib
from math import *
from decimal import *
from scipy import linalg

# Sklearn imports

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import inspect

# Method for implementation of the UKF estimator based on simple model for the batetry bank

# Negative Current is for Discharging
# Positive current is for Charging

# State Equations 
#  SoC                              SoC(k+1)=SoC(k)+((eff*T)/C)*I(k) + Noise
#  Resistor for discharge           R-(k+1) = R-(k) + Noise
#  Resistor for charge              R+(k+1) = R+(k) + Noise

# Observation equation
#  Voltage            =  Voc (Soc)  +  sig*R-*I  +   (1-sig)*R+*I;

def socEstimation(p_soc,v_bat,i_prev,i_now,t_sampling,e_acum_in,states_in,rxx_in,c_n,f) :
    
    # Initialization
    
    # Local vars
    e_acum = e_acum_in;
    
    # Output vars
    states_out = None;
    rxx_out = None;
    err = None;
    e_acum_out = None;
    
    ##-------------------- Estimator turns off and it returns the previous values-----------------% 
    
    # Condition disabled, originally abs(i_now) <= 0.1
    if False :  
        
        print('Current too small to be considered.');
        
        states_out = states_in;
        rxx_out = rxx_in;
        err = 0; 
        e_acum_out = e_acum_in;
        
        print('Returning values by default.');
        
        return states_out, rxx_out, err, e_acum_out;
    
    #-------------------------------------- Turns on the filter-----------------------------------%
    #---------------------------------- Efficiency initialization---------------------------------%
    
    # Efficiency = 1 for discharge; Less than one for charge
    
    if i_prev <= 0 :     
        n_0 = 1;
    else :
        n_0 = 0.95;
    
    #---------------------------------States and noise initialization-----------------------------%
    
    #  States Initialization
    x_med = [states_in];
    
    #  Noise Process Initialization 
    rxx = [rxx_in];                                
    
    #------------------------Definition of the Process state Equations----------------------------% 
    
    # First coeff of each equation  
    a = np.eye(p_soc.n,p_soc.n);
    # Capacity in Amperes-seconds
    b = [((t_sampling*n_0)/(3600*c_n)), 0, 0];
    
    #------------------------------------- UKF Algorithm------------------------------------------% 
    #---------------------------------- OFCL implementation---------------------------------------% 
    #----- The process and observation noises are update according with acumlative error (e_acum)
    #----- See equation in document and flowchart
    
    
    # Enhanced OFCL
    
    if  e_acum <= p_soc.e_th :
        
        g_std = [max(p_soc.p1*np.sqrt(p_soc.rww[0,0], dtype=np.float128),1.2e-15),
                 max(p_soc.p2*np.sqrt(p_soc.rww[1,1], dtype=np.float128),1e-10),
                 max(p_soc.p2*np.sqrt(p_soc.rww[2,2], dtype=np.float128),1e-9)];

        p_soc.rww = np.array(np.diag([g_std[0]**2, g_std[1]**2, g_std[2]**2]), dtype=np.float128);

    else :

        g_std = [min(p_soc.q1*np.sqrt(p_soc.rww[0,0], dtype=np.float128),1.2e-3),
                 min(p_soc.q2*np.sqrt(p_soc.rww[1,1], dtype=np.float128),1e-1),
                 min(p_soc.q2*np.sqrt(p_soc.rww[2,2], dtype=np.float128),1e-1)];

        p_soc.rww = np.array(np.diag([g_std[0]**2, g_std[1]**2, g_std[2]**2]), dtype=np.float128);
    
    #---------------------Calculate the sigmas point for the states--------------------------------% 
    
    try :
        
        sxx = linalg.cholesky(rxx[0], lower=True, overwrite_a=False, check_finite=True);
        sxx = [[Decimal(x) for x in y] for y in sxx];
        sxx = np.array([[np.float128(x) for x in y] for y in sxx]);
        
    except Exception as e:
        
        print('Cholesky decomposition failed due to: '+ str(e));
        print('Returning values by default.');
        
        states_out = states_in;
        rxx_out = rxx_in;
        err = 0; 
        e_acum_out = e_acum_in;
        
        return states_out, rxx_out, err, e_acum_out;
    
    # Concatenate Cholesky factors
    
    x_ = np.concatenate(([np.zeros(len(x_med[0]))], sxx, -sxx), axis=0).transpose();
    
    # Multiply Gamma with Cholesky factors
    x_ =  p_soc.gamma*x_;
    
    # Calculation of Sigma points
    x_0 = matlib.repmat(np.array(x_med[0]),x_.shape[1],1).transpose();
    
    x_ = np.add(x_,x_0, dtype=np.float128);
    
    #-----------------------------States Prediction------------------------------------------------% 
    
    x = np.add(a.dot(x_),matlib.repmat(b,x_.shape[1],1).transpose()*i_prev, dtype=np.float128);
    x_med.append(np.sum(matlib.repmat(p_soc.wm,p_soc.n,1)*x,axis=1, dtype=np.float128));
    
    #------------------------Error Prediciton  of the States and covariance------------------------%
    
    x_e = np.subtract(x,matlib.repmat(x_med[1],x_.shape[1],1).transpose(), dtype=np.float128);
    rxx.append(np.add(p_soc.wm[0]*(np.array([x_e[:,0]])*np.array([x_e[:,0]]).transpose()),p_soc.rww, dtype=np.float128));
    
    index = 1;
    
    while(index < 2*p_soc.n+1) :
        rxx[1] = np.add(p_soc.wm[index]*(np.array([x_e[:,index]])*np.array([x_e[:,index]]).transpose()), rxx[1], dtype=np.float128);
        index += 1;
    
    #---------------------Calculate the sigmas point for the Observations--------------------------%
    
    try :
        
        sxx = linalg.cholesky(rxx[1], lower=True, overwrite_a=False, check_finite=True);
        sxx = [[Decimal(x) for x in y] for y in sxx];
        sxx = np.array([[np.float128(x) for x in y] for y in sxx], dtype=np.float128);
        
    except Exception as e:
        
        print('Cholesky decomposition failed due to: '+ str(e));
        print('Returning values by default.');
        
        states_out = states_in;
        rxx_out = rxx_in;
        err = 0; 
        e_acum_out = e_acum_in;
        
        return states_out, rxx_out, err, e_acum_out;
    
    # Concatenate Cholesky factors
    x_ = np.concatenate(([np.zeros(len(x_med[1]))], sxx, -sxx), axis=0).transpose();
    
    # Multiply Gamma with Cholesky factors
    x_ =  p_soc.gamma*x_;
    
    # Calculation of Sigma points
    x_0 = matlib.repmat(np.array(x_med[1]),x_.shape[1],1).transpose();
    
    x_ = np.add(x_,x_0, dtype=np.float128);
    
    #--------------------------------Observation Prediction ---------------------------------------%
    #----------VoC vs SoC is calculate at the laboratory method------------------------------------%
       
    voc = (3.755*np.power(x_[0],3, dtype=np.float128) - 5.059*np.power(x_[0],2, dtype=np.float128) + 3.959*x_[0] + 17.064)*f;
       
    # Identify the sign according to the current
    
    if i_now <= 0 :
        sig=1;        
    else :
        sig=0;        
    
    # Calculate the voltage in the battery according to the simple model (to upgrade)
    
    v_model =  np.array([voc + sig*x_[1]*i_now + (1-sig)*x_[2]*i_now], dtype=np.float128);

    # Average output
    
    v_avg = np.array(p_soc.wm, dtype=np.float128).dot(v_model.transpose());
            
    #--------------------- Residual prediction (Residuos de la predicción)-----------------------%
    
    v_e = np.subtract(v_model,matlib.repmat(v_avg,x_.shape[1],1).transpose(), dtype=np.float128);
    
    ryy = np.add(p_soc.wc[0]*(np.array([v_e[:,0]])*np.array([v_e[:,0]]).transpose()),p_soc.rvv, dtype=np.float128);
    
    index = 1;
    
    while(index < 2*p_soc.n+1) :
        
        ryy = np.add(p_soc.wc[index]*(np.array([v_e[:,index]])*np.array([v_e[:,index]]).transpose()),ryy, dtype=np.float128);
        index += 1;
    
    #------------------------------------Gain---------------------------------------------------%
    
    rxy =  p_soc.wc[0]*(np.array([x_e[:,0]])*np.array([v_e[:,0]]).transpose());
    
    index = 1;
    
    while(index < 2*p_soc.n+1) :
        rxy = np.add(p_soc.wc[index]*(np.array([x_e[:,index]])*np.array([v_e[:,index]]).transpose()),rxy, dtype=np.float128);
        index += 1;
    
    k = np.divide(rxy,ryy, dtype=np.float128); 
    
    #--------------------------------------Update-----------------------------------------------%
    
    # Calculate the error
    e = np.subtract(v_bat,v_avg, dtype=np.float128);
    
    x_med[1] =  np.add(x_med[1], k[0]*e, dtype=np.float128);
    x_med[1][0] = min(x_med[1][0],1); 
    x_med[1][0] = max(x_med[1][0],0.001); 
    x_med[1][1] =  max(x_med[1][1],0.001); 
    x_med[1][2] =  max(x_med[1][2],0.001);
    
    # Update the covariance of rxx
    
    rxx[1] =  np.subtract(rxx[1],np.array(k)*ryy*np.array(k).transpose(), dtype=np.float128); 
    
    # Update output variables
    
    states_out = x_med[1];
    rxx_out = rxx[1];
    err = e; 
    e_acum_out = e_acum;
    
    # Return the updated states
    
    return states_out, rxx_out, err, e_acum_out;

# Method for calculating the maximum power for charging and discharging                                                                                                                   
# based on paper High-Performance Battery-Pack Power Estimation Using a Dynamic Cell Model
# by Greogory L. Plet
#
# This script is based on the simple model for battery bank 
# Combine the SoC and Voltage Methods

def getMaximumPower (states_in,soc_max,soc_min,c_n,i_max,v_max,v_min,p_max,f):

    # SoC 
    soc = states_in[0];     

    # Discharging Resistor 0.001
    r_dis = states_in[1];  

    # Charging Resistor
    r_ch = states_in[2];               

    # Relation VoC vs SoC obtained at laboratory process
    # f is the parameter to ajust the VoC vs Soc to the number of batteries connected in serie
    # according with the experiment at the lab for obtained the relation Voc and SoC 

    voc = (3.755*np.power(soc,3) - 5.059*np.power(soc,2) + 3.959*soc + 17.064)*f;

    # Derivate of the dVOC/Soc
    d_voc = (3*3.755*np.power(soc,2) - 2*5.059*soc + 3.959)*f;            

    # Capacity in Ampere-second
    c_n = 3600*c_n;                                     

    # Efficiency for charging
    eff = 0.95;                                            

    # Interval Time
    dt_vol =  0.001*60;                   

    dt_soc_dis = (36*60);                    

    dt_soc_ch = (40*60); 

    #For avoiding numerical error
    
    if soc >= soc_max:
        soc = soc_max;

    if soc <= soc_min:
        soc = soc_min;
            
    # Maximum current allowed for discharging according Equation (9) 
    # Voltage Method 

    i_max_dis_vol = (v_min - voc)/(r_dis + ((dt_vol/c_n)*d_voc));     
    
    # Maximum current allowed for charging according Equation (8) 
    # Voltage Method                       

    i_max_ch_vol = (v_max - voc)/(r_ch + (((eff*dt_vol)/c_n)*d_voc));         

    # Maximum current allowed for discharging according Equation (5) 
    # SoC Method                                                                       
    
    i_max_dis_soc = -(soc - soc_min)/(dt_soc_dis/c_n);      
    
    # Maximum current allowed for charging according Equation (6)  
    # SoC Method

    i_max_ch_soc = -(soc - soc_max)/((dt_soc_ch*eff)/c_n);              

    # Combination of methods 
    # Equations (10) and (11)
    # min in equation (10)is modified for max due the current sign 

    i_max_dis = max(-i_max, max(i_max_dis_soc, i_max_dis_vol));                

    i_max_ch = max(0, max(i_max_ch_soc, i_max_ch_vol));

    if soc > soc_max:
        i_max_ch = 0;

    if soc < soc_min:
        i_max_dis =0;
    
    # Predicted future Voltage Equations (12) and (13)

    v_b_dis = voc + ((dt_vol/c_n)*d_voc)*i_max_dis + r_dis * i_max_dis; 

    v_b_ch = voc + (((dt_vol*eff)/c_n)*d_voc)*i_max_ch + r_ch * i_max_ch;

    #  Maximum Power allowed for discharging with Predicted future voltage
    
    p_max_dis = min(p_max,-v_b_dis* i_max_dis);                               

    # Maximum Power allowed for charging with Predicted future voltage
    
    p_max_ch = min(p_max, v_b_ch*i_max_ch);                                
    
    return p_max_ch, p_max_dis

# Basic control based on rules - excluding PmaxCh and PmaxDisch for the first version
# p_pv - Photovoltaic Power
# p_load - Load Power
# p_grid - Max power to be bought to the Main Grid
# p_max - Power limit provided by the inverter/charger

def getBatteryPowerReference(p_pv,p_load,p_grid,soc,soc_max,soc_min,p_max_ch,p_max_dis,p_max):
    
    # Negative Power indicates Charging Process
    # Positive Power indicates Discharging Process
    # Net Power = Load Power - PV Power - Battery Power;

    error = p_load - p_pv - p_grid;

    # Point to check the available power (see in document Rules-based controller flowchart)
    if error <= 0 :
        
        # Check the maximum SoC
        if soc >= soc_max :
            pb_ref = 0;
        # Waiting (stop working battery bank)
        else :
            pb_ref =-min(abs(error),p_max_ch);
    
    # The reference power plus PV is not enough to supply demand.
    else :
        
        # Check the minimun Soc for use the battery bank
        if soc >= soc_min :
            pb_ref= min(error,p_max_dis);
        # The SoC is less than SoC min and it is necessary charging
        else :
            pb_ref =max(-p_pv,-p_max);
            
    return pb_ref;

class basicEmsEstimator(BaseEstimator, RegressorMixin):
    
    """ 
    A estimator based on rules wich contains the EMS configurations
    to administer a photovoltaic system with batteries using criteria of
    SoC, PmaxCh and PmaxDis.
    
    Parameters
    ----------
    p_soc : socParams, default='None' - Description to write
    p_bat : batteryParams, default='None' - Description to write
    states_in : list, default='None' - Description to write
    rxx_in : list, default='None' - Description to write
    e_acum_in = float, default='None' - Description to write
    soc_max : float, default='None' - Description to write
    soc_min : float, default='None' - Description to write
    t_sampling : int, default='None' - Description to write
    c_n : float, default='None' - Description to write
    i_max : float, default='None' - Description to write
    v_max : float, default='None' - Description to write
    v_min : float, default='None' - Description to write
    p_max : float, default='None' - Description to write
    f : int, default='None' - Description to write
    """
    
    def __init__(self,p_soc = None, p_bat = None, states_in = np.array([0.2,1,1]),
                 rxx_in = np.diag(np.array([1.2e-15, 1.2e-10, 1.2e-10])),
                 e_acum_in = 0, soc_max = 0.8, soc_min = 0.2, t_sampling = 60,
                 c_n = 0, i_max = 0, v_max = 0, v_min = 0, p_max = 0, f = 0):
        
        # Gather all the passed parameters
        args, _, _,values = inspect.getargvalues(inspect.currentframe())
        
        # Remove parameter 'self' 
        values.pop("self")

        # Init the attributes of the class
        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(self, X, y = None):
        
        """
        Implementation of a fitting function. This section could help when needed to
        tune the rww parameter of SoC or other necessary.
        
        Features per row
        ----------
        X[0] -> v_bat = Present measure of voltage in the battery terminals
        X[1] -> i_prev = Last measure of current
        X[2] -> i_now = Present measure of current
        X[3] -> pv = Present measure of photovoltaic power
        X[4] -> load = Present measure of load consumption
        X[5] -> reference = Present reference of power from the Utility
        
        Outputs
        ----------
        y[0] -> soc = Estimation of State of Charge for the next timeslot
        y[1] -> r_ch = Estimation of Resistance of Charge for the next timeslot
        y[2] -> r_dis = Estimation of Resistance of Discharge for the next timeslot
        y[3] -> err = Computation of observed error
        y[4] -> e_acum_out = Acummulated error in the ouput
        y[5] -> p_max_ch = Estimation of Maximum Power of Charge for the next timeslot
        y[6] -> p_max_dis = Estimation of Maximum Power of Discharge for the next timeslot
        y[7] -> p_b_ref = Prediction of reference power for batteries
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
            
        Returns
        -------
        self : object
            Returns self.
        """
        
        # Check input and target vectors correctness
        # ToDo
        
        # Configure parameters when fitted
        
        self.is_fitted_ = True
        
        # `fit` should always return `self`
        return self

    def predict(self, X, y = None):
        
        """
        Implementation of a predicting function. With input features estimate
        the value of reference power for batteries.
        
        Features per row
        ----------
        X[0] -> v_bat = Present measure of voltage in the battery terminals
        X[1] -> i_prev = Last measure of current
        X[2] -> i_now = Present measure of current
        X[3] -> p_pv = Present measure of photovoltaic power
        X[4] -> p_load = Present measure of load consumption
        X[5] -> p_ref = Present reference of power from the Utility
        
        Outputs
        ----------
        y[0] -> soc = Estimation of State of Charge for the next timeslot
        y[1] -> r_ch = Estimation of Resistance of Charge for the next timeslot
        y[2] -> r_dis = Estimation of Resistance of Discharge for the next timeslot
        y[3] -> err = Computation of observed error
        y[4] -> e_acum_out = Acummulated error in the ouput
        y[5] -> p_max_ch = Estimation of Maximum Power of Charge for the next timeslot
        y[6] -> p_max_dis = Estimation of Maximum Power of Discharge for the next timeslot
        y[7] -> p_b_ref = Prediction of reference power for batteries
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
            
        Returns
        -------
        y : ndarray, shape (n_samples, n_outputs)
            Returns an array of predictions
        """
        
        # Define the Output vector
        
        y = [];
        
        # Verification before prediction
        
        check_is_fitted(self, 'is_fitted_')
        
        # Perform the prediction over the input array
        
        for x in X :
            
            # Define a vector of results
            
            y_reg = [];
            
            # Extract the values of the input array
            
            v_bat = x[0];
            i_prev = x[1];
            i_now = x[2];
            p_pv = x[3];
            p_load = x[4];
            p_ref = x[5];
            
            # Perform estimations
        
            # SoC Estimation
        
            states_out, rxx_out, err, e_acum_out = self.__computeSocEstimation(v_bat,i_prev,i_now);
        
            # Maximum power Estimation
        
            p_max_ch, p_max_dis = self.__computeMaximumPower();
        
            # Maximum power Estimation
        
            p_b_ref = self.__computeBatteryPowerReference(p_pv,p_load,p_ref,p_max_ch, p_max_dis);
        
            # Update estimator parameters
        
            self.states_in = states_out;
            self.rxx_in = rxx_out;
            self.e_acum_in = e_acum_out*(1-1/(2**3)) + abs(err)*(1/(2**3));
            
            # Append results to output vector
            
            y_reg.append(states_out[0]);
            y_reg.append(states_out[1]);
            y_reg.append(states_out[2]);
            y_reg.append(err);
            y_reg.append(e_acum_out);
            y_reg.append(p_max_ch);
            y_reg.append(p_max_dis);
            y_reg.append(p_b_ref);
            
            # Append sub result to output vector
            
            y.append(y_reg);
        
        # Return the vector of predictions
        
        return y;
    
    """
    --------------------------------------------------------------------------------------------------
    Definition of specific EMS methods
    --------------------------------------------------------------------------------------------------
    """  
    
    def __computeSocEstimation(self,v_bat,i_prev,i_now) : 
        return socEstimation(self.p_soc,v_bat,i_prev,i_now,self.t_sampling,self.e_acum_in,self.states_in,self.rxx_in,self.c_n,self.f);
    
    def __computeMaximumPower(self): 
        return getMaximumPower(self.states_in,self.soc_max,self.soc_min,self.c_n,self.i_max,self.v_max,self.v_min,self.p_max,self.f);
        
    def __computeBatteryPowerReference(self,p_pv,p_load,p_ref,p_max_ch,p_max_dis): 
        return getBatteryPowerReference(p_pv,p_load,p_ref,self.states_in[0],self.soc_max,self.soc_min,p_max_ch,p_max_dis,self.p_max);

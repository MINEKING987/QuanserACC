# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports


import os
import random
import signal
import numpy as np
from threading import Thread
import time
import cv2
import pyqtgraph as pg
import math
import image_interpretation
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.products.qcar import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images
import threading


tf = 100
startDelay = 1
controllerUpdateRate = 100
vel_ref = 0.5
K_p = 0.1
K_i = 1
k_d = 0
K_stanley = 1
nodeSequence = [10,4,20,10]

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Initial setup

roadmap = SDCSRoadMap(leftHandTraffic=False,useSmallMap=False)
waypointSequence = roadmap.generate_path(nodeSequence)
x_offset = 0.13
y_offset = 1.67
initialPose = [x_offset, y_offset, 0.001]

if not IS_PHYSICAL_QCAR:
    exec(open("Setup_Competition.py").read())
    pass

global KILL_THREAD
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)
#endregion

class SpeedController:

    def __init__(self, kp=0, ki=0,kd = 0):
        self.maxThrottle = 0.3

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ei = 0
        self.prev_err = 0


    # ==============  SECTION A -  Speed Control  ====================
    def update(self, v, v_ref, dt):

        e = v_ref - v
        self.ei += dt*e
        if dt == 0:
            de = 0
        else:
            de = (e - self.prev_err)/dt
        self.prev_err = e

        return np.clip(
            self.kp*e + self.ki*self.ei+self.kd*de,
            -self.maxThrottle,
            self.maxThrottle
        )
        
        return 0

class SteeringController:

    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi/6

        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0

        self.k = k
        self.cyclic = cyclic

        self.p_ref = (0, 0)
        self.th_ref = 0

    # ==============  SECTION B -  Steering Control  ====================
    def update(self, p, th, speed):
        wp_1 = self.wp[:, np.mod(self.wpi, self.N-1)]
        wp_2 = self.wp[:, np.mod(self.wpi+1, self.N-1)]
        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        try:
            v_uv = v / v_mag
        except ZeroDivisionError:
            return 0

        tangent = np.arctan2(v_uv[1], v_uv[0])

        s = np.dot(p-wp_1, v_uv)

        if s >= v_mag:
            if  self.cyclic or self.wpi < self.N-2:
                self.wpi += 1

        ep = wp_1 + v_uv*s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent-th)

        self.p_ref = ep
        self.th_ref = tangent

        return [np.clip(
            wrap_to_pi(psi + np.arctan2(self.k*ect, speed)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle),self.wpi]
    

def is_turning(waypoints, turn_threshold=np.pi/720):
    if len(waypoints) < 3:
        raise ValueError("Need at least 3 waypoints to determine a turn."+str(waypoints))

    first_direction = waypoints[1, :] - waypoints[0, :]  # Direction vector from 1st to 2nd point
    last_direction = waypoints[-1, :] - waypoints[-2, :]  # Direction vector from last to second last point

    # Check if magnitudes are close to zero (straight line segments)
    if np.linalg.norm(first_direction) < 1e-6 and np.linalg.norm(last_direction) < 1e-6:
        return False  # Likely a straight path

    angle_diff = np.arccos(np.dot(first_direction / np.linalg.norm(first_direction), 
                                last_direction / np.linalg.norm(last_direction)))
    return angle_diff > turn_threshold

def check_red_thread(imageproc, stop_variable):
    while True:
        stop_variable[0] = imageproc.check_red()

def controlLoop():
    #region controlLoop setup
    imageproc = image_main()
    global KILL_THREAD
    u = 0
    delta = 0
    # used to limit data sampling to 10hz
    countMax = controllerUpdateRate / 10
    count = 0
    #endregion

    #region Controller initialization
    speedController = SpeedController(
        kp=K_p,
        ki=K_i,
        kd=k_d
    )
    steeringController = SteeringController(
        waypoints=waypointSequence,
        k=K_stanley
    )
    #endregion

    #region QCar interface setup
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    ekf = QCarEKF(x_0=initialPose)
    gps = QCarGPS(initialPose=initialPose)
    #endregion

    with qcar, gps:
        t0 = time.time()
        t=0
        freq_div = 0
        wpi = 0
        stop = [False]
        N = len(waypointSequence[0, :])
        # Start the continuous stop thread
        stop_thread = threading.Thread(target=check_red_thread, args=(imageproc, stop))
        stop_thread.daemon = True  # Daemonize the thread so it automatically stops when the main thread exits
        stop_thread.start()
        while (t < tf+startDelay) and (not KILL_THREAD):
            #region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t-tp
            #endregion

            #region : Read from sensors and update state estimates
            qcar.read()
            if gps.readGPS():
                y_gps = np.array([
                    gps.position[0],
                    gps.position[1],
                    gps.orientation[2]
                ])
                ekf.update(
                    [qcar.motorTach, delta],
                    dt,
                    y_gps,
                    qcar.gyroscope[2],
                )
            else:
                ekf.update(
                    [qcar.motorTach, delta],
                    dt,
                    None,
                    qcar.gyroscope[2],
                )

            x = ekf.x_hat[0,0]
            y = ekf.x_hat[1,0]
            th = ekf.x_hat[2,0]
            p = ( np.array([x, y])
                + np.array([np.cos(th), np.sin(th)]) * 0.2)
            v = qcar.motorTach
            #endregion

            #region : Update controllers and write to car
            if t < startDelay:
                u = 0
                delta = 0
            else:
                #region : Speed controller update
                wp_1 = waypointSequence[:, np.mod(wpi-4, N-1)]
                wp_2 = waypointSequence[:, np.mod(wpi-1, N-1)]
                wp_3 = waypointSequence[:, np.mod(wpi+2, N-1)]
                wp = [wp_1,wp_2,wp_3]
                wp = np.array(wp)
                # turning = is_turning(wp)
                if (p[1] <= -1.65+y_offset and p[1] > -1.75+y_offset): 
                    u = speedController.update(v, 0.02 if stop[0] else vel_ref, dt)
                else:
                    u = speedController.update(v, vel_ref, dt)
                #endregion

                #region : Steering controller update
                [delta,wpi] = steeringController.update(p, th, v)
                #endregion

            qcar.write(u, delta)
            #endregion

            #region : Update Scopes
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay

                # Speed control scope
                speedScope.axes[0].sample(t_plot, [v, vel_ref])
                speedScope.axes[1].sample(t_plot, [vel_ref-v])
                speedScope.axes[2].sample(t_plot, [u])

                # Steering control scope
                steeringScope.axes[4].sample(t_plot, [[p[0],p[1]]])

                p[0] = ekf.x_hat[0,0]
                p[1] = ekf.x_hat[1,0]

                x_ref = steeringController.p_ref[0]
                y_ref = steeringController.p_ref[1]
                th_ref = steeringController.th_ref

                x_ref = gps.position[0]
                y_ref = gps.position[1]
                th_ref = gps.orientation[2]

                steeringScope.axes[0].sample(t_plot, [p[0], x_ref])
                steeringScope.axes[1].sample(t_plot, [p[1], y_ref])
                steeringScope.axes[2].sample(t_plot, [th, th_ref])
                steeringScope.axes[3].sample(t_plot, [delta])


                arrow.setPos(p[0], p[1])
                arrow.setStyle(angle=180-th*180/np.pi)

                count = 0
            #endregion
            continue

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
def image_main():


    cameraInterfacingLab = image_interpretation.ImageInterpretation(
        imageSize=[[820,410], [640,480]],
        frameRate=np.array([30, 30]),
        streamInfo=[3, "RGB"],
        chessDims=6,
        boxSize=1
    )

    cameraMatrix  = np.array([
        [495.84,   0.00, 408.03],
        [0.00, 454.60, 181.21],
        [0.00,   0.00,   1.00]
    ])

    distortionCoefficients = np.array([
        -0.57513,
        0.37175,
        -0.00489,
        -0.00277,
        -0.11136
    ])

    cameraInterfacingLab.CSICamIntrinsics = cameraMatrix
    cameraInterfacingLab.CSIDistParam     = distortionCoefficients
    cameraInterfacingLab.d435CamIntrinsics = cameraMatrix
    cameraInterfacingLab.d435DistParam     = distortionCoefficients

    return cameraInterfacingLab

        


#region : Setup and run experiment
if __name__ == '__main__':
    #region : Setup scopes
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30

    #Scope for monitoring speed controller
    speedScope = MultiScope(
        rows=3,
        cols=1,
        title='Vehicle Speed Control',
        fps=fps
    )
    speedScope.addAxis(
        row=0,
        col=0,
        timeWindow=tf,
        yLabel='Vehicle Speed [m/s]',
        yLim=(0, 1)
    )
    speedScope.axes[0].attachSignal(name='v_meas', width=2)
    speedScope.axes[0].attachSignal(name='v_ref')
    speedScope.addAxis(
        row=1,
        col=0,
        timeWindow=tf,
        yLabel='Speed Error [m/s]',
        yLim=(-0.5, 0.5)
    )
    speedScope.axes[1].attachSignal()
    speedScope.addAxis(
        row=2,
        col=0,
        timeWindow=tf,
        xLabel='Time [s]',
        yLabel='Throttle Command [%]',
        yLim=(-0.3, 0.3)
    )
    speedScope.axes[2].attachSignal()

    # Scope for monitoring steering controller
    steeringScope = MultiScope(
        rows=4,
        cols=2,
        title='Vehicle Steering Control',
        fps=fps
    )
    steeringScope.addAxis(
        row=0,
        col=0,
        timeWindow=tf,
        yLabel='x Position [m]',
        yLim=(-2.5, 2.5)
    )
    steeringScope.axes[0].attachSignal(name='x_meas')
    steeringScope.axes[0].attachSignal(name='x_ref')
    steeringScope.addAxis(
        row=1,
        col=0,
        timeWindow=tf,
        yLabel='y Position [m]',
        yLim=(-1, 5)
    )
    steeringScope.axes[1].attachSignal(name='y_meas')
    steeringScope.axes[1].attachSignal(name='y_ref')
    steeringScope.addAxis(
        row=2,
        col=0,
        timeWindow=tf,
        yLabel='Heading Angle [rad]',
        yLim=(-3.5, 3.5)
    )
    steeringScope.axes[2].attachSignal(name='th_meas')
    steeringScope.axes[2].attachSignal(name='th_ref')
    steeringScope.addAxis(
        row=3,
        col=0,
        timeWindow=tf,
        yLabel='Steering Angle [rad]',
        yLim=(-0.6, 0.6)
    )
    steeringScope.axes[3].attachSignal()
    steeringScope.axes[3].xLabel = 'Time [s]'
    steeringScope.addXYAxis(
        row=0,
        col=1,
        rowSpan=4,
        xLabel='x Position [m]',
        yLabel='y Position [m]',
        xLim=(-2.5, 2.5),
        yLim=(-1, 5)
    )
    im = cv2.imread(
        images.SDCS_CITYSCAPE,
        cv2.IMREAD_GRAYSCALE
    )
    steeringScope.axes[4].attachImage(
        scale=(-0.002035, 0.002035),
        offset=(1125,2365),
        rotation=180,
        levels=(0, 255)
    )
    steeringScope.axes[4].images[0].setImage(image=im)
    referencePath = pg.PlotDataItem(
        pen={'color': (85,168,104), 'width': 2},
        name='Reference'
    )
    steeringScope.axes[4].plot.addItem(referencePath)
    referencePath.setData(waypointSequence[0, :],waypointSequence[1, :])
    steeringScope.axes[4].attachSignal(name='Estimated', width=2)
    arrow = pg.ArrowItem(
        angle=180,
        tipAngle=60,
        headLen=10,
        tailLen=10,
        tailWidth=5,
        pen={'color': 'w', 'fillColor': [196,78,82], 'width': 1},
        brush=[196,78,82]
    )
    arrow.setPos(initialPose[0], initialPose[1])
    steeringScope.axes[4].plot.addItem(arrow)
    #endregion

    #region : Setup control thread, then run experiment
    controlThread = Thread(target=controlLoop)
    controlThread.start()

    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
    #endregion
    if not IS_PHYSICAL_QCAR:
        # qlabs_setup.terminate()
        pass

    input('Experiment complete. Press any key to exit...')
#endregion
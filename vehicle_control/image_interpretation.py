# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports
"""
image_interpretation.py
Skills activity code for image interpretation lab guide.
Students will perform camera calibration along with line detection.
Please review Lab Guide - Image Interpretation PDF
"""
from pal.products.qcar import QCarCameras,QCarRealSense, IS_PHYSICAL_QCAR
from hal.utilities.image_processing import ImageProcessing
import time
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : ImageInterpretation Class Setup

class ImageInterpretation():

    def __init__(self,
            imageSize,
            frameRate,
            streamInfo,
            chessDims,
            boxSize):

        # Camera calibration constants:
        self.NUMBER_IMAGES = 15

        # List of variables given by students
        self.imageSize      = imageSize
        self.chessboardDim  = [chessDims,chessDims]
        self.frameRate      = frameRate
        self.boxSize        = boxSize
        self.sampleRate     = 1/self.frameRate
        self.calibFinished  = False
        self.model =  YOLO('qs-tl-stop.pt')
        self.red = False
        # List of camera intrinsic properties :
        self.CSICamIntrinsics =  np.array([[318.86  ,  0.00 , 401.34],[  0.00 , 312.14 , 201.50],[  0.00 ,   0.00   , 1.00]])

        self.CSIDistParam = np.array([[-0.9033 , 1.5314 ,-0.0173, 0.0080 ,-1.1659]])

        self.d435CamIntrinsics = np.array([[455.20,0.00,308.53],[0.00,459.43,213.56],[0.00,0.00,1.00]])
        self.d435DistParam = np.array( [[-5.1135e-01,  5.4549, -2.2593e-02, -6.2131e-03, -2.0190e+01]])
        self.streamD435 = np.zeros((self.imageSize[1][0],self.imageSize[1][1]))
        self.streaCSI = np.zeros((self.imageSize[0][0],self.imageSize[0][1]))
        enableCameras = [False, False, False, False]
        enableCameras[streamInfo[0]] = True
        self.frontCSI = QCarCameras(
            frameWidth  = self.imageSize[0][0],
            frameHeight = self.imageSize[0][1],
            frameRate   = self.frameRate[0],
            enableRight = enableCameras[0],
            enableBack  = enableCameras[1],
            enableLeft  = enableCameras[2],
            enableFront = enableCameras[3]
        )
        self.d435Color = QCarRealSense(
            mode=streamInfo[1],
            frameWidthRGB  = self.imageSize[1][0],
            frameHeightRGB = self.imageSize[1][1],
            frameRateRGB   = self.frameRate[1]
        )
        self.camCalibTool = ImageProcessing()

    def line_detection(self, cameraType):
        if cameraType == "csi":

            self.frontCSI.readAll()
            image = self.frontCSI.csi[3].imageData
            cameraIntrinsics = self.CSICamIntrinsics
            cameraDistortion = self.CSIDistParam

        if cameraType == "D435":

            self.d435Color.read_RGB()
            image = self.d435Color.imageBufferRGB
            cameraIntrinsics = self.d435CamIntrinsics
            cameraDistortion = self.d435DistParam

        # ============= SECTION C1 - Image Correction =============
        imageShape = np.shape(image)
        undistortedImage = self.camCalibTool.undistort_img(
            image,
            cameraIntrinsics,
            cameraDistortion
        )
        return undistortedImage
        
    def check_red(self):
        image = self.line_detection("csi")
        result = self.model(image,verbose = False)[0]
        detections = sv.Detections.from_ultralytics(result)
        if "red" in list(detections.data.values())[0]:
            self.red = True
        if "green" in list(detections.data.values())[0]:
            self.red = False

        return self.red
    
    def stop_cameras(self):
        # Stopping the image feed for both cameras
        self.frontCSI.terminate()
        self.d435Color.terminate()

#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Main
def main():
    try:
        '''
        INPUTS:
        imageSize           = [[L,W],[L,W]] 2x2 array which specifies the
                                resultion for the CSI camera and the D435
        frameRate           = [CSIframeRate, D435frameRate] 2x1 array for
                                image frame rates for CSI, D435 camera
        streamInfo          = [CSIIndex,D435Stream] 2x1 array to specify the
                                CSI camera index and the D435 image stream
        chessDims           = Specify the number of cells of the square
                                chessboard used for calibration
        boxSize             = Float value to specify the size of the grids
                                in the chessboard. Be mindful of the physical
                                units being used
        '''

        # ======== SECTION A - Student Inputs for Image Interpretation ===========
        cameraInterfacingLab = ImageInterpretation(
            imageSize=[[820,410], [640,480]],
            frameRate=np.array([30, 30]),
            streamInfo=[3, "RGB"],
            chessDims=6,
            boxSize=1
        )

        ''' Students decide the activity they would like to do in the
        ImageInterpretation Lab

        List of current activities:
        - Calibrate   (interfacing skill activity)
        - Line Detect (line detection skill activity)
        '''
        camMode = "Line Detect"

        # ========= SECTION D - Camera Intrinsics and Distortion Coeffs. =========
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

        if camMode == "Calibrate":
            try:
                cameraInterfacingLab.camera_calibration()
                if cameraInterfacingLab.calibFinished == True \
                        and camMode == "Calibrate":
                    print("calibration process done, stopping cameras...")
                    cameraInterfacingLab.stop_cameras()

            except KeyboardInterrupt:
                cameraInterfacingLab.stop_cameras()

        if camMode == "Line Detect":
            try:
                text = "Specify the camera used for line detection (csi/D435): "
                cameraType = input(text)
                if cameraType == "csi" :
                    # cameraInterfacingLab.CSICamIntrinsics = cameraMatrix
                    # cameraInterfacingLab.CSIDistParam     = distortionCoefficients
                    cameraInterfacingLab.line_detection(cameraType)

                elif cameraType =="D435":
                    # cameraInterfacingLab.d435CamIntrinsics = cameraMatrix
                    # cameraInterfacingLab.d435DistParam     = distortionCoefficients
                    cameraInterfacingLab.line_detection(cameraType)
                else:
                    print("Invalid camera type")

            except KeyboardInterrupt:
                cameraInterfacingLab.stop_cameras()
    finally:
        if not IS_PHYSICAL_QCAR:
            import qlabs_setup
            qlabs_setup.terminate()
        input('Experiment complete. Press any key to exit...')


#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Run
if __name__ == '__main__':
    main()
#endregion
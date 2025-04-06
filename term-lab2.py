# NOTE: For testing, based on the current variables (and my testing) road96.png is an example of a True Positive, road99.png is a False negative, road5.png is a False Positive

# RESULTS
# min_conf = 0.41
# Total Images Processed: 877
# False Positives: 9
# False Negatives: 61
# Total Time Taken: 20209 ms


import cv2 as cv
import re
import numpy as np
import scipy as sp
from scipy import signal
import easyocr
import xml.etree.ElementTree as ET # XML Parser, Docs: https://docs.python.org/3/library/xml.etree.elementtree.html
from os import listdir # Read files in directory
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import time
import argparse

reader = easyocr.Reader(['en'])
print("")
print("")



# Template Code
T = cv.imread('data/traffic-stop-signs/template-1-1.png')
T = cv.cvtColor(T, cv.COLOR_BGR2RGB)
T = cv.resize(T, (64,64))

# Template Code (Traffic Light)
TL = cv.imread('data/traffic-stop-signs/traffic-light-template.jpg')
TL = cv.cvtColor(T, cv.COLOR_BGR2RGB)
TL = cv.resize(T, (64,64))

# Template Code (Speed Limit)
TS = cv.imread('data/traffic-stop-signs/speed-limit.jpg')
TS = cv.cvtColor(T, cv.COLOR_BGR2RGB)
TS = cv.resize(T, (64,64))

# Minimum Confidence (Folder command line arg)
min_conf = 0.35 # I observed more (false positives + false negatives) at values below/above 0.41
min_conf_speed = 0.35

# SETTINGS
textStops = True
stopConf = 0.8


# https://medium.com/@adityamahajan.work/easyocr-a-comprehensive-guide-5ff1cb850168
def find_text(I, speedBoundingBox):
    foundStop = False
    foundSpeed = False
    boundingBoxesOverlap = False
    result = reader.readtext(I)
    for (bbox, text, prob) in result:
        print(f'Text: {text}, Probability: {prob}')
        if str.lower(text) == "stop":
            if prob > stopConf:
                foundStop = True
        s_no_whitespace = re.sub(r"\s+", "", text)  # Remove all whitespace
        pattern = r"^\d{1,3}$"
        if(bool(re.match(pattern, s_no_whitespace)) and prob > stopConf and boundingBoxesOverlap == False):
            foundSpeed = text
            easyOCRX = [bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]]
            easyOCRY = [bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]]

            if type(speedBoundingBox) != bool:
                print("Not bool")
                minX = speedBoundingBox[0]
                minY = speedBoundingBox[1]
                maxX = speedBoundingBox[0] + speedBoundingBox[2]
                maxY = speedBoundingBox[1] + speedBoundingBox[3]

                print("Cond 1: ", max(easyOCRX), ">=", minX)
                print("Cond 2: ", min(easyOCRX), "<=", maxX)
                print("Cond 3: ", max(easyOCRY), ">=", minY)
                print("Cond 4: ", min(easyOCRY), "<=", maxY)
                if (max(easyOCRX) >= minX or min(easyOCRX) <= maxX) and (max(easyOCRY) >= minY or min(easyOCRY) <= maxY):
                    print("All conditions met!")
                    boundingBoxesOverlap = True

                print("Speed text found at:", bbox)
                print("Speed sign found at:", speedBoundingBox)
    print("Returning: ", boundingBoxesOverlap)
    return foundStop, foundSpeed, boundingBoxesOverlap


def find_stop_sign(T, I, conf):
    """
    Given a traffic stop sign template T and an image I, returns the bounding box 
    for the detected stop sign.
    
    A bounding box is defined as follows: [top, left, height, width]
    
    You may return an empty bounding box [0,0,1,1] to indicate that a 
    stop sign wasn't found.
    """
    
    # The following hardcoded value uses:
    #
    # T = 'data/traffic-stop-signs/template-1-1.png'
    # I = 'data/traffic-stop-signs/traffic-stop-1.jpg
    #
    # You need to implement this method to work with other templates 
    # and images

    method = 'cv.TM_CCOEFF_NORMED' # Method used to detect similarity
    I_S, levels = make_square(I)
    boo = gen_gaussian_pyramid(I_S, levels) # Generate Guassian Pyrimad of image
    currentBestVal = float('-inf') # Keeping track of the currently found highest confidence (will update as we compare each level)
    currentBestImage = None
    loc, val, R, lvl, iterations = None, None, None, None, 0

    #print("Image List Length:", len(boo))
    for i in boo:
        if i.shape[0] < T.shape[0]: # Actual Image should not be smaller than the template (not having this continue statement results in everything being detected as a stop sign)
             #Shape dimension of the image is smaller than the template, don't check this iteration
            continue
        R_temp = cv.matchTemplate(i, T, eval(method))
        loc_temp, val_temp = find_loc_and_value_in_R(R_temp, use_max=True)
        if max(currentBestVal, val_temp) == val_temp: # If the new value found by the function is greater than the current max, update all the values to reflect that
            currentBestVal = val_temp
            currentBestImage = i
            loc = loc_temp
            val = val_temp
            R = R_temp
            lvl = iterations
        iterations += 1

    loc, val = find_loc_and_value_in_R(R, use_max=True)

    w_t,h_t = (T.shape[0] * (2**lvl)), (T.shape[1] * (2**lvl))


    if(val < conf): # Check if the highest confidence value found is above the threshold set
        return np.array([0, 0, 1, 1]).astype(int) # Under threshold
    else:
        return np.array([loc[0] * 2**lvl, loc[1] * 2**lvl, w_t, h_t]).astype(int) # Scale loc values back up based on the level of the pyramid.


def highlight(R, T, I, use_max=True):
    """
    Finds the location of maximum (or minimum) matching response, and 
    draws a rectangle on the image around this location.  The
    size of the rectangle is determined by template T.
    
    Returns an image with the drawn rectangle.  Also returns the loc and
    the value (maximum or minimum as the case may be).  The original image
    remains unchanged.
    """
    
    W, H = I.shape[0], I.shape[1]
    w, h = T.shape[0], T.shape[1]
    wr, hg = R.shape[0], R.shape[1]
        
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(R)
    loc = max_loc if use_max else min_loc
    val = max_val if use_max else min_val
    
    loc1 = loc + np.array([h//2, w//2])               # Size of R is different from I 
    tl = loc1 - np.array([h//2, w//2])
    br = loc1 + np.array([h//2, w//2])
    I_ = np.copy(I)
    c = (1.0, 0, 0) if I_.dtype == 'float32' else (255, 0, 0)
    cv.rectangle(I_, tuple(tl), tuple(br), c, 4)
    return I_, loc, val


def find_loc_and_value_in_R(R, use_max=True):
    """
    Finds the location of maximum (or minimum) matching response.
    """
    
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(R)
    loc = max_loc if use_max else min_loc
    val = max_val if use_max else min_val
    
    return loc, val


def draw_rect(I, bbox):

    I_ = np.copy(I)
    c = (1.0, 0, 0) if I_.dtype == 'float32' else (255, 0, 0)
    cv.rectangle(I_, bbox, c, 4)
    return I_


def gen_gaussian_pyramid(I, levels):
    G = I.copy()
    alphas = [1]
    betas = [0]
    #if contrast == True:
        #alphas = [0.5,0.8,0.9,1,1.1,1.2,1.5] # Contrast
    #if brightness == True:
        #betas = [-20,-10,0,10,20] # Brightness
    gpI = []

    for x in alphas:
        for y in betas:
            gpI.append(cv.convertScaleAbs(G, alpha=x, beta=y))

    for i in range(levels):
        G = cv.pyrDown(G)
        for x in alphas:
            for y in betas:
                gpI.append(cv.convertScaleAbs(G, alpha=x, beta=y))

    return gpI



def visualize_guassian_pyramid(gpI):
    I = gpI[0]
    h, w = I.shape[0], I.shape[1]
    
    if len(I.shape) == 3:
        result = np.empty([h, 2*w, I.shape[2]], dtype=I.dtype)
    else:
        result = np.empty([h, 2*w], dtype=I.dtype)
    
    x = 0
    for I_ in gpI:
        if len(I.shape) == 3:
            h, w, _ = I_.shape
            result[:h,x:x+w,:] = I_
        else:
            h, w = I_.shape
            result[:h,x:x+w] = I_
        x += w
    
    return result



def make_square(I):
    h = I.shape[0]
    w = I.shape[1]
    
    n_levels = int(np.ceil(np.log(np.max([h,w]))/np.log(2))) # NOTE: I modified np.int to int based on error msg: Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information. The aliases was originally deprecated in NumPy 1.20
    new_h = np.power(2, n_levels)
    new_w = new_h
    
    if len(I.shape) == 3:
        tmp = np.zeros([new_h, new_w, I.shape[2]], dtype=I.dtype)
        tmp[:h,:w,:] = I
    else:
        tmp = np.zeros([new_h, new_w], dtype=I.dtype)
        tmp[:h,:w] = I

    return tmp, n_levels


# Used for --detectall
def processImages(folder):

    # Variables to track analytics
    totalImages = 0
    falsePositives = 0
    falseNegatives = 0
    timeTaken = 0

    OCRFlipCount = 0
    correctFlips = 0
    wrongFlips = 0

    OCRFlipCountSpeed = 0
    correctFlipsSpeed = 0
    wrongFlipsSpeed = 0


    falsePositivesTraffic = 0
    falseNegativesTraffic = 0

    falsePositivesSpeed = 0
    falseNegativesSpeed = 0

    speedsFound = {}

    wrongFlips = []

    
    print(f"{'filename':<15} | {'stop detected':^15} | {'stop ground truth':^15} | {'traffic detected':^20} | {'traffic ground truth':^20} | {'sp limit detected':^20} | {'sp limit ground truth':^22}") # Print header
    print("")

    start = time.time() # Starting time
    for imagePathOrig in listdir(folder): # Iterate every image path within the folder path provided
        print("")
        totalImages += 1

        imagePath = imagePathOrig

        groundTruth = "No" # Will update to yes later if found
        groundTruthTraffic = "No"
        groundTruthSpeed = "No"
        detected = "N/A"
        detectedTraffic = "N/A"
        detectedSpeed = "N/A"

        imagePath = folder + "/" + imagePath
        image = cv.imread(imagePath) # Read from the imagePath
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Put Image through Image Pyrimad
        bbox = find_stop_sign(T, image, min_conf)
        bboxTL = find_stop_sign(TL, image, min_conf)
        bboxTS = find_stop_sign(TS, image, min_conf_speed)


        #print(bboxTS)

        xml_path = imagePath.replace("images","annotations") # Modify the image path to find the corresponding XML
        xml_path = xml_path.replace("png","xml")

        tree = ET.parse(xml_path) # Parse the XML
        root = tree.getroot()   

        groundTruth = "No"

        for child in root:
            if str(child.tag) == "object":
                if child[0].text == "stop":
                    groundTruth = "Yes"
                if child[0].text == "trafficlight":
                    groundTruthTraffic = "Yes"
                if child[0].text == "speedlimit":
                    groundTruthSpeed = "Yes"

        returnVal = False
        returnValSpeed = False

        speedBoundingBox = bboxTS
        if np.all(speedBoundingBox == np.array([0, 0, 1, 1]).astype(int)):
            speedBoundingBox = False

        if textStops:
            returnVal, returnValSpeed, bbOverlap = find_text(image, speedBoundingBox)

        if np.all(bbox == np.array([0, 0, 1, 1]).astype(int)): # Did we determine there to be a stop sign with a high enough confidence interval?
            detected = "No" # Not detected
            if returnVal == True:
                detected = "Yes"
                OCRFlipCount += 1
                if groundTruth == "Yes":
                    correctFlips += 1
                else:
                    wrongFlips += 1

            elif groundTruth == "Yes": # Ground Truth says yes though, therefore false negative
                falseNegatives += 1
        else:
            detected = "Yes" # Is Detected
            if groundTruth == "No": # Ground Truth says no though, therefore false positive
                falsePositives += 1


        if np.all(bboxTL == np.array([0, 0, 1, 1]).astype(int)): # Did we determine there to be a stop sign with a high enough confidence interval?
            detectedTraffic = "No" # Not detected
            if groundTruthTraffic == "Yes": # Ground Truth says yes though, therefore false negative
                falseNegativesTraffic += 1
        else:
            detectedTraffic = "Yes" # Is Detected
            if groundTruthTraffic == "No": # Ground Truth says no though, therefore false positive
                falsePositivesTraffic += 1


        if np.all(bboxTS == np.array([0, 0, 1, 1]).astype(int)): # Did we determine there to be a stop sign with a high enough confidence interval?
            detectedSpeed = "No" # Not detected

            if returnValSpeed != False:
                print("bbOverlap:", bbOverlap)
                if bbOverlap == True:
                    speedsFound[imagePathOrig] = str(returnValSpeed) + "  |  " + str(bbOverlap)
                detectedSpeed = "Yes"
                OCRFlipCountSpeed += 1
                if groundTruthSpeed == "Yes":
                    correctFlipsSpeed += 1
                else:
                    wrongFlipsSpeed += 1
                    falsePositivesSpeed += 1
                    wrongFlips.append(imagePathOrig)
                    #speedsFound[imagePathOrig] = "INCORRECT FLIP | " + str(returnValSpeed) + "  |  " + str(bbOverlap)

            elif groundTruthSpeed == "Yes": # Ground Truth says yes though, therefore false negative
                falseNegativesSpeed += 1
        else:
            detectedSpeed = "Yes" # Is Detected
            if returnValSpeed != False:
                print("bbOverlap:", bbOverlap)
                if bbOverlap == True:
                    speedsFound[imagePathOrig] = str(returnValSpeed) + "  |  " + str(bbOverlap)
            if groundTruthSpeed == "No": # Ground Truth says no though, therefore false positive
                falsePositivesSpeed += 1


        print(f"{imagePathOrig:<15} | {detected:^15} | {groundTruth:^15} | {detectedTraffic:^20} | {groundTruthTraffic:^20} | {detectedSpeed:^20} | {groundTruthSpeed:^22}") # Print table row with image path, detected (yes or no), ground truth (yes or no)
        
    end = time.time()

    # Print out the summary
    print("")
    print("Summary:")
    print("Total Images Processed:", totalImages)
    print("Min Confidence (Stop and Lights):", min_conf)
    print("Min Confidence (Speed):", min_conf_speed)
    print("")
    print("Stop Sign OCR:", str(textStops))
    print("Stop Sign OCR Min Confidence:", stopConf)
    print("Stop Sign OCR Flips Negative->Positive:", OCRFlipCount)
    print("Correct Flips:", correctFlips)
    print("Incorrect Flips:", wrongFlips)
    print("")
    print("False Positives (Stop Sign):", falsePositives)
    print("False Negatives (Stop Sign):", falseNegatives)
    print("Detections Flipped Negative->Positive due to OCR:", OCRFlipCount)
    print("")
    print("False Positives (Traffic Lights):", falsePositivesTraffic)
    print("False Negatives (Traffic Lights):", falseNegativesTraffic)
    print("")
    print("False Positives (Speed Limit):", falsePositivesSpeed)
    print("False Negatives (Speed Limit):", falseNegativesSpeed)
    print("")
    print("Speed OCR Min Confidence:", stopConf)
    print("Speed OCR Flips Negative->Positive:", OCRFlipCountSpeed)
    print("Correct Flips:", correctFlipsSpeed)
    print("Incorrect Flips:", wrongFlipsSpeed)
    print("Incorrect Speed Flips: ", wrongFlips)
    print("")
    print("Total Time Taken:", int((end-start)*1000), "ms") # Convert seconds to ms (*1000)

    filename = "speeds.txt"

    try:
        with open(filename, 'w') as f:
            for key, value in speedsFound.items():
                f.write(f"{key}: {value}\n")
        print(len(speedsFound), " entries added to text document!")
        print(f"Dictionary successfully written to {filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

        




# Used for --show-image and --detect
# topLeft = False for --show-image, True for --detect
def showImage(imagePath, topLeft=False):
    image = cv.imread(imagePath)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(10,10))

    # If --detect
    if topLeft:
        bbox = find_stop_sign(T, image) # Find bounding box
        print("")
        print(f'(Predicted) Bounding Box (xmin,ymin,size,size) = {bbox}')
        print("")
        left_text = "Stop Sign Detected!"
        if np.all(bbox == np.array([0, 0, 1, 1]).astype(int)):
            left_text = "NO Stop Sign Detected!"
        image = draw_rect(image, bbox) # Display bounding box as rectangle on image
        plt.text(.01, .99, left_text, ha='left', va='top', color='white',  # As per lab documentation for --detect, top left should show if stop sign was detected or not
         path_effects=[path_effects.Stroke(linewidth=3, foreground='black'), 
                       path_effects.Normal()])

    xml_path = imagePath.replace("images","annotations")
    xml_path = xml_path.replace("png","xml")

    tree = ET.parse(xml_path)
    root = tree.getroot()   

    print("(Actual/XML) Traffic Sign Name:", root[4][0].text)
    print("(Actual/XML) Traffic Sign Bounding Box Location: xmin ", root[4][5][0].text, " | ymin ", root[4][5][1].text, " | xmax ", root[4][5][2].text, " | ymax ", root[4][5][3].text)
    print("")

    plt.imshow(image, cmap='gray');    

    plt.show()


# Parser code for arguments
parser = argparse.ArgumentParser(description='Traffic Stop Sign Detection')

group = parser.add_mutually_exclusive_group(required=True)

group.add_argument('--show-image', help='Display image with XML info') 
group.add_argument('--detectall', help='Iterate through the provided folder and show statistics') 
group.add_argument('--detect', help='Iterate through the provided folder and show statistics') 

args = parser.parse_args()

# Run correct function based on argument
if args.show_image:
    showImage(args.show_image, False)

if args.detect:
    showImage(args.detect, True)

if args.detectall:
    processImages(args.detectall)


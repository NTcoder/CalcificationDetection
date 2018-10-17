import numpy as np
import cv2
from matplotlib import pyplot as plt

filename = 'Fig-5-Mammogram-shows-egg-shell.png'
#filename = 'raw-slide1-upper.jpg'
#filename = 'raw-mammogram.jpg'
#filename = 'raw-slide4-upper.jpg'
#filename = '15996tn.jpg'
filename = 'ES1-breast-620.jpg'
#filename = '3CC_0.jpg'
#filename = 'Fullscreen capture 12122014 24745 PM.bmp.jpg'
#filename = '1-s2.0-S1658361208700568-gr5.jpg'
#filename = 'e6154-calcifications.png'

img = cv2.imread(filename,0)
cv2.imshow('original',img)


#STEP 1 Image Enhancement
##################################################
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

cv2.imwrite('clahe_2.jpg',cl1)


# STEP 2 Image Segmentation/Feature response
######################################################
img = cv2.imread('clahe_2.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)


# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#cv2.imshow('kmeans',res2)

#cv2.imwrite('kmeans.jpg',res2)


#Step 3 Feature Extraction
##############################################################
hist = cv2.calcHist([res2],[0],None,[256],[0,256])
#plt.hist(res2.ravel(),256,[0,256])

## feature extraction
hist_X = []
max = 0
for i in range(len(hist)):
	if (hist[i] != 0):
		#print (i,hist[i])
		second = max
		max = i
print(second, max)
#print(res2)
#mask = [][][]
area_H = 0
area_M = 0
for i in range(len(res2)):
	for j in range(len(res2[i])):
		for k in range(len(res2[i][j])):
			if res2[i][j][k] == max:
				#mask [i][j][k] = 2
				res2[i][j][k] = 255
				area_H += 1
				area_M += 1
			elif res2[i][j][k] == second:
				#mask [i][j][k] = 1
				res2[i][j][k] = 127
				area_M +=1
			else:
				#mask [i][j][k] = 0
				res2[i][j][k] = 0

#backtorgb = cv2.cvtColor(res2,cv2.COLOR_GRAY2RGB)




##################################################
# Step 4classification
print("Number of Pixels under moderate risk:", area_M)
print("Number of Pixels under High risk:", area_H)
cv2.imshow('calcification',res2)

#plt.imshow(hull, cmap = plt.get_cmap('gray'))

#plt.show()		
#pprint.pprint(hist)

#img_gray = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
#new=[[[0,0,255%j] for j in i] for i in img_gray]
#dt = np.dtype('f8')
#new=np.array(new,dtype=dt)

#cv2.imwrite('img.jpg',new)
#plt.plot(hist)
#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
import cv2



def colourtype(H,S,V):
	if V<75:
		return('Black')
	elif V>190 and S<27:
		return('White')
	elif S<53 and V<185:
		return('Grey')
	else:
		if H<14:
			return('Red')
		elif H<25:
			return('Orange') 
		elif H<34:
			return('Yellow')
		elif H<73:
			return('Green')
		elif H<102:
			return('Aqua')
		elif H<127:
			return('Blue')
		elif H<149:
			return('Purple')
		elif H<175:
			return('Pink')
		else:
			return('Red')
		

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

color_dict = {'Black':(0,0,0),'White':(0,0,255),'Grey':(0,0,120),'Red':(0,255,255),'Orange':(20,255,255),'Yellow':(30,255,255),'Green':(55,255,255),'Aqua':(85,255,255),'Blue':(155,255,255),'Purple':(138,255,255),'Pink':(161,255,255)}
hsv_dict = {'Black':0,'White':0,'Grey':0,'Red':0,'Orange':0,'Yellow':0,'Green':0,'Aqua':0,'Blue':0,'Purple':0,'Pink':0}

while 1:
	_,frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.1,2)
	fh,fw,_ = frame.shape

	for x,y,w,h in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		sd = 1.4	# Distance from top of face to top of shirt region, based on detected face height.
		sx = 0.6	# Width of shirt region compared to the detected face
		sy = 0.6	# Height of shirt region compared to the detected face

		shx = x + int(0.5 + (1.0-sx) * w)
		shy = y + int(sd * h) + int(0.5 - (1-sy)*h)
		shw = int(sx*w)
		shh = int(sy*h)

		bottom = shy + shh
		if bottom > fh:
			sd = 0.95
			sy = 0.3
			shy = y + int(sy * h) + int(0.5 - (1-sy)*h)
			shh = int(sy*h)

		bottom = shy + shh
		if bottom > fh:
			bottom = fh
			ssh = bottom - shy
			if ssh <= 1:
				continue

		cv2.rectangle(frame,(shx,shy),(shx+shw,shy+shh),(255,255,255),2)
		hsv1 = frame[shy:shy+shh,shx:shx+shw]
		try:
			hsv = cv2.cvtColor(hsv1,cv2.COLOR_BGR2HSV)
			for i in range(np.shape(hsv)[1]):
				for j in range(np.shape(hsv)[0]):
					z = hsv[i,j]
					z = colourtype(z[0],z[1],z[2])
					hsv_dict[z] += 1

			maxn = max(hsv_dict,key = hsv_dict.get)

			cv2.putText(frame,maxn,(shx,shy+shh+12),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2,cv2.LINE_AA)
		except:
			pass
	
		cv2.imshow('Frame',frame)

	k = cv2.waitKey(30) and 0xff
	if k==27:
		break
cap.release()
cv2.destroyAllWindows()







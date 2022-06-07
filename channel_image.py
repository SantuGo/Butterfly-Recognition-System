import cv2

butterfly = cv2.imread("butterfly/ADONIS/03.jpg")


b = butterfly[:, :, 0]
g = butterfly[:, :, 1]
r = butterfly[:, :, 2]

butterfly[:, :, 2] = 0
butterfly[:, :, 0] = 0
cv2.imshow(" butterflyb0g0", butterfly)
cv2.waitKey()
cv2.destroyAllWindows()
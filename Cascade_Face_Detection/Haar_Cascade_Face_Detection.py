
# coding: utf-8

# # Face Detection using OpenCV3
# Based on the tutorial found at https://realpython.com/blog/python/face-recognition-with-python/

# In[120]:

# For displaying images inline in jupyter
# % matplotlib inline
import cv2  # Why would they name the OpenCV3 library cv2?
print('OpenCV version:', cv2.__version__)
import matplotlib.pyplot as plt


# In[121]:

def show_image(img, title, cmap=None):
    """
    Utility function to display an image with a title and a given color mapping inline with Jupyter Notebook. 
    """
    plt.figure(figsize=(12,12))
    plt.imshow(img, interpolation='bicubic', cmap=cmap)
    plt.title(title, fontsize=24)
    plt.axis('off')
    plt.style.use('seaborn-dark')
    plt.show()


# In[122]:

def reverse_channels(image):
    """
    OpenCV3 opens color images in BGR format. A computer display requires images to be RGB.
    This utility function takes an image in either BGR or RGB format and returns that same
    image in the opposite format.
    """
    return cv2.merge(cv2.split(image)[::-1])


# In[123]:

gray_abba = cv2.imread('abba.png')
me_and_boys = cv2.imread('IMG_3510.png', flags=cv2.IMREAD_COLOR)
gray_me_and_boys = cv2.cvtColor(me_and_boys, cv2.COLOR_BGR2GRAY)
adversarial = cv2.imread('adversarial.jpg')


# In[124]:

show_image(img=gray_abba, title='Abba Grayscale Test', cmap='gray')


# In[125]:

show_image(img=gray_me_and_boys, title='Me and Boys Grayscale Test', cmap='gray')


# In[126]:

show_image(img=adversarial, title='Adversarial Test', cmap='gray')


# ## Create a Cascade Classifier
# Use the Haar Cascade to initialize the Haar Cascade Classifier object.

# In[127]:

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# ## Try to detect the faces!
# I found that this is highly dependent on the parameters. So you'll have to play around with the `scaleFactor` and the `minSize` until you end up with the right combination for your image.

# In[128]:

me_and_boys_faces = face_cascade.detectMultiScale(
    gray_me_and_boys,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(750, 750),  # The image I provided was much larger than the others
    flags = cv2.CASCADE_SCALE_IMAGE
)

print('Me and boys: {num_faces} faces found'.format(num_faces=len(me_and_boys_faces)))

abba_faces = face_cascade.detectMultiScale(
    gray_abba,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30),  # As you can see, the Haar Cascade is sensitive to scale
    flags=cv2.CASCADE_SCALE_IMAGE
)

print('       Abba: {num_faces} faces found'.format(num_faces=len(abba_faces)))

adversarial_faces = face_cascade.detectMultiScale(
    adversarial,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print('Adversarial: {num_faces} faces found'.format(num_faces=len(adversarial_faces)))


# In[129]:

abba = reverse_channels(abba)
me_and_boys = reverse_channels(me_and_boys)


# ## Draw boxes around the faces
# If the detector has found some faces, then let's plot some green boxes around them.

# In[130]:

for (x, y, w, h) in abba_faces:
    cv2.rectangle(gray_abba, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
for (x, y, w, h) in me_and_boys_faces:
    cv2.rectangle(me_and_boys, (x, y), (x+w, y+h), (0, 255, 0), 15)

for (x, y, w, h) in adversarial_faces:
    cv2.rectangle(adversarial, (x, y), (x+w, y+h), (0, 255, 0), 2)


# ## Did it work?
# The results really aren't that bad. However, I had to put at least a little effort into adjusting the hyperparameters to prevent tiny face-like artifacts from being detected; so, in my view, this isn't a method we can count on in general.

# In[131]:

show_image(abba, 'Abba Face Detection Test')


# In[132]:

show_image(me_and_boys, 'Me and Boys Face Detection Test')


# In[133]:

show_image(adversarial, 'Adversarial Face Detection Test')


# In[ ]:




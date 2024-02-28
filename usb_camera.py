from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk,ImageDraw
import cv2
import os,time,sys
from datetime import datetime
from check_image_quality import check_image_quality,crop_and_save_image
from dlib import get_frontal_face_detector,shape_predictor
from encrypt_aes import encrypt,add_metadata

#remove all file from tmp
# dir = 'tmp/'
# for f in os.listdir(dir):
#     os.remove(os.path.join(dir, f))
# only for test

try:
    os.system("v4l2-ctl  --set-ctrl=exposure_auto=2")
    os.system("v4l2-ctl  --set-ctrl=exposure_absolute=500")
    os.system("onboard &")
except:
    print("could not run")
    

detector = get_frontal_face_detector()
predictor = shape_predictor("shape_files/shape_predictor_68_face_landmarks.dat")


CAMERA_PORT = 0
IMAGEWIDTH = 3840
IMAGEHEIGHT = 2160

#Propriedades de configuracao da camera
# 3 = width da camera, 4 = height da camera
CAMERA_PROP_WIDTH = 3
CAMERA_PROP_HEIGHT = 4


#capture = cv2.VideoCapture()
#capture.open(1 + cv2.CAP_DSHOW)

#fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
#capture.set(cv2.CAP_PROP_FOURCC, fourcc)
#capture.set(cv2.CAP_PROP_FPS, 30)
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)


cap = None
fps = 10
camera_panel = None
capture_identifier = None
frame_cleared =False
image_list = []
original_image_list = []
image_name_list=[]
subject_directory = None
capture_count=0
overall_processing_time = 0

data_directory = "./data/"

if not os.path.exists(data_directory):
    os.mkdir(data_directory)


# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the 0idth and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

def check_input_field():
    if sub_name_var.get() and sub_id_var.get():
        return True
    else : 
        messagebox.showinfo("Required", "Subject Name and Id Required")
        return False

#create new folder if not exist
def create_folder_for_subject():
    global subject_directory,data_directory

    if check_input_field() is False:
        return False

    n =sub_name_var.get()
    if not os.path.exists(data_directory+n):
        os.mkdir(data_directory+n)

    subject_directory = data_directory+n+'/'


# remove all child in camera frame
def remove_cameraframe_child():
    global camera_panel

    #print("remove camera frame child")
    for widgets in camera_frame.winfo_children():
        widgets.destroy()

    camera_panel=None


def open_popup(img):
    top= Toplevel(root)
    top.geometry("750x750")
    top.title("Image Saved")

    #for display and grid
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(img)
    #img = img.resize((150,150)) # new width & height
    img = ImageTk.PhotoImage(image=img)
    #for display and grid

    #display frame on gui 
    label= Label(top, image= img)
    label.image= img
    label.pack()

    return 1



# save image on click image
def click_on_image(img_index):
    try:
        print("image name saved:",image_name_list[int(img_index)])

        crop_and_save_image(original_image_list[int(img_index)],subject_directory+image_name_list[int(img_index)],detector,predictor)

        #cv2.imwrite(subject_directory+image_name_list[int(img_index)],original_image_list[int(img_index)])
        add_metadata(subject_directory+image_name_list[int(img_index)],image_name_list[int(img_index)])
        encrypt(subject_directory+image_name_list[int(img_index)])
        #image = maintain_aspect_ratio_resize(image, width=IMAGEWIDTH)
        #cv2.imwrite(subject_directory+image_name,maintain_aspect_ratio_resize(original_image_list[int(img_index)],width=IMAGEWIDTH),[cv2.IMWRITE_JPEG_QUALITY, 100])

        #messagebox.showinfo("Image Saved", "Thank You, Image Saved")
        open_popup(original_image_list[int(img_index)])
    except:
       messagebox.showinfo("Image Could Not Saved", "Thank You")

    remove_cameraframe_child()


def draw_box(image):
    height,width,depth = image.shape

    start_x = int(width*30/100)
    start_y = int(height*15/100)

    end_x = int(width*70/100)
    end_y = int(height*80/100)


    color = (255, 0, 0)
    return cv2.rectangle(image,(start_x,start_y), (end_x,end_y), color, 4)



def draw_box_pil(img):
    w, h = 150, 150
    shape = [(40, 40), (w - 10, h - 10)]
    
    img = ImageDraw.Draw(img)
    return img.rectangle(shape, outline ="red")
    
    #img1.rectangle(shape, fill ="# ffff33", outline ="red")


def scan():
    print("calling")
    global cap,fps,capture_identifier,camera_panel,image_list,original_image_list,capture_count
    ret, img = cap.read()
    now = datetime.now() 
    image_name = now.strftime("%Y_%m_%d_%H_%M_%S_%f.jpg")

    if ret:
        img = cv2.flip(img,1) 
        orginal_img = img.copy()
        check_img = img.copy()
        
        img = maintain_aspect_ratio_resize(img,150)
        img = draw_box(img)


        #for display and grid
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(img)
        #img = img.resize((150,150)) # new width & height
        img = ImageTk.PhotoImage(image=img)
        #for display and grid
        

        #display frame on gui 
        camera_panel.config(image=img)
        camera_panel.tkimg = img #

        start_time = time.monotonic()
        res,msg = check_image_quality(check_img,image_name,detector,predictor)

        if res is True:
            image_list.append(img)
            original_image_list.append(orginal_img)
            image_name_list.append(image_name)
            capture_count= capture_count + 1
            message_label.config(text="Look at the camera please, captured: "+str(capture_count)+" out of 4",bg="green")
        else:
            message_label.config(text=msg+' \ncaptured '+str(capture_count)+" out of 4",bg="red")
            
        print('##per_image_quality_check_time_seconds:: ', time.monotonic() - start_time)

        if len(image_list)>3:
            print("image more than 5")
            stop_scan()

        if camera_panel:
            capture_identifier = camera_panel.after(fps, scan) # change 25 to other value to adjust FPS


def start_camera_capture():
    global cap,camera_panel,capture_count,overall_processing_time

    overall_processing_time = time.monotonic()

    if not check_input_field():
        return 0

    message_label.config(text="Look at the camera please...",bg="green")
    create_folder_for_subject()
    remove_cameraframe_child()
    
    #clear variable
    capture_count=0
    image_list.clear()
    original_image_list.clear()
    image_name_list.clear()
    
    #add new camera panel
    camera_panel = Label(camera_frame)
    camera_panel.pack()


    if cap is None:
        #cap = cv2.VideoCapture(2)
        if len(sys.argv) > 1 and sys.argv[1] == 'c':
            cap = cv2.VideoCapture(CAMERA_PORT)
        else:
            #cap = cv2.VideoCapture("v4l2src num-buffers=3000 ! video/x-raw,format=UYVY,width=1280,height=720,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink  ")
            cap = cv2.VideoCapture("v4l2src ! video/x-raw,format=UYVY,width=1920,height=1080 ! videoconvert ! video/x-raw,format=BGR ! appsink  ")

        scan() # start the capture loop
    else:
        print('capture already started')




def plot_grid_image():

    message_label.config(text="Select image you want to save...")
    
    i=0
    col=1 # start from column 1
    row=2 # 2 images in a row

    for img in image_list:
        b1 = Button(camera_frame, text="select image",command=lambda m=str(i): click_on_image(m))
        b1.grid(row=row,column=col)
        b1.image = img
        b1['image']=img # garbage collection 
         
        if(col==2):   
            row=row+1 
            col=1     
        else:         
            col=col+1   

        i=i+1

    print('##overall_process_time_seconds: ', time.monotonic() - overall_processing_time)


def stop_scan():

    global cap,camera_panel,capture_identifier,capture_count
    message_label.config(text="Fill out the information and click start...")
    
    #if camera_panel:
    #camera_panel.after_cancel(capture_identifier)
    remove_cameraframe_child()

    if cap is not None:
        cap.release()
        cap = None
        print('capture stop') 
        plot_grid_image()
    else:
        print('capture not started')  
    
    #clear capture variable
    capture_count=0
    image_list.clear()



###########################
### GUI START
###########################

root = Tk()
root.title('CITER FACE QUALITY ASSESMENT')
#root.geometry('900x600')
root.configure(background="black")
# root.attributes('-fullscreen', True)
width= root.winfo_screenwidth() 
height= root.winfo_screenheight()
# #setting tkinter window size
root.geometry("%dx%d" % (width, height))


#first frame start
message_frame = Frame(root,bg='black')
message_frame.pack()

message_label=Label(message_frame,bg="black",fg="white", text="Fill out the form and click start button to start",font=('Aerial 15 bold'))
message_label.grid(row=0, column=2,pady=10,sticky="nsew")
#first frame end


#form frame start
form_frame =Frame(root,bg='black')
form_frame.pack(pady=5)

sub_name_var=StringVar()
sub_id_var=StringVar()
#sub_name_var.set("test_subject")
#sub_id_var.set("1") 

Label(form_frame,text="Subject Name",bg='black',fg='white').pack(side=LEFT)
subject_name = Entry(form_frame,textvariable=sub_name_var).pack(side=LEFT,padx=5)
Label(form_frame,text="Subject Id",bg='black',fg='white').pack(side=LEFT,padx=5)
subject_id   = Entry(form_frame,textvariable=sub_id_var).pack(side=LEFT,padx=5)


Button(form_frame,text='Stop',command=stop_scan).pack(side=LEFT)
Button(form_frame,text="Start" ,command=start_camera_capture).pack(side=RIGHT)
#form frame end


# camera frame start 
camera_frame =Frame(root,bg="black")
camera_frame.pack()

#add new camera panel
camera_panel = Label(camera_frame,bg='black')
camera_panel.pack()
#camera frame end

frameCnt = 152
#frames = [PhotoImage(file='face1.gif',format = 'gif -index %i' %(i)) for i in range(frameCnt)]
frames = []

def update(ind):
    frame = frames[ind]
    ind += 1
    if ind == frameCnt:
        ind = 0
    label.configure(image=frame)
    root.after(200, update, ind)

label = Label(root,bg='#000000')
label.pack()
#root.after(0, update, 0)

#root.iconbitmap('/home/baset/Activity/clarkson/gui/final/icon.ico')
root.iconphoto(False, PhotoImage(file='asset/icon.png'))
root.mainloop()

if cap:
    cap.release()

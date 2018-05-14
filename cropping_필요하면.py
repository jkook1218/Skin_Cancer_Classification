from PIL import Image
import os
# Define constants (mostly paths of directories)
DIR = "C://Users//korea//Desktop//project//anam//tensforflow_test//data//HIGH//" #227
DIR_LOW = "C://Users//korea//Desktop//project//anam//tensforflow_test//data//LOW//" #568
HIGH_CROPPED_SAVE_DIR = "C://Users//korea//Desktop//project//anam//tensforflow_test//data//HIGH_CROPPED//" #189
LOW_CROPPED_SAVE_DIR =  "C://Users//korea//Desktop//project//anam//tensforflow_test//data//LOW_CROPPED//" #459
IMG_DIR_LIST = os.listdir(DIR)
len_h =224
len_w =224
len_c = 3
nclass = 2 # HIGH & LOW
# print( DIR + DATE_DIR_LIST[0] + "//" )

# with Image.open("C://Users//korea//Desktop//project//anam//KU_Colon//data//6.01//etc//00471795_2.jpg") as img:
#     width, height = img.size

# print(width)
# print(height)

# def open_high_img(dir,seq) : #(날짜 + high)
#     img_select_dir = os.listdir(dir)
#     open_high_dir =  dir + "//" + img_select_dir[2] + "//"
#     high_dir_list = os.listdir(open_high_dir)
#     return  open_high_dir + high_dir_list[seq]
#
# def open_low_img(dir,seq) : #(날짜 + low)
#     img_select_dir = os.listdir(dir)
#     open_low_dir = dir + "//" + img_select_dir[3] + "//"
#     low_dir_list = os.listdir(open_low_dir)
#     return open_low_dir + low_dir_list[seq]

# print(IMG_DIR_LIST[0])
# print(IMG_DIR_LIST[1])
# print(IMG_DIR_LIST[2])
# print(IMG_DIR_LIST[3])
# print(IMG_DIR_LIST[4])
# print(IMG_DIR_LIST[5])
# print(IMG_DIR_LIST[6])

HIGH_error_file_list = []
for img_num in range(0,len(IMG_DIR_LIST)):
    img_file_dir = DIR + IMG_DIR_LIST[img_num]
    try:
        with Image.open(img_file_dir) as img:
            width, height = img.size
            if width == 1128 :

                area = (30, 30,1128, 964) #쪼개해야얄 픽셀
                cropped_img = img.crop(area)
                resize_img = cropped_img.resize((224,224))
                resize_img.save(IMG_DIR_LIST[img_num])
                os.rename(IMG_DIR_LIST[img_num], HIGH_CROPPED_SAVE_DIR + IMG_DIR_LIST[img_num])
                print(IMG_DIR_LIST[img_num] + " is completed 1128!!!")

            elif width == 640 :

                area = (87, 20,614, 480)
                cropped_img = img.crop(area) #쪼개야할 픽셀
                resize_img = cropped_img.resize((224, 224))
                resize_img.save(IMG_DIR_LIST[img_num])
                os.rename(IMG_DIR_LIST[img_num], HIGH_CROPPED_SAVE_DIR + IMG_DIR_LIST[img_num])
                print(IMG_DIR_LIST[img_num] + " is completed 640!!!")
            else:
                area = (87, 20, 614, 480)
                cropped_img = img.crop(area)  # 쪼개야할 픽셀
                resize_img = cropped_img.resize((224, 224))
                resize_img.save(IMG_DIR_LIST[img_num])
                os.rename(IMG_DIR_LIST[img_num], HIGH_CROPPED_SAVE_DIR + IMG_DIR_LIST[img_num])
                print(IMG_DIR_LIST[img_num] + " is completed 640!!!")
    except OSError:
        print(IMG_DIR_LIST[img_num] + " is error file")
        HIGH_error_file_list.append(IMG_DIR_LIST[img_num])
        pass
print(HIGH_error_file_list)
print("Completed")

# LOW_error_file_list = []
# for img_num in range(0,len(IMG_DIR_LIST)):
#     img_file_dir = DIR_LOW + IMG_DIR_LIST[img_num]
#     try:
#         with Image.open(img_file_dir) as img:
#             width, height = img.size
#             if width == 1128 :
#
#                 area = (30, 30,1128, 964) #쪼개해야할 픽셀
#                 cropped_img = img.crop(area)
#                 resize_img = cropped_img.resize((224,224))
#                 resize_img.save(IMG_DIR_LIST[img_num])
#                 os.rename(IMG_DIR_LIST[img_num], LOW_CROPPED_SAVE_DIR + IMG_DIR_LIST[img_num])
#                 print(IMG_DIR_LIST[img_num] + " is completed 1128!!!")
#
#             elif width == 640 :
#
#                 area = (87, 20,614, 480)
#                 cropped_img = img.crop(area) #쪼개야할 픽셀
#                 resize_img = cropped_img.resize((224, 224))
#                 resize_img.save(IMG_DIR_LIST[img_num])
#                 os.rename(IMG_DIR_LIST[img_num], LOW_CROPPED_SAVE_DIR + IMG_DIR_LIST[img_num])
#                 print(IMG_DIR_LIST[img_num] + " is completed 640!!!")
#             else :
#                 area = (87, 20, 614, 480)
#                 cropped_img = img.crop(area)  # 쪼개야할 픽셀
#                 resize_img = cropped_img.resize((224, 224))
#                 resize_img.save(IMG_DIR_LIST[img_num])
#                 os.rename(IMG_DIR_LIST[img_num], LOW_CROPPED_SAVE_DIR + IMG_DIR_LIST[img_num])
#                 print(IMG_DIR_LIST[img_num] + " is completed 640!!!")
#     except OSError:
#         print(IMG_DIR_LIST[img_num] + " is error file")
#         LOW_error_file_list.append(IMG_DIR_LIST[img_num])
#         pass
# print(LOW_error_file_list)
# print("Completed")


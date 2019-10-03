"""

class CreateProject
    Create all folders necessary to start project from scratch or finetune an existing model
    Copy all image files into the newly created project folder

"""

import os
from shutil import copyfile
import tkinter.messagebox

class CreateProject:

    def __init__(self, project_name):

        self.project_name = project_name

    def create_folders(self, path):

        if not os.path.lexists(path + os.sep + self.project_name):

            os.mkdir(path + os.sep + self.project_name)

            os.mkdir(path + os.sep + self.project_name + os.sep + "aug_label")
            os.mkdir(path + os.sep + self.project_name + os.sep + "aug_merge")
            os.mkdir(path + os.sep + self.project_name + os.sep + "aug_train")
            os.mkdir(path + os.sep + self.project_name + os.sep + "aug_weights")
            os.mkdir(path + os.sep + self.project_name + os.sep + "merge")
            os.mkdir(path + os.sep + self.project_name + os.sep + "npydata")

            os.mkdir(path + os.sep + self.project_name + os.sep + "train")

            os.mkdir(path + os.sep + self.project_name + os.sep + "train" + os.sep + "image")
            os.mkdir(path + os.sep + self.project_name + os.sep + "train" + os.sep + "label")

            os.mkdir(path + os.sep + self.project_name + os.sep + "train" + os.sep + "RawImgs")

            os.mkdir(path + os.sep + self.project_name + os.sep + "train" + os.sep + "RawImgs" + os.sep + "image")
            os.mkdir(path + os.sep + self.project_name + os.sep + "train" + os.sep + "RawImgs" + os.sep + "label")

            return True

        else:

            tkinter.messagebox.showinfo("Note", "MitoSegNet_Project folder already exists!")
            return False

    def copy_data(self, path, orgpath, labpath):

        image_dest_path = path + os.sep + self.project_name + os.sep + "train" + os.sep + "RawImgs" + os.sep + "image"
        label_dest_path = path + os.sep + self.project_name + os.sep + "train" + os.sep + "RawImgs" + os.sep + "label"

        file_list = os.listdir(orgpath)
        file_list_lab = os.listdir(labpath)

        print(file_list)
        print(file_list_lab)

        if file_list_lab[0].startswith("ground_truth_"):

            new_file_list_lab = []
            for lab_file in file_list_lab:

                if lab_file.startswith("ground_truth_"):

                    new_lab_file = lab_file[13:]
                    new_file_list_lab.append(new_lab_file)

            if len(file_list) == len(new_file_list_lab):

                file_list.sort()
                new_file_list_lab.sort()

                print(file_list)
                print(new_file_list_lab)

                if file_list == new_file_list_lab:

                    for files in file_list:

                        copyfile(orgpath + os.sep + files, image_dest_path + os.sep + files)
                        copyfile(labpath + os.sep + "ground_truth_" + files, label_dest_path + os.sep + files)

                else:
                    print("File names of raw and ground truth images are not identical")

        else:

            if file_list == file_list_lab:

                for files, lab_files in zip(file_list, file_list_lab):

                    copyfile(orgpath + os.sep + files, image_dest_path + os.sep + files)
                    copyfile(labpath + os.sep + lab_files, label_dest_path + os.sep + lab_files)

            else:
                print("File names of raw and ground truth images are not identical")






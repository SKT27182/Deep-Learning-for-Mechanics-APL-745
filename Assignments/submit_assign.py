#!/home/shailja/.virtualenv/my_env/bin/python3

# Script to create a markdown by taking one picture at a time and below that take the
# text from the text file, repeat this for all the pictures

# Script mus be runned just inside the dir where all the assignment directories are kept

# assuming that each assignment directory contain, figures dir: inside which all the figures are kept and
# discription of each pic is in the discription.md file

# give the assignment directory name as an argument
# python3 to_pdf.py assign_2

import os
import sys
import zipfile
from files.files import ImageToMarkdown
import subprocess




class AssignmentMarkdown:

    def __init__(self, assign_dir, fig_folder="figures", description_file="discrip.txt"):
        """
        Parameters
        ----------

        assign_dir : str
            Path of the assignment file

        fig_folder : str
            Path of the dir where all the plots are saved, default: figures

        description_file : str
            Description file name, default: discription.txt

        """
        self.assign_dir = assign_dir
        # figure folder path relative to Assignment dir
        self.fig_folder = fig_folder
        self.desp_file = description_file

    def pick_line(self):
        """
        Return the list of lines from the text file, separated by ### from the description_file inside the figures dir
        """

        with open(os.path.join(self.assign_dir, self.fig_folder, self.desp_file), "r") as f:
            lines = f.read().split("###")

        return lines[1:]

    def get_probelm_no(self, pic):
        # assuming that pics name are in the form 'figures/0102.png'
        # where first two number tells about the Problem no.
        # and last two about the image number

        return int(pic.split("/")[-1][:2])

    # Returnt the list of pictures path as list
    def get_pics_text_list(self, exclude_fig=["discrip.txt"]):
        """
        Parameters
        ----------

        exclude_fig : list
            List of the plots or files in fig_folder to exclude, default: discription.txt

        Return the list of pictures path as list
        """

        texts = self.pick_line()

        pics = []
        current_prob = 0
        pic_paths = sorted(os.listdir(
            os.path.join(self.assign_dir, self.fig_folder)))

        ind = -1
        for pic in pic_paths:
            if pic not in exclude_fig:
                ind += 1
                if self.get_probelm_no(pic) == current_prob:
                    print(f"Adding : Image {pic} ")
                    pics.append(os.path.join(self.fig_folder, pic))
                else:
                    current_prob += 1

                    print("###################################")
                    print(f"Adding Heading for Problem: {current_prob}")

                    texts.insert(ind, f"Problem: {current_prob}")
                    pics.append("")

                    print(f"Adding : Image {pic} ")
                    pics.append(os.path.join(self.fig_folder, pic))
                    # b/c in image we added twice one blank sting and one pic
                    ind += 1

        return texts, pics

    def create_zip(self, zip_file, exclude_folders, exclude_file):
        try:
            # Create a zip file
            with zipfile.ZipFile(os.path.join(self.assign_dir,zip_file), 'w') as zip_obj:
                # Create a set to keep track of added files
                added_files = set()
                # Iterate over all the files in directory
                for foldername, subfolders, files in os.walk(self.assign_dir):
                    # Exclude the subfolders
                    subfolders[:] = [
                        subfolder for subfolder in subfolders if subfolder not in exclude_folders]
                    for file in files:
                        if file not in exclude_file:
                            # Create complete filepath of file in directory
                            file_path = os.path.join(foldername, file)
                            # Check if file has already been added
                            if file_path not in added_files:
                                # Add file to zip
                                zip_obj.write(file_path)
                                # Add file to the set
                                added_files.add(file_path)
        except FileNotFoundError:
            print(f"{self.assign_dir} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")


    def create_pdf(self):
        os.chdir(self.assign_dir)
        files = os.listdir()
        for file in files:
            if file.endswith("Figures.md"):
                output_file = "Figures.pdf"
                if os.path.exists(output_file):
                    answer = input(f'{output_file} already exists, do you want to overwrite it? (y/n) ')
                    if answer.lower() != 'y':
                        continue
                command = f'pandoc --pdf-engine=xelatex -V geometry:"left=3cm,right=3cm,top=1.5cm,bottom=1.5cm" {file} -o {output_file}'
                subprocess.run(command, shell=True)
            elif file.endswith(".ipynb"):
                output_file = os.path.splitext(file)[0] + '.pdf'
                if os.path.exists(output_file):
                    answer = input(f'{output_file} already exists, do you want to overwrite it? (y/n) ')
                    if answer.lower() != 'y':
                        continue
                os.system(f"jupyter nbconvert --to pdf {file} --output {output_file}")
        os.chdir("..")



def main(assignment, zip_name, exclude_folders, exclude_files, mark_down_fig, fig_folder, template=None):

    # creating an instance of AssignmentMarkdown class
    assing = AssignmentMarkdown(assign_dir=assignment, fig_folder=fig_folder)

    # get the list of texts and pics paths
    texts, pics = assing.get_pics_text_list()

    # Creating an instance of ImageToMarkdown class which will make a markdwon file if not exists otherwise append
    to_markdown = ImageToMarkdown(
        texts_list=texts, images_list=pics, markdown_file=os.path.join(assignment, mark_down_fig))
    to_markdown.convert(template)

    # creating a pdf file
    assing.create_pdf()

    assing.create_zip(zip_file=zip_name,
                      exclude_folders=exclude_folders, exclude_file=exclude_files)



assignment =  sys.argv[1]
zip_name = os.environ.get("zip_name")

# figure dir path

mark_down_fig = "Figures.md"
fig_folder = "figures"

exclude_folders = [
    ".ipynb_checkpoints",
    "__pycache__",
    ".pytest_cache",
    "figures",
    "data"
]

exclude_files = [
    "submit_assn.py",
    "test.py",
    "test.ipynb",
    "Figures.md",
    zip_name
]


assignment_num = assignment[-1]

template = f"""---
title: "Assignment {assignment_num} Plots"
author: "Shailja Kant Tiwari"
header-includes:
  - \\usepackage{{amssymb,amsmath,geometry}}
  - \\setmainfont{{TeX Gyre Schola}}
  - \\setmathfont{{TeX Gyre Schola Math}}
output:
  pdf_document
---

# Assignment {assignment_num}

"""



main(assignment=assignment, zip_name=zip_name,
     exclude_folders=exclude_folders, exclude_files=exclude_files, 
     mark_down_fig=mark_down_fig, fig_folder=fig_folder, template=template)
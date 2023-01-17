#!/home/shailja/.virtualenv/my_env/bin/python3

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import subprocess
import zipfile


# take a assingment no. from the command line
assn = sys.argv[1]

# specify the directory where the figures are located
img_dir = os.path.join(assn, 'figures/')
# check if the directory exists
if not os.path.exists(img_dir):
    print('The directory {} does not exist'.format(img_dir))
    sys.exit(1)




# create a pdf file with the name 'figures.pdf' also covert the .ipynb file to .pdf
fig_save = os.path.join(assn, 'Figures.pdf')
with PdfPages(fig_save) as pdf:
    for filename in sorted(os.listdir(img_dir)):
        # check if the file is a png or pdf file
        if filename.endswith('.png') or filename.endswith('.pdf'):
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(plt.imread(img_dir + filename))
            plt.axis('off')
            # Get the file name without the extension
            file_name = os.path.splitext(filename)[0]
            # Replace _ with space
            file_name = file_name.replace("_", " ")
            # Add the heading
            plt.suptitle(file_name)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)  


# now make a zip file of the assignment
# create a zip file with the name '2021PHS7218.zip'
zip_file = '2021PHS7218.zip'
with zipfile.ZipFile(os.path.join(assn,zip_file), 'w') as zip_obj:
    for foldername, subfolders, filenames in os.walk(assn):
        for filename in filenames:
            # check if the file is a pdf or ipynb file
            if filename.endswith('.pdf') or filename.endswith('.ipynb'):
                file_path = os.path.join(foldername, filename)
                relative_path = os.path.relpath(file_path, assn)
                zip_obj.write(file_path, relative_path)

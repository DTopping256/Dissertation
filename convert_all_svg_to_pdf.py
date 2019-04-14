import os
import subprocess
from tqdm import tqdm

filenames_converted = list(map(lambda x: x[:-4], filter(lambda x: ".pdf" in x, os.listdir(os.getcwd()))))
files_to_convert = list(filter(lambda x: x[-4:] == ".svg" and x[:-4] not in filenames_converted, os.listdir(os.getcwd())))
print(len(files_to_convert), "new .svg files found.")

print("Converting all svg files into pdf...")
for svg in tqdm(files_to_convert):
  pdfname = svg[:-4]+".pdf"
  cmd_str = 'inkscape --file="{}" --export-area-drawing --without-gui --export-pdf="{}"'.format(svg, pdfname)
  subprocess.run(cmd_str)
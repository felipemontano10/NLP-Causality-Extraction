#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date Created: Feb - 16 - 2020
Date Modified: feb - 16 - 2020
Last modified: Felipe Montano
"""

import pandas as pd
import sys
import io
import glob, os
import re
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams


#List of pdf documents
pdf = []
os.chdir("/......./Task 1. Hypotheses retrieval/Doc PDFs")
for file in glob.glob("*.pdf"):
    pdf.append(file)

#Def the function
def pdfparser(data):
    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data_out =  retstr.getvalue()
    b = data[:-4]+".txt"
    f = open(b, 'w')
    f.write(data_out)
    f.close
    # return data
  
#Get the .txt files
for i in pdf:
    pdfparser(i)

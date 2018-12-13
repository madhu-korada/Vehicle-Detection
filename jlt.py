# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 17:50:25 2018

@author: madhu
"""
"""import numpy as np
a = np.transpose(np.array([1, 2, 3]))
a
b = np.transpose(np.array([2, 3]))
c = np.append(a,b,axis=1)



from openpyxl import Workbook
nfc_east = ('DAL', 'WAS', 'PHI', 'NYG')
wb = Workbook()
ws = wb.active
for row, i in enumerate(nfc_east):
    column_cell = 'A'
    ws[column_cell+str(row+2)] = str(i)

wb.save("row_creation_loop.xlsx")"""

from pandas import Series, ExcelWriter

writer = ExcelWriter('output.xls')
for counter in range(10):
   sheet_name = 'Sheet%s' % counter
   Series.to_frame(name = 'exam').to_excel(writer, sheet_name=sheet_name)

writer.save()





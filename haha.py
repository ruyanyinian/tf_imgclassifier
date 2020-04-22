import re

# 'F:\\DL_Data\\Animal\\test_data\\elephant\\IMG_0796\\src_images\\000036.jpg 0.61686623 class 4 punguin GT 0 elephant'

string = r"F:\\DL_Data\\Animal\\test_data\\elephant\\IMG_0796\\src_images\\000036.jpg 0.61686623 class 4 punguin GT 0 elephant"
print(re.findall(r"\d", string))

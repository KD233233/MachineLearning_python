# #coding:gbk
# import os
# from PIL import Image
#
# class Pgm(object):
#     def is_pgm_file(self,in_path):
#         if not os.path.isfile(in_path):
#             return False
#         if in_path is not str and not in_path.endswith(".pgm"):
#             return False
#         return True
#     def pgm2jpg(self,in_path,out_path):
#         '''保存图片'''
#         if not self.is_pgm_file(in_path):
#             raise Exception(f'{in_path}不是一个PGM文件')
#         img = Image.open(in_path)
#         img.save(out_path)
#
#
